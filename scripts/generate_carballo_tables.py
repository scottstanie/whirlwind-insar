#!/usr/bin/env python3
"""Generate approximate Carballo/Fieguth probability lookup tables.

This script documents the offline model behind the old ``ww-orig`` cost
tables. It is a reconstruction, not a bit-for-bit recovery of the original
generator.

The generated tables store probability fields, not final integer costs:

    p0(alpha_hat, gamma_hat, L) = P(delta_k = 0)
    p1(alpha_hat, gamma_hat, L) = P(delta_k = +1)

Runtime cost is then:

    cost = int(100 * max(-log(p1 / p0), 0))

The reconstruction follows Carballo and Fieguth (2000):

1. Evaluate Lee's multilook interferometric phase PDF.
2. Convolve the phase-noise PDF with its reversed copy to obtain a
   phase-gradient noise PDF on [-2*pi, 2*pi].
3. Integrate that PDF over the residual intervals for delta_k = 0 and +1.
4. Optionally marginalize the true slope alpha through the Gaussian slope-error
   approximation in Carballo equation (15).

Important limitations:

* The saved ``ww-orig`` tables vary with nlooks even at sample coherence 0. A
  pure true-coherence Lee model cannot reproduce that exactly. The old generator
  likely also used the Bayesian true-coherence/sample-coherence integration
  sketched in Geoff's notes.
* The saved tables only cover nlooks in [1, 80] and use fill_value=nan. New
  production tables should either extend that axis or explicitly clamp/fallback.
* This implementation is written for clarity and reproducibility, not speed.

Example:

    python scripts/generate_carballo_tables.py \\
        --out-dir /tmp/carballo_tables \\
        --compare-dir /Users/staniewi/repos/whirlwind/src/whirlwind_orig

To emit raw little-endian f32 blobs compatible with the current Rust
``spline_lut.rs``:

    python scripts/generate_carballo_tables.py \\
        --out-dir /tmp/carballo_tables --write-rust-bins

To regenerate the embedded Rust blobs (the only way they are consumed - the Rust
reads the five `.bin` files baked into the binary at build time via
``include_bytes!`` in ``cost/spline_lut.rs``; rebuild the wheel afterwards):

    python scripts/generate_carballo_tables.py \\
        --out-dir /tmp/carballo_tables \\
        --write-rust-bins \\
        --rust-bin-dir crates/whirlwind-core/src/cost
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammaln, hyp2f1


TAU = 2.0 * math.pi
PI = math.pi


def lee_phase_pdf(
    phi: np.ndarray,
    gamma: float,
    nlooks: float,
    *,
    gamma_eps: float,
) -> np.ndarray:
    """Lee 1994 multilook interferometric phase PDF.

    This mirrors the stable form used in ``crates/whirlwind-core/src/cost``.
    ``gamma`` is clipped below 1 to avoid the singular delta-function limit.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if gamma <= 0.0:
        return np.full_like(phi, 1.0 / TAU, dtype=np.float64)

    g = float(np.clip(gamma, 0.0, 1.0 - gamma_eps))
    n = float(nlooks)
    g2 = g * g
    beta = g * np.cos(phi)
    b2 = beta * beta
    one_minus_b2 = np.maximum(1.0 - b2, 1e-300)
    one_minus_g2 = max(1.0 - g2, 1e-300)

    term1 = np.empty_like(phi, dtype=np.float64)
    direct = b2 < 0.5
    term1[direct] = (
        one_minus_g2**n / TAU * hyp2f1(n, 1.0, 0.5, b2[direct])
    )

    log_pref = n * math.log(one_minus_g2) - (n + 0.5) * np.log(
        one_minus_b2[~direct]
    )
    term1[~direct] = (
        np.exp(np.clip(log_pref, -745.0, 700.0))
        / TAU
        * hyp2f1(0.5 - n, -0.5, 0.5, b2[~direct])
    )

    log_term2 = (
        gammaln(n + 0.5)
        - gammaln(n)
        + n * math.log(one_minus_g2)
        - (n + 0.5) * np.log(one_minus_b2)
    )
    term2 = (
        np.sign(beta)
        * np.abs(beta)
        * np.exp(np.clip(log_term2, -745.0, 700.0))
        / (2.0 * math.sqrt(PI))
    )

    pdf = np.nan_to_num(term1 + term2, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(pdf, 0.0)


def gradient_noise_cdf(
    gamma: float,
    nlooks: float,
    *,
    phase_samples: int,
    gamma_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CDF of phase-gradient noise on approximately [-2*pi, 2*pi]."""
    phi = np.linspace(-PI, PI, phase_samples, endpoint=False, dtype=np.float64)
    phi += PI / phase_samples
    dphi = TAU / phase_samples

    pdf = lee_phase_pdf(phi, gamma, nlooks, gamma_eps=gamma_eps)
    norm = pdf.sum() * dphi
    if norm <= 0.0 or not np.isfinite(norm):
        pdf = np.full_like(phi, 1.0 / TAU)
    else:
        pdf = pdf / norm

    grad_pdf = fftconvolve(pdf, pdf[::-1], mode="full") * dphi
    x = np.arange(-(phase_samples - 1), phase_samples, dtype=np.float64) * dphi
    grad_pdf = np.maximum(np.nan_to_num(grad_pdf, nan=0.0), 0.0)
    grad_pdf /= max(grad_pdf.sum() * dphi, 1e-300)

    cdf = np.concatenate(
        [[0.0], np.cumsum((grad_pdf[:-1] + grad_pdf[1:]) * 0.5 * dphi)]
    )
    cdf /= max(cdf[-1], 1e-300)
    return x, cdf


def interp_cdf(x: np.ndarray, cdf: np.ndarray, value: np.ndarray) -> np.ndarray:
    return np.interp(value, x, cdf, left=0.0, right=1.0)


def raw_residual_probability(
    x: np.ndarray,
    cdf: np.ndarray,
    alpha: np.ndarray,
    residual: int,
) -> np.ndarray:
    """Probability for residual 0 or +1 for true slope alpha.

    ``x`` is phase-gradient noise, so the unwrapped gradient is alpha + x.
    """
    if residual == 0:
        lower = -PI - alpha
        upper = PI - alpha
    elif residual == 1:
        lower = PI - alpha
        upper = np.full_like(alpha, 2.0 * PI, dtype=np.float64)
    else:
        raise ValueError("only residual 0 and +1 are generated")

    prob = interp_cdf(x, cdf, upper) - interp_cdf(x, cdf, lower)
    return np.maximum(prob, 0.0)


def slope_error_sigma(sample_corr: float, window_pixels: int) -> float:
    """Gaussian slope-error sigma from Carballo equation (15).

    ``window_pixels`` is the number of pixels in the square slope-estimation
    window, not the side length. The paper's example uses 5x5 => 25.
    Whirlwind's runtime smoothing uses 7x7 => 49. The default CLI value is 64
    because it empirically fits the bundled tables better for this simplified
    reconstruction.
    """
    gamma = float(sample_corr)
    if gamma <= 1e-6:
        return PI / math.sqrt(3.0)

    nwin = int(window_pixels)
    rho = 0.0
    for k in range(2, nwin + 1):
        log_abs = (
            gammaln(nwin + 1)
            - gammaln(nwin - k + 1)
            - gammaln(k + 1)
            - nwin * gamma * (k - 1.0) / k
        )
        rho += ((-1.0) ** k) * math.exp(min(log_abs, 700.0))
    rho = float(np.clip(rho / nwin, 0.0, 1.0))

    var = rho * PI * PI / 3.0 + (1.0 - rho) * 6.0 / (
        gamma * nwin * (nwin - 1.0)
    )
    return math.sqrt(max(var, 0.0))


def marginalized_probability(
    x: np.ndarray,
    cdf: np.ndarray,
    alpha_hat: float,
    gamma_hat: float,
    residual: int,
    *,
    alpha_quad: int,
    slope_tail_sigma: float,
    slope_window_pixels: int,
    use_slope_marginalization: bool,
) -> float:
    if not use_slope_marginalization:
        alpha = np.array([alpha_hat], dtype=np.float64)
        return float(raw_residual_probability(x, cdf, alpha, residual)[0])

    sigma = slope_error_sigma(gamma_hat, slope_window_pixels)
    if sigma < 1e-6:
        alpha = np.array([alpha_hat], dtype=np.float64)
        return float(raw_residual_probability(x, cdf, alpha, residual)[0])

    half_width = slope_tail_sigma * sigma
    alpha = np.linspace(
        alpha_hat - half_width, alpha_hat + half_width, alpha_quad, dtype=np.float64
    )
    weight = np.exp(-0.5 * ((alpha_hat - alpha) / sigma) ** 2)
    prob = raw_residual_probability(x, cdf, alpha, residual)
    return float(np.trapezoid(prob * weight, alpha) / np.trapezoid(weight, alpha))


def default_axes(
    phase_count: int,
    corr_count: int,
    nlooks_count: int,
    nlooks_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.linspace(-PI, PI, phase_count, dtype=np.float64)
    corr = np.linspace(0.0, 1.0, corr_count, dtype=np.float64)
    nlooks = np.geomspace(1.0, nlooks_max, nlooks_count).astype(np.float64)
    return phase, corr, nlooks


def generate_tables(args: argparse.Namespace) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    phase, corr, nlooks = default_axes(
        args.phase_count, args.corr_count, args.nlooks_count, args.nlooks_max
    )
    p0 = np.empty((phase.size, corr.size, nlooks.size), dtype=np.float64)
    p1 = np.empty_like(p0)

    total = corr.size * nlooks.size
    started = time.time()
    done = 0
    for il, looks in enumerate(nlooks):
        for ic, coherence in enumerate(corr):
            x, cdf = gradient_noise_cdf(
                float(coherence),
                float(looks),
                phase_samples=args.phase_samples,
                gamma_eps=args.gamma_eps,
            )
            for ia, alpha_hat in enumerate(phase):
                p0[ia, ic, il] = marginalized_probability(
                    x,
                    cdf,
                    float(alpha_hat),
                    float(coherence),
                    0,
                    alpha_quad=args.alpha_quad,
                    slope_tail_sigma=args.slope_tail_sigma,
                    slope_window_pixels=args.slope_window_pixels,
                    use_slope_marginalization=not args.no_slope_marginalization,
                )
                p1[ia, ic, il] = marginalized_probability(
                    x,
                    cdf,
                    float(alpha_hat),
                    float(coherence),
                    1,
                    alpha_quad=args.alpha_quad,
                    slope_tail_sigma=args.slope_tail_sigma,
                    slope_window_pixels=args.slope_window_pixels,
                    use_slope_marginalization=not args.no_slope_marginalization,
                )

            done += 1
            if args.verbose:
                elapsed = time.time() - started
                print(
                    f"[{done:3d}/{total}] L={looks:.4g} gamma={coherence:.3f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    # Match the old file's small negative roundoff behavior poorly; for new
    # tables, probabilities should remain finite and non-negative.
    p0 = np.clip(p0, 0.0, 1.0)
    p1 = np.clip(p1, 0.0, 1.0)
    return phase, corr, nlooks, p0, p1


def save_rgi_npz(path: Path, grid: tuple[np.ndarray, ...], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        grid_0=grid[0],
        grid_1=grid[1],
        grid_2=grid[2],
        values=values,
        method=np.array("linear"),
        bounds_error=np.array(False),
        fill_value=np.array(np.nan),
    )


def write_rust_bins(
    out_dir: Path,
    phase: np.ndarray,
    corr: np.ndarray,
    nlooks: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "carballo_grid_phase.bin": phase,
        "carballo_grid_corr.bin": corr,
        "carballo_grid_nlooks.bin": nlooks,
        "carballo_p0.bin": p0,
        "carballo_p1.bin": p1,
    }
    for name, arr in files.items():
        (out_dir / name).write_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def load_old_tables(compare_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p0_file = compare_dir / "carballo-pdf-0-spline.npz"
    p1_file = compare_dir / "carballo-pdf-1-spline.npz"
    old0 = np.load(p0_file, allow_pickle=False)
    old1 = np.load(p1_file, allow_pickle=False)
    return (
        old0["grid_0"],
        old0["grid_1"],
        old0["grid_2"],
        old0["values"],
        old1["values"],
    )


def report_comparison(
    generated: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    compare_dir: Path,
) -> None:
    phase, corr, nlooks, p0, p1 = generated
    old_phase, old_corr, old_nlooks, old_p0, old_p1 = load_old_tables(compare_dir)
    if not (
        np.array_equal(phase, old_phase)
        and np.array_equal(corr, old_corr)
        and np.allclose(nlooks, old_nlooks)
        and p0.shape == old_p0.shape
    ):
        print("\nComparison skipped: generated axes do not match saved table axes.")
        return

    def metrics(name: str, got: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> None:
        diff = got[mask] - ref[mask]
        print(
            f"  {name:<12} rmse={np.sqrt(np.mean(diff * diff)):.6g} "
            f"max_abs={np.max(np.abs(diff)):.6g} mean={np.mean(diff):.6g}"
        )

    finite = np.isfinite(old_p0) & np.isfinite(old_p1)
    interior = finite & (corr[None, :, None] > 0.0) & (corr[None, :, None] < 1.0)
    print(f"\nComparison against {compare_dir}:")
    metrics("p0 all", p0, old_p0, finite)
    metrics("p1 all", p1, old_p1, finite)
    metrics("p0 interior", p0, old_p0, interior)
    metrics("p1 interior", p1, old_p1, interior)

    eps = 1e-30
    cost_got = -np.log(np.maximum(p1, eps) / np.maximum(p0, eps))
    cost_ref = -np.log(np.maximum(old_p1, eps) / np.maximum(old_p0, eps))
    cost_mask = finite & np.isfinite(cost_ref) & np.isfinite(cost_got)
    metrics("llr all", cost_got, cost_ref, cost_mask)
    metrics("llr interior", cost_got, cost_ref, cost_mask & interior)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=Path("/Users/staniewi/repos/whirlwind/src/whirlwind_orig"),
        help="Directory containing saved ww-orig carballo-pdf-*-spline.npz tables.",
    )
    parser.add_argument("--write-rust-bins", action="store_true")
    parser.add_argument(
        "--rust-bin-dir",
        type=Path,
        help=(
            "Directory for Rust .bin files. Defaults to --out-dir when "
            "--write-rust-bins is set."
        ),
    )
    parser.add_argument("--phase-count", type=int, default=31)
    parser.add_argument("--corr-count", type=int, default=11)
    parser.add_argument("--nlooks-count", type=int, default=11)
    parser.add_argument("--nlooks-max", type=float, default=80.0)
    parser.add_argument(
        "--phase-samples",
        type=int,
        default=8192,
        help="Samples for Lee PDF and FFT convolution.",
    )
    parser.add_argument(
        "--alpha-quad",
        type=int,
        default=1001,
        help="Quadrature samples for slope-error marginalization.",
    )
    parser.add_argument(
        "--slope-window-pixels",
        type=int,
        default=64,
        help="N in Carballo eq. 15. Try 25, 49, 64 when fitting old tables.",
    )
    parser.add_argument("--slope-tail-sigma", type=float, default=8.0)
    parser.add_argument("--gamma-eps", type=float, default=1e-6)
    parser.add_argument("--no-slope-marginalization", action="store_true")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.rust_bin_dir is not None and not args.write_rust_bins:
        parser.error("--rust-bin-dir requires --write-rust-bins")
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    phase, corr, nlooks, p0, p1 = generate_tables(args)
    grid = (phase, corr, nlooks)

    save_rgi_npz(args.out_dir / "carballo-pdf-0-spline.npz", grid, p0)
    save_rgi_npz(args.out_dir / "carballo-pdf-1-spline.npz", grid, p1)
    if args.write_rust_bins:
        rust_bin_dir = args.rust_bin_dir or args.out_dir
        write_rust_bins(rust_bin_dir, phase, corr, nlooks, p0, p1)

    print(f"Wrote tables to {args.out_dir} in {time.time() - started:.1f}s")
    if args.write_rust_bins:
        print(f"Wrote Rust LUT blobs to {rust_bin_dir}")
    print(f"  p0 range [{p0.min():.6g}, {p0.max():.6g}]")
    print(f"  p1 range [{p1.min():.6g}, {p1.max():.6g}]")

    if not args.no_compare and args.compare_dir.exists():
        report_comparison((phase, corr, nlooks, p0, p1), args.compare_dir)


if __name__ == "__main__":
    main()
