#!/usr/bin/env python3
"""Generate/export ww-orig Carballo/Touzi probability lookup tables.

This script has two deliberately separate modes:

1. ``--source-table-dir``: byte-parity export mode. It reads the surviving
   ww-orig ``carballo-pdf-{0,1}-spline`` tables (``.npz`` or the first-commit
   ``.pkl`` RegularGridInterpolator pickles) and writes the five little-endian
   ``f32`` blobs embedded by Rust. This is the supported way to recreate
   ``crates/whirlwind-core/src/cost/carballo_*.bin`` from preserved artifacts.

2. analytic reconstruction mode (default): computes a readable reference
   implementation of the documented model from Geoff's unwrapping notes:

    f(phi_N | gamma_hat, L) = (1/C) * integral_0^1
            f_Lee(phi_N | gamma, L) * f_Touzi(gamma_hat | gamma, L) d gamma   (1)

i.e. the Lee (1994) multilook phase-noise PDF, with TWO unknowns marginalized
under uniform priors:

  * the local signal **slope** alpha               -> Carballo (1994) eq. (15)
  * the **true coherence** gamma given the sample
    coherence gamma_hat, via the Touzi (1999)
    coherence-estimator density f(gamma_hat|gamma)  -> the part beyond Carballo

The phase-noise-difference PDF is the self-convolution of (1) (IID assumption),
and the residual probabilities p0 = P(Delta k = 0) and p1 = P(Delta k = +1)
integrate that difference PDF over the wrap intervals. Runtime cost is then:

    cost = int(100 * max(-log(p1 / p0), 0))

==============================================================================
IMPORTANT - THIS IS A MODEL REFERENCE, NOT A BIT-FOR-BIT REGENERATOR.
==============================================================================

The ORIGINAL analytic generator that produced the shipping ``.npz``/``.bin``
tables is NOT preserved in any repo (only its OUTPUT, its CONSUMER
``whirlwind/src/whirlwind_orig/_cost.py``, and Geoff's notes survive). The
analytic mode reconstructs the *documented math* and reproduces the qualitative
fingerprint of the coherence marginalization (the tables vary with ``L`` even
at ``gamma_hat = 0``, which a pure Lee/slope model cannot do), but it does
**not** reproduce the shipping tables numerically. Run with ``--compare-dir``
to print the residual RMSE and the gamma_hat=0 fingerprint.

The surviving saved tables are therefore the AUTHORITATIVE source for byte
reproduction; the analytic mode is for documentation, diagnostics, and future
model work. ``--write-rust-bins`` is guarded: it refuses to overwrite an
existing embedded-blob directory with non-identical output unless you pass
``--allow-overwrite-embedded``.

Example - recreate the embedded Rust blobs from saved ww-orig tables:

    python scripts/generate_carballo_tables.py \\
        --source-table-dir /Users/staniewi/repos/whirlwind/src/whirlwind_orig \\
        --out-dir /tmp/carballo_tables \\
        --write-rust-bins \\
        --verify-rust-bin-dir crates/whirlwind-core/src/cost

Example - inspect the model and its gap vs the shipping tables:

    python scripts/generate_carballo_tables.py \\
        --out-dir /tmp/carballo_tables \\
        --compare-dir /Users/staniewi/repos/whirlwind/src/whirlwind_orig

To turn OFF the coherence marginalization (slope-only, pure-Carballo baseline):

    python scripts/generate_carballo_tables.py \\
        --out-dir /tmp/carballo_tables --no-coherence-marginalization
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle
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
    """Lee 1994 multilook interferometric phase PDF at TRUE coherence ``gamma``.

    This mirrors the stable form used in ``crates/whirlwind-core/src/cost``.
    ``gamma`` is clipped below 1 to avoid the singular delta-function limit.
    At gamma <= 0 the phase is uniform (1/2pi), independent of ``nlooks``.
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
    term1[direct] = one_minus_g2**n / TAU * hyp2f1(n, 1.0, 0.5, b2[direct])

    log_pref = n * math.log(one_minus_g2) - (n + 0.5) * np.log(one_minus_b2[~direct])
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


def coherence_posterior_weights(
    gamma_hat: float,
    nlooks: float,
    gamma_nodes: np.ndarray,
) -> np.ndarray:
    """Posterior shape over TRUE coherence given the SAMPLE coherence (Touzi 1999).

    The Touzi (1999) coherence-estimate density is

        f(gamma_hat | gamma, L) = 2(L-1)(1-gamma^2)^L
              * gamma_hat (1-gamma_hat^2)^(L-2) * 2F1(L, L; 1; gamma^2 gamma_hat^2)

    With a uniform prior on the true coherence, the posterior p(gamma|gamma_hat,L)
    is proportional to this. The ``gamma_hat``/``L``-only prefactors are constant
    in ``gamma`` and cancel in the gamma-normalization, leaving the gamma-dependent
    shape

        w(gamma) ~ (1-gamma^2)^L * 2F1(L, L; 1; gamma^2 gamma_hat^2)

    which is well-defined at gamma_hat = 0 -> (1-gamma^2)^L (i.e. it concentrates
    near gamma = 0 as L grows). Returned weights are unnormalized.
    """
    g = np.asarray(gamma_nodes, dtype=np.float64)
    L = float(nlooks)
    z = (g * gamma_hat) ** 2
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        w = (1.0 - g * g) ** L * hyp2f1(L, L, 1.0, z)
    w = np.where(np.isfinite(w), w, 0.0)
    return np.maximum(w, 0.0)


def phase_pdf_given_sample_coherence(
    phi: np.ndarray,
    gamma_hat: float,
    nlooks: float,
    *,
    gamma_eps: float,
    gamma_quad: int,
    marginalize_coherence: bool,
) -> np.ndarray:
    """Phase-noise PDF conditioned on the SAMPLE coherence, eq. (1).

    When ``marginalize_coherence`` is False this is just the Lee PDF with the
    sample coherence plugged in as if it were the true coherence (pure-Carballo
    baseline).
    """
    if not marginalize_coherence:
        return lee_phase_pdf(phi, gamma_hat, nlooks, gamma_eps=gamma_eps)

    g_nodes = np.linspace(0.0, 1.0 - gamma_eps, gamma_quad)
    w = coherence_posterior_weights(gamma_hat, nlooks, g_nodes)
    wsum = np.trapezoid(w, g_nodes)
    if wsum <= 0.0 or not np.isfinite(wsum):
        return lee_phase_pdf(phi, gamma_hat, nlooks, gamma_eps=gamma_eps)

    acc = np.zeros_like(phi, dtype=np.float64)
    for gi, wi in zip(g_nodes, w):
        if wi <= 0.0:
            continue
        acc += wi * lee_phase_pdf(phi, float(gi), nlooks, gamma_eps=gamma_eps)
    return acc / wsum


def gradient_noise_cdf(
    gamma_hat: float,
    nlooks: float,
    *,
    phase_samples: int,
    gamma_eps: float,
    gamma_quad: int,
    marginalize_coherence: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CDF of phase-gradient (difference) noise on approximately [-2pi, 2pi]."""
    phi = np.linspace(-PI, PI, phase_samples, endpoint=False, dtype=np.float64)
    phi += PI / phase_samples
    dphi = TAU / phase_samples

    pdf = phase_pdf_given_sample_coherence(
        phi,
        gamma_hat,
        nlooks,
        gamma_eps=gamma_eps,
        gamma_quad=gamma_quad,
        marginalize_coherence=marginalize_coherence,
    )
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

    var = rho * PI * PI / 3.0 + (1.0 - rho) * 6.0 / (gamma * nwin * (nwin - 1.0))
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


def generate_tables(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                gamma_quad=args.gamma_quad,
                marginalize_coherence=not args.no_coherence_marginalization,
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


def load_rgi_npz(path: Path) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    method = str(data["method"])
    bounds_error = bool(data["bounds_error"])
    fill_value = float(data["fill_value"])
    if method != "linear" or bounds_error or not np.isnan(fill_value):
        raise SystemExit(
            f"{path} is not the expected linear, bounds_error=False, fill_value=nan "
            "RegularGridInterpolator export."
        )
    grid = (data["grid_0"], data["grid_1"], data["grid_2"])
    return grid, data["values"]


def load_rgi_pickle(path: Path) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    method = getattr(obj, "method", "linear")
    bounds_error = bool(getattr(obj, "bounds_error", False))
    fill_value = float(getattr(obj, "fill_value", np.nan))
    if method != "linear" or bounds_error or not np.isnan(fill_value):
        raise SystemExit(
            f"{path} is not the expected linear, bounds_error=False, fill_value=nan "
            "RegularGridInterpolator pickle."
        )
    if not hasattr(obj, "grid") or not hasattr(obj, "values"):
        raise SystemExit(f"{path} does not look like a RegularGridInterpolator pickle.")
    grid = tuple(np.asarray(axis, dtype=np.float64) for axis in obj.grid)
    if len(grid) != 3:
        raise SystemExit(f"{path} has {len(grid)} grid axes; expected 3.")
    return grid, np.asarray(obj.values, dtype=np.float64)


def load_saved_tables(
    table_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load surviving ww-orig spline tables from .npz or first-commit .pkl files."""
    npz0 = table_dir / "carballo-pdf-0-spline.npz"
    npz1 = table_dir / "carballo-pdf-1-spline.npz"
    pkl0 = table_dir / "carballo-pdf-0-spline.pkl"
    pkl1 = table_dir / "carballo-pdf-1-spline.pkl"

    if npz0.exists() and npz1.exists():
        grid0, p0 = load_rgi_npz(npz0)
        grid1, p1 = load_rgi_npz(npz1)
    elif pkl0.exists() and pkl1.exists():
        grid0, p0 = load_rgi_pickle(pkl0)
        grid1, p1 = load_rgi_pickle(pkl1)
    else:
        raise SystemExit(
            f"{table_dir} must contain carballo-pdf-0/1-spline.npz "
            "or carballo-pdf-0/1-spline.pkl."
        )

    for i, (a, b) in enumerate(zip(grid0, grid1, strict=True)):
        if not np.array_equal(a, b):
            raise SystemExit(f"p0/p1 grid axis {i} differs in {table_dir}.")

    phase, corr, nlooks = (np.asarray(axis, dtype=np.float64) for axis in grid0)
    expected_shape = (phase.size, corr.size, nlooks.size)
    if p0.shape != expected_shape or p1.shape != expected_shape:
        raise SystemExit(
            f"Saved table shape mismatch in {table_dir}: expected {expected_shape}, "
            f"got p0={p0.shape}, p1={p1.shape}."
        )
    return (
        phase,
        corr,
        nlooks,
        np.ascontiguousarray(p0, dtype=np.float64),
        np.ascontiguousarray(p1, dtype=np.float64),
    )


def rust_bin_payloads(
    phase: np.ndarray,
    corr: np.ndarray,
    nlooks: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
) -> dict[str, bytes]:
    files = {
        "carballo_grid_phase.bin": phase,
        "carballo_grid_corr.bin": corr,
        "carballo_grid_nlooks.bin": nlooks,
        "carballo_p0.bin": p0,
        "carballo_p1.bin": p1,
    }
    return {
        name: np.ascontiguousarray(arr, dtype="<f4").tobytes()
        for name, arr in files.items()
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _looks_like_embedded_blob_dir(out_dir: Path) -> bool:
    """True if ``out_dir`` already holds the authoritative embedded LUT blobs."""
    return (out_dir / "carballo_p0.bin").exists() and (
        out_dir / "carballo_p1.bin"
    ).exists()


def write_rust_bins(
    out_dir: Path,
    phase: np.ndarray,
    corr: np.ndarray,
    nlooks: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    allow_overwrite_embedded: bool,
) -> None:
    payloads = rust_bin_payloads(phase, corr, nlooks, p0, p1)
    if _looks_like_embedded_blob_dir(out_dir) and not allow_overwrite_embedded:
        diffs = [
            name
            for name, data in payloads.items()
            if not (out_dir / name).exists() or (out_dir / name).read_bytes() != data
        ]
        if not diffs:
            # Re-exporting the authoritative saved tables should be a metadata no-op.
            return
        raise SystemExit(
            f"Refusing to overwrite existing LUT blobs in {out_dir} with "
            f"non-identical output ({', '.join(diffs)}).\n"
            "If you really mean to replace the shipping cost model, re-validate "
            "parity end to end and pass "
            "--allow-overwrite-embedded."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        path = out_dir / name
        if path.exists() and path.read_bytes() == data:
            continue
        path.write_bytes(data)


def load_old_tables(
    compare_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_saved_tables(compare_dir)


def report_rust_bin_parity(
    generated: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rust_bin_dir: Path,
) -> None:
    phase, corr, nlooks, p0, p1 = generated
    payloads = rust_bin_payloads(phase, corr, nlooks, p0, p1)
    print(f"\nRust .bin parity against {rust_bin_dir}:")
    all_match = True
    for name, data in payloads.items():
        path = rust_bin_dir / name
        if not path.exists():
            print(f"  {name:<28} MISSING")
            all_match = False
            continue
        ref = path.read_bytes()
        match = ref == data
        all_match &= match
        status = "OK" if match else "DIFF"
        print(f"  {name:<28} {status}  sha256={sha256_bytes(data)}")
    if all_match:
        print("  all five blobs match byte-for-byte")


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

    # Fingerprint of the coherence marginalization: at sample coherence 0 the
    # tables vary with L (a pure Lee/slope model would be L-independent there).
    ia = int(np.argmin(np.abs(phase - 0.0)))
    print("\n  Fingerprint  p1(alpha~0, gamma_hat=0) across nlooks:")
    print("    nlooks:", np.array2string(np.round(nlooks, 2), max_line_width=200))
    print(
        "    shipping:",
        np.array2string(np.round(old_p1[ia, 0, :], 4), max_line_width=200),
    )
    print(
        "    this run:", np.array2string(np.round(p1[ia, 0, :], 4), max_line_width=200)
    )
    print(
        "  (shipping shows a strong L-dependence from the Touzi coherence "
        "marginalization; this reference reproduces its direction, not its "
        "magnitude - the original generator is not preserved.)"
    )


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
    parser.add_argument(
        "--source-table-dir",
        type=Path,
        help=(
            "Read authoritative saved ww-orig carballo-pdf-*-spline tables "
            "(.npz or first-commit .pkl) instead of computing the analytic "
            "reconstruction. Use this mode to recreate the embedded Rust blobs."
        ),
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
    parser.add_argument(
        "--verify-rust-bin-dir",
        type=Path,
        help="Compare this run's would-be Rust blobs byte-for-byte against a dir.",
    )
    parser.add_argument(
        "--allow-overwrite-embedded",
        action="store_true",
        help=(
            "Permit --write-rust-bins to overwrite a directory that already "
            "holds non-identical embedded blobs. Off by default."
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
        "--gamma-quad",
        type=int,
        default=257,
        help="Quadrature nodes for the true-coherence (Touzi) marginalization.",
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
    parser.add_argument(
        "--no-coherence-marginalization",
        action="store_true",
        help=(
            "Disable the Touzi true-coherence marginalization (slope-only, "
            "pure-Carballo baseline). The sample coherence is plugged straight "
            "into the Lee density as if it were the true coherence."
        ),
    )
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
    if args.source_table_dir is None:
        mode = "analytic reconstruction"
        phase, corr, nlooks, p0, p1 = generate_tables(args)
    else:
        mode = f"authoritative source-table export from {args.source_table_dir}"
        phase, corr, nlooks, p0, p1 = load_saved_tables(args.source_table_dir)
    grid = (phase, corr, nlooks)

    save_rgi_npz(args.out_dir / "carballo-pdf-0-spline.npz", grid, p0)
    save_rgi_npz(args.out_dir / "carballo-pdf-1-spline.npz", grid, p1)
    rust_bin_dir = args.rust_bin_dir or args.out_dir
    if args.write_rust_bins:
        write_rust_bins(
            rust_bin_dir,
            phase,
            corr,
            nlooks,
            p0,
            p1,
            allow_overwrite_embedded=args.allow_overwrite_embedded,
        )

    print(f"Wrote tables to {args.out_dir} in {time.time() - started:.1f}s")
    print(f"  mode: {mode}")
    if args.write_rust_bins:
        print(f"Wrote Rust LUT blobs to {rust_bin_dir}")
    if args.source_table_dir is None:
        print(f"  coherence marginalization: {not args.no_coherence_marginalization}")
    print(f"  p0 range [{p0.min():.6g}, {p0.max():.6g}]")
    print(f"  p1 range [{p1.min():.6g}, {p1.max():.6g}]")

    generated = (phase, corr, nlooks, p0, p1)
    if (
        args.source_table_dir is None
        and not args.no_compare
        and args.compare_dir.exists()
    ):
        report_comparison(generated, args.compare_dir)
    if args.verify_rust_bin_dir is not None:
        report_rust_bin_parity(generated, args.verify_rust_bin_dir)


if __name__ == "__main__":
    main()
