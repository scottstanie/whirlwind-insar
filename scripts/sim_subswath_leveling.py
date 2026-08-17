#!/usr/bin/env python3
"""Simulated NISAR subswath leveling: truth is known, so the cycles are absolute.

Everything measured on real GUNW frames is scored against the production
unwrap, which is itself a SNAPHU result and demonstrably wrong on some frames --
so "off by N cycles" there is relative to a reference we do not trust. Here the
truth is constructed, so the error maps are absolute.

Scene
-----
* ``troposim.turbulence.simulate`` for a realistic power-law atmosphere (metres
  of delay, rescaled to a chosen RMS), which supplies the *curvature* a single
  fitted plane cannot absorb.
* plus a linear ramp of a chosen size (the ionospheric bulk phase).
* converted to L-band radians, cut into subswath stripes by a NISAR-like mask,
  and observed through ``whirlwind.simulate_ifg`` (Lee-PDF noise at a given
  coherence and looks).

Arms
----
``none``        bridge disabled -- the do-nothing baseline. Without this guard a
                method can look good merely by not making things worse.
``bridge``      current default: minimum-jump re-leveling across the gap.
``remove_ramp`` PR #99's pre-pass (needs a build that has it).
``gapfill``     fill the gaps, widen the mask, let the MCF choose the integers.

The predictor to watch is ``cyc/gap``: the true phase change across a gap, in
cycles, measured from truth. Minimum-jump assumes it is zero, so it must start
failing once it exceeds 0.5 -- this scene lets you watch that happen.

Usage::

    python scripts/sim_subswath_leveling.py --ramp-cycles 0 2 5 10 20 \
        --figure-ramp 10 --out nisar-pngs/sim_leveling.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from exp_gap_interp import fill_gaps  # noqa: E402

TWOPI = 2.0 * np.pi
LAMBDA_L_BAND_M = 0.2385  # NISAR L-band centre wavelength


def subswath_mask(shape: tuple[int, int], n_stripes: int, gap_px: int) -> np.ndarray:
    """NISAR-like fixed-PRF mask: vertical stripes separated by nodata gaps."""
    m, n = shape
    stripe = (n - (n_stripes - 1) * gap_px) // n_stripes
    mask = np.zeros(shape, dtype=bool)
    for s in range(n_stripes):
        a = s * (stripe + gap_px)
        mask[:, a : a + stripe] = True
    return mask


def make_truth(
    shape: tuple[int, int],
    ramp_cycles: float,
    tropo_rms_m: float,
    beta: float,
    resolution: float,
    seed: int,
) -> np.ndarray:
    """Turbulence (rescaled to an RMS in metres) plus a linear ramp, in radians."""
    from troposim import turbulence

    tropo = np.asarray(
        turbulence.simulate(shape=shape, beta=beta, resolution=resolution, seed=seed)
    ).reshape(shape)
    tropo -= tropo.mean()
    if tropo.std() > 0:
        tropo *= tropo_rms_m / tropo.std()
    phase_tropo = tropo * (2 * TWOPI / LAMBDA_L_BAND_M)  # metres of delay -> radians

    m, n = shape
    jj = np.arange(n)[None, :] / max(n - 1, 1)
    ii = np.arange(m)[:, None] / max(m - 1, 1)
    # Most of the ramp across range (columns), a third of it across azimuth.
    ramp = TWOPI * ramp_cycles * (jj + ii / 3.0)
    return (phase_tropo + ramp).astype(np.float32)


def true_cross_gap_cycles(truth: np.ndarray, mask: np.ndarray) -> float:
    """Median |true phase change| across a gap, in cycles -- the failure predictor."""
    from exp_gap_relevel import _runs

    vals = []
    for i in range(0, mask.shape[0], 8):
        runs = _runs(mask[i])
        for (_a1, b1), (a2, _b2) in zip(runs, runs[1:]):
            vals.append(abs(truth[i, a2] - truth[i, b1 - 1]) / TWOPI)
    return float(np.median(vals)) if vals else float("nan")


def cycle_accuracy(est: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    """Fraction of valid pixels on the correct 2*pi cycle, after one global shift."""
    d = (np.asarray(est, np.float64) - truth)[valid]
    d = d - TWOPI * np.round(np.median(d) / TWOPI)
    return float(np.mean(np.abs(d) < np.pi))


def run_arms(
    truth: np.ndarray,
    mask: np.ndarray,
    coherence: float,
    nlooks: float,
    seed: int,
    arms: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    import whirlwind as ww

    gamma = np.full(truth.shape, coherence, np.float32)
    igram, corr = ww.simulate_ifg(truth, gamma, int(nlooks), seed)
    igram = np.asarray(igram, np.complex64)
    corr = np.asarray(corr, np.float32)
    igram[~mask] = 0
    corr_s = np.where(mask, corr, 0).astype(np.float32)
    wrapped = np.angle(igram).astype(np.float32)

    out: dict[str, np.ndarray] = {}
    for arm in arms:
        # Arm syntax: [ramp+]{none|bridge|fill<method>}
        use_ramp = arm.startswith("ramp+")
        base = arm[len("ramp+") :] if use_ramp else arm
        if base.startswith("fill"):
            method = base[len("fill") :] or "nearest"
            zf, filled = fill_gaps(np.where(mask, wrapped, np.nan), mask, method)
            c2 = corr_s.copy()
            c2[filled] = 0.05
            kw = {"remove_ramp": True} if use_ramp else {}
            unw, _cc = ww.unwrap(zf, c2, nlooks, mask | filled, **kw)
        else:
            kw = {"bridge": base != "none"}
            if use_ramp:
                kw["remove_ramp"] = True
            unw, _cc = ww.unwrap(igram, corr_s, nlooks, mask, **kw)
        out[arm] = np.asarray(unw, np.float32)
    return out, wrapped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=int, nargs=2, default=(800, 800))
    p.add_argument("--n-stripes", type=int, default=7)
    p.add_argument("--gap-px", type=int, default=26)
    p.add_argument("--ramp-cycles", type=float, nargs="*", default=[0, 2, 5, 10, 20])
    p.add_argument("--tropo-rms-m", type=float, default=0.04)
    # beta 8/3 = mid-scale, ramp-like turbulence; 3.5-3.8 puts the power at low
    # frequencies -> big smooth blobs that are decidedly NOT a plane, which is
    # the case a single fitted ramp cannot absorb.
    p.add_argument("--beta", type=float, nargs="*", default=[8 / 3])
    p.add_argument("--figure-beta", type=float, default=None)
    p.add_argument("--resolution", type=float, default=60.0)
    p.add_argument("--coherence", type=float, default=0.7)
    p.add_argument("--nlooks", type=float, default=78.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--figure-ramp", type=float, default=10.0)
    p.add_argument("--out", type=Path, default=Path("nisar-pngs/sim_leveling.png"))
    p.add_argument(
        "--arms", nargs="*", default=["none", "bridge", "remove_ramp", "gapfill"]
    )
    args = p.parse_args()

    shape = (args.shape[0], args.shape[1])
    mask = subswath_mask(shape, args.n_stripes, args.gap_px)
    print(
        f"scene {shape}, {args.n_stripes} stripes, {args.gap_px}px gaps, "
        f"valid={mask.mean():.2f}, tropo={args.tropo_rms_m * 100:.0f}cm rms, "
        f"coh={args.coherence}, nlooks={args.nlooks:.0f}\n"
    )
    hdr = f"{'beta':>5} {'ramp(cyc)':>9} {'cyc/gap':>8} | " + " ".join(
        f"{a:>12}" for a in args.arms
    )
    print(hdr)
    print("-" * len(hdr))

    fig_beta = args.figure_beta if args.figure_beta is not None else args.beta[0]
    fig_data = None
    for beta in args.beta:
        for rc in args.ramp_cycles:
            truth = make_truth(
                shape, rc, args.tropo_rms_m, beta, args.resolution, args.seed
            )
            cyc_gap = true_cross_gap_cycles(truth, mask)
            res, wrapped = run_arms(
                truth, mask, args.coherence, args.nlooks, args.seed, args.arms
            )
            scores = {a: cycle_accuracy(res[a], truth, mask) for a in args.arms}
            print(
                f"{beta:>5.2f} {rc:>9.1f} {cyc_gap:>8.2f} | "
                + " ".join(f"{scores[a]:>12.4f}" for a in args.arms)
            )
            if abs(rc - args.figure_ramp) < 1e-9 and abs(beta - fig_beta) < 1e-9:
                fig_data = (beta, rc, cyc_gap, truth, wrapped, res, scores)
        print("-" * len(hdr))

    if fig_data is None:
        print("\n(no --figure-ramp match; skipping figure)")
        return

    beta, rc, cyc_gap, truth, wrapped, res, scores = fig_data
    ncol = 1 + len(args.arms)
    fig, axes = plt.subplots(
        2, ncol, figsize=(3.5 * ncol, 7.6), constrained_layout=True
    )
    show = lambda a: np.where(mask, a, np.nan)  # noqa: E731
    # interpolation="nearest" everywhere: resampling a wrapped field averages
    # across the +/-pi branch cut and turns clean fringes into mush.
    kw = dict(interpolation="nearest")

    t_cyc = show(truth) / TWOPI
    t_cyc = t_cyc - np.nanmedian(t_cyc)
    lo, hi = np.nanpercentile(t_cyc, [0.5, 99.5])
    im = axes[0, 0].imshow(t_cyc, cmap="RdBu_r", vmin=lo, vmax=hi, **kw)
    axes[0, 0].set_title(
        f"TRUTH (cycles)\nbeta {beta:g}, ramp {rc:g} cyc, {cyc_gap:.2f} cyc/gap",
        fontsize=9,
    )
    fig.colorbar(im, ax=axes[0, 0], shrink=0.75)
    im = axes[1, 0].imshow(
        show(wrapped), cmap="twilight", vmin=-np.pi, vmax=np.pi, **kw
    )
    axes[1, 0].set_title("wrapped input (rad)", fontsize=9)
    fig.colorbar(im, ax=axes[1, 0], shrink=0.75)

    for c, arm in enumerate(args.arms, start=1):
        u = show(res[arm]) / TWOPI
        u = u - np.nanmedian(u)
        axes[0, c].imshow(u, cmap="RdBu_r", vmin=lo, vmax=hi, **kw)
        axes[0, c].set_title(f"{arm}: unwrapped\n(same scale as truth)", fontsize=9)

        d = (np.asarray(res[arm], np.float64) - truth) / TWOPI
        d = d - np.round(np.median(d[mask]))
        dl = max(1.0, float(np.nanpercentile(np.abs(show(d)), 99.5)))
        im = axes[1, c].imshow(show(d), cmap="RdBu", vmin=-dl, vmax=dl, **kw)
        axes[1, c].set_title(
            f"{arm} MINUS truth (cycles)\non-cycle {scores[arm] * 100:.1f}%", fontsize=9
        )
        fig.colorbar(im, ax=axes[1, c], shrink=0.75)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Simulated NISAR subswaths (beta={beta:g}): turbulence + ramp, known truth\n"
        "top = unwrapped phase (shared scale), bottom = error against truth",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nWrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
