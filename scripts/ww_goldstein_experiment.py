"""Pre-filter the wrapped phase with Goldstein, then run ww."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np


def goldstein_filter(ig: np.ndarray, alpha: float = 0.5, win: int = 32, step: int = 16) -> np.ndarray:
    """Standard Goldstein adaptive phase filter.

    Operates on overlapping blocks, multiplying the FFT magnitude by
    `|F|^alpha` and writing back the (real, imag) phase product.
    """
    h, w = ig.shape
    out = np.zeros_like(ig)
    wsum = np.zeros((h, w), dtype=np.float32)
    # 2D Hann window for overlap-add.
    win1d = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win) / (win - 1)))
    win2d = (win1d[:, None] * win1d[None, :]).astype(np.float32)
    for i0 in range(0, h - win + 1, step):
        for j0 in range(0, w - win + 1, step):
            block = ig[i0 : i0 + win, j0 : j0 + win]
            F = np.fft.fft2(block)
            mag = np.abs(F)
            F_filt = F * (mag**alpha)
            block_filt = np.fft.ifft2(F_filt).astype(np.complex64)
            out[i0 : i0 + win, j0 : j0 + win] += block_filt * win2d
            wsum[i0 : i0 + win, j0 : j0 + win] += win2d
    mask = wsum > 0
    out[mask] = out[mask] / wsum[mask]
    return out.astype(np.complex64)


def macos_rss_to_gb(maxrss):
    return (maxrss / (1024**3)) if sys.platform == "darwin" else (maxrss / (1024**2))


def peak_rss_gb():
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(macos_rss_to_gb(s), macos_rss_to_gb(c))


def aligned_diff(a, b, mask):
    d = b - a
    k = int(np.round(float(np.nanmedian(d[mask])) / (2 * np.pi)))
    return d - 2 * np.pi * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/nisar-comparison"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--reference", default="snaphu_plain")
    ap.add_argument("--nlooks", type=float, default=100.0)
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--step", type=int, default=16)
    args = ap.parse_args()

    import whirlwind as ww

    inp = args.out / "input"
    out_dir = args.out / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    ig = np.load(inp / "ig.npy")
    coh = np.load(inp / "coh.npy")
    mask = np.load(inp / "mask.npy")
    print(f"[{args.label}] input loaded")

    # Normalize ig so Goldstein operates on unit-magnitude phase only —
    # otherwise SLC amplitude dominates the FFT and Goldstein becomes a
    # low-pass that smears phase.
    mag = np.abs(ig)
    ig_unit = np.where(mag > 0, ig / np.maximum(mag, 1e-30), 0).astype(np.complex64)
    print(f"[{args.label}] Goldstein α={args.alpha} win={args.win} step={args.step}...")
    t_g = time.perf_counter()
    ig_filt = goldstein_filter(ig_unit, alpha=args.alpha, win=args.win, step=args.step)
    print(f"[{args.label}] Goldstein took {time.perf_counter() - t_g:.1f}s")
    # Re-apply mask after filtering (Goldstein bleeds across NaN borders).
    ig_filt[~mask] = 0

    t0 = time.perf_counter()
    unw, cc = ww.unwrap_with_conncomp(
        ig_filt, coh, float(args.nlooks), mask=mask, cost_threshold=args.threshold
    )
    elapsed = time.perf_counter() - t0
    peak = peak_rss_gb()
    print(f"[{args.label}] ww {elapsed:.1f}s peak {peak:.2f} GB")

    np.save(out_dir / "unw.npy", unw)
    np.save(out_dir / "conncomp.npy", cc)

    ref_unw = np.load(args.out / args.reference / "unw.npy")
    ref_cc = np.load(args.out / args.reference / "conncomp.npy")
    eval_mask = (cc > 0) & (ref_cc > 0) & mask
    diff = aligned_diff(unw, ref_unw, eval_mask)
    v = diff[eval_mask & np.isfinite(diff)]
    abs_v = np.abs(v)
    metrics = dict(
        label=args.label,
        n_eval=int(v.size),
        rms_rad=float(np.sqrt(np.mean(v**2))),
        frac_within_pi_2=float((abs_v < np.pi / 2).mean()),
        frac_at_2pi=float(((abs_v >= np.pi) & (abs_v < 3 * np.pi)).mean()),
        frac_at_4pi=float(((abs_v >= 3 * np.pi) & (abs_v < 5 * np.pi)).mean()),
        frac_at_6pi_plus=float((abs_v >= 5 * np.pi).mean()),
        ww_coverage=float(((cc > 0) & mask).mean()),
        eval_coverage=float(eval_mask.mean()),
        elapsed_sec=elapsed,
        peak_rss_gb=peak,
        alpha=args.alpha,
        win=args.win,
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    axes[0].imshow(np.angle(np.where(mask, ig_filt, np.nan)), cmap="twilight",
                   vmin=-np.pi, vmax=np.pi, interpolation="none")
    axes[0].set_title(f"Goldstein-filtered phase  α={args.alpha}")
    d_show = np.where(eval_mask, diff, np.nan)
    axes[1].imshow(d_show, cmap="RdBu_r", vmin=-2 * np.pi, vmax=2 * np.pi, interpolation="none")
    axes[1].set_title(f"snaphu - ww  RMS={metrics['rms_rad']:.2f}  "
                      f"within±π/2={100 * metrics['frac_within_pi_2']:.1f}%")
    axes[2].hist(v, bins=200, range=(-4 * np.pi, 4 * np.pi), log=True)
    for k in (-2, -1, 0, 1, 2):
        axes[2].axvline(2 * np.pi * k, color="r", lw=0.5, alpha=0.4)
    axes[2].set_title("diff hist")
    for ax in axes[:2]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out_dir / "diff.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
