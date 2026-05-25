"""Make summary figures from a whirlwind-rs unwrap_stack output.

Produces:
  fig_<name>_wrapped_vs_unwrapped.png    : one IG, wrapped + ours + dolphin's
  fig_<name>_closure_rms.png              : per-pixel closure RMS map (≈ 0)
  fig_<name>_posterior_std.png            : per-date posterior std heatmap
  fig_<name>_diff_vs_dolphin.png          : per-pixel mod-2π diff to dolphin
  fig_<name>_per_ig_metrics.png           : per-IG agreement vs dolphin
  fig_<name>_displacement_timeseries.png  : θ_d(t) at a few sample pixels

Inputs:
  --ours    /path/to/whirlwind/out (must have report.json)
  --dolphin /path/to/dolphin (optional; needed for the diff/metrics figures)
  --out     /path/to/figure/dir
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

print = functools.partial(print, flush=True)


def _read(path: Path, win: Window | None = None) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1, window=win)


def _read_complex(path: Path, win: Window | None = None) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1, window=win)
    if not np.iscomplexobj(a):
        a = np.exp(1j * np.nan_to_num(a.astype(np.float32))).astype(np.complex64)
    return a


def _parse_date(s: str) -> datetime:
    # "YYYYMMDDhhmmss"
    return datetime.strptime(s[:14], "%Y%m%d%H%M%S")


def _phase_cmap():
    """Cyclic colormap for wrapped phase."""
    return "twilight_shifted"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--dolphin", type=Path, default=None,
                   help="dolphin output dir (optional, needed for diff plots)")
    p.add_argument("--out", type=Path, required=True,
                   help="figure output directory")
    p.add_argument("--name", default="palos_verdes",
                   help="prefix for output filenames")
    p.add_argument("--n-sample-pixels", type=int, default=5,
                   help="number of pixels to plot in the timeseries figure")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.ours / "report.json").read_text())
    edges = report["edges"]
    dates = report["dates"]
    ref = report["reference_pixel"]
    window = report.get("window")
    crop = Window(window[1], window[0], window[3] - window[1], window[2] - window[0]) if window else None  # type: ignore[call-arg]

    print(f"[fig] {len(edges)} IGs over {len(dates)} dates")
    print(f"[fig] reference pixel: ({ref['i']}, {ref['j']})")

    # ---- Figure 1: wrapped vs ours vs dolphin SNAPHU for one representative IG ----
    pick = edges[len(edges) // 2]  # middle of stack
    name_ig = f"{pick['from']}_{pick['to']}"
    ig_path = Path(report["dolphin_dir"]) / "interferograms" / f"{name_ig}.int.tif"
    ours_path = args.ours / "corrected" / f"{name_ig}.unw.tif"

    igram = _read_complex(ig_path, crop)
    wrapped = np.angle(igram)
    ours = _read(ours_path)

    cols = 3 if args.dolphin else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5), constrained_layout=True)
    im0 = axes[0].imshow(wrapped, cmap=_phase_cmap(), vmin=-np.pi, vmax=np.pi)
    axes[0].set_title(f"Wrapped phase\n{name_ig[:8]}_{name_ig[15:23]}")
    plt.colorbar(im0, ax=axes[0], shrink=0.7, label="rad")
    im1 = axes[1].imshow(ours, cmap="RdBu_r", vmin=np.percentile(ours, 1), vmax=np.percentile(ours, 99))
    axes[1].set_title("whirlwind-rs unwrapped\n(closure-corrected, anchored)")
    plt.colorbar(im1, ax=axes[1], shrink=0.7, label="rad")
    if args.dolphin:
        unw_path = args.dolphin / "unwrapped" / f"{name_ig}.unw.tif"
        if unw_path.exists():
            theirs = _read(unw_path, crop)
            theirs_a = theirs - theirs[ref["i"], ref["j"]]
            im2 = axes[2].imshow(theirs_a, cmap="RdBu_r",
                                  vmin=np.percentile(ours, 1), vmax=np.percentile(ours, 99))
            axes[2].set_title("dolphin SNAPHU unwrapped\n(anchored at our reference)")
            plt.colorbar(im2, ax=axes[2], shrink=0.7, label="rad")
        else:
            axes[2].set_visible(False)
    out_path = args.out / f"fig_{args.name}_wrapped_vs_unwrapped.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[fig] wrote {out_path.name}")

    # ---- Figure 2: closure RMS map ----
    rms = _read(args.ours / "closure_rms.tif")
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(rms, cmap="magma", vmin=0, vmax=max(1e-3, float(np.percentile(rms, 99))))
    ax.set_title(f"Closure RMS after correction (≈ 0 by construction)\nmean = {rms.mean():.2e} rad")
    plt.colorbar(im, ax=ax, shrink=0.7, label="rad")
    fig.savefig(args.out / f"fig_{args.name}_closure_rms.png", dpi=120)
    plt.close(fig)
    print(f"[fig] wrote fig_{args.name}_closure_rms.png")

    # ---- Figure 3: per-date posterior std heatmap ----
    with rasterio.open(args.ours / "date_phase_std.tif") as src:
        std_cube = src.read()  # (D, m, n)
    # Spatially-averaged per-date std
    per_date_med = np.array([
        float(np.median(std_cube[d][std_cube[d] > 0])) if (std_cube[d] > 0).any() else 0.0
        for d in range(std_cube.shape[0])
    ])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    # Heatmap of (date, pixel-row) of per-date std (averaged across columns)
    per_date_row = std_cube.mean(axis=2)  # (D, m)
    im = axes[0].imshow(per_date_row, aspect="auto", cmap="viridis", origin="lower")
    axes[0].set_xlabel("image row")
    axes[0].set_ylabel("acquisition index")
    axes[0].set_title("posterior std per date (col-averaged)")
    plt.colorbar(im, ax=axes[0], shrink=0.7, label="rad")
    dts = [_parse_date(d) for d in dates]
    axes[1].plot(dts, per_date_med, "o-")
    axes[1].set_ylabel("median posterior std (rad)")
    axes[1].set_xlabel("acquisition date")
    axes[1].set_title("scene-median posterior std per acquisition")
    axes[1].grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(args.out / f"fig_{args.name}_posterior_std.png", dpi=120)
    plt.close(fig)
    print(f"[fig] wrote fig_{args.name}_posterior_std.png")

    # ---- Figure 4 + 5: per-IG diff vs dolphin SNAPHU (if available) ----
    # We compare against dolphin/unwrapped/*.unw.tif — the per-IG SNAPHU
    # outputs, which are the right apples-to-apples baseline for our per-IG
    # closure-corrected unwrap. Dolphin's timeseries/*.tif files are
    # SBAS-inverted DISPLACEMENT in meters (single-reference network), a
    # different mathematical object that requires unit conversion + network
    # inversion to compare and isn't done here.
    if args.dolphin:
        unw_dir = args.dolphin / "unwrapped"
        pct_within = []
        abs_rms = []
        names = []
        baseline_days = []
        diff_panel = None
        diff_panel_name = None

        for e in edges:
            n = f"{e['from']}_{e['to']}"
            tp = unw_dir / f"{n}.unw.tif"
            op = args.ours / "corrected" / f"{n}.unw.tif"
            if not (tp.exists() and op.exists()):
                continue
            ours = _read(op)
            theirs = _read(tp, crop)
            theirs_a = theirs - theirs[ref["i"], ref["j"]]
            valid = np.isfinite(ours) & np.isfinite(theirs_a) & (theirs != 0)
            if not valid.any():
                continue
            d = ours - theirs_a
            d_mod = np.angle(np.exp(1j * d))
            pct = float(100 * np.mean(np.abs(d_mod[valid]) < np.pi / 2))
            rms = float(np.sqrt(np.mean(d[valid] ** 2)))
            pct_within.append(pct)
            abs_rms.append(rms)
            names.append(n)
            d_t = (_parse_date(e["to"]) - _parse_date(e["from"])).total_seconds() / 86400
            baseline_days.append(d_t)
            if diff_panel is None and abs(rms - np.median(abs_rms or [rms])) < 1e-9:
                diff_panel = d_mod
                diff_panel_name = n

        if diff_panel is None and pct_within:
            target = float(np.median(abs_rms))
            mid_idx = int(np.argmin(np.abs(np.array(abs_rms) - target)))
            n = names[mid_idx]
            op = args.ours / "corrected" / f"{n}.unw.tif"
            tp = unw_dir / f"{n}.unw.tif"
            ours = _read(op)
            theirs = _read(tp, crop)
            theirs_a = theirs - theirs[ref["i"], ref["j"]]
            diff_panel = np.angle(np.exp(1j * (ours - theirs_a)))
            diff_panel_name = n

        if diff_panel is not None and diff_panel_name is not None:
            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            abs_diff = np.abs(diff_panel)
            im = ax.imshow(abs_diff, cmap="hot",
                           vmin=0, vmax=max(0.01, float(np.percentile(abs_diff, 99))))
            short = f"{diff_panel_name[:8]}_{diff_panel_name[15:23]}"
            ax.set_title(f"|whirlwind − dolphin SNAPHU| mod 2π  for {short}\n"
                         f"max = {abs_diff.max():.4g} rad, median = {np.median(abs_diff):.2e} rad")
            plt.colorbar(im, ax=ax, shrink=0.7, label="rad")
            fig.savefig(args.out / f"fig_{args.name}_diff_vs_dolphin.png", dpi=120)
            plt.close(fig)
            print(f"[fig] wrote fig_{args.name}_diff_vs_dolphin.png")

        # Per-IG metrics scatter
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        axes[0].scatter(baseline_days, pct_within, alpha=0.6)
        axes[0].set_xlabel("temporal baseline (days)")
        axes[0].set_ylabel("% of pixels within π/2 of dolphin (mod 2π)")
        axes[0].set_title(f"per-IG mod-2π agreement  (median {np.median(pct_within):.1f}%)")
        axes[0].set_ylim(min(99, min(pct_within) - 1), 100.5)
        axes[0].grid(alpha=0.3)
        axes[1].scatter(baseline_days, abs_rms, alpha=0.6)
        axes[1].set_xlabel("temporal baseline (days)")
        axes[1].set_ylabel("absolute RMS diff vs dolphin (rad)")
        axes[1].set_title(f"per-IG absolute disagreement  (median {np.median(abs_rms):.2f} rad)")
        axes[1].grid(alpha=0.3)
        fig.savefig(args.out / f"fig_{args.name}_per_ig_metrics.png", dpi=120)
        plt.close(fig)
        print(f"[fig] wrote fig_{args.name}_per_ig_metrics.png")

    # ---- Figure 6: timeseries at sample pixels ----
    with rasterio.open(args.ours / "date_phases.tif") as src:
        theta = src.read()  # (D, m, n)
    rng = np.random.default_rng(0)
    h, w = theta.shape[1], theta.shape[2]
    sample_ij = []
    for _ in range(args.n_sample_pixels):
        for _ in range(50):
            i, j = rng.integers(0, h), rng.integers(0, w)
            if not np.any(np.isnan(theta[:, i, j])):
                sample_ij.append((int(i), int(j)))
                break
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    dts = [_parse_date(d) for d in dates]
    for (i, j) in sample_ij:
        ax.plot(dts, theta[:, i, j], "o-", alpha=0.7, label=f"({i},{j})")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("acquisition date")
    ax.set_ylabel("relative phase (rad, anchored at reference)")
    ax.set_title(f"acquisition phase θ_d at {len(sample_ij)} random pixels")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(args.out / f"fig_{args.name}_displacement_timeseries.png", dpi=120)
    plt.close(fig)
    print(f"[fig] wrote fig_{args.name}_displacement_timeseries.png")

    print(f"\n[done] figures in {args.out}")


if __name__ == "__main__":
    main()
