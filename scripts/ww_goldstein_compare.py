"""Final side-by-side: baseline ww vs ww+Goldstein vs snaphu_plain."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


OUT = Path("/tmp/nisar-comparison")
INPUT_MASK = np.load(OUT / "input" / "mask.npy")
IG = np.load(OUT / "input" / "ig.npy")


def panel(ax, arr, cc, ref_med, title, vmin=-12, vmax=12):
    a = np.where(cc > 0, arr - ref_med, np.nan)
    ax.imshow(a, cmap="twilight", vmin=vmin, vmax=vmax, interpolation="none")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    bl = np.load(OUT / "baseline" / "unw.npy"), np.load(OUT / "baseline" / "conncomp.npy")
    gd = np.load(OUT / "exp5_goldstein_w64" / "unw.npy"), np.load(OUT / "exp5_goldstein_w64" / "conncomp.npy")
    sn = np.load(OUT / "snaphu_plain" / "unw.npy"), np.load(OUT / "snaphu_plain" / "conncomp.npy")

    eval_mask = INPUT_MASK & (bl[1] > 0) & (gd[1] > 0) & (sn[1] > 0)
    ref_med_bl = float(np.nanmedian(bl[0][eval_mask]))
    ref_med_gd = float(np.nanmedian(gd[0][eval_mask]))
    ref_med_sn = float(np.nanmedian(sn[0][eval_mask]))

    fig, axes = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)

    wrap = np.angle(IG)
    axes[0, 0].imshow(np.where(INPUT_MASK, wrap, np.nan), cmap="twilight",
                      vmin=-np.pi, vmax=np.pi, interpolation="none")
    axes[0, 0].set_title("Wrapped phase (input)")
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    panel(axes[0, 1], bl[0], bl[1], ref_med_bl, "ww baseline (asymmetric Carballo)")
    panel(axes[0, 2], gd[0], gd[1], ref_med_gd, "ww + Goldstein α=0.7 win=64")
    panel(axes[1, 0], sn[0], sn[1], ref_med_sn, "snaphu_plain (smooth cost)")

    def adiff(a, b):
        d = b - a
        k = int(np.round(float(np.nanmedian(d[eval_mask])) / (2 * np.pi)))
        d = d - 2 * np.pi * k
        d[~eval_mask] = np.nan
        return d

    d_bl = adiff(bl[0], sn[0])
    rms_bl = float(np.sqrt(np.nanmean(d_bl**2)))
    pct_bl = 100 * float((np.abs(d_bl[eval_mask]) < np.pi / 2).mean())
    axes[1, 1].imshow(d_bl, cmap="RdBu_r", vmin=-2 * np.pi, vmax=2 * np.pi, interpolation="none")
    axes[1, 1].set_title(f"snaphu − baseline    RMS={rms_bl:.2f} rad   {pct_bl:.1f}% within π/2")
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])

    d_gd = adiff(gd[0], sn[0])
    rms_gd = float(np.sqrt(np.nanmean(d_gd**2)))
    pct_gd = 100 * float((np.abs(d_gd[eval_mask]) < np.pi / 2).mean())
    axes[1, 2].imshow(d_gd, cmap="RdBu_r", vmin=-2 * np.pi, vmax=2 * np.pi, interpolation="none")
    axes[1, 2].set_title(f"snaphu − Goldstein   RMS={rms_gd:.2f} rad   {pct_gd:.1f}% within π/2")
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])

    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    fig.savefig(plots / "goldstein_vs_baseline_vs_snaphu.png", dpi=150)
    plt.close(fig)
    print(f"wrote {plots / 'goldstein_vs_baseline_vs_snaphu.png'}")
    print(f"baseline RMS={rms_bl:.2f} {pct_bl:.1f}% within π/2")
    print(f"+goldstein RMS={rms_gd:.2f} {pct_gd:.1f}% within π/2")


if __name__ == "__main__":
    main()
