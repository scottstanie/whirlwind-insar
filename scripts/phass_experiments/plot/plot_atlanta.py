"""Downsampled K-field panels for the Atlanta S-1 scene: OPERA reference vs
whirlwind modes. K = round((unw - wrapped)/2pi). Each panel shares the OPERA
color scale so over-unwrapping (whirlwind's huge K excursions) is obvious.

Usage:  plot_atlanta.py [mode ...]   (default: baseline reuse goldstein convex)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio

OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
ATL = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
LAM = 0.05546576
TAU = np.float32(2 * np.pi)
STRIDE = 8

modes = sys.argv[1:] or ["baseline", "reuse", "goldstein", "convex"]


def ds(a):
    return a[::STRIDE, ::STRIDE]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase = rasterio.open(ATL / "opera.int.phs.tif").read(1).astype(np.float32)
    disp = rasterio.open(ATL / "opera.displacement.tif").read(1).astype(np.float32)
    coh = rasterio.open(ATL / "opera.int.cor.tif").read(1).astype(np.float32)
    wrapped = np.angle(np.exp(1j * phase)).astype(np.float32)
    mask = np.isfinite(phase) & np.isfinite(coh) & np.isfinite(disp) & (coh > 0) & (coh < 1.0)
    k_ref = np.round((disp * (4 * np.pi / LAM) - wrapped) / TAU)
    k_ref[~mask] = np.nan

    panels = [("OPERA (SNAPHU) ref", ds(k_ref))]
    for mode in modes:
        p = OUT / f"atlanta_{mode}.npz"
        if not p.exists():
            continue
        k = np.load(p)["k"].astype(np.float32)
        k[~mask] = np.nan
        panels.append((mode, ds(k)))

    # Shared scale from the reference (so whirlwind blow-ups are visible).
    vmax = float(np.nanpercentile(np.abs(ds(k_ref)), 99.5)) or 3.0
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (label, k) in zip(axes, panels):
        im = ax.imshow(k, vmin=-vmax, vmax=vmax, cmap="twilight", interpolation="nearest")
        rng = np.nanmax(k) - np.nanmin(k)
        ax.set_title(f"{label}\nK range [{np.nanmin(k):.0f},{np.nanmax(k):.0f}] std={np.nanstd(k):.1f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Atlanta S-1: integer ambiguity K (shared OPERA scale)", fontsize=12)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "atlanta_k_panel.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
