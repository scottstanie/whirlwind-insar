"""Headline plot: NISAR whole-image vs tiled vs SNAPHU 9x9 reference.

Top row:    integer ambiguity K for SNAPHU / whirlwind whole-image / tiled.
Bottom row: |dK| error vs SNAPHU on the cc=1 mainland (the runaway is obvious
            in whole-image, cleaned up by tiling).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = np.float32(2 * np.pi)
STRIDE = 6


def modal(d):
    d = d[np.isfinite(d)].astype(int)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import whirlwind as ww

    ig = (
        rasterio.open(N / "20251224_20260117.int.looked.tif")
        .read(1)
        .astype(np.complex64)
    )
    coh = (
        rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif")
        .read(1)
        .astype(np.float32)
    )
    sunw = (
        rasterio.open(N / "20251224_20260117.snaphu_9x9.unw.tif")
        .read(1)
        .astype(np.float32)
    )
    scc = (
        rasterio.open(N / "20251224_20260117.snaphu_9x9.cc.tif")
        .read(1)
        .astype(np.uint32)
    )
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig[~mask] = 0
    coh = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    sk = np.round((sunw - wrapped) / TAU)
    region = (scc == 1) & mask

    t = time.perf_counter()
    uw, _cc = ww.unwrap(ig, coh, nlooks=100.0, mask=mask)
    tw = time.perf_counter() - t
    t = time.perf_counter()
    ut, _cc = ww.unwrap(
        ig, coh, nlooks=100.0, mask=mask, tile_size=512, tile_overlap=64
    )
    tt = time.perf_counter() - t

    def kf(unw):
        k = np.round((unw - wrapped) / TAU)
        k[~mask] = np.nan
        # center vs SNAPHU on the mainland
        d = (k - sk)[region]
        k = k - modal(d)
        return k

    kw, kt = kf(uw), kf(ut)
    sk_disp = sk.copy().astype(np.float32)
    sk_disp[~mask] = np.nan

    def matchpct(k):
        d = (k - sk)[region]
        d = d[np.isfinite(d)]
        return (np.round(d) == 0).mean() * 100

    ds = lambda a: a[::STRIDE, ::STRIDE]
    allk = np.concatenate([sk_disp[region][::50], (sk_disp[region])[::50]])
    lo, hi = np.nanpercentile(sk_disp[mask], [1, 99])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        ("SNAPHU 9x9 (reference)", sk_disp, None),
        (f"whirlwind whole-image\n{matchpct(kw):.1f}% match, {tw:.0f}s", kw, None),
        (f"whirlwind tiled 512\n{matchpct(kt):.1f}% match, {tt:.1f}s", kt, None),
    ]
    for ax, (title, k, _) in zip(axes[0], panels):
        im = ax.imshow(
            ds(k), vmin=lo, vmax=hi, cmap="twilight", interpolation="nearest"
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # bottom: |dK| error maps on the mainland
    def errmap(k):
        e = np.abs(k - sk)
        e[~region] = np.nan
        return e

    axes[1, 0].axis("off")
    for ax, (title, k) in zip(
        axes[1, 1:], [("whole-image |dK| vs SNAPHU", kw), ("tiled |dK| vs SNAPHU", kt)]
    ):
        im = ax.imshow(
            ds(errmap(k)), vmin=0, vmax=3, cmap="inferno", interpolation="nearest"
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        "NISAR no-Goldstein unwrap: whole-image vs tiled (K = round((unw−wrapped)/2π))",
        fontsize=13,
    )
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "nisar_tiled_panel.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
