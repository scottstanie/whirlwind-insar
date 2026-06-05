"""Compute K-agreement of dolphin's PHASS output vs SNAPHU 9x9 on NISAR.

Also saves a 2-panel plot (dolphin PHASS vs SNAPHU 9x9 K-fields) to the
phass_experiments plots dir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

NISAR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
DOLPH = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/dolphin_phass"
)
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")

TAU = np.float32(2 * np.pi)


def main() -> None:
    with rasterio.open(NISAR / "20251224_20260117.int.looked.tif") as src:
        ig = src.read(1).astype(np.complex64)
    with rasterio.open(NISAR / "20251224_20260117.int.coh.looked.cleaned.tif") as src:
        coh = src.read(1).astype(np.float32)
    with rasterio.open(NISAR / "20251224_20260117.snaphu_9x9.unw.tif") as src:
        snaphu_unw = src.read(1).astype(np.float32)
    with rasterio.open(NISAR / "20251224_20260117.snaphu_9x9.cc.tif") as src:
        snaphu_cc = src.read(1).astype(np.uint32)
    with rasterio.open(DOLPH / "20251224_20260117.unw.tif") as src:
        dolph_unw = src.read(1).astype(np.float32)
    with rasterio.open(DOLPH / "20251224_20260117.unw.conncomp.tif") as src:
        dolph_cc = src.read(1).astype(np.uint32)

    wrapped = np.angle(ig).astype(np.float32)
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)

    snaphu_k = np.round((snaphu_unw - wrapped) / TAU).astype(np.int32)
    dolph_k = np.round((dolph_unw - wrapped) / TAU).astype(np.int32)

    snaphu_main = (snaphu_cc == 1) & mask
    common = snaphu_main  # whirlwind/dolphin's mask is the same input mask

    n_common = int(common.sum())
    print(f"SNAPHU mainland (cc=1 ∩ mask): {n_common:,} px")
    print(
        f"dolphin PHASS conncomp coverage: {(dolph_cc>0).mean()*100:.2f}%, "
        f"n_components={int(dolph_cc.max())}"
    )

    dk = dolph_k[common] - snaphu_k[common]
    center = int(np.bincount(dk - dk.min()).argmax() + dk.min())
    dk_c = dk - center
    m0 = float((dk_c == 0).sum()) / n_common * 100
    m1 = float((np.abs(dk_c) == 1).sum()) / n_common * 100
    m2 = float((np.abs(dk_c) >= 2).sum()) / n_common * 100
    print(
        f"dolphin PHASS K=match: {m0:.2f}%  (|dK|=1: {m1:.2f}%, "
        f"|dK|≥2: {m2:.2f}%, global offset {center:+d})"
    )

    # Side-by-side plot vs SNAPHU
    import matplotlib.pyplot as plt

    panels = [
        ("SNAPHU 9x9", snaphu_k, snaphu_cc > 0),
        ("dolphin PHASS", dolph_k, dolph_cc > 0),
    ]
    k_all = np.concatenate([p[1][p[2]] for p in panels if p[2].any()])
    lo, hi = float(np.quantile(k_all, 0.005)), float(np.quantile(k_all, 0.995))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, (label, k, valid) in zip(axes, panels):
        kp = k.astype(np.float32).copy()
        kp[~valid] = np.nan
        im = ax.imshow(kp, vmin=lo, vmax=hi, cmap="twilight", interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("nisar: integer cycles K - dolphin PHASS vs SNAPHU 9x9", fontsize=11)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "nisar_dolphin_phass_vs_snaphu.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
