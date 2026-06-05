"""K-field panels comparing baseline / reuse / convex on NISAR.

Output: /Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots/nisar_convex_panel.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
NISAR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
TAU = np.float32(2 * np.pi)

with rasterio.open(NISAR / "20251224_20260117.int.looked.tif") as src:
    wrapped = np.angle(src.read(1).astype(np.complex64)).astype(np.float32)
with rasterio.open(NISAR / "20251224_20260117.snaphu_9x9.unw.tif") as src:
    snaphu_unw = src.read(1).astype(np.float32)
with rasterio.open(NISAR / "20251224_20260117.snaphu_9x9.cc.tif") as src:
    snaphu_cc = src.read(1).astype(np.uint32)
snaphu_k = np.round((snaphu_unw - wrapped) / TAU).astype(np.int32)
mainland = snaphu_cc == 1


def k_centered(k_ww):
    dk = k_ww[mainland] - snaphu_k[mainland]
    ctr = int(np.bincount(dk - dk.min()).argmax() + dk.min())
    return (k_ww - ctr).astype(np.int32), ctr, dk - ctr


variants = [
    ("SNAPHU 9x9 (reference)", snaphu_k, 1020.0, 0),
    (
        "baseline (unit-cap MCF)",
        np.load(OUT / "nisar_baseline.npz")["k"].astype(np.int32),
        float(np.load(OUT / "nisar_baseline.npz")["elapsed"]),
        None,
    ),
    (
        "reuse (PHASS-style)",
        np.load(OUT / "nisar_reuse.npz")["k"].astype(np.int32),
        float(np.load(OUT / "nisar_reuse.npz")["elapsed"]),
        None,
    ),
    (
        "convex (SNAPHU-style)",
        np.load(OUT / "nisar_convex.npz")["k"].astype(np.int32),
        float(np.load(OUT / "nisar_convex.npz")["elapsed"]),
        None,
    ),
]

# Center each K-field (except SNAPHU) on its mode-of-mainland-mismatch
panels = []
for name, k, wall, _ in variants:
    if name.startswith("SNAPHU"):
        panels.append((name, k, wall, None))
    else:
        k_c, _, dk = k_centered(k)
        match = (dk == 0).mean() * 100
        panels.append((name + f"\n{wall:.0f} s · K-match {match:.2f} %", k_c, wall, dk))

all_k = np.concatenate([p[1][mainland] for p in panels])
klo, khi = np.percentile(all_k, [1, 99])

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(
    "NISAR α=0 (no Goldstein)  ·  K-fields top; Δ K vs SNAPHU bottom (mainland only)",
    fontsize=13,
)

for col, (name, k, wall, dk) in enumerate(panels):
    axes[0, col].imshow(k, vmin=klo, vmax=khi, cmap="twilight", interpolation="nearest")
    if col == 0:
        axes[0, col].set_title(name + f"\n{wall:.0f} s")
    else:
        axes[0, col].set_title(name)
    if dk is None:
        axes[1, col].imshow(mainland, cmap="Greys", interpolation="nearest")
        axes[1, col].set_title(f"SNAPHU cc=1 mainland\n({mainland.sum():,} px)")
    else:
        diff = np.where(mainland, k - snaphu_k, np.nan)
        dmag = 4
        norm = mcolors.TwoSlopeNorm(vmin=-dmag, vcenter=0, vmax=dmag)
        im = axes[1, col].imshow(
            diff, norm=norm, cmap="RdBu_r", interpolation="nearest"
        )
        axes[1, col].set_title(f"Δ K vs SNAPHU")
        if col == 3:
            fig.colorbar(im, ax=axes[1, :], shrink=0.85, label="Δ K (cycles)")

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

out = PLOTS / "nisar_convex_panel.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
