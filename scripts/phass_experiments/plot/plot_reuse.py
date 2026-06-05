"""Side-by-side plots: SNAPHU vs whirlwind baseline vs whirlwind reuse.

Produces per-scene panels:
  - K field (integer cycle count vs SNAPHU ref)
  - K-difference vs SNAPHU on the mainland (cc=1 for NISAR, full for PV)
  - Unwrapped phase

Output: plots/{scene}_reuse_panel.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments")
OUT = ROOT / "outputs"
PLOTS = ROOT / "plots"
NISAR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
PV = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes"
    "/Palos_Verdes_C13_RO23_SP/network_output/20251129_20251205"
)

TAU = np.float32(2 * np.pi)


def load_scene(scene: str):
    if scene == "nisar":
        with rasterio.open(NISAR / "20251224_20260117.int.looked.tif") as src:
            ig = src.read(1).astype(np.complex64)
        with rasterio.open(
            NISAR / "20251224_20260117.int.coh.looked.cleaned.tif"
        ) as src:
            coh = src.read(1).astype(np.float32)
        with rasterio.open(NISAR / "20251224_20260117.snaphu_9x9.unw.tif") as src:
            snaphu_unw = src.read(1).astype(np.float32)
        with rasterio.open(NISAR / "20251224_20260117.snaphu_9x9.cc.tif") as src:
            snaphu_cc = src.read(1).astype(np.uint32)
        wrapped = np.angle(ig).astype(np.float32)
        snaphu_k = np.round((snaphu_unw - wrapped) / TAU).astype(np.int32)
        mainland = snaphu_cc == 1
    elif scene == "pv":
        with rasterio.open(
            PV / "CAPELLA_C13_C13_SP_PHS_HH_20251129T183328_20251205T162657.tif"
        ) as src:
            wrapped = src.read(1).astype(np.float32)
        ref = np.load(OUT / "pv_snaphu.npz")
        snaphu_unw = ref["unw"].astype(np.float32)
        snaphu_cc = ref["cc"].astype(np.uint32)
        snaphu_k = ref["k"].astype(np.int32)
        mainland = snaphu_cc == 1
    else:
        raise ValueError(scene)
    return wrapped, snaphu_unw, snaphu_k, snaphu_cc, mainland


def k_match_pct(k_ww: np.ndarray, snaphu_k: np.ndarray, mainland: np.ndarray) -> float:
    dk = k_ww[mainland] - snaphu_k[mainland]
    center = int(np.bincount(dk - dk.min()).argmax() + dk.min())
    return float(((dk - center) == 0).mean() * 100)


def plot_scene(scene: str, snaphu_wall_s: float, baseline_wall_s: float):
    wrapped, snaphu_unw, snaphu_k, snaphu_cc, mainland = load_scene(scene)
    baseline = np.load(OUT / f"{scene}_baseline.npz")
    reuse = np.load(OUT / f"{scene}_reuse.npz")

    k_base = baseline["k"].astype(np.int32)
    k_reuse = reuse["k"].astype(np.int32)
    base_elapsed = float(baseline["elapsed"])
    reuse_elapsed = float(reuse["elapsed"])

    base_match = k_match_pct(k_base, snaphu_k, mainland)
    reuse_match = k_match_pct(k_reuse, snaphu_k, mainland)

    # Center each K-field on its own mode so absolute global offset doesn't drift the colormap
    def center_k(k):
        valid = mainland
        if not valid.any():
            return k
        dk = k[valid] - snaphu_k[valid]
        center = int(np.bincount(dk - dk.min()).argmax() + dk.min())
        return k - center

    k_base_c = center_k(k_base)
    k_reuse_c = center_k(k_reuse)

    # Difference-from-SNAPHU panels (mainland only; rest grey)
    diff_base = (k_base_c - snaphu_k).astype(float)
    diff_reuse = (k_reuse_c - snaphu_k).astype(float)
    diff_base = np.where(mainland, diff_base, np.nan)
    diff_reuse = np.where(mainland, diff_reuse, np.nan)

    # Shared color range for K fields
    all_k = np.concatenate(
        [snaphu_k[mainland], k_base_c[mainland], k_reuse_c[mainland]]
    )
    klo, khi = np.percentile(all_k, [1, 99])

    # Shared color range for diff panels (symmetric)
    dvals = np.concatenate(
        [diff_base[~np.isnan(diff_base)], diff_reuse[~np.isnan(diff_reuse)]]
    )
    dmag = max(1, int(np.ceil(np.nanpercentile(np.abs(dvals), 99))))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"{scene.upper()}  ·  α=0 (no Goldstein)  ·  K-agreement vs SNAPHU 9x9",
        fontsize=13,
    )

    # Top row: K fields
    axes[0, 0].imshow(
        snaphu_k, vmin=klo, vmax=khi, cmap="twilight", interpolation="nearest"
    )
    axes[0, 0].set_title(f"SNAPHU 9x9 (reference)\n{snaphu_wall_s:.0f} s")
    axes[0, 1].imshow(
        k_base_c, vmin=klo, vmax=khi, cmap="twilight", interpolation="nearest"
    )
    axes[0, 1].set_title(
        f"whirlwind baseline (unit-cap MCF)\n{base_elapsed:.1f} s  ·  K-match {base_match:.2f}%"
    )
    axes[0, 2].imshow(
        k_reuse_c, vmin=klo, vmax=khi, cmap="twilight", interpolation="nearest"
    )
    axes[0, 2].set_title(
        f"whirlwind reuse (PHASS-style)\n{reuse_elapsed:.1f} s  ·  K-match {reuse_match:.2f}%"
    )

    # Bottom row: difference panels (mainland)
    axes[1, 0].imshow(mainland, cmap="Greys", interpolation="nearest")
    axes[1, 0].set_title(f"SNAPHU cc=1 mainland\n({mainland.sum():,} px)")
    norm = mcolors.TwoSlopeNorm(vmin=-dmag, vcenter=0, vmax=dmag)
    axes[1, 1].imshow(diff_base, norm=norm, cmap="RdBu_r", interpolation="nearest")
    axes[1, 1].set_title("Δ K vs SNAPHU (baseline − SNAPHU)")
    im = axes[1, 2].imshow(
        diff_reuse, norm=norm, cmap="RdBu_r", interpolation="nearest"
    )
    axes[1, 2].set_title("Δ K vs SNAPHU (reuse − SNAPHU)")
    fig.colorbar(im, ax=axes[1, :], shrink=0.85, label="Δ K (cycles)")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    out_path = PLOTS / f"{scene}_reuse_panel.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{scene}] wrote {out_path}")


if __name__ == "__main__":
    # Wall times for context - SNAPHU's own run on each scene.
    # PV SNAPHU was 12.3 s (single tile). NISAR SNAPHU 9x9 tiled was 17 min.
    plot_scene("pv", snaphu_wall_s=12.3, baseline_wall_s=0.7)
    plot_scene("nisar", snaphu_wall_s=17 * 60, baseline_wall_s=75.0)
