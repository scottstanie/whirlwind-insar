"""Visualize the bridging fix on one frame from the npz saved by
diag_bridge_isce3_compare.py. Six panels: production unwrapped, then the
per-integration-region cycle error (vs production, relative to the largest
region) for raw / whirlwind-bridge / isce3-bridge, plus coherence. The error
panels make the inter-region gauge jumps visible - exactly what the per-component
agreement metric hides.

Usage: python scripts/plot_bridge_compare.py [FRAME=A_016]
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TWOPI = 2.0 * np.pi
BASE = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final"


def region_cycle_error(u, prod_unw, region, ref_lab, valid):
    """Per-region integer cycle offset vs production, relative to the reference
    region (0 = correctly levelled), painted back onto the pixels."""
    amb = np.rint((u - prod_unw) / TWOPI)
    g = np.rint(np.median(amb[valid & (region == ref_lab)]))
    err = np.full(u.shape, np.nan)
    for lab in range(1, int(region.max()) + 1):
        m = valid & (region == lab)
        if m.sum() < 50:
            continue
        err[region == lab] = np.rint(np.median(amb[m])) - g
    return np.where(valid, err, np.nan)


def main():
    frame = sys.argv[1] if len(sys.argv) > 1 else "A_016"
    d = np.load(f"{BASE}/{frame}_bridge_compare.npz")
    prod_unw = d["prod_unw"]
    mask = d["mask"]
    coh = d["coh"]
    region = d["region"]
    valid = mask & np.isfinite(prod_unw)
    sizes = np.bincount(region.ravel())
    ref_lab = int(np.argmax(sizes[1:]) + 1)

    lo, hi = np.nanpercentile(prod_unw[valid], [2, 98])
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    im = ax[0, 0].imshow(
        np.where(valid, prod_unw, np.nan), cmap="viridis", vmin=lo, vmax=hi
    )
    ax[0, 0].set_title("NISAR GUNW unwrapped (rad)")
    fig.colorbar(im, ax=ax[0, 0], shrink=0.7)
    im = ax[0, 1].imshow(np.where(valid, coh, np.nan), cmap="gray", vmin=0, vmax=1)
    ax[0, 1].set_title("coherence")
    fig.colorbar(im, ax=ax[0, 1], shrink=0.7)
    ax[0, 2].axis("off")

    panels = [
        (ax[1, 0], "raw", d["raw"]),
        (ax[1, 1], "whirlwind bridge (new)", d["ww_bridge"]),
        (ax[1, 2], "isce3 bridge", d["isce3_bridge"]),
    ]
    for a, name, u in panels:
        err = region_cycle_error(u, prod_unw, region, ref_lab, valid)
        amb = np.rint((u - prod_unw) / TWOPI)
        g = np.rint(np.median(amb[valid & (region == ref_lab)]))
        absagree = float(np.mean((amb[valid] - g) == 0)) * 100
        im = a.imshow(err, cmap="RdBu", vmin=-3, vmax=3, interpolation="nearest")
        a.set_title(
            f"{name}\nregion cycle error vs production "
            f"(absolute agreement {absagree:.1f}%)"
        )
        fig.colorbar(im, ax=a, shrink=0.7, label="cycles off")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])

    fig.suptitle(
        f"{frame}: inter-region gauge bridging - raw vs whirlwind vs isce3", fontsize=14
    )
    out = f"docs/figures/bridge_compare_{frame}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure -> {out}", flush=True)
    import shutil

    shutil.copy(out, f"{BASE}/bridge_compare_{frame}.png")


if __name__ == "__main__":
    main()
