"""A_016: whirlwind vs production, showing the COST-FUNCTION story.

The key claim (motivates the 'high-coherence-cut penalty' option): whirlwind
places branch cuts THROUGH coherent pixels (cheap shortest path under its
coherence cost), while production routes cuts AROUND coherent areas (SNAPHU's
statistical cost makes coherent cuts expensive). Both are valid unwrappings;
they differ by a +1 winding.

2x3 panels:
 (a) coherence            (b) production unwrapped   (c) whirlwind unwrapped
 (d) PRODUCTION cuts/coh  (e) WHIRLWIND cuts/coh     (f) ambiguity diff (cycles)
Branch cuts (flow!=0) are dilated + overlaid in RED on a gray coherence map.
A zoom inset on the transition band is added to (d)/(e).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

LEARN = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")
A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/a016_diag")
OUT.mkdir(parents=True, exist_ok=True)
TAU = 2 * np.pi


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def cut_pixels(u, ig, valid):
    """Pixels incident to a branch-cut edge (flow != 0)."""
    H, W = u.shape
    fh = np.rint((u[:, 1:] - u[:, :-1] - wrap(ig[:, 1:] - ig[:, :-1])) / TAU)
    fv = np.rint((u[1:, :] - u[:-1, :] - wrap(ig[1:, :] - ig[:-1, :])) / TAU)
    cut = np.zeros((H, W), bool)
    ch = valid[:, :-1] & valid[:, 1:] & (fh != 0)
    cut[:, :-1] |= ch
    cut[:, 1:] |= ch
    cv = valid[:-1, :] & valid[1:, :] & (fv != 0)
    cut[:-1, :] |= cv
    cut[1:, :] |= cv
    return cut


def overlay(ax, coh, cut, valid, title, zoom=None):
    bg = np.where(valid, coh, np.nan)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    cd = binary_dilation(cut, iterations=2)
    rgba = np.zeros((*cut.shape, 4))
    rgba[cd] = [1, 0, 0, 1]
    ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if zoom:
        y0, y1, x0, x1 = zoom
        ax.add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, ec="cyan", fc="none", lw=1.5)
        )


def main() -> None:
    d = np.load(LEARN / "ww_gunw_bench" / A016 / "full_arrays.npz")
    mask = d["mask"]
    prod = d["prod_unw"].astype(np.float64)
    pcc = d["prod_cc"]
    unw = d["ww_unw"].astype(np.float64)
    ig = d["ig"].astype(np.float64)
    coh = d["coh"].astype(np.float64)
    valid = mask & np.isfinite(unw) & np.isfinite(prod)
    reg = valid & (pcc > 0)
    a = np.rint((unw - prod) / TAU)
    a = a - modal(a[reg])

    # DIFFERING cuts: edges where ww cut (flow!=0) but production did NOT (flow==0)
    H, W = unw.shape

    def flow(u):
        fh = np.rint((u[:, 1:] - u[:, :-1] - wrap(ig[:, 1:] - ig[:, :-1])) / TAU)
        fv = np.rint((u[1:, :] - u[:-1, :] - wrap(ig[1:, :] - ig[:-1, :])) / TAU)
        return fh, fv

    wfh, wfv = flow(unw)
    pfh, pfv = flow(prod)
    ww_only = np.zeros((H, W), bool)
    mh = valid[:, :-1] & valid[:, 1:] & (wfh != 0) & (pfh == 0)
    ww_only[:, :-1] |= mh
    ww_only[:, 1:] |= mh
    mv = valid[:-1, :] & valid[1:, :] & (wfv != 0) & (pfv == 0)
    ww_only[:-1, :] |= mv
    ww_only[1:, :] |= mv

    # bbox of ww-only cuts for the zoom
    ys, xs = np.where(ww_only)
    y0, y1 = max(0, ys.min() - 80), min(H, ys.max() + 80)
    x0, x1 = max(0, xs.min() - 80), min(W, xs.max() + 80)

    fig, ax = plt.subplots(2, 3, figsize=(20, 13), constrained_layout=True)
    lo, hi = np.nanpercentile(np.where(reg, prod, np.nan), [2, 98])
    im = ax[0, 0].imshow(np.where(reg, prod, np.nan), cmap="viridis", vmin=lo, vmax=hi)
    ax[0, 0].set_title("PRODUCTION unwrapped (SNAPHU)")
    fig.colorbar(im, ax=ax[0, 0], shrink=0.6)
    im = ax[0, 1].imshow(np.where(reg, unw, np.nan), cmap="viridis", vmin=lo, vmax=hi)
    ax[0, 1].set_title("WHIRLWIND unwrapped (tile512) - valid, but right side wound +1")
    fig.colorbar(im, ax=ax[0, 1], shrink=0.6)
    im = ax[0, 2].imshow(np.where(reg, a, np.nan), cmap="RdBu", vmin=-2, vmax=2)
    ax[0, 2].set_title("ambiguity diff ww-prod (cycles): the +1 winding region")
    fig.colorbar(im, ax=ax[0, 2], shrink=0.6)

    # full frame: coherence + the DIFFERING (ww-only) cuts in red
    overlay(
        ax[1, 0],
        coh,
        ww_only,
        valid,
        "cuts ww made but SNAPHU did NOT (red), on coherence",
        zoom=(y0, y1, x0, x1),
    )
    # zoom of the same
    cohz = np.where(valid, coh, np.nan)[y0:y1, x0:x1]
    ax[1, 1].imshow(cohz, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    cz = binary_dilation(ww_only[y0:y1, x0:x1], iterations=1)
    rgba = np.zeros((*cz.shape, 4))
    rgba[cz] = [1, 0, 0, 1]
    ax[1, 1].imshow(rgba, interpolation="nearest")
    ax[1, 1].set_title(
        f"ZOOM: ww-only cuts cross COHERENT pixels (bright)\n(box from left panel)"
    )
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    # histogram of coherence on ww-only cuts vs scene
    chmin = np.minimum(coh[:, :-1], coh[:, 1:])
    cvmin = np.minimum(coh[:-1, :], coh[1:, :])
    cc = np.concatenate([chmin[mh], cvmin[mv]])
    ax[1, 2].hist(
        coh[valid],
        bins=40,
        range=(0, 1),
        density=True,
        alpha=0.5,
        label="scene (valid)",
    )
    ax[1, 2].hist(
        cc,
        bins=40,
        range=(0, 1),
        density=True,
        alpha=0.6,
        color="red",
        label="ww-only cut pixels",
    )
    ax[1, 2].axvline(
        np.median(cc), color="red", ls="--", label=f"median {np.median(cc):.2f}"
    )
    ax[1, 2].set_title(
        "coherence: ww-only cuts vs scene\n(cuts sit on HIGH coherence = cost-suboptimal vs SNAPHU)"
    )
    ax[1, 2].set_xlabel("coherence")
    ax[1, 2].legend()

    p = OUT / "a016_cut_comparison.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(
        f"ww-only cut pixels: {int(ww_only.sum()):,}; median coherence there = {np.median(cc):.3f} (scene median {np.median(coh[valid]):.3f})"
    )

    # quantify coherence on cuts for the caption
    H, W = unw.shape

    def cutcoh(u):
        fh = np.rint((u[:, 1:] - u[:, :-1] - wrap(ig[:, 1:] - ig[:, :-1])) / TAU)
        fv = np.rint((u[1:, :] - u[:-1, :] - wrap(ig[1:, :] - ig[:-1, :])) / TAU)
        ch = np.minimum(coh[:, :-1], coh[:, 1:])
        cv = np.minimum(coh[:-1, :], coh[1:, :])
        m = valid[:, :-1] & valid[:, 1:] & (fh != 0)
        m2 = valid[:-1, :] & valid[1:, :] & (fv != 0)
        cc = np.concatenate([ch[m], cv[m2]])
        return cc.mean(), np.median(cc), 100 * np.mean(cc > 0.7)

    wm, wmd, w70 = cutcoh(unw)
    pm, pmd, p70 = cutcoh(prod)
    print(
        f"whirlwind cuts: mean coh={wm:.3f} median={wmd:.3f}  {w70:.0f}% through coh>0.7"
    )
    print(
        f"production cuts: mean coh={pm:.3f} median={pmd:.3f}  {p70:.0f}% through coh>0.7"
    )
    print(f"PLOT: {p}")


if __name__ == "__main__":
    main()
