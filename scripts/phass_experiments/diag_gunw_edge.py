"""Why do the coarse 2pi-offset blocks appear? Test the EDGE hypothesis on the
A_016 GUNW: are the offset-block pixels (amb!=0 within production comp1)
clustered near the VALID-DATA boundary and/or tile boundaries?

Cheap: operates on the saved bench arrays, no new unwrap. Also writes a map of
where the blocks are (block pixels vs valid mask vs tile grid).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

BENCH = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_bench")
A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = 2 * np.pi
TS, OV = 512, 64
STEP = TS - OV  # 448


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(BENCH / A016 / "full_arrays.npz")
    mask = d["mask"]; prod = d["prod_unw"]; pcc = d["prod_cc"]; wwa = d["ww_aligned"]; coh = d["coh"]
    comp1 = mask & (pcc == 1)
    amb = np.rint((wwa - prod) / TAU)
    a1 = amb[comp1]; off_val = a1 - modal(a1)
    block = np.zeros_like(comp1)
    block[comp1] = (off_val != 0)
    okreg = np.zeros_like(comp1)
    okreg[comp1] = (off_val == 0)
    H, W = mask.shape

    # distance of each valid pixel to the nearest INVALID pixel (valid-edge dist)
    edist = distance_transform_edt(mask)
    print(f"comp1={comp1.sum():,}  block={block.sum():,} ({100*block.sum()/comp1.sum():.1f}%)", flush=True)
    print("valid-edge distance (px):  region        median   p90   frac<=16px", flush=True)
    for nm, reg in [("block", block), ("ok", okreg)]:
        ed = edist[reg]
        print(f"   {nm:10s}  {np.median(ed):8.1f} {np.percentile(ed,90):6.1f}   {100*np.mean(ed<=16):5.1f}%", flush=True)

    # distance to nearest tile-boundary LINE (either axis), tiles at 0,448,896,...
    rstarts = list(range(0, H, STEP)); cstarts = list(range(0, W, STEP))
    rl = np.min([np.abs(np.arange(H) - s) for s in rstarts], axis=0)
    cl = np.min([np.abs(np.arange(W) - s) for s in cstarts], axis=0)
    tdist = np.minimum(rl[:, None], cl[None, :])
    print("tile-boundary distance (px):  region      median   frac<=4px (on a seam)", flush=True)
    for nm, reg in [("block", block), ("ok", okreg)]:
        td = tdist[reg]
        print(f"   {nm:10s}  {np.median(td):8.1f}   {100*np.mean(td<=4):5.1f}%", flush=True)

    # connected blocks: how many, sizes; are they whole edge tiles or sub-tile?
    from scipy.ndimage import label
    lab, n = label(block)
    sizes = np.bincount(lab.ravel())[1:]
    big = np.sort(sizes)[::-1][:8]
    print(f"block connected pieces: {n}; largest sizes px = {big.tolist()}", flush=True)
    # for the biggest block, bbox + touches-valid-edge?
    li = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lab == li)
    touches = (edist[lab == li].min() <= 2)
    print(f"  biggest block: {sizes[li-1]:,}px bbox rows {ys.min()}..{ys.max()} cols {xs.min()}..{xs.max()} "
          f"touches-valid-edge={touches} median-edge-dist={np.median(edist[lab==li]):.0f}", flush=True)

    # map
    disp = np.full((H, W), 0.0)
    disp[mask] = 0.3
    disp[okreg] = 0.55
    disp[block] = 1.0
    s = 3
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(disp[::s, ::s], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax[0].set_title("A_016: white=offset block, grey=ok comp1, dark=other valid", fontsize=12)
    ax[1].imshow(np.where(mask, coh, np.nan)[::s, ::s], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax[1].set_title("coherence", fontsize=12)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    out = PLOTS / "diag_gunw_A016_blocks.png"
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
