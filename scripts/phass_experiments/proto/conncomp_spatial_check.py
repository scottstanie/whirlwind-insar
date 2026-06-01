"""Are mid-size components compact geographic ISLANDS or scattered speckle?
Use bounding-box fill-fraction (size / bbox area). Compact island ~ high fill;
scattered/stringy speckle ~ low fill. Also overlay against production
components: do whirlwind mid-size comps sit inside production's kept islands?
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

A016 = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/a016_diag/a016_default_fixed.npz"
CLEAN = (
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_reuse/"
    "NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_20251017T124857_"
    "20251029T124836_20251029T124858_X05010_N_P_J_001/full_arrays.npz"
)


def labels(unw, mask):
    u = unw.astype(np.float32)
    m = mask & np.isfinite(u)
    h, w = u.shape
    idx = np.full((h, w), -1, np.int64)
    nnodes = int(m.sum())
    idx[m] = np.arange(nnodes)
    rows, cols = [], []
    a = m[:, :-1] & m[:, 1:] & (np.abs(u[:, :-1] - u[:, 1:]) < np.pi)
    rows.append(idx[:, :-1][a]); cols.append(idx[:, 1:][a])
    b = m[:-1, :] & m[1:, :] & (np.abs(u[:-1, :] - u[1:, :]) < np.pi)
    rows.append(idx[:-1, :][b]); cols.append(idx[1:, :][b])
    r = np.concatenate(rows); c = np.concatenate(cols)
    g = coo_matrix((np.ones(r.size, np.uint8), (r, c)), shape=(nnodes, nnodes))
    ncomp, lab = connected_components(g, directed=False)
    full = np.full((h, w), -1, np.int64)
    full[m] = lab
    return full, ncomp


def analyze(name, unw, mask, coh, prod_cc):
    lab, ncomp = labels(unw, mask)
    sizes = np.bincount(lab[lab >= 0])
    order = np.argsort(sizes)[::-1]

    print(f"\n{'='*78}\n{name}")
    print(f"  {'rank':>4} {'size':>10} {'coh':>6} {'fill%':>7} {'bbox':>13} "
          f"{'in_prodCC?':>10}")
    for rank, comp in enumerate(order[:25]):
        sel = lab == comp
        sz = int(sel.sum())
        if sz < 50:
            break
        ys, xs = np.where(sel)
        bh = ys.max() - ys.min() + 1
        bw = xs.max() - xs.min() + 1
        fill = 100.0 * sz / (bh * bw)
        mc = float(coh[sel].mean())
        # which production CC label dominates these pixels (0 = dropped/invalid)
        pvals = prod_cc[sel]
        if pvals.size:
            pos = pvals[pvals > 0]
            if pos.size > 0:
                lbls, cnts = np.unique(pos, return_counts=True)
                dom = int(lbls[cnts.argmax()])
                fracpos = 100.0 * pos.size / pvals.size
                prodstr = f"L{dom}({fracpos:.0f}%)"
            else:
                prodstr = "DROPPED"
        else:
            prodstr = "-"
        print(f"  {rank+1:>4} {sz:>10,} {mc:>6.3f} {fill:>6.1f}% "
              f"{f'{bh}x{bw}':>13} {prodstr:>10}")


def main():
    with np.load(A016) as z:
        analyze("A_016 (fragmented)", z["unw"], z["mask"], z["coh"], z["pcc"])
    with np.load(CLEAN) as z:
        analyze("CLEAN GUNW A_013", z["ww_unw"], z["mask"], z["coh"],
                z["prod_cc"].astype(np.int64))


if __name__ == "__main__":
    main()
