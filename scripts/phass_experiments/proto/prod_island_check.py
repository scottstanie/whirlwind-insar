"""What do PRODUCTION's kept components look like (coherence, size, location),
and what does whirlwind do with those same pixels? This tests the context's
claim that production keeps small coherent islands that the 1% floor would drop.
Also: cumulative size-rank curve to locate the natural separating gap.
"""
from __future__ import annotations

import numpy as np

A016 = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/a016_diag/a016_default_fixed.npz"
CLEAN = (
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_reuse/"
    "NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_20251017T124857_"
    "20251029T124836_20251029T124858_X05010_N_P_J_001/full_arrays.npz"
)


def prod_summary(name, prod_cc, coh, mask):
    print(f"\n{'='*78}\n{name}  PRODUCTION components")
    nvalid = int(mask.sum())
    for lbl in np.unique(prod_cc):
        if lbl <= 0:
            continue
        sel = prod_cc == lbl
        sz = int(sel.sum())
        mc = float(coh[sel].mean())
        print(f"  prod L{lbl}: size={sz:>10,} ({100*sz/nvalid:6.3f}% valid)  "
              f"mean_coh={mc:.3f}")
    # pixels valid in mask but prod dropped (prod_cc<=0 within mask)
    dropped = mask & (prod_cc <= 0)
    print(f"  valid px dropped by production (prod_cc<=0): {int(dropped.sum()):,} "
          f"({100*dropped.sum()/nvalid:.3f}% valid)  "
          f"mean_coh(dropped)={float(coh[dropped].mean()) if dropped.any() else float('nan'):.3f}")


def gap_curve(name, sizes):
    sizes = np.sort(sizes)[::-1]
    print(f"\n  {name}: top-20 size-rank + ratio to next:")
    for i in range(min(20, sizes.size - 1)):
        ratio = sizes[i] / max(sizes[i + 1], 1)
        flag = "  <== GAP" if ratio > 5 else ""
        print(f"    rank {i+1:>3}: {int(sizes[i]):>10,}  (x{ratio:6.1f} over next){flag}")


def main():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    def wwsizes(unw, mask):
        u = unw.astype(np.float32); m = mask & np.isfinite(u)
        h, w = u.shape; idx = np.full((h, w), -1, np.int64)
        idx[m] = np.arange(int(m.sum()))
        rr, cc = [], []
        a = m[:, :-1] & m[:, 1:] & (np.abs(u[:, :-1] - u[:, 1:]) < np.pi)
        rr.append(idx[:, :-1][a]); cc.append(idx[:, 1:][a])
        b = m[:-1, :] & m[1:, :] & (np.abs(u[:-1, :] - u[1:, :]) < np.pi)
        rr.append(idx[:-1, :][b]); cc.append(idx[1:, :][b])
        r = np.concatenate(rr); c = np.concatenate(cc)
        g = coo_matrix((np.ones(r.size, np.uint8), (r, c)),
                       shape=(int(m.sum()),) * 2)
        _, lab = connected_components(g, directed=False)
        return np.bincount(lab)

    with np.load(A016) as z:
        prod_summary("A_016", z["pcc"], z["coh"], z["mask"])
        gap_curve("A_016 whirlwind", wwsizes(z["unw"], z["mask"]))
    with np.load(CLEAN) as z:
        prod_summary("CLEAN A_013", z["prod_cc"].astype(np.int64), z["coh"], z["mask"])
        gap_curve("A_013 whirlwind", wwsizes(z["ww_unw"], z["mask"]))


if __name__ == "__main__":
    main()
