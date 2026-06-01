"""Drill into the size GAP and use COHERENCE to confirm whether mid-size
components are real coherent islands or decorrelated noise speckle.

For each component we record (size, mean_coh). Real islands should sit at high
coherence; noise speckle at low coherence. This tells us whether an absolute
pixel floor cleanly separates the two populations.
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
    _, lab = connected_components(g, directed=False)
    full = np.full((h, w), -1, np.int64)
    full[m] = lab
    return full, m


def per_comp(name, unw, mask, coh):
    lab, m = labels(unw, mask)
    flat_lab = lab[m]
    flat_coh = coh[m]
    n = flat_lab.max() + 1
    sizes = np.bincount(flat_lab, minlength=n)
    coh_sum = np.bincount(flat_lab, weights=flat_coh, minlength=n)
    mean_coh = coh_sum / np.maximum(sizes, 1)
    order = np.argsort(sizes)[::-1]
    sizes = sizes[order]; mean_coh = mean_coh[order]

    print(f"\n{'='*78}\n{name}")
    # mean coh of components grouped by size bucket
    print("  mean coherence of components by size bucket:")
    edges = [(1, 9), (10, 99), (100, 999), (1000, 9999), (10000, 10**12)]
    for lo, hi in edges:
        sel = (sizes >= lo) & (sizes <= hi)
        if sel.sum() == 0:
            continue
        # size-weighted mean coh across comps in bucket
        wc = np.average(mean_coh[sel], weights=sizes[sel])
        hilbl = ">=1e4" if hi > 1e11 else str(hi)
        print(f"    {f'{lo}-{hilbl}':>12}  ncomp={int(sel.sum()):>6}  "
              f"px-wtd mean_coh={wc:.3f}  "
              f"comp-mean_coh range [{mean_coh[sel].min():.3f},{mean_coh[sel].max():.3f}]")

    # list the mid/large components (>=100 px) with size + coh, to eyeball the gap
    print("\n  components >= 100 px (size, mean_coh):")
    sel = sizes >= 100
    for s, ch in zip(sizes[sel][:40], mean_coh[sel][:40]):
        print(f"    {int(s):>10,} px   coh={ch:.3f}")
    if sel.sum() > 40:
        print(f"    ... ({int(sel.sum())-40} more components >=100 px)")

    # how many components >=100px are 'high coh' (>0.5) vs low
    big = sizes >= 50
    hi_coh = big & (mean_coh > 0.5)
    lo_coh = big & (mean_coh <= 0.5)
    print(f"\n  among comps >=50 px: high-coh(>0.5)={int(hi_coh.sum())} "
          f"(px={int(sizes[hi_coh].sum()):,}), "
          f"low-coh(<=0.5)={int(lo_coh.sum())} (px={int(sizes[lo_coh].sum()):,})")


def main():
    with np.load(A016) as z:
        per_comp("A_016 (fragmented)", z["unw"], z["mask"], z["coh"])
    with np.load(CLEAN) as z:
        per_comp("CLEAN GUNW A_013", z["ww_unw"], z["mask"], z["coh"])


if __name__ == "__main__":
    main()
