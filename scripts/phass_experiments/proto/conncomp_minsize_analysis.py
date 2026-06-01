"""Empirical component-size distribution analysis for the conncomp min-size policy.

Computes connected components from SAVED unwrapped arrays (no ww.unwrap re-run)
using the tear-based definition |d(unw)| < pi, matching make_report_figures.ww_conncomp
but at FULL resolution (stride=1) so small islands are not artificially merged/destroyed
by downsampling. Reports the full size distribution and where the 1% / absolute-floor
cutlines fall.
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def tear_components(unw, mask):
    """Full-res connected components of the unwrapped surface.
    Edge between valid neighbours iff |d(unw)| < pi. Returns label image
    (0=invalid) and array of component sizes sorted descending."""
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
    counts = np.bincount(lab)
    return ncomp, np.sort(counts)[::-1], nnodes


def label_sizes(cc):
    """Sizes of each nonzero label in a production conncomp array, desc."""
    cc = cc[cc > 0]
    if cc.size == 0:
        return np.array([], dtype=np.int64)
    counts = np.bincount(cc)
    counts = counts[counts > 0]
    return np.sort(counts)[::-1]


def summarize(name, sizes, n_valid, pixel_m=80.0):
    print(f"\n===== {name} =====")
    print(f"n_valid = {n_valid:,}")
    print(f"num components (>=1 px) = {len(sizes)}")
    frac1 = 0.01 * n_valid
    print(f"1% floor (current ww min_size_frac=0.01) = {frac1:,.0f} px")
    for absfloor in (100, 200, 500, 1000):
        kept_abs = int((sizes >= absfloor).sum())
        print(f"  abs floor {absfloor:>5} px: keep {kept_abs} comps")
    kept_1pct = int((sizes >= frac1).sum())
    print(f"  1% frac floor: keep {kept_1pct} comps")
    # cumulative coverage
    tot = sizes.sum()
    print(f"  total labeled px = {tot:,}")
    print("  --- largest components (size px, %valid, cum%valid) ---")
    cum = 0
    for i, s in enumerate(sizes[:25]):
        cum += s
        print(f"   #{i+1:>3}: {s:>10,} px  {100*s/n_valid:7.3f}%  cum {100*cum/n_valid:7.3f}%")
    # what is dropped by 1% but kept by abs floors
    for absfloor in (100, 200, 500, 1000):
        between = sizes[(sizes >= absfloor) & (sizes < frac1)]
        if len(between):
            print(f"  comps in [{absfloor}, {frac1:.0f}) px (dropped by 1%, kept by abs {absfloor}): "
                  f"n={len(between)}, total {between.sum():,} px = {100*between.sum()/n_valid:.3f}% valid, "
                  f"max={between.max():,}, min={between.min():,}")
    # noise-scale: how many tiny fragments
    for hi in (5, 10, 25, 50, 100, 200):
        n_tiny = int((sizes < hi).sum())
        print(f"  fragments < {hi:>4} px: count={n_tiny:>6}  (total {sizes[sizes<hi].sum():,} px)")
    return sizes


def main():
    A016 = '/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/a016_diag/a016_default_fixed.npz'
    d = np.load(A016)
    mask = d['mask']
    n_valid = int(mask.sum())

    # whirlwind unwrapped
    nc_ww, sz_ww, nv = tear_components(d['unw'], mask)
    summarize('A_016 whirlwind unw (tear components, full res)', sz_ww, n_valid)

    # production unwrapped (tear components)
    nc_pr, sz_pr, _ = tear_components(d['prod'], mask)
    summarize('A_016 production unw (tear components, full res)', sz_pr, n_valid)

    # production NATIVE conncomp labels (what SNAPHU actually emitted)
    sz_pcc = label_sizes(d['pcc'])
    summarize('A_016 production NATIVE conncomp labels (pcc)', sz_pcc, n_valid)

    # clean GUNW
    g = ('NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_'
         '20251017T124857_20251029T124836_20251029T124858_X05010_N_P_J_001')
    base = '/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_reuse/' + g
    dc = np.load(base + '/full_arrays.npz')
    maskc = dc['mask']
    nvc = int(maskc.sum())
    nc_wwc, sz_wwc, _ = tear_components(dc['ww_unw'], maskc)
    summarize('CLEAN A_013 whirlwind unw (tear components)', sz_wwc, nvc)
    sz_pccc = label_sizes(dc['prod_cc'])
    summarize('CLEAN A_013 production NATIVE conncomp (prod_cc)', sz_pccc, nvc)


if __name__ == '__main__':
    main()
