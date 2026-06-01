"""Empirical no-2pi-tear connected-component SIZE DISTRIBUTION for the
min-size-floor policy decision (conncomp.rs min_size_frac / max_ncomps).

Reuses the ww_conncomp tear definition from make_report_figures.py: two valid
neighbours are in the same component iff |d(unw)| < pi. Computed at FULL
resolution (stride=1) because we care about absolute physical pixel sizes
(80 m), not the downsampled display panels.

Light scipy only -- no ww.unwrap.

Run: /Users/staniewi/miniforge3/envs/mapping-312/bin/python \
       scripts/phass_experiments/conncomp_size_distribution.py
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


def component_sizes(unw, mask, stride=1):
    """Return sorted-descending array of component sizes (pixel counts) over the
    valid mask, where a 2pi tear (|d unw| >= pi) breaks connectivity. stride=1
    => full resolution."""
    u = unw[::stride, ::stride].astype(np.float32)
    m = (mask[::stride, ::stride]) & np.isfinite(u)
    h, w = u.shape
    idx = np.full((h, w), -1, np.int64)
    nnodes = int(m.sum())
    idx[m] = np.arange(nnodes)
    if nnodes == 0:
        return np.array([], np.int64), 0

    rows, cols = [], []
    a = m[:, :-1] & m[:, 1:] & (np.abs(u[:, :-1] - u[:, 1:]) < np.pi)
    rows.append(idx[:, :-1][a]); cols.append(idx[:, 1:][a])
    b = m[:-1, :] & m[1:, :] & (np.abs(u[:-1, :] - u[1:, :]) < np.pi)
    rows.append(idx[:-1, :][b]); cols.append(idx[1:, :][b])
    r = np.concatenate(rows); c = np.concatenate(cols)
    g = coo_matrix((np.ones(r.size, np.uint8), (r, c)), shape=(nnodes, nnodes))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    sizes = np.sort(sizes)[::-1]
    return sizes, nnodes


def buckets(sizes):
    edges = [(1, 9), (10, 99), (100, 999), (1000, 9999), (10000, 10**12)]
    out = []
    for lo, hi in edges:
        sel = (sizes >= lo) & (sizes <= hi)
        out.append((lo, hi, int(sel.sum()), int(sizes[sel].sum())))
    return out


def analyze(name, unw, mask):
    nvalid = int(mask.sum())
    sizes, nnodes = component_sizes(unw, mask, stride=1)
    print(f"\n{'='*78}\n{name}")
    print(f"  shape={mask.shape}  valid px={nvalid:,}  graph-nodes(valid&finite)={nnodes:,}")
    print(f"  total components = {sizes.size:,}")
    print(f"  largest 15 component sizes: {sizes[:15].tolist()}")
    px_m = 80.0
    print(f"\n  SIZE BUCKETS (px):")
    print(f"    {'bucket':>14} {'#comps':>8} {'sum px':>14} {'%valid px':>10}")
    for lo, hi, ncomp, npx in buckets(sizes):
        hilbl = ">=10000" if hi > 10**11 else str(hi)
        print(f"    {f'{lo}-{hilbl}':>14} {ncomp:>8} {npx:>14,} {100*npx/nvalid:>9.3f}%")

    print(f"\n  KEEP/DROP at candidate floors (component kept iff size >= floor):")
    print(f"    {'floor':>22} {'#kept':>6} {'#dropped':>9} {'kept px':>13} "
          f"{'dropped px':>12} {'%drop px':>9} {'side(km)':>9}")
    # absolute floors + fraction floors
    abs_floors = [10, 50, 100, 200, 500, 1000]
    frac_floors = [1e-4, 5e-4, 1e-3, 1e-2]
    rows = [("abs", f) for f in abs_floors] + [("frac", f) for f in frac_floors]
    for kind, fl in rows:
        thr = fl if kind == "abs" else int(np.ceil(fl * nvalid))
        kept = sizes[sizes >= thr]
        dropped = sizes[sizes < thr]
        side_km = (np.sqrt(thr) * px_m) / 1000.0
        if kind == "abs":
            lbl = f"abs {fl} px"
        else:
            lbl = f"{fl:g} ({thr:,} px)"
        print(f"    {lbl:>22} {kept.size:>6} {dropped.size:>9} "
              f"{int(kept.sum()):>13,} {int(dropped.sum()):>12,} "
              f"{100*dropped.sum()/nvalid:>8.4f}% {side_km:>8.2f}")
    return sizes, nvalid


def main():
    with np.load(A016) as z:
        a_sizes, a_nv = analyze("A_016 (fragmented, islands) -- whirlwind unw",
                                z["unw"], z["mask"])
        # also production conncomp on A_016 for reference
        pcc = z["pcc"]; mask = z["mask"]
        psizes = np.bincount(pcc[(pcc > 0)].ravel())
        psizes = np.sort(psizes[psizes > 0])[::-1]
        print(f"\n  [A_016 PRODUCTION (SNAPHU) native conncomp sizes]: {psizes.tolist()}")
        print(f"    -> smallest kept prod comp = {int(psizes.min()):,} px "
              f"= {100*psizes.min()/a_nv:.3f}% of valid")

    with np.load(CLEAN) as z:
        c_sizes, c_nv = analyze("CLEAN GUNW A_013 -- whirlwind unw",
                                z["ww_unw"], z["mask"])
        pcc = z["prod_cc"]
        psizes = np.bincount(pcc[(pcc > 0)].ravel())
        psizes = np.sort(psizes[psizes > 0])[::-1]
        print(f"\n  [A_013 PRODUCTION (SNAPHU) native conncomp sizes]: {psizes.tolist()}")
        if psizes.size:
            print(f"    -> smallest kept prod comp = {int(psizes.min()):,} px "
                  f"= {100*psizes.min()/c_nv:.3f}% of valid")


if __name__ == "__main__":
    main()
