"""Validate the best-of-both fix: keep the tile-512 fine solve (detail/quality)
but re-level its no-jump regions to a BIG-TILE (2048) solve that spans the neck
(correct region levels). If snapping 512's regions to the 2048 anchor recovers
~97% on A_016, then "anchor = big-tile full-res solve" is the fix (replacing the
neck-aliasing multilook anchor).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
BENCH = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_bench") / A016 / "full_arrays.npz"
ANCH = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_variants/tile2048") / A016 / "full_arrays.npz"
TAU = 2 * np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def main() -> None:
    d = np.load(BENCH); a = np.load(ANCH)
    mask = d["mask"]; prod = d["prod_unw"]; pcc = d["prod_cc"]; coh = d["coh"]
    fine = d["ww_unw"].astype(np.float64)        # tile-512 (57.5%)
    anchor = a["ww_unw"].astype(np.float64)       # tile-2048 (97%)
    reg = mask & (pcc > 0)

    def match(u):
        x = np.rint((u - prod) / TAU)[reg]; x = x[np.isfinite(x)]; x = x - modal(x)
        return 100 * np.mean(np.abs(x) < 0.5)
    print(f"fine(512)={match(fine):.2f}%  anchor(2048)={match(anchor):.2f}%", flush=True)

    # segment fine into no-jump regions
    valid = mask & np.isfinite(fine) & np.isfinite(anchor)
    H, W = fine.shape
    idx = np.full((H, W), -1, np.int64); idx[valid] = np.arange(int(valid.sum()))
    n = int(valid.sum())
    rows = []; cols = []
    er = valid[:, :-1] & valid[:, 1:] & (np.rint((fine[:, 1:] - fine[:, :-1]) / TAU) == 0)
    rows.append(idx[:, :-1][er]); cols.append(idx[:, 1:][er])
    ed = valid[:-1, :] & valid[1:, :] & (np.rint((fine[1:, :] - fine[:-1, :]) / TAU) == 0)
    rows.append(idx[:-1, :][ed]); cols.append(idx[1:, :][ed])
    r = np.concatenate(rows); c = np.concatenate(cols)
    g = coo_matrix((np.ones(r.size, np.uint8), (r, c)), shape=(n, n))
    nreg, lab = connected_components(g, directed=False)
    region = np.full((H, W), -1, np.int64); region[valid] = lab

    # per-region coherence-weighted mode of round((anchor-fine)/2pi); snap fine
    out = fine.copy()
    krel = np.rint((anchor - fine) / TAU)
    w = np.where(valid, coh, 0.0)
    sizes = np.bincount(lab)
    # accumulate weighted votes per (region, k)
    from collections import defaultdict
    votes = defaultdict(lambda: defaultdict(float))
    vi, vj = np.where(valid)
    kk = krel[valid]; ww = w[valid]; rr = region[valid]
    for ri, k, wt in zip(rr, kk, ww):
        if np.isfinite(k):
            votes[int(ri)][int(k)] += float(wt)
    off = {}
    for ri, vm in votes.items():
        off[ri] = max(vm.items(), key=lambda kv: kv[1])[0]
    snap = np.zeros((H, W))
    flat = np.array([off.get(int(region[i, j]), 0) if valid[i, j] else 0 for i, j in zip(*np.where(valid))])
    out[valid] = fine[valid] + TAU * flat
    print(f"after snap-512-regions-to-2048-anchor: cc>0 match={match(out):.2f}%  ({len(off)} regions snapped)", flush=True)


if __name__ == "__main__":
    main()
