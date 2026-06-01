"""Prototype the "reoptimize at the end" region reconciliation on A_016's
tiled-512 output (57.5%). The fine solve splits production-comp1 into a LEFT
(correct) and RIGHT (+1 cycle) region across a low-coherence neck; the tile-grid
reconciliation can't decide the neck because each tile-seam there is weak.

Fix idea: segment the fine result into no-jump regions, build a region graph
whose edges aggregate the coherence weight over the FULL shared boundary (so the
weak neck's many pixels sum to a confident vote), then choose per-region integer
2pi offsets by a coherence-weighted-mode vote anchored to the largest region.
If this recovers ~97%+, the region-reconciliation fix is validated -> port to Rust.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
ORIG = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_bench") / A016 / "full_arrays.npz"
TAU = 2 * np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def match(unw, prod, reg):
    a = np.rint((unw - prod) / TAU)[reg]; a = a[np.isfinite(a)]; a = a - modal(a)
    return 100 * np.mean(np.abs(a) < 0.5)


def main() -> None:
    d = np.load(ORIG)
    mask = d["mask"]; prod = d["prod_unw"]; pcc = d["prod_cc"]; coh = d["coh"]; unw = d["ww_unw"].astype(np.float64)
    valid = mask & np.isfinite(unw)
    H, W = unw.shape
    cc_reg = mask & (pcc > 0)
    print(f"A_016 {unw.shape}  before: cc>0 match={match(unw, prod, cc_reg):.2f}%", flush=True)

    # 1) segment into no-jump regions (edge kept iff round(dunw/2pi)==0)
    idx = np.full((H, W), -1, np.int64); idx[valid] = np.arange(int(valid.sum()))
    n = int(valid.sum())
    rows, cols = [], []
    er = valid[:, :-1] & valid[:, 1:] & (np.rint((unw[:, 1:] - unw[:, :-1]) / TAU) == 0)
    rows.append(idx[:, :-1][er]); cols.append(idx[:, 1:][er])
    ed = valid[:-1, :] & valid[1:, :] & (np.rint((unw[1:, :] - unw[:-1, :]) / TAU) == 0)
    rows.append(idx[:-1, :][ed]); cols.append(idx[1:, :][ed])
    r = np.concatenate(rows); c = np.concatenate(cols)
    g = coo_matrix((np.ones(r.size, np.uint8), (r, c)), shape=(n, n))
    nreg, lab1d = connected_components(g, directed=False)
    region = np.full((H, W), -1, np.int64); region[valid] = lab1d
    sizes = np.bincount(lab1d)
    big = np.where(sizes >= 2000)[0]  # regions worth leveling; tiny speckle ignored
    print(f"  {nreg} regions; {big.size} with >=2000 px (cover {100*sizes[big].sum()/n:.1f}% of valid)", flush=True)

    # 2) region adjacency: aggregate coherence weight + signed jump over FULL boundary
    isbig = np.zeros(nreg, bool); isbig[big] = True
    edge_w = {}   # (rA,rB) -> total weight ; edge_j sum -> for weighted-mode of integer jump
    def add(ia, ib, ja, jb, kk, ww):
        ra = region[ia, ja]; rb = region[ib, jb]
        if ra == rb or ra < 0 or rb < 0 or not isbig[ra] or not isbig[rb]:
            return
        key = (ra, rb) if ra < rb else (rb, rb if False else ra)
        key = (min(ra, rb), max(ra, rb))
        sgn = 1 if ra < rb else -1
        # jump so that o_low - o_high desired = sgn*k? store votes per integer
        edge_w.setdefault(key, {})
        kkey = int(kk) * (1 if ra < rb else -1)  # k from low->high region
        edge_w[key][kkey] = edge_w[key].get(kkey, 0.0) + float(ww)
    # horizontal boundary jumps
    hb = valid[:, :-1] & valid[:, 1:]
    kk = np.rint((unw[:, 1:] - unw[:, :-1]) / TAU)
    w = np.minimum(coh[:, :-1], coh[:, 1:])
    ii, jj = np.where(hb & (kk != 0))
    for a_i, a_j in zip(ii, jj):
        add(a_i, a_i, a_j, a_j + 1, kk[a_i, a_j], w[a_i, a_j])
    vb = valid[:-1, :] & valid[1:, :]
    kk2 = np.rint((unw[1:, :] - unw[:-1, :]) / TAU)
    w2 = np.minimum(coh[:-1, :], coh[1:, :])
    ii, jj = np.where(vb & (kk2 != 0))
    for a_i, a_j in zip(ii, jj):
        add(a_i, a_i + 1, a_j, a_j, kk2[a_i, a_j], w2[a_i, a_j])
    print(f"  region-graph edges: {len(edge_w)}", flush=True)

    # collapse each edge to (best integer jump low->high, total weight)
    adj = {}
    for (ra, rb), votes in edge_w.items():
        kbest = max(votes.items(), key=lambda kv: kv[1])[0]
        wt = sum(votes.values())
        adj.setdefault(ra, []).append((rb, kbest, wt))
        adj.setdefault(rb, []).append((ra, -kbest, wt))

    # 3) anchored coherence-weighted-mode vote (like coarse_refine else-branch)
    anchor = big[np.argmax(sizes[big])]
    off = {anchor: 0}
    order = [rr for rr in big if rr != anchor]
    for _ in range(300):
        changed = False
        for rr in order:
            votes = {}
            for nb, kk_, wt in adj.get(rr, []):
                if nb in off:
                    # want unw[rr] + 2pi*o_rr ~ unw[nb]+2pi*o_nb ; boundary jump
                    # (unw[high]-unw[low]) ~ 2pi*kk so o_low - o_high = kk (low->high)
                    cand = off[nb] + kk_   # o_rr = o_nb + (jump from nb->rr)
                    votes[cand] = votes.get(cand, 0.0) + wt
            if votes:
                best = max(votes.items(), key=lambda kv: kv[1])[0]
                if off.get(rr) != best:
                    off[rr] = best; changed = True
        if not changed:
            break

    out = unw.copy()
    for rr, o in off.items():
        if o != 0:
            out[region == rr] += TAU * o
    print(f"  after region-reopt: cc>0 match={match(out, prod, cc_reg):.2f}%", flush=True)


if __name__ == "__main__":
    main()
