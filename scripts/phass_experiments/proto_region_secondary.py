"""SNAPHU-style region-graph SECONDARY optimization as a post-pass on the
tile512 output. THE decisive prototype for A_016.

Key fix over the failed region_reopt (which segmented by NO-JUMP edges and merged
the drift): segment by BRANCH-CUT edges. An edge p->q is "reliable" (kept) iff the
integrated step equals the wrapped step:  round((unw_q-unw_p - wrap(ig_q-ig_p))/2pi)==0.
A +1 integer drift between two regions MUST be realized as net flow across a cut
line (MCF produces integer flows), so reliable-edge components SEPARATE left/right
where no-jump merged them.

Then build a region graph; each edge (A,B) aggregates over the FULL shared boundary
the coherence-weighted vote of the branch-cut integer d (= the relative-integer
correction). Solve per-region offsets anchored to the largest region (max-spanning-
tree by boundary weight, propagate). Apply, measure cc>0 match.

Run on A_016 (target >=95%) and A_013/D_074 (must NOT drop below their tile512).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

LEARN = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")
FRAMES = {
    "A_016": "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001",
    "A_013": "NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_20251017T124857_20251029T124836_20251029T124858_X05010_N_P_J_001",
    "D_074": "NISAR_L2_PR_GUNW_003_005_D_074_004_4000_SH_20251017T132342_20251017T132345_20251029T132342_20251029T132346_X05010_N_P_J_001",
}
TAU = 2 * np.pi
MIN_REGION = 2000  # px; smaller regions ride with their best neighbor (ignored as nodes)


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def match(unw, prod, reg):
    a = np.rint((unw - prod) / TAU)[reg]
    a = a[np.isfinite(a)]
    a = a - modal(a)
    return 100 * np.mean(np.abs(a) < 0.5)


def load(name):
    # prefer the linear tile512 saved arrays (ww_gunw_bench); has ig, unw, coh, mask, prod, pcc
    d = np.load(LEARN / "ww_gunw_bench" / FRAMES[name] / "full_arrays.npz")
    return (d["mask"], d["prod_unw"].astype(np.float64), d["prod_cc"],
            d["coh"].astype(np.float64), d["ig"].astype(np.float64),
            d["ww_unw"].astype(np.float64))


def region_secondary(unw, ig, coh, mask, thlo=0.0):
    """Return corrected unw via branch-cut region segmentation + full-boundary
    coherence-weighted relative-integer leveling."""
    H, W = unw.shape
    valid = mask & np.isfinite(unw)
    idx = np.full((H, W), -1, np.int64)
    idx[valid] = np.arange(int(valid.sum()))
    n = int(valid.sum())

    # branch-cut integer d on each 4-conn edge: round((dunw - wrap(dig))/2pi)
    # horizontal (p=(i,j) -> q=(i,j+1))
    vh = valid[:, :-1] & valid[:, 1:]
    dh = np.rint((unw[:, 1:] - unw[:, :-1] - wrap(ig[:, 1:] - ig[:, :-1])) / TAU)
    wh = np.minimum(coh[:, :-1], coh[:, 1:])
    vv = valid[:-1, :] & valid[1:, :]
    dv = np.rint((unw[1:, :] - unw[:-1, :] - wrap(ig[1:, :] - ig[:-1, :])) / TAU)
    wv = np.minimum(coh[:-1, :], coh[1:, :])

    # 1) reliable-edge connected components (d==0 AND coh>=thlo)
    rows, cols = [], []
    rel_h = vh & (dh == 0) & (np.minimum(coh[:, :-1], coh[:, 1:]) >= thlo)
    rows.append(idx[:, :-1][rel_h]); cols.append(idx[:, 1:][rel_h])
    rel_v = vv & (dv == 0) & (np.minimum(coh[:-1, :], coh[1:, :]) >= thlo)
    rows.append(idx[:-1, :][rel_v]); cols.append(idx[1:, :][rel_v])
    r = np.concatenate(rows); c = np.concatenate(cols)
    g = coo_matrix((np.ones(r.size, np.uint8), (r, c)), shape=(n, n))
    nreg, lab1d = connected_components(g, directed=False)
    region = np.full((H, W), -1, np.int64); region[valid] = lab1d
    sizes = np.bincount(lab1d)
    big = set(np.where(sizes >= MIN_REGION)[0].tolist())
    cover = 100 * sizes[list(big)].sum() / n if big else 0

    # 2) region-pair boundary: aggregate coherence-weighted votes of d (the
    #    relative integer to apply: o_q - o_p should cancel d). Edges where d!=0
    #    OR low-coh connect different reliable regions.
    votes = defaultdict(lambda: defaultdict(float))  # (rA,rB)->{k:weight}, rA<rB, k = o_low - o_high target
    def acc(ra, rb, d, w):
        if ra == rb or ra not in big or rb not in big:
            return
        # want o[ra] + (unw bias) ... store signed jump from low->high region index
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        sgn = 1.0 if ra < rb else -1.0
        votes[(lo, hi)][sgn * d] += w
    # iterate all boundary edges (d may be 0 across a low-coh split too; include all inter-region)
    ii, jj = np.where(vh & (region[:, :-1] != region[:, 1:]))
    for a_i, a_j in zip(ii.tolist(), jj.tolist()):
        acc(region[a_i, a_j], region[a_i, a_j + 1], dh[a_i, a_j], wh[a_i, a_j])
    ii, jj = np.where(vv & (region[:-1, :] != region[1:, :]))
    for a_i, a_j in zip(ii.tolist(), jj.tolist()):
        acc(region[a_i, a_j], region[a_i + 1, a_j], dv[a_i, a_j], wv[a_i, a_j])

    # collapse to best integer + weight per region pair
    adj = defaultdict(list)
    for (lo, hi), vmap in votes.items():
        kbest = max(vmap.items(), key=lambda kv: kv[1])[0]   # d low->high
        wt = sum(vmap.values())
        # correction: we want unw[hi]+2pi*o[hi] - (unw[lo]+2pi*o[lo]) to drop the
        # branch-cut integer d, i.e. o[hi]-o[lo] = -kbest. store as offset relation.
        adj[lo].append((hi, -kbest, wt))
        adj[hi].append((lo, kbest, wt))

    # 3) anchored max-spanning propagation (most reliable edges first)
    anchor = max(big, key=lambda rr: sizes[rr])
    off = {anchor: 0}
    # Prim-like: grow from anchor along highest-weight edges
    import heapq
    heap = [(-wt, anchor, nb, k) for (nb, k, wt) in adj[anchor]]
    heapq.heapify(heap)
    while heap:
        negw, src, nb, k = heapq.heappop(heap)
        if nb in off:
            continue
        off[nb] = off[src] + k
        for (nb2, k2, wt2) in adj[nb]:
            if nb2 not in off:
                heapq.heappush(heap, (-wt2, nb, nb2, k2))

    out = unw.copy()
    nshift = 0
    for rr, o in off.items():
        if o != 0:
            out[region == rr] += TAU * o
            nshift += 1
    return out, nreg, len(big), cover, nshift


def main() -> None:
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(FRAMES)
    for name in names:
        mask, prod, pcc, coh, ig, unw = load(name)
        reg = mask & (pcc > 0) & np.isfinite(unw)
        before = match(unw, prod, reg)
        for thlo in (0.0, 0.3):
            out, nreg, nbig, cover, nshift = region_secondary(unw, ig, coh, mask, thlo=thlo)
            after = match(out, prod, reg)
            print(f"{name}: thlo={thlo}  before={before:.2f}%  after={after:.2f}%  "
                  f"(regions={nreg}, big={nbig} cover {cover:.1f}%, shifted {nshift})", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
