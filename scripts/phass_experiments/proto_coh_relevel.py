"""Coherence-confident region re-leveling: snap a coherent block that got the
wrong integer (e.g. A_016's water-tile block, +1) to the mainland, while leaving
genuinely-ambiguous low-coherence ISLANDS alone.

Operates on the multi-shift result. Segment by reliable (no-cut) edges -> regions.
For each adjacent region pair, the shared boundary is the set of CUT edges between
them; record the coherence-weighted dominant cut integer and the boundary's MEAN
coherence. Anchor = largest region; propagate integer offsets via a spanning tree,
traversing ONLY boundaries whose mean coherence exceeds COH_BOUNDARY (a cut through
coherent terrain is wrong and confidently fixable; a low-coherence moat is a true
ambiguity and is NOT crossed -> island keeps its level).

Validate: A_016 block fixed, island untouched, match up; clean scene = no-op.
"""
from __future__ import annotations

import heapq
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

TAU = 2 * np.pi
COH_BOUNDARY = 0.5      # only level across boundaries whose mean coherence exceeds this
MIN_REGION = 200        # ignore tiny speckle regions as graph nodes


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def relevel(unw, ig, coh, valid):
    H, W = unw.shape
    idx = np.full((H, W), -1, np.int64)
    idx[valid] = np.arange(int(valid.sum()))
    n = int(valid.sum())
    # cut integer on each edge
    dh = np.rint((unw[:, 1:] - unw[:, :-1] - wrap(ig[:, 1:] - ig[:, :-1])) / TAU)
    dv = np.rint((unw[1:, :] - unw[:-1, :] - wrap(ig[1:, :] - ig[:-1, :])) / TAU)
    vh = valid[:, :-1] & valid[:, 1:]
    vv = valid[:-1, :] & valid[1:, :]
    # reliable (no-cut) edges -> regions
    rh = vh & (dh == 0); rv = vv & (dv == 0)
    r = np.concatenate([idx[:, :-1][rh], idx[:-1, :][rv]])
    c = np.concatenate([idx[:, 1:][rh], idx[1:, :][rv]])
    g = coo_matrix((np.ones(r.size, np.uint8), (r, c)), shape=(n, n))
    nreg, lab1d = connected_components(g, directed=False)
    region = np.full((H, W), -1, np.int64); region[valid] = lab1d
    sizes = np.bincount(lab1d)
    big = set(np.where(sizes >= MIN_REGION)[0].tolist())

    # region-pair boundaries (cut edges): coherence-weighted cut integer + mean coh
    votes = defaultdict(lambda: defaultdict(float))   # (lo,hi)->{k:weight}
    cohsum = defaultdict(float); cohcnt = defaultdict(int)
    chm = np.minimum(coh[:, :-1], coh[:, 1:]); cvm = np.minimum(coh[:-1, :], coh[1:, :])

    def add(ra, rb, dval, cc):
        if ra == rb or ra not in big or rb not in big:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        s = 1.0 if ra < rb else -1.0
        votes[(lo, hi)][s * dval] += cc
        cohsum[(lo, hi)] += cc; cohcnt[(lo, hi)] += 1

    ii, jj = np.where(vh & (dh != 0) & (region[:, :-1] != region[:, 1:]))
    for a_i, a_j in zip(ii.tolist(), jj.tolist()):
        add(region[a_i, a_j], region[a_i, a_j + 1], dh[a_i, a_j], chm[a_i, a_j])
    ii, jj = np.where(vv & (dv != 0) & (region[:-1, :] != region[1:, :]))
    for a_i, a_j in zip(ii.tolist(), jj.tolist()):
        add(region[a_i, a_j], region[a_i + 1, a_j], dv[a_i, a_j], cvm[a_i, a_j])

    adj = defaultdict(list)
    for (lo, hi), vmap in votes.items():
        mean_coh = cohsum[(lo, hi)] / max(1, cohcnt[(lo, hi)])
        if mean_coh < COH_BOUNDARY:        # low-coh moat -> do NOT cross (island stays)
            continue
        kbest = max(vmap.items(), key=lambda kv: kv[1])[0]  # cut integer lo->hi
        wt = cohsum[(lo, hi)]
        adj[lo].append((hi, -kbest, wt))   # o_hi = o_lo - kbest  (cancel the cut)
        adj[hi].append((lo, kbest, wt))

    anchor = max(big, key=lambda rr: sizes[rr])
    off = {anchor: 0}
    heap = [(-wt, anchor, nb, k) for (nb, k, wt) in adj[anchor]]
    heapq.heapify(heap)
    while heap:
        _, src, nb, k = heapq.heappop(heap)
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
            out[region == rr] += TAU * o; nshift += 1
    return out, nshift


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "a016"
    if target == "a016":
        d = np.load(Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/a016_diag/a016_default_fixed.npz"))
        unw = d["unw"].astype(np.float64); prod = d["prod"].astype(np.float64); pcc = d["pcc"]
        coh = d["coh"].astype(np.float64); mask = d["mask"]; ig = d["ig"].astype(np.float64)
    else:  # a clean scene from the reuse arrays
        fr = {"A_013": "NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_20251017T124857_20251029T124836_20251029T124858_X05010_N_P_J_001",
              "D_074": "NISAR_L2_PR_GUNW_003_005_D_074_004_4000_SH_20251017T132342_20251017T132345_20251029T132342_20251029T132346_X05010_N_P_J_001"}[target]
        d = np.load(Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_reuse") / fr / "full_arrays.npz")
        unw = d["ww_unw"].astype(np.float64); prod = d["prod_unw"].astype(np.float64); pcc = d["prod_cc"]
        coh = d["coh"].astype(np.float64); mask = d["mask"]; ig = d["ig"].astype(np.float64)
    reg = mask & (pcc > 0) & np.isfinite(unw) & np.isfinite(prod)

    def match(u):
        a = np.rint((u - prod) / TAU)[reg]; a = a[np.isfinite(a)]; return 100 * np.mean(np.abs(a - modal(a)) < 0.5)

    before = match(unw)
    out, nshift = relevel(unw, ig, coh, mask & np.isfinite(unw))
    after = match(out)
    # block + island specifics for a016
    if target == "a016":
        a0 = np.rint((unw - prod) / TAU); a0 = a0 - modal(a0[reg])
        a1 = np.rint((out - prod) / TAU); a1 = a1 - modal(a1[reg])
        blk = np.zeros_like(mask); blk[1600:1803, 2834:2944] = True
        isl = np.zeros_like(mask); isl[943:1736, 1133:1722] = True
        print(f"  block off: before={int((blk&reg&(np.abs(a0)>=1)).sum()):,} after={int((blk&reg&(np.abs(a1)>=1)).sum()):,}")
        print(f"  island off: before={int((isl&reg&(np.abs(a0)>=1)).sum()):,} after={int((isl&reg&(np.abs(a1)>=1)).sum()):,} (should stay ~same)")
    print(f"{target}: match before={before:.2f}%  after={after:.2f}%  (regions shifted={nshift})")


if __name__ == "__main__":
    main()
