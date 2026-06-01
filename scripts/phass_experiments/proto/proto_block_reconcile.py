"""DATA-DRIVEN validation of the full-boundary reconcile (synthesis Branch ii).

Upper bound (proto): best per-512-block integer shift of the tile512 output = 99.74%
on A_016 => the per-tile SOLVES are fine, only the per-tile INTEGER OFFSETS (seams)
are wrong. Question: can a reconcile recover the right integers from DATA ALONE
(no production)?

Method (proxy on the saved composite): treat BxB blocks as nodes. Between adjacent
blocks, the relative integer = coherence-weighted MODE over the FULL shared 1-px
boundary of the wrapped-gradient-aware integer
    d_pq = round((unw_q - unw_p - wrap(ig_q - ig_p)) / 2pi)
(the branch-cut integer; shifting block B by -d cancels the systematic seam jump).
Anchor the largest block at 0; propagate along a MAX-WEIGHT spanning tree (most
reliable seams first). Apply per-block shift; measure cc>0 match.

If A_016 reaches >=95% and clean frames stay >=their tile512, the full-boundary
reconcile is validated data-drivenly -> port to Rust at TILE granularity.
"""
from __future__ import annotations

import heapq
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LEARN = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")
FRAMES = {
    "A_016": "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001",
    "A_013": "NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_20251017T124857_20251029T124836_20251029T124858_X05010_N_P_J_001",
    "D_074": "NISAR_L2_PR_GUNW_003_005_D_074_004_4000_SH_20251017T132342_20251017T132345_20251029T132342_20251029T132346_X05010_N_P_J_001",
}
TAU = 2 * np.pi


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def match(unw, prod, reg):
    a = np.rint((unw - prod) / TAU)[reg]
    a = a[np.isfinite(a)]
    return 100 * np.mean(np.abs(a - modal(a)) < 0.5)


def reconcile_blocks(unw, ig, coh, valid, B, kmax=4):
    H, W = unw.shape
    nbi, nbj = (H + B - 1) // B, (W + B - 1) // B
    bid = lambda bi, bj: bi * nbj + bj
    bsize = np.zeros(nbi * nbj)
    for bi in range(nbi):
        for bj in range(nbj):
            bsize[bid(bi, bj)] = valid[bi*B:(bi+1)*B, bj*B:(bj+1)*B].sum()

    # relative integer + weight between horizontally/vertically adjacent blocks,
    # from the coherence-weighted mode of d over the shared boundary column/row.
    adj = defaultdict(list)

    def seam(pblk, qblk, p_unw, q_unw, p_ig, q_ig, w):
        # d for each boundary pixel where both valid
        ok = np.isfinite(p_unw) & np.isfinite(q_unw)
        if ok.sum() < 20:
            return
        d = np.rint((q_unw - p_unw - wrap(q_ig - p_ig)) / TAU)[ok].astype(int)
        ww = w[ok]
        votes = defaultdict(float)
        for dv, wv in zip(d, ww):
            votes[dv] += wv
        kbest = max(votes.items(), key=lambda kv: kv[1])[0]
        wt = sum(votes.values())
        # want q shifted by -kbest relative to p:  o[q]-o[p] = -kbest
        adj[pblk].append((qblk, -kbest, wt))
        adj[qblk].append((pblk, kbest, wt))

    for bi in range(nbi):
        for bj in range(nbj):
            # right neighbor: boundary column j=(bj+1)*B-1 (in p) vs j=(bj+1)*B (in q)
            jr = (bj + 1) * B
            if jr < W:
                rows = slice(bi*B, min((bi+1)*B, H))
                p_un = unw[rows, jr-1]; q_un = unw[rows, jr]
                p_ig = ig[rows, jr-1]; q_ig = ig[rows, jr]
                w = np.minimum(coh[rows, jr-1], coh[rows, jr])
                seam(bid(bi,bj), bid(bi,bj+1), p_un, q_un, p_ig, q_ig, w)
            ib = (bi + 1) * B
            if ib < H:
                colsl = slice(bj*B, min((bj+1)*B, W))
                p_un = unw[ib-1, colsl]; q_un = unw[ib, colsl]
                p_ig = ig[ib-1, colsl]; q_ig = ig[ib, colsl]
                w = np.minimum(coh[ib-1, colsl], coh[ib, colsl])
                seam(bid(bi,bj), bid(bi+1,bj), p_un, q_un, p_ig, q_ig, w)

    # anchored max-weight spanning-tree propagation
    anchor = int(np.argmax(bsize))
    off = {anchor: 0}
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
    for bi in range(nbi):
        for bj in range(nbj):
            o = off.get(bid(bi, bj), 0)
            if o:
                sl = (slice(bi*B, min((bi+1)*B, H)), slice(bj*B, min((bj+1)*B, W)))
                m = valid[sl]
                out[sl] = np.where(m, out[sl] + o * TAU, out[sl])
    return out


def main() -> None:
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(FRAMES)
    for name in names:
        d = np.load(LEARN / "ww_gunw_bench" / FRAMES[name] / "full_arrays.npz")
        mask = d["mask"]; prod = d["prod_unw"].astype(np.float64); pcc = d["prod_cc"]
        coh = d["coh"].astype(np.float64); ig = d["ig"].astype(np.float64); unw = d["ww_unw"].astype(np.float64)
        valid = mask & np.isfinite(unw)
        reg = mask & (pcc > 0) & np.isfinite(unw)
        before = match(unw, prod, reg)
        for B in (256, 512):
            out = reconcile_blocks(unw, ig, coh, valid, B)
            after = match(out, prod, reg)
            print(f"{name}: B={B}  before={before:.2f}%  after={after:.2f}%", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
