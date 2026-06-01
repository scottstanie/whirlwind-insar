"""Does a FINER coarse anchor level A_016's left/right correctly?

The tiled pipeline anchors region integers to a whole-image coarse solve at lk=8.
diag_gunw_anchor showed the lk=8 anchor is SYSTEMATICALLY +1 on the right half
(100% wrong) — so the cascade inherits the error. Per the unifying principle
(fewer full-res edges across the neck => wrong integer cheaper), a finer anchor
(smaller lk => more edges) may level it correctly.

Sweep lk in {2,3,4,6,8}; for each, coarse-unwrap the x-lk multilooked whole image
under BOTH costs, upsample, and report the dominant anchor-vs-production ambiguity
on the LEFT (correct) and RIGHT (drifted) halves of production comp1. A finer lk
that gives left=0 AND right=0 would fix A_016 via the existing cascade.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import whirlwind as ww

BENCH = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_bench")
A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
TAU = 2 * np.pi
FRONTIER = 2075  # left<frontier correct, right>=frontier drifts at tile512


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def ml(igc, coh, mask, L):
    h, w = igc.shape
    H, W = h // L, w // L
    igo = np.zeros((H, W), np.complex64); co = np.zeros((H, W), np.float32); mo = np.zeros((H, W), bool)
    for ci in range(H):
        sl_i = slice(ci * L, (ci + 1) * L)
        for cj in range(W):
            sl_j = slice(cj * L, (cj + 1) * L)
            sel = mask[sl_i, sl_j]
            nsel = int(sel.sum())
            if 2 * nsel >= L * L:
                z = igc[sl_i, sl_j][sel].sum()
                igo[ci, cj] = z / abs(z) if abs(z) > 0 else 0
                co[ci, cj] = coh[sl_i, sl_j][sel].mean()
                mo[ci, cj] = True
    return igo, co, mo


def dom(a):
    a = a[np.isfinite(a)].astype(int)
    if a.size == 0:
        return 0, 0.0
    vals, cnts = np.unique(a, return_counts=True)
    d = vals[np.argmax(cnts)]
    return int(d), 100.0 * float(np.mean(a == 0))


def main() -> None:
    d = np.load(BENCH / A016 / "full_arrays.npz")
    mask = d["mask"]; prod = d["prod_unw"]; pcc = d["prod_cc"]; coh = d["coh"]
    ig = d["ig"].astype(np.float32)
    igc = np.exp(1j * ig).astype(np.complex64); igc[~mask] = 0
    cohw = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    comp1 = mask & (pcc == 1) & np.isfinite(prod)
    cols = np.arange(prod.shape[1])[None, :]
    left = comp1 & (cols < FRONTIER)
    right = comp1 & (cols >= FRONTIER)
    print(f"A_016 comp1: left={left.sum():,} right={right.sum():,}\n", flush=True)

    for L in (2, 3, 4, 6, 8):
        igo, co, mo = ml(igc, cohw, mask, L)
        for cost in ("reuse", "linear"):
            if cost == "reuse":
                cunw = ww.unwrap_reuse(igo, co, 50.0 * L * L, mo)
            else:
                cunw = ww.unwrap(igo, co, 50.0 * L * L, mo, tile_size=100000, tile_overlap=64)
            cunw = np.asarray(cunw, np.float64)
            H, W = prod.shape
            ii = np.minimum(np.arange(H) // L, cunw.shape[0] - 1)
            jj = np.minimum(np.arange(W) // L, cunw.shape[1] - 1)
            up = cunw[np.ix_(ii, jj)]
            anch = np.where(mask, up, np.nan)
            amb = np.rint((anch - prod) / TAU)
            amb = amb - modal(amb[comp1])
            dl, ml_ = dom(amb[left]); dr, mr_ = dom(amb[right])
            flag = "  <-- LEVELS BOTH" if (dl == 0 and dr == 0) else ""
            print(f"  lk={L} {cost:6s} coarse={igo.shape}: left dom={dl:+d}(match0 {ml_:4.0f}%)  "
                  f"right dom={dr:+d}(match0 {mr_:4.0f}%){flag}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
