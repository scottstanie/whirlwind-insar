"""Why doesn't the coarse anchor bridge A_016's two halves? Reconstruct the
multilook-8 coarse igram from the saved A_016 wrapped phase, unwrap the coarse
whole-image (== compute_coarse_anchor), upsample, and check whether the coarse
anchor levels the LEFT (ok) and RIGHT (offset) halves of production comp1
consistently vs production. Also check coarse-mask connectivity across the gap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BENCH = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_bench")
A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
TAU = 2 * np.pi
L = 8


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def ml(igc, coh, mask, L):
    h, w = igc.shape
    H, W = h // L, w // L
    igo = np.zeros((H, W), np.complex64); co = np.zeros((H, W), np.float32); mo = np.zeros((H, W), bool)
    for ci in range(H):
        for cj in range(W):
            sel = mask[ci*L:(ci+1)*L, cj*L:(cj+1)*L]
            n = int(sel.sum())
            if 2*n >= L*L:
                z = igc[ci*L:(ci+1)*L, cj*L:(cj+1)*L][sel].sum()
                igo[ci, cj] = z/abs(z) if abs(z) > 0 else 0
                co[ci, cj] = coh[ci*L:(ci+1)*L, cj*L:(cj+1)*L][sel].mean()
                mo[ci, cj] = True
    return igo, co, mo


def main() -> None:
    import whirlwind as ww
    d = np.load(BENCH / A016 / "full_arrays.npz")
    mask = d["mask"]; prod = d["prod_unw"]; pcc = d["prod_cc"]; coh = d["coh"]
    ig = d["ig"].astype(np.float32)  # wrapped phase (real)
    igc = np.exp(1j * ig).astype(np.complex64); igc[~mask] = 0
    cohw = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)

    igo, co, mo = ml(igc, cohw, mask, L)
    H, W = igo.shape
    print(f"coarse {igo.shape}  coarse-valid={mo.sum():,}", flush=True)

    # connectivity of the coarse mask (does it bridge the gap at col~2284 -> coarse ~285?)
    from scipy.ndimage import label
    lab, n = label(mo)
    sizes = np.bincount(lab.ravel())[1:]
    print(f"coarse mask: {n} connected pieces; largest={100*sizes.max()/mo.sum():.1f}% of coarse-valid", flush=True)

    # unwrap the coarse whole-image (this IS compute_coarse_anchor's core)
    canchor = ww.unwrap(igo, co, nlooks=50.0*L*L, mask=mo)
    up = np.kron(canchor, np.ones((L, L), np.float32))
    hh, wwd = up.shape
    m = mask[:hh, :wwd]; pr = prod[:hh, :wwd]; cc = pcc[:hh, :wwd]
    anch = np.where(m, up, np.nan)

    # Does the coarse anchor agree with production's RELATIVE level between the
    # left (ok) and right (offset) halves of comp1? Compare anchor-vs-prod
    # ambiguity on each side after a single global offset.
    comp1 = m & (cc == 1)
    amb = np.rint((anch - pr) / TAU)
    g = modal(amb[comp1])
    amb = amb - g
    left = comp1 & (np.arange(wwd)[None, :] < 2284)
    right = comp1 & (np.arange(wwd)[None, :] >= 2284)
    for nm, reg in [("left(<2284)", left), ("right(>=2284)", right)]:
        a = amb[reg]; a = a[np.isfinite(a)]
        if a.size:
            vals, cnts = np.unique(a.astype(int), return_counts=True)
            dom = vals[np.argmax(cnts)]
            print(f"  coarse ANCHOR vs prod, {nm}: dominant ambiguity={dom:+d}  match0={100*np.mean(a==0):.1f}%  "
                  f"|a|>=1={100*np.mean(np.abs(a)>=1):.1f}%", flush=True)
    print("If left dom=0 and right dom=+-1 -> the coarse anchor ITSELF mis-levels the two halves "
          "(the gap is wider than the x8 multilook can bridge / the coarse solve splits them).", flush=True)


if __name__ == "__main__":
    main()
