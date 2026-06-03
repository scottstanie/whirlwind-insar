#!/usr/bin/env python3
"""Minimal: ww default TILED unwrap on D_077 (full or crop), per-comp match vs
production + runtime. Used to A/B the adaptive coarse-anchor multilook
(WHIRLWIND_ANCHOR_LK env)."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

TWOPI = 2.0 * np.pi
WD = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")


def percomp_match(test, prod_unw, wrapped, prod_cc, valid):
    amb = np.rint((test - wrapped) / TWOPI) - np.rint((prod_unw - wrapped) / TWOPI)
    in_comp = valid & (prod_cc > 0)
    if not in_comp.any():
        return float("nan")
    off = np.zeros(amb.shape)
    for lab in np.unique(prod_cc[in_comp]):
        m = valid & (prod_cc == lab)
        off[m] = np.rint(np.median(amb[m]))
    return float(np.mean((amb - off)[in_comp] == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=0)
    args = ap.parse_args()
    sn = np.load(WD / "snaphu_ref" / "D_077.npz")
    prod_unw = sn["prod_unw"].astype(np.float32)
    prod_cc = sn["prod_cc"].astype(np.int64)
    coh = np.nan_to_num(sn["coh"]).astype(np.float32)
    mask = sn["mask"]
    wrapped = np.nan_to_num(sn["wrapped"]).astype(np.float32)
    M, N = prod_unw.shape
    if args.size:
        s = args.size; i0, j0 = (M - s) // 2, (N - s) // 2
        sl = (slice(i0, i0 + s), slice(j0, j0 + s))
        prod_unw, prod_cc, coh, mask, wrapped = (a[sl].copy() for a in (prod_unw, prod_cc, coh, mask, wrapped))
        M, N = s, s
    ig = np.exp(1j * wrapped).astype(np.complex64)
    import whirlwind as ww
    t0 = time.perf_counter()
    u, cc = ww.unwrap(ig, coh, 16.0, mask)  # default tiling
    dt = time.perf_counter() - t0
    u = np.asarray(u, np.float32)
    v = mask & np.isfinite(u)
    pc = percomp_match(u, prod_unw, wrapped, prod_cc, v)
    recall = float(np.mean(np.asarray(cc)[mask] > 0))
    print(f"ANCHOR_LK={os.environ.get('WHIRLWIND_ANCHOR_LK','adaptive'):>8s}  {M}x{N}  "
          f"tiled per-comp={pc*100:6.2f}%  recall={recall*100:.1f}%  ncc={int(np.asarray(cc).max())}  {dt:.1f}s")


if __name__ == "__main__":
    main()
