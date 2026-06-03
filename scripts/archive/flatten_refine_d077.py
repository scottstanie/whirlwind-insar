#!/usr/bin/env python3
"""Test the 'fast tiled mostly-right -> whole-image refine' hybrid (the
anchor/flatten idea). For an anchor field A (oracle=production, or ww's own
tiled solve), flatten the wrapped phase by A, unwrap the residual WHOLE-IMAGE
(no tiling -> no seams), and add A back. If `final` matches production while
being seamless, the anchored whole-image solve beats the runaway and the tiling
artifacts vanish. No Rust changes — pure orchestration of ww.unwrap.

    WHIRLWIND_TILE_SOLVER=reuse env -u CONDA_PREFIX uv run --no-sync \
        python scripts/flatten_refine_d077.py --size 1024
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

TWOPI = 2.0 * np.pi
WD = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")


def wrap(x):
    return (x + np.pi) % TWOPI - np.pi


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
    ap.add_argument("--size", type=int, default=0, help="0 = full frame; else center crop")
    args = ap.parse_args()

    sn = np.load(WD / "snaphu_ref" / "D_077.npz")
    prod_unw = sn["prod_unw"].astype(np.float32)
    prod_cc = sn["prod_cc"].astype(np.int64)
    coh = np.nan_to_num(sn["coh"]).astype(np.float32)
    mask = sn["mask"]
    wrapped = np.nan_to_num(sn["wrapped"]).astype(np.float32)
    M, N = prod_unw.shape
    if args.size:
        s = args.size
        i0, j0 = (M - s) // 2, (N - s) // 2
        sl = (slice(i0, i0 + s), slice(j0, j0 + s))
        prod_unw, prod_cc, coh, mask, wrapped = (a[sl].copy() for a in (prod_unw, prod_cc, coh, mask, wrapped))
        M, N = s, s
    ig = np.exp(1j * wrapped).astype(np.complex64)
    big = max(M, N) + 100
    valid = mask & np.isfinite(prod_unw)

    import whirlwind as ww

    def match(field):
        v = mask & np.isfinite(field)
        return percomp_match(np.asarray(field, np.float32), prod_unw, wrapped, prod_cc, v)

    print(f"D_077 {M}x{N}")

    # Baselines
    t0 = time.perf_counter()
    u_tiled, _ = ww.unwrap(ig, coh, 16.0, mask)  # default tiling
    print(f"  tiled (anchor)        : {match(u_tiled)*100:6.2f}%  ({time.perf_counter()-t0:.1f}s)")
    t0 = time.perf_counter()
    u_whole, _ = ww.unwrap(ig, coh, 16.0, mask, tile_size=big, tile_overlap=0)
    print(f"  whole-image (runaway) : {match(u_whole)*100:6.2f}%  ({time.perf_counter()-t0:.1f}s)")

    # Hybrid: flatten by anchor, refine residual whole-image, add back.
    for name, anchor in [("oracle=production", prod_unw.astype(np.float32)),
                         ("tiled-solve", np.asarray(u_tiled, np.float32))]:
        a = np.nan_to_num(anchor)
        flat = wrap(wrapped - a).astype(np.float32)
        ig_flat = np.exp(1j * flat).astype(np.complex64)
        t0 = time.perf_counter()
        resid, _ = ww.unwrap(ig_flat, coh, 16.0, mask, tile_size=big, tile_overlap=0)
        final = np.asarray(resid, np.float32) + a
        dt = time.perf_counter() - t0
        print(f"  flatten+refine [{name:16s}]: {match(final)*100:6.2f}%  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
