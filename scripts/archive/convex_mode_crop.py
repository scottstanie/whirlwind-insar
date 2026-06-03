#!/usr/bin/env python3
"""Test a single WHIRLWIND_CONVEX_SOLVE mode (env, cached per process) on
center crops of D_077. Reports per-component match vs production + runtime, so
we can see whether ssp/cancel lift the crops the default `pd` solver fails on.

    WHIRLWIND_TILE_SOLVER=convex WHIRLWIND_CONVEX_SOLVE=cancel \
      env -u CONDA_PREFIX uv run --no-sync python scripts/convex_mode_crop.py --sizes 512 1024
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from pathlib import Path

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
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    args = ap.parse_args()
    mode = os.environ.get("WHIRLWIND_CONVEX_SOLVE", "pd")

    sn = np.load(WD / "snaphu_ref" / "D_077.npz")
    prod_unw = sn["prod_unw"].astype(np.float32)
    prod_cc = sn["prod_cc"].astype(np.int64)
    coh = np.nan_to_num(sn["coh"]).astype(np.float32)
    mask = sn["mask"]
    wrapped = np.nan_to_num(sn["wrapped"]).astype(np.float32)
    M, N = prod_unw.shape
    ig_full = np.exp(1j * wrapped).astype(np.complex64)

    import whirlwind as ww
    print(f"CONVEX_SOLVE={mode}  D_077 {M}x{N}")
    print(f"{'crop':>6s} {'per-comp':>9s} {'recall':>8s} {'sec':>7s}")
    for s in args.sizes:
        i0, j0 = (M - s) // 2, (N - s) // 2
        sl = (slice(i0, i0 + s), slice(j0, j0 + s))
        ig = np.ascontiguousarray(ig_full[sl]); co = np.ascontiguousarray(coh[sl])
        mk = np.ascontiguousarray(mask[sl]); pu = np.ascontiguousarray(prod_unw[sl])
        pc = np.ascontiguousarray(prod_cc[sl]); wr = np.ascontiguousarray(wrapped[sl])
        t0 = time.perf_counter()
        u, cc = ww.unwrap(ig, co, 16.0, mk, tile_size=s + 100, tile_overlap=0)
        dt = time.perf_counter() - t0
        u = np.asarray(u, np.float32)
        v = mk & np.isfinite(u)
        m_ww = percomp_match(u, pu, wr, pc, v)
        recall = float(np.mean(np.asarray(cc)[mk] > 0)) if mk.any() else float("nan")
        print(f"{s:>6d} {m_ww*100:>8.2f}% {recall*100:>7.2f}% {dt:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
