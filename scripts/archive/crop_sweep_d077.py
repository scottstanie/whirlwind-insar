#!/usr/bin/env python3
"""Scale test: run ww convex (whole-image) AND snaphu on CENTER CROPS of D_077
of increasing size, scoring per-component match vs the production crop. If
ww-convex match is high on small crops and collapses as size grows, the
whole-image failure is the SOLVER-at-scale (batched-augment staleness), not the
cost. snaphu (a sound convex solver) is the control: it should stay high.

    WHIRLWIND_TILE_SOLVER=convex env -u CONDA_PREFIX uv run --no-sync \
        python scripts/crop_sweep_d077.py
"""
from __future__ import annotations

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
    sn = np.load(WD / "snaphu_ref" / "D_077.npz")
    prod_unw = sn["prod_unw"].astype(np.float32)
    prod_cc = sn["prod_cc"].astype(np.int64)
    coh = np.nan_to_num(sn["coh"]).astype(np.float32)
    mask = sn["mask"]
    wrapped = np.nan_to_num(sn["wrapped"]).astype(np.float32)
    M, N = prod_unw.shape
    ig_full = np.exp(1j * wrapped).astype(np.complex64)

    import whirlwind as ww
    solver = os.environ.get("WHIRLWIND_TILE_SOLVER", "reuse")
    do_snaphu = solver == "convex"
    if do_snaphu:
        import snaphu

    print(f"D_077 {M}x{N}  ww-solver={solver}")
    print(f"{'crop':>6s} {'ww_convex_whole':>16s} {'snaphu':>10s} {'ww_recall':>10s} {'sec':>6s}")
    for s in [256, 512, 1024, 2048, 3072]:
        if s > min(M, N):
            continue
        i0 = (M - s) // 2
        j0 = (N - s) // 2
        sl = (slice(i0, i0 + s), slice(j0, j0 + s))
        ig = np.ascontiguousarray(ig_full[sl])
        co = np.ascontiguousarray(coh[sl])
        mk = np.ascontiguousarray(mask[sl])
        pu = np.ascontiguousarray(prod_unw[sl])
        pc = np.ascontiguousarray(prod_cc[sl])
        wr = np.ascontiguousarray(wrapped[sl])
        big = s + 100

        t0 = time.perf_counter()
        u, cc = ww.unwrap(ig, co, 16.0, mk, tile_size=big, tile_overlap=0)
        dt = time.perf_counter() - t0
        u = np.asarray(u, np.float32)
        v = mk & np.isfinite(u)
        m_ww = percomp_match(u, pu, wr, pc, v)
        recall = float(np.mean(np.asarray(cc)[mk] > 0)) if mk.any() else float("nan")

        m_sn = float("nan")
        if do_snaphu:
            try:
                us, _ = snaphu.unwrap(ig, co, 16.0, cost="smooth", init="mcf", mask=mk)
                us = np.asarray(us, np.float32)
                m_sn = percomp_match(us, pu, wr, pc, mk & np.isfinite(us))
            except Exception as e:
                m_sn = float("nan")
                print("   snaphu crop err:", e)
        print(f"{s:>6d} {m_ww*100:>15.2f}% {m_sn*100:>9.2f}% {recall*100:>9.2f}% {dt:>6.1f}")


if __name__ == "__main__":
    main()
