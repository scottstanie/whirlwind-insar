#!/usr/bin/env python3
"""Run snaphu single-tile (NISAR settings) on one GUNW frame and SAVE the
unwrapped output + conncomp to npz, alongside the inputs the ww bench used, so
we can compute ww's convex objective on snaphu's flow and plot a 3-way
production / snaphu / whirlwind comparison.

    env -u CONDA_PREFIX uv run --no-sync python scripts/snaphu_save_d077.py \
        --local-h5 <WD>/nisar_gunw/*D_077*.h5 --out <WD>/snaphu_ref/D_077.npz --nlooks 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_nisar_gunw_whirlwind import (  # noqa: E402
    TWOPI,
    gunw_paths,
    mask_to_bool,
    read_array,
    wrap_phase,
)


def percomp_match(test_unw, prod_unw, wrapped, prod_cc, valid):
    amb = np.rint((test_unw - wrapped) / TWOPI) - np.rint((prod_unw - wrapped) / TWOPI)
    in_comp = valid & (prod_cc > 0)
    if not in_comp.any():
        return float("nan")
    off = np.zeros(amb.shape, np.float64)
    for lab in np.unique(prod_cc[in_comp]):
        m = valid & (prod_cc == lab)
        off[m] = np.rint(np.median(amb[m]))
    return float(np.mean((amb - off)[in_comp] == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nlooks", type=float, default=16.0)
    args = ap.parse_args()

    import snaphu

    with h5py.File(args.local_h5, "r") as h:
        p = gunw_paths(h, None)
        prod_unw = read_array(h[p["unw"]], np.float32)
        coh = read_array(h[p["coh_unw"]], np.float32)
        prod_cc = h[p["cc"]][()].astype(np.int64, copy=False)
        mask_arr = h[p["mask"]][()] if p["mask"] in h else None
    mask = (
        mask_to_bool(mask_arr, "water_only", prod_unw.shape)
        & np.isfinite(prod_unw)
        & np.isfinite(coh)
    )
    wrapped = wrap_phase(prod_unw).astype(np.float32)
    ig = np.exp(1j * wrapped).astype(np.complex64)
    coh_c = np.clip(np.nan_to_num(coh), 0.0, 1.0).astype(np.float32)

    print(f"snaphu {snaphu.__version__} cost=smooth init=mcf min_region_size=300 shape={ig.shape}", flush=True)
    t0 = time.perf_counter()
    unw, cc = snaphu.unwrap(
        ig, coh_c, args.nlooks, cost="smooth", init="mcf",
        mask=mask, min_region_size=300,
    )
    dt = time.perf_counter() - t0
    unw = np.asarray(unw, np.float32)
    cc = np.asarray(cc).astype(np.int32)
    valid = mask & np.isfinite(unw)
    pc = percomp_match(unw, prod_unw, wrapped, prod_cc, valid)
    print(f"snaphu DONE {dt:.1f}s  per-comp-match={pc*100:.2f}%  ncc={int(cc.max())}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        snaphu_unw=unw, snaphu_cc=cc,
        prod_unw=prod_unw, prod_cc=prod_cc.astype(np.int32),
        coh=coh.astype(np.float32), mask=mask,
        wrapped=wrapped, runtime_s=np.float64(dt), percomp=np.float64(pc),
        nlooks=np.float64(args.nlooks),
    )
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
