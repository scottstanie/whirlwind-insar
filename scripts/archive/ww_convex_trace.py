#!/usr/bin/env python3
"""Whole-image convex solve on one GUNW frame with WHIRLWIND_DEBUG tracing, to
see whether the primal-dual/SSP solve drains all excess at NISAR scale or
strands it (early-termination). Run with:

    WHIRLWIND_TILE_SOLVER=convex WHIRLWIND_DEBUG=1 \
    env -u CONDA_PREFIX uv run --no-sync python scripts/ww_convex_trace.py \
        --local-h5 <WD>/nisar_gunw/*D_077*.h5 --nlooks 16
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
    gunw_paths,
    mask_to_bool,
    read_array,
    wrap_phase,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-h5", type=Path, required=True)
    ap.add_argument("--nlooks", type=float, default=16.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    with h5py.File(args.local_h5, "r") as h:
        p = gunw_paths(h, None)
        prod_unw = read_array(h[p["unw"]], np.float32)
        coh = read_array(h[p["coh_unw"]], np.float32)
        mask_arr = h[p["mask"]][()] if p["mask"] in h else None
    mask = (
        mask_to_bool(mask_arr, "water_only", prod_unw.shape)
        & np.isfinite(prod_unw)
        & np.isfinite(coh)
    )
    wrapped = wrap_phase(prod_unw).astype(np.float32)
    ig = np.exp(1j * wrapped).astype(np.complex64)
    coh_c = np.clip(np.nan_to_num(coh), 0.0, 1.0).astype(np.float32)
    big = max(prod_unw.shape) + 1000

    import whirlwind as ww
    print(f"shape={ig.shape}  whole-image tile_size={big}  solver=convex(env)", flush=True)
    t0 = time.perf_counter()
    unw, cc = ww.unwrap(ig, coh_c, args.nlooks, mask, tile_size=big, tile_overlap=0)
    dt = time.perf_counter() - t0
    unw = np.asarray(unw, np.float32)
    print(f"DONE {dt:.1f}s  ncc={int(np.asarray(cc).max())}  "
          f"ww_unw range=[{np.nanmin(unw):.1f},{np.nanmax(unw):.1f}]", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, ww_unw=unw, ww_cc=np.asarray(cc).astype(np.int32))
        print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
