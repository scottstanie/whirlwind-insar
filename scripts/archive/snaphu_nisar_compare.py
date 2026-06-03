#!/usr/bin/env python3
"""Run snaphu-py with NISAR's GUNW unwrap settings (single tile, cost=smooth,
init=mcf — the isce3 defaults in share/nisar/defaults/insar.yaml) on a few GUNW
frames to get a RUNTIME baseline (and a quality reference) for the whirlwind
comparison. snaphu-py 0.4.1's defaults already match NISAR; we only override
min_region_size=300.

    env -u CONDA_PREFIX uv run --no-sync python scripts/snaphu_nisar_compare.py \
        --local-h5 <WD>/nisar_gunw/*D_077*.h5 <WD>/nisar_gunw/*A_016*.h5 --nlooks 16
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
    """Per-(production)-component ambiguity match — same metric as the ww bench."""
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
    ap.add_argument("--local-h5", nargs="+", type=Path, required=True)
    ap.add_argument("--nlooks", type=float, default=16.0)
    args = ap.parse_args()

    import snaphu

    print(f"snaphu {snaphu.__version__}  (cost=smooth, init=mcf, ntiles=(1,1), min_region_size=300)\n")
    for path in args.local_h5:
        with h5py.File(path, "r") as h:
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
        coh = np.clip(np.nan_to_num(coh), 0.0, 1.0).astype(np.float32)

        t0 = time.perf_counter()
        unw, cc = snaphu.unwrap(
            ig, coh, args.nlooks, cost="smooth", init="mcf",
            mask=mask, min_region_size=300,
        )
        dt = time.perf_counter() - t0
        unw = np.asarray(unw, np.float32)
        valid = mask & np.isfinite(unw)
        pc = percomp_match(unw, prod_unw, wrapped, prod_cc, valid)
        frame = path.name.split("_004_4000")[0].split("GUNW_")[-1]
        print(
            f"{frame}: snaphu single-tile  {dt:6.1f}s  "
            f"per-comp-match-vs-prod={pc * 100:5.1f}%  shape={ig.shape}  ncc={int(np.asarray(cc).max())}",
            flush=True,
        )


if __name__ == "__main__":
    main()
