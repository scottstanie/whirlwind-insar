"""Run single-tile SNAPHU (snaphu-py direct, cost=smooth init=mcf, ntiles=(1,1),
single_tile_reoptimize=True) on ONE GUNW frame, timed, printing the SAME per-comp
line as run_native_one.py so the 4-way sweep scores it identically. snaphu-py
writes temp files (no in-memory GDAL dataset), avoiding the tophu MEM/GA_Update bug.

Usage: python scripts/snaphu_one.py <h5path>   (base miniforge3 env: has snaphu 0.4.1)
Wrap in `/usr/bin/time -l` for peak RSS.
"""
import sys
import os
import re
import time

import numpy as np
import h5py

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase, percomp_match
import snaphu

h5path = sys.argv[1]
ntiles = int(sys.argv[2]) if len(sys.argv) > 2 else 1   # 1 = single-tile; 9 = 9x9 + reoptimize
frame = re.search(r"_([AD]_\d{3})_", h5path).group(1)
with h5py.File(h5path, "r") as h:
    pol, prod, coh, pcc, marr = gunw_layers(h)
mask = water_only_mask(marr, prod.shape) & np.isfinite(prod) & np.isfinite(coh)
wrapped = np.where(mask, wrap_phase(prod), 0.0).astype(np.float32)
ig = np.exp(1j * wrapped).astype(np.complex64)
coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)

# single_tile_reoptimize is a no-op at ntiles=(1,1) (snaphu/_unwrap.py gates it on
# `not single_tile`), so the 1-tile run is a clean single pass; at 9x9 it is
# SNAPHU's production path (tiled solve + a whole-image reoptimize pass).
overlap = 0 if ntiles == 1 else 200
# Single-tile is inherently one graph (1 core). The tiled PRODUCTION path
# parallelizes tiles, so give it the cores it would actually use (whirlwind itself
# runs 12 threads) — handicapping SNAPHU's production config would be unfair.
nproc = 1 if ntiles == 1 else (os.cpu_count() or 8)
t0 = time.perf_counter()
unw, cc = snaphu.unwrap(
    ig, coh_in, nlooks=16.0, cost="smooth", init="mcf", mask=mask,
    ntiles=(ntiles, ntiles), tile_overlap=overlap, nproc=nproc,
    single_tile_reoptimize=True,
)
dt = time.perf_counter() - t0
unw = np.asarray(unw, np.float32)
ncc = int(np.asarray(cc).max())
valid = mask & np.isfinite(unw)
pc = percomp_match(unw, prod, wrapped, pcc, valid)
tag = "snaphu" if ntiles == 1 else f"snaphu{ntiles}x{ntiles}"
print(f"{frame}: {tag:10s} {dt:6.1f}s  per-comp-match-vs-prod={pc * 100:5.1f}%  "
      f"ncc={ncc}  shape={ig.shape}", flush=True)
