"""Measure whirlwind's rayon benefit: run the single-tile unwrap (bridge off) on
one frame at whatever WHIRLWIND_NUM_THREADS is set, printing threads + time +
per-comp. Loop externally over thread counts. The cost/residue/conncomp build is
rayon-parallel; the PD/SSP solver is largely serial - so the speedup depends on
how cost-build-dominated (residue-light) vs solver-dominated (residue-heavy) the
frame is.

Usage: WHIRLWIND_NUM_THREADS=1 python scripts/rayon_bench.py 005_D_077
"""

import sys
import glob
import time

import numpy as np
import h5py

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase, percomp_match
import whirlwind as ww

frame = sys.argv[1]
h5 = glob.glob(
    f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5"
)[0]
with h5py.File(h5, "r") as h:
    pol, prod, coh, pcc, marr = gunw_layers(h)
mask = water_only_mask(marr, prod.shape) & np.isfinite(prod) & np.isfinite(coh)
wr = np.where(mask, wrap_phase(prod), 0.0).astype(np.float32)
ig = np.exp(1j * wr).astype(np.complex64)
ci = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)

t0 = time.perf_counter()
u, _ = ww.unwrap(ig, ci, 16.0, mask, bridge=False)
dt = time.perf_counter() - t0
u = np.asarray(u, np.float32)
pc = percomp_match(u, prod, wr, pcc, mask & np.isfinite(u))
print(
    f"{frame} threads={ww.num_threads():2d}  {dt:6.1f}s  per-comp={pc * 100:5.1f}%",
    flush=True,
)
