"""Profile the PD-vs-SSP cost split of whirlwind's single-tile linear unwrap by
sweeping WHIRLWIND_LINEAR_PD_ITERS on D_077 (the slowest NISAR frame). If more PD
iterations make it FASTER, the multi-source SSP fallback dominates wall-clock (the
ATBD's hypothesis); if slower, PD dominates. One heavy unwrap at a time.

Usage (base miniforge3 env): python scripts/prof_pdssp.py [FRAME=D_077] [PD_ITERS...]
"""
import glob
import time
import os
import sys

import numpy as np
import h5py

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase, percomp_match
import whirlwind as ww

frame = sys.argv[1] if len(sys.argv) > 1 else "D_077"
pd_iters = sys.argv[2:] or ["8", "64", "256"]
h5 = glob.glob(f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5")[0]
with h5py.File(h5, "r") as h:
    pol, prod, coh, pcc, marr = gunw_layers(h)
mask = water_only_mask(marr, prod.shape) & np.isfinite(prod) & np.isfinite(coh)
wr = np.where(mask, wrap_phase(prod), 0.0).astype(np.float32)
ig = np.exp(1j * wr).astype(np.complex64)
ci = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)

for pdi in pd_iters:
    os.environ["WHIRLWIND_LINEAR_PD_ITERS"] = pdi
    t0 = time.perf_counter()
    u, _ = ww.unwrap(ig, ci, 16.0, mask, bridge=False)
    dt = time.perf_counter() - t0
    u = np.asarray(u, np.float32)
    pc = percomp_match(u, prod, wr, pcc, mask & np.isfinite(u))
    print(f"{frame} PD_ITERS={pdi:>3}  {dt:6.1f}s  per-comp={pc*100:5.1f}%", flush=True)
