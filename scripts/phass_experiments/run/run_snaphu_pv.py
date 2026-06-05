"""Run SNAPHU smooth on Palos Verdes as a reference for the PHASS experiments.

NISAR already has a snaphu_9x9 reference saved next to its input TIFF; PV
doesn't, so generate one here. Single-tile (PV is small, 871x864).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import rasterio
import snaphu

OUT_DIR = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs"
)
PV = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes"
    "/Palos_Verdes_C13_RO23_SP/network_output/20251129_20251205"
)

with rasterio.open(
    PV / "CAPELLA_C13_C13_SP_PHS_HH_20251129T183328_20251205T162657.tif"
) as src:
    phase = src.read(1).astype(np.float32)
with rasterio.open(
    PV / "CAPELLA_C13_C13_SP_COH_HH_20251129T183328_20251205T162657.tif"
) as src:
    coh = src.read(1).astype(np.float32)

ig = np.exp(1j * phase).astype(np.complex64)
mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0)
ig[~mask] = 0
coh[~mask] = 0
coh = np.clip(coh, 0.0, 1.0)
print(f"shape={ig.shape}  valid={mask.sum():,}", flush=True)

t0 = time.perf_counter()
unw, cc = snaphu.unwrap(
    ig,
    coh,
    nlooks=5.0,
    cost="smooth",
    mask=mask.astype(np.uint8),
    nproc=os.cpu_count() or 1,
)
elapsed = time.perf_counter() - t0
print(
    f"snaphu done in {elapsed:.1f}s  n_components={int(cc.max())}  "
    f"coverage={(cc>0).mean()*100:.2f}%",
    flush=True,
)

tau = np.float32(2 * np.pi)
k_int = np.round((np.asarray(unw) - phase) / tau).astype(np.int32)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / "pv_snaphu.npz"
np.savez_compressed(
    out,
    unw=np.asarray(unw, dtype=np.float32),
    cc=np.asarray(cc, dtype=np.uint32),
    k=k_int,
    elapsed=float(elapsed),
)
print(f"wrote {out}")
