"""Run a chosen whirlwind mode on the Atlanta S-1 scene (OPERA reference).

The IFG is Sentinel-1 C-band (λ ≈ 5.6 cm). The "OPERA displacement"
file is the SNAPHU-unwrapped phase converted to LOS displacement in
meters; rewrap-from-displacement gives `opera.int.phs.tif`. To compare
K-fields against the OPERA reference we convert displacement → radians
via `phase = displacement · 4π / λ`.

Lots of NaN in the inputs (about 39 % of pixels masked).

Usage:  run_atlanta.py <mode>
  mode ∈ {baseline, reuse, convex, convex_raw}
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

OUT_DIR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
ATL = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
LAMBDA_S1 = 0.05546576  # meters, Sentinel-1 C-band

mode = sys.argv[1]
if mode == "convex_raw":
    os.environ["WHIRLWIND_CONVEX_OFFSET_RAW"] = "1"

import whirlwind as ww  # noqa: E402

with rasterio.open(ATL / "opera.int.phs.tif") as src:
    phase = src.read(1).astype(np.float32)
with rasterio.open(ATL / "opera.int.cor.tif") as src:
    coh = src.read(1).astype(np.float32)

# Mask: finite phase and coh, coh in (0, 1), non-zero IG. NaNs out.
mask = (
    np.isfinite(phase) & np.isfinite(coh)
    & (coh > 0) & (coh < 1.0)
)
ig = np.exp(1j * np.where(mask, phase, 0.0)).astype(np.complex64)
ig[~mask] = 0
coh = np.where(mask, coh, 0).astype(np.float32)
coh = np.clip(coh, 0.0, 1.0)
# Roughly Sentinel-1 IW boxcar looks (Reuter 2025 OPERA: typically 5-10).
# OPERA standard SLC-DISP-S1 uses 5×10 (range×azimuth) = 50 looks.
nlooks = 50.0
print(f"[atlanta/{mode}] shape={phase.shape}  valid={mask.sum():,} ({mask.mean()*100:.1f}%)", flush=True)

t0 = time.perf_counter()
if mode == "baseline":
    unw = ww.unwrap(ig, coh, nlooks=nlooks, mask=mask)
elif mode == "reuse":
    unw = ww.unwrap_reuse(ig, coh, nlooks=nlooks, mask=mask)
elif mode in ("convex", "convex_raw"):
    unw = ww.unwrap_convex(ig, coh, nlooks=nlooks, mask=mask)
else:
    sys.exit(2)
elapsed = time.perf_counter() - t0
print(f"[atlanta/{mode}] done in {elapsed:.1f}s", flush=True)

# K = round((unw - wrapped)/2π); use the same wrapped that whirlwind saw.
tau = np.float32(2 * np.pi)
wrapped = np.angle(ig).astype(np.float32)
k_int = np.round((unw - wrapped) / tau).astype(np.int32)
out_path = OUT_DIR / f"atlanta_{mode}.npz"
np.savez(out_path, unw=unw, k=k_int, mask=mask, elapsed=np.float32(elapsed))
print(f"[atlanta/{mode}] wrote {out_path}", flush=True)
