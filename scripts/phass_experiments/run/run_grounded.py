"""Run whirlwind unwrap_grounded (Carballo cost + virtual ground node) on a scene.

Tests the reviewer's recommendation that the grounded variant of the
coherence-cost path (currently CRLB-only) should help on real NISAR data
the same way it fixed the ignored `diagonal_ramp_512` regression for
clean ramps.

Usage:  run_grounded.py <scene> <ground_cost>
  scene ∈ {nisar, pv}
  ground_cost ∈ ints, typically 0..200

Result: <OUT>/<scene>_grounded_<ground_cost>.npz   (keys: unw, k, elapsed)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
NISAR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
PV = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes"
          "/Palos_Verdes_C13_RO23_SP/network_output/20251129_20251205")

scene = sys.argv[1]
ground_cost = int(sys.argv[2])

# No env vars — pure default Carballo + ground.
print(f"[{scene}/grounded_{ground_cost}] start", flush=True)

import rasterio  # noqa: E402
import whirlwind as ww  # noqa: E402

if scene == "nisar":
    with rasterio.open(NISAR / "20251224_20260117.int.looked.tif") as src:
        ig = src.read(1).astype(np.complex64)
    with rasterio.open(NISAR / "20251224_20260117.int.coh.looked.cleaned.tif") as src:
        coh = src.read(1).astype(np.float32)
    nlooks = 100.0
elif scene == "pv":
    with rasterio.open(PV / "CAPELLA_C13_C13_SP_PHS_HH_20251129T183328_20251205T162657.tif") as src:
        phase = src.read(1).astype(np.float32)
    with rasterio.open(PV / "CAPELLA_C13_C13_SP_COH_HH_20251129T183328_20251205T162657.tif") as src:
        coh = src.read(1).astype(np.float32)
    ig = np.exp(1j * phase).astype(np.complex64)
    nlooks = 5.0
else:
    sys.exit(2)

mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
ig[~mask] = 0
coh[~mask] = 0
coh = np.clip(coh, 0.0, 1.0)
print(f"[{scene}/grounded_{ground_cost}] shape={ig.shape}  valid={mask.sum():,}", flush=True)

t0 = time.perf_counter()
unw = ww.unwrap_grounded(ig, coh, nlooks=nlooks, mask=mask, ground_cost=ground_cost)
elapsed = time.perf_counter() - t0

wrapped = np.angle(ig).astype(np.float32)
tau = np.float32(2 * np.pi)
k_int = np.round((unw - wrapped) / tau).astype(np.int32)

print(f"[{scene}/grounded_{ground_cost}] done in {elapsed:.1f}s", flush=True)

out_path = OUT_DIR / f"{scene}_grounded_{ground_cost}.npz"
np.savez(out_path, unw=unw, k=k_int, elapsed=np.float32(elapsed))
print(f"[{scene}/grounded_{ground_cost}] wrote {out_path}", flush=True)
