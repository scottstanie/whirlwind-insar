"""Run unwrap_reuse with the *faithful* PHASS cost (γ²·100 + 255-cliff).

`WHIRLWIND_PHASS_FAITHFUL_GOOD_CORR=<G>` enables the PHASS C++ cost
recipe exactly (cost is γ² for γ ≤ G, then 2.55 above, which after the
COST_SCALE=100 multiplier produces the 0-100 / 255 integer pair that
PhassUnwrapper.cc:119-141 emits). The faithful recipe was pathological
under unit-capacity SSP (>5× baseline runtime on PV and NISAR), but
under reuse the cliff should be digestible because used arcs already
have reduced cost 0.

Usage:  run_reuse_faithful.py <scene> [<good_corr>]
  scene ∈ {nisar, pv}
  good_corr ∈ (0, 1], default 0.7 (PHASS's own default)

Writes: <OUT>/<scene>_reuse_faithful_<good>.npz
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
good = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
os.environ["WHIRLWIND_PHASS_FAITHFUL_GOOD_CORR"] = f"{good}"
tag = f"reuse_faithful_{good:.2f}".replace(".", "_")

print(f"[{scene}/{tag}] start (WHIRLWIND_PHASS_FAITHFUL_GOOD_CORR={good})", flush=True)

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
print(f"[{scene}/{tag}] shape={ig.shape}  valid={mask.sum():,}", flush=True)

t0 = time.perf_counter()
unw = ww.unwrap_reuse(ig, coh, nlooks=nlooks, mask=mask)
elapsed = time.perf_counter() - t0

wrapped = np.angle(ig).astype(np.float32)
tau = np.float32(2 * np.pi)
k_int = np.round((unw - wrapped) / tau).astype(np.int32)

print(f"[{scene}/{tag}] done in {elapsed:.1f}s", flush=True)
out_path = OUT_DIR / f"{scene}_{tag}.npz"
np.savez(out_path, unw=unw, k=k_int, elapsed=np.float32(elapsed))
print(f"[{scene}/{tag}] wrote {out_path}", flush=True)
