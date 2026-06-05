"""Run unwrap_reuse with `WHIRLWIND_HARD_CUT_THRESH` set, on a chosen scene.

Same wrapper as `run_reuse.py`, but the env var is set *before* importing
whirlwind so the Rust OnceLock that caches the threshold picks it up.

Usage:  run_reuse_hardcut.py <scene> [<threshold_rad>]
  scene ∈ {nisar, pv}
  threshold ∈ rad (default 1.0 - PHASS's own `phase_diff_th`)

Tests the hypothesis from the 2026-05-28 follow-up that the residual
gap to dolphin PHASS K-agreement (NISAR 92.70 → 97.93 %) is closable
by adding PHASS's geometric tie-breaker (hard cost-zero cuts at high
raw-phase-gradient arcs) on top of the now-working reuse path. The
prior attempt at the same threshold on top of unit-cap MCF was
catastrophic (PV: 472 s vs 0.7 s baseline, K-match 47 %); the reuse
path should be able to digest the zero-cost subgraph because used arcs
already carry multi-unit flow.

Writes: <OUT>/<scene>_reuse_hardcut_<thresh>.npz
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs"
)
NISAR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
PV = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes"
    "/Palos_Verdes_C13_RO23_SP/network_output/20251129_20251205"
)

scene = sys.argv[1]
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
os.environ["WHIRLWIND_HARD_CUT_THRESH"] = f"{threshold}"
tag = f"reuse_hardcut_{threshold:.1f}".replace(".", "_")

print(f"[{scene}/{tag}] start (WHIRLWIND_HARD_CUT_THRESH={threshold})", flush=True)

import rasterio  # noqa: E402
import whirlwind as ww  # noqa: E402

if scene == "nisar":
    with rasterio.open(NISAR / "20251224_20260117.int.looked.tif") as src:
        ig = src.read(1).astype(np.complex64)
    with rasterio.open(NISAR / "20251224_20260117.int.coh.looked.cleaned.tif") as src:
        coh = src.read(1).astype(np.float32)
    nlooks = 100.0
elif scene == "pv":
    with rasterio.open(
        PV / "CAPELLA_C13_C13_SP_PHS_HH_20251129T183328_20251205T162657.tif"
    ) as src:
        phase = src.read(1).astype(np.float32)
    with rasterio.open(
        PV / "CAPELLA_C13_C13_SP_COH_HH_20251129T183328_20251205T162657.tif"
    ) as src:
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
