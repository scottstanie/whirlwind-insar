"""Run whirlwind α=0 once with a chosen cost variant on a chosen scene.

Usage:  run_one.py <scene> <mode>
  scene ∈ {nisar, pv}
  mode  ∈ {baseline, hard_cut, phass_cost, phass_full}

Env vars are set BEFORE importing whirlwind (OnceLock caches them in Rust).
Result: <OUT_DIR>/<scene>_<mode>.npz   (keys: unw, cc, k, elapsed)
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
mode = sys.argv[2]

# Each mode is (env vars to set, *cost_threshold scale relative to scene default).
# cost_threshold_scale = 1.0 keeps the per-scene default; PHASS-cost modes use
# 0.25 because PHASS γ² saturated cost has magnitudes ~3-4× smaller than the
# Carballo cost the per-scene default was tuned against.
MODES = {
    "baseline":      ({}, 1.0),
    # PHASS uses 1.0 rad; we found that catastrophically slow and wrong at α=0
    # (PV: 472s, K-match 41% vs 87% baseline). 2.0 rad fires only on
    # near-wrap-line gradients and is the practical setting.
    "hard_cut":      ({"WHIRLWIND_HARD_CUT_THRESH": "2.0"}, 1.0),
    "hard_cut_lo":   ({"WHIRLWIND_HARD_CUT_THRESH": "1.0"}, 1.0),     # PHASS default
    "phass_cost":    ({"WHIRLWIND_PHASS_COST": "0.5"}, 0.25),
    "phass_full":    ({"WHIRLWIND_PHASS_COST": "0.5",
                       "WHIRLWIND_HARD_CUT_THRESH": "2.0"}, 0.25),
}
if mode not in MODES:
    print(f"unknown mode {mode!r}; choose from {list(MODES)}", file=sys.stderr)
    sys.exit(1)
env_vars, cost_threshold_scale = MODES[mode]
os.environ.update(env_vars)

print(f"[{scene}/{mode}] env:",
      {k: v for k, v in os.environ.items() if k.startswith("WHIRLWIND_")},
      flush=True)

# Imports after env-var setup so the Rust OnceLock catches the values.
import rasterio  # noqa: E402
import whirlwind as ww  # noqa: E402

if scene == "nisar":
    with rasterio.open(NISAR / "20251224_20260117.int.looked.tif") as src:
        ig = src.read(1).astype(np.complex64)
    with rasterio.open(NISAR / "20251224_20260117.int.coh.looked.cleaned.tif") as src:
        coh = src.read(1).astype(np.float32)
    nlooks = 100.0
    cost_threshold = int(200 * cost_threshold_scale)
elif scene == "pv":
    # PV inputs are real-valued phase + coherence; synthesize a unit-magnitude
    # complex IG so the whirlwind API can consume it.
    with rasterio.open(PV / "CAPELLA_C13_C13_SP_PHS_HH_20251129T183328_20251205T162657.tif") as src:
        phase = src.read(1).astype(np.float32)
    with rasterio.open(PV / "CAPELLA_C13_C13_SP_COH_HH_20251129T183328_20251205T162657.tif") as src:
        coh = src.read(1).astype(np.float32)
    ig = np.exp(1j * phase).astype(np.complex64)
    nlooks = 5.0
    cost_threshold = int(100 * cost_threshold_scale)
else:
    sys.exit(2)

mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
ig[~mask] = 0
coh[~mask] = 0
coh = np.clip(coh, 0.0, 1.0)
print(f"[{scene}/{mode}] shape={ig.shape}  valid={mask.sum():,}  "
      f"cost_threshold={cost_threshold}", flush=True)

t0 = time.perf_counter()
unw, cc = ww.unwrap_with_conncomp(
    ig, coh, nlooks=nlooks, mask=mask,
    cost_threshold=cost_threshold,
    min_size_frac=0.001, max_ncomps=10,
    goldstein_alpha=0.0,  # α=0: no Goldstein. The whole point.
)
elapsed = time.perf_counter() - t0

# K = round((unw − wrapped)/2π) on the *original* wrapped phase.
wrapped = np.angle(ig).astype(np.float32)
tau = np.float32(2 * np.pi)
k_int = np.round((unw - wrapped) / tau).astype(np.int32)

print(f"[{scene}/{mode}] done in {elapsed:.1f}s  "
      f"n_components={int(cc.max())}  coverage={(cc>0).mean()*100:.2f}%",
      flush=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / f"{scene}_{mode}.npz"
np.savez_compressed(out, unw=unw, cc=cc, k=k_int, elapsed=float(elapsed))
print(f"[{scene}/{mode}] wrote {out}", flush=True)
