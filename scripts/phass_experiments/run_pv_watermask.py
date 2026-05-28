"""Run reuse (or baseline) on PV with the SLC-resolution water mask
applied to the input IFG.

The PV IFG is at 871×864 (multilooked); the water mask is at SLC
resolution 12197×19012. Decimation factors are 14 (rows) × 22 (cols)
— close to exact division. We downsample with `min` over each window:
an IFG pixel is treated as land iff *every* SLC pixel within its
14×22 footprint is land. Conservative — we don't unwrap any pixel
that touches water.

Without this, PV's input mask has median coh 0.129 (38 % of pixels
with coh < 0.1), almost all of which is ocean decorrelation. With
the faithful PHASS cost (γ²·100 + 255-cliff) those low-coh arcs
get cost 0-4, creating a huge low-cost subgraph that pathologized
the solver. Applying the water mask first should make the faithful
cost test (and the rest) actually tractable on PV.

Usage:  run_pv_watermask.py <mode>
  mode ∈ {baseline, reuse, reuse_faithful, reuse_faithful_07}
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PV_ROOT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes/Palos_Verdes_C13_RO23_SP")
PV = PV_ROOT / "network_output/20251129_20251205"
WM = PV_ROOT / "e2e_output_20260519/stack_output/geometry/water_mask.rdr.tif"

mode = sys.argv[1]
if mode == "reuse_faithful":
    os.environ["WHIRLWIND_PHASS_FAITHFUL_GOOD_CORR"] = "0.7"

import rasterio  # noqa: E402
import whirlwind as ww  # noqa: E402

with rasterio.open(PV / "CAPELLA_C13_C13_SP_PHS_HH_20251129T183328_20251205T162657.tif") as src:
    phase = src.read(1).astype(np.float32)
with rasterio.open(PV / "CAPELLA_C13_C13_SP_COH_HH_20251129T183328_20251205T162657.tif") as src:
    coh = src.read(1).astype(np.float32)
with rasterio.open(WM) as src:
    wm = src.read(1)
ig = np.exp(1j * phase).astype(np.complex64)
nlooks = 5.0
m_ifg, n_ifg = ig.shape

# Decimate water mask from SLC to IFG resolution by integer downsampling
# with min-over-window. `decim_r/c` are the strict integer factors;
# whatever pixels are leftover at the bottom/right edge get dropped.
decim_r = wm.shape[0] // m_ifg
decim_c = wm.shape[1] // n_ifg
trim_r = decim_r * m_ifg
trim_c = decim_c * n_ifg
wm_trim = wm[:trim_r, :trim_c].reshape(m_ifg, decim_r, n_ifg, decim_c)
# min over each (decim_r, decim_c) window: pixel is land iff every SLC px in
# its footprint is land (= 1).
wm_ifg = wm_trim.min(axis=(1, 3))
print(f"Water mask downsampled to {wm_ifg.shape} via {decim_r}×{decim_c} min-window", flush=True)
print(f"  IFG-grid land pixels: {(wm_ifg == 1).sum():,} / {wm_ifg.size:,}  ({(wm_ifg == 1).mean()*100:.1f}%)", flush=True)

coh_mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
land_mask = wm_ifg == 1
mask = coh_mask & land_mask
ig[~mask] = 0
coh[~mask] = 0
coh = np.clip(coh, 0.0, 1.0)
print(f"[{mode}] coh+land mask: {mask.sum():,} valid (was {coh_mask.sum():,} coh-only)", flush=True)

t0 = time.perf_counter()
if mode == "baseline":
    unw = ww.unwrap(ig, coh, nlooks=nlooks, mask=mask)
elif mode in ("reuse", "reuse_faithful"):
    unw = ww.unwrap_reuse(ig, coh, nlooks=nlooks, mask=mask)
else:
    sys.exit(2)
elapsed = time.perf_counter() - t0

wrapped = np.angle(ig).astype(np.float32)
tau = np.float32(2 * np.pi)
k_int = np.round((unw - wrapped) / tau).astype(np.int32)

print(f"[{mode}] done in {elapsed:.2f}s", flush=True)
tag = f"pv_watermask_{mode}"
out_path = OUT_DIR / f"{tag}.npz"
np.savez(out_path, unw=unw, k=k_int, mask=mask, elapsed=np.float32(elapsed))
print(f"[{mode}] wrote {out_path}", flush=True)
