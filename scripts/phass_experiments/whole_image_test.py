"""Whole-image (single-tile, NO reconcile) test — the 'best-quality default'
hypothesis (S.S.): like SNAPHU single-tile, solve the whole frame in one MCF so
there are no tiles to reconcile and the distributed-drift mechanism cannot exist.

Tests whole-image under BOTH costs:
  - REUSE : ww.unwrap_reuse(...)                       (corner-safe, PHASS flow-reuse)
  - LINEAR: ww.unwrap(..., tile_size=HUGE)             (routes to crate::unwrap, linear Carballo)

Strictly ONE unwrap at a time. Prints match (cc>0), runtime, peak RSS. A_016
(the fragmented problem frame) runs FIRST to gauge memory before the rest.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psutil

import whirlwind as ww

GUNW = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/whole_image_test")
OUT.mkdir(parents=True, exist_ok=True)
TAU = 2 * np.pi
UNW = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
HUGE = 100000  # tile_size >= dims -> whole-image linear path in unwrap_tiled

FRAMES = {
    "A_016": "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001.h5",
    "D_074": "NISAR_L2_PR_GUNW_003_005_D_074_004_4000_SH_20251017T132342_20251017T132345_20251029T132342_20251029T132346_X05010_N_P_J_001.h5",
    "A_013": "NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_20251017T124857_20251029T124836_20251029T124858_X05010_N_P_J_001.h5",
    "A_018": "NISAR_L2_PR_GUNW_003_005_A_018_004_4000_SH_20251017T125111_20251017T125127_20251029T125111_20251029T125131_X05010_N_P_J_001.h5",
    "A_035": "NISAR_L2_PR_GUNW_003_006_A_035_004_4000_SH_20251017T144100_20251017T144117_20251029T144100_20251029T144117_X05009_N_P_J_001.h5",
}
NLOOKS = 16.0


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def read_frame(path: Path):
    with h5py.File(path, "r") as h5:
        grp = h5[UNW]
        pols = [k for k, v in grp.items() if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"}]
        pol = sorted(pols)[0]
        prod = h5[f"{UNW}/{pol}/unwrappedPhase"][()].astype(np.float32)
        coh = h5[f"{UNW}/{pol}/coherenceMagnitude"][()].astype(np.float32)
        pcc = h5[f"{UNW}/{pol}/connectedComponents"][()].astype(np.int64)
        maskd = h5[f"{UNW}/mask"][()] if f"{UNW}/mask" in h5 else None
    valid = (maskd != 127) if maskd is not None else np.ones(prod.shape, bool)
    valid &= np.isfinite(prod) & np.isfinite(coh)
    return wrap(prod).astype(np.float32), coh, valid, prod, pcc


def main() -> None:
    frames = sys.argv[1].split(",") if len(sys.argv) > 1 else list(FRAMES)
    proc = psutil.Process(os.getpid())
    rows = []
    for name in frames:
        ig, coh, valid, prod, pcc = read_frame(GUNW / FRAMES[name])
        igc = np.ascontiguousarray(np.exp(1j * ig), dtype=np.complex64)
        cohc = np.ascontiguousarray(coh, dtype=np.float32)
        maskc = np.ascontiguousarray(valid, dtype=bool)
        reg = valid & (pcc > 0)
        for cost in ("reuse", "linear"):
            gc.collect()
            avail0 = psutil.virtual_memory().available / 1e9
            rss0 = proc.memory_info().rss / 1e6
            t0 = time.perf_counter()
            if cost == "reuse":
                unw = ww.unwrap_reuse(igc, cohc, NLOOKS, maskc)
            else:
                unw = ww.unwrap(igc, cohc, NLOOKS, maskc, tile_size=HUGE, tile_overlap=64)
            dt = time.perf_counter() - t0
            rss1 = proc.memory_info().rss / 1e6
            u = np.asarray(unw, np.float64)
            r = reg & np.isfinite(u)
            a = np.rint((u - prod) / TAU)[r]
            a = a - modal(a)
            match = 100.0 * float(np.mean(np.abs(a) < 0.5))
            row = {"frame": name, "cost": cost, "match_cc": round(match, 2),
                   "runtime_s": round(dt, 1), "peak_rss_mb": round(rss1, 0),
                   "avail_gb_before": round(avail0, 1), "shape": list(u.shape)}
            rows.append(row)
            print(f"  WHOLE  {name}  {cost:6s}: match={match:6.2f}%  {dt:7.1f}s  rss={rss1:7.0f}MB  (avail {avail0:.1f}GB)", flush=True)
            del unw, u, a
            gc.collect()
        del ig, coh, valid, prod, pcc, igc, cohc, maskc, reg
        gc.collect()
        print("", flush=True)
    (OUT / "whole.json").write_text(json.dumps(rows, indent=2))
    print(f"JSON: {OUT/'whole.json'}", flush=True)


if __name__ == "__main__":
    main()
