"""Reuse-solver tile-size sweep on the 5 NISAR GUNW frames.

Decisive test for the A_016 fix: does a BIGGER tile (which fixes A_016's
distributed drift) REGRESS the clean frames now that the default solver is the
corner-safe REUSE (the old big-tile regression was the LINEAR cost's runaway)?

Runs strictly ONE unwrap at a time (serial loop in one process) to respect the
laptop's one-heavy-unwrap-at-a-time memory limit. Records cc>0 ambiguity match,
runtime, and peak RSS delta. Writes a JSON table; prints a summary.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import psutil

import whirlwind as ww

GUNW = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/reuse_tile_sweep")
OUT.mkdir(parents=True, exist_ok=True)
TAU = 2 * np.pi
UNW = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"

FRAMES = {
    "A_013": "NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_20251017T124857_20251029T124836_20251029T124858_X05010_N_P_J_001.h5",
    "A_016": "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001.h5",
    "A_018": "NISAR_L2_PR_GUNW_003_005_A_018_004_4000_SH_20251017T125111_20251017T125127_20251029T125111_20251029T125131_X05010_N_P_J_001.h5",
    "A_035": "NISAR_L2_PR_GUNW_003_006_A_035_004_4000_SH_20251017T144100_20251017T144117_20251029T144100_20251029T144117_X05009_N_P_J_001.h5",
    "D_074": "NISAR_L2_PR_GUNW_003_005_D_074_004_4000_SH_20251017T132342_20251017T132345_20251029T132342_20251029T132346_X05010_N_P_J_001.h5",
}
SIZES = [512, 1024, 2048]
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
    ig = wrap(prod).astype(np.float32)
    return ig, coh, valid, prod, pcc


def main() -> None:
    proc = psutil.Process(os.getpid())
    rows = []
    for size in SIZES:
        for name, fn in FRAMES.items():
            ig, coh, valid, prod, pcc = read_frame(GUNW / fn)
            igc = np.ascontiguousarray(np.exp(1j * ig), dtype=np.complex64)
            cohc = np.ascontiguousarray(coh, dtype=np.float32)
            maskc = np.ascontiguousarray(valid, dtype=bool)
            gc.collect()
            rss0 = proc.memory_info().rss / 1e6
            t0 = time.perf_counter()
            unw, _cc = ww.unwrap(igc, cohc, NLOOKS, maskc, tile_size=size, tile_overlap=64)
            dt = time.perf_counter() - t0
            rss1 = proc.memory_info().rss / 1e6
            unw = np.asarray(unw, np.float64)
            reg = valid & (pcc > 0) & np.isfinite(unw)
            a = np.rint((unw - prod) / TAU)[reg]
            a = a - modal(a)
            match = 100.0 * float(np.mean(np.abs(a) < 0.5))
            row = {"frame": name, "tile": size, "match_cc": round(match, 3),
                   "runtime_s": round(dt, 1), "rss_mb": round(rss1 - rss0, 0),
                   "shape": list(unw.shape)}
            rows.append(row)
            print(f"  tile={size:5d}  {name}: match={match:6.2f}%  {dt:6.1f}s  +{rss1-rss0:6.0f}MB", flush=True)
            del ig, coh, valid, prod, pcc, igc, cohc, maskc, unw, a
            gc.collect()
        print("", flush=True)
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2))
    # pivot table
    print("\n=== match_cc (%) by frame x tile ===", flush=True)
    print(f"{'frame':8s} " + " ".join(f"t{s:<6d}" for s in SIZES), flush=True)
    for name in FRAMES:
        cells = []
        for s in SIZES:
            r = next((x for x in rows if x["frame"] == name and x["tile"] == s), None)
            cells.append(f"{r['match_cc']:7.2f}" if r else "   n/a ")
        print(f"{name:8s} " + " ".join(cells), flush=True)
    print(f"\nJSON: {OUT/'sweep.json'}", flush=True)


if __name__ == "__main__":
    main()
