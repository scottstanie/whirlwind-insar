"""Bench the ACTUAL default path (ww.unwrap, tile_size=0 -> auto-512 -> gated
multi-shift unwrap_tiled_robust) on the 5 NISAR GUNW frames. Confirms A_016 is
fixed by the gate firing, and clean frames are untouched (gate does not fire).
Serial (one heavy unwrap at a time)."""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

import whirlwind as ww

GUNW = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw")
TAU = 2 * np.pi
UNW = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
FRAMES = {
    "A_013": "NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_20251017T124857_20251029T124836_20251029T124858_X05010_N_P_J_001.h5",
    "A_016": "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001.h5",
    "A_018": "NISAR_L2_PR_GUNW_003_005_A_018_004_4000_SH_20251017T125111_20251017T125127_20251029T125111_20251029T125131_X05010_N_P_J_001.h5",
    "A_035": "NISAR_L2_PR_GUNW_003_006_A_035_004_4000_SH_20251017T144100_20251017T144117_20251029T144100_20251029T144117_X05009_N_P_J_001.h5",
    "D_074": "NISAR_L2_PR_GUNW_003_005_D_074_004_4000_SH_20251017T132342_20251017T132345_20251029T132342_20251029T132346_X05010_N_P_J_001.h5",
}


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def main() -> None:
    print(f"{'frame':8} {'match_cc':>9} {'runtime':>9}")
    for name, fn in FRAMES.items():
        with h5py.File(GUNW / fn) as h5:
            pol = sorted(k for k, v in h5[UNW].items() if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"})[0]
            prod = h5[f"{UNW}/{pol}/unwrappedPhase"][()].astype(np.float64)
            coh = h5[f"{UNW}/{pol}/coherenceMagnitude"][()].astype(np.float32)
            pcc = h5[f"{UNW}/{pol}/connectedComponents"][()].astype(np.int64)
            md = h5[f"{UNW}/mask"][()]
        valid = (md != 127) & np.isfinite(prod) & np.isfinite(coh)
        igc = np.ascontiguousarray(np.exp(1j * wrap(prod)), np.complex64)
        cohc = np.ascontiguousarray(coh, np.float32)
        mk = np.ascontiguousarray(valid, bool)
        reg = valid & (pcc > 0)
        t0 = time.perf_counter()
        _u, _cc = ww.unwrap(igc, cohc, 16.0, mk)  # TRUE default (tile_size=0)
        u = np.asarray(_u, np.float64)
        dt = time.perf_counter() - t0
        a = np.rint((u - prod) / TAU)[reg]; a = a - modal(a)
        m = 100 * np.mean(np.abs(a) < 0.5)
        print(f"{name:8} {m:8.2f}% {dt:8.1f}s", flush=True)
        del prod, coh, pcc, valid, igc, cohc, mk, reg, u, a


if __name__ == "__main__":
    main()
