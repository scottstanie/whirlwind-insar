"""Validate the multi-shift + min-high-coherence-cut SELECTOR.

Finding: the correct tile-grid offset (which avoids seam artifacts AND the wrong
winding) is the one whose unwrapping has the FEWEST high-coherence branch cuts
(a good unwrap never cuts coherent terrain). So: run N shifted tilings, pick the
one minimizing the coherence-weighted count of cuts through coh>THR.

Validate: (1) on A_016 argmin-hicut picks a high-match shift; (2) clean scenes
(A_013,D_074,A_018,A_035) stay high under the selector (no regression). Reports
per-shift match + hicut, and what the selector would choose vs the best possible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

import whirlwind as ww

GUNW = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw")
TAU = 2 * np.pi
UNW = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
TS, OV = 512, 64
THR = 0.7
SHIFTS = [0, 112, 224, 336]
FRAMES = {
    "A_016": "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001.h5",
    "A_013": "NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_20251017T124857_20251029T124836_20251029T124858_X05010_N_P_J_001.h5",
    "A_018": "NISAR_L2_PR_GUNW_003_005_A_018_004_4000_SH_20251017T125111_20251017T125127_20251029T125111_20251029T125131_X05010_N_P_J_001.h5",
    "A_035": "NISAR_L2_PR_GUNW_003_006_A_035_004_4000_SH_20251017T144100_20251017T144117_20251029T144100_20251029T144117_X05009_N_P_J_001.h5",
    "D_074": "NISAR_L2_PR_GUNW_003_005_D_074_004_4000_SH_20251017T132342_20251017T132345_20251029T132342_20251029T132346_X05010_N_P_J_001.h5",
}


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def read(name):
    with h5py.File(GUNW / FRAMES[name]) as h5:
        pol = sorted(k for k, v in h5[UNW].items() if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"})[0]
        prod = h5[f"{UNW}/{pol}/unwrappedPhase"][()].astype(np.float64)
        coh = h5[f"{UNW}/{pol}/coherenceMagnitude"][()].astype(np.float64)
        pcc = h5[f"{UNW}/{pol}/connectedComponents"][()].astype(np.int64)
        md = h5[f"{UNW}/mask"][()]
    valid = (md != 127) & np.isfinite(prod) & np.isfinite(coh)
    return wrap(prod).astype(np.float32), coh, valid, prod, pcc


def unw_shift(ig, coh, valid, s):
    H, W = ig.shape
    igc = np.exp(1j * ig).astype(np.complex64)
    if s == 0:
        return np.asarray(ww.unwrap(np.ascontiguousarray(igc), np.ascontiguousarray(coh, np.float32),
                                    16.0, np.ascontiguousarray(valid, bool), tile_size=TS, tile_overlap=OV), np.float64)
    pig = np.zeros((H + s, W + s), np.complex64); pco = np.zeros((H + s, W + s), np.float32); pmk = np.zeros((H + s, W + s), bool)
    pig[s:, s:] = igc; pco[s:, s:] = coh.astype(np.float32); pmk[s:, s:] = valid
    u = ww.unwrap(np.ascontiguousarray(pig), np.ascontiguousarray(pco), 16.0,
                  np.ascontiguousarray(pmk), tile_size=TS, tile_overlap=OV)
    return np.asarray(u, np.float64)[s:, s:]


def main() -> None:
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(FRAMES)
    for name in names:
        ig, coh, valid, prod, pcc = read(name)
        reg = valid & (pcc > 0) & np.isfinite(prod)
        Wp = ig.astype(np.float64)

        def hicut(u):
            fh = np.abs(np.rint((u[:, 1:] - u[:, :-1] - wrap(Wp[:, 1:] - Wp[:, :-1])) / TAU))
            fv = np.abs(np.rint((u[1:, :] - u[:-1, :] - wrap(Wp[1:, :] - Wp[:-1, :])) / TAU))
            ch = np.minimum(coh[:, :-1], coh[:, 1:]); cv = np.minimum(coh[:-1, :], coh[1:, :])
            vh = valid[:, :-1] & valid[:, 1:]; vv = valid[:-1, :] & valid[1:, :]
            return float((fh[vh & (ch > THR)] * ch[vh & (ch > THR)]).sum() + (fv[vv & (cv > THR)] * cv[vv & (cv > THR)]).sum())

        def match(u):
            a = np.rint((u - prod) / TAU)[reg]; a = a[np.isfinite(a)]; return 100 * np.mean(np.abs(a - modal(a)) < 0.5)

        rows = []
        for s in SHIFTS:
            u = unw_shift(ig, coh, valid, s)
            rows.append((s, match(u), hicut(u)))
        sel = min(rows, key=lambda r: r[2])      # selector: min high-coh cuts
        best = max(rows, key=lambda r: r[1])     # oracle: max match
        base = next(r for r in rows if r[0] == 0)
        print(f"=== {name} ===", flush=True)
        for s, m, h in rows:
            tag = "  <- SELECTED" if s == sel[0] else ("  (best)" if s == best[0] else "")
            print(f"  shift={s:3d}: match={m:6.2f}%  hicut={h:12.0f}{tag}", flush=True)
        print(f"  SELECTOR picks shift={sel[0]} -> {sel[1]:.2f}%  | base(shift0)={base[1]:.2f}%  | oracle-best={best[1]:.2f}%", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
