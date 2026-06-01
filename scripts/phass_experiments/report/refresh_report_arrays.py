"""Refresh report arrays on the current build (reuse default + gated multi-shift):
 - NISAR: default ww.unwrap (tile512 -> robust; gate won't fire on this clean
   scene) vs SNAPHU 9x9. Saves nisar_anchor_{unw,sk,scc,mask,wrapped}.npy.
 - A_016 GUNW: default ww.unwrap (gate FIRES -> multi-shift) — saves the fixed
   result + reference for the new A_016 report panel.
Serial; one heavy unwrap at a time.
"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import rasterio

import whirlwind as ww

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
GUNW = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
A016OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/a016_diag")
OUT.mkdir(parents=True, exist_ok=True); A016OUT.mkdir(parents=True, exist_ok=True)
TAU = 2 * np.pi
UNW = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001.h5"


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def do_nisar():
    ig = rasterio.open(N / "20251224_20260117.int.looked.tif").read(1).astype(np.complex64)
    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    sunw = rasterio.open(N / "20251224_20260117.snaphu_9x9.unw.tif").read(1).astype(np.float32)
    scc = rasterio.open(N / "20251224_20260117.snaphu_9x9.cc.tif").read(1).astype(np.uint32)
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig = np.where(mask, ig, 0).astype(np.complex64)
    coh = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    sk = np.round((sunw - wrapped) / TAU)
    t0 = time.perf_counter()
    _unw, _cc = ww.unwrap(ig, coh, 100.0, mask)  # TRUE default
    unw = np.asarray(_unw, np.float32)
    dt = time.perf_counter() - t0
    kw = np.round((unw - wrapped) / TAU); kw[~mask] = np.nan
    mainland = (scc == 1) & mask
    d = (kw - sk)[mainland]; d = d[np.isfinite(d)]; d = d - modal(d)
    print(f"NISAR default: {dt:.1f}s mainland match={100*np.mean(np.abs(d)<0.5):.2f}%", flush=True)
    np.save(OUT / "nisar_default_unw.npy", unw)
    np.save(OUT / "nisar_anchor_sk.npy", sk.astype(np.float32))
    np.save(OUT / "nisar_anchor_scc.npy", scc)
    np.save(OUT / "nisar_anchor_mask.npy", mask)
    np.save(OUT / "nisar_anchor_wrapped.npy", wrapped)


def do_a016():
    with h5py.File(GUNW / A016) as h5:
        pol = sorted(k for k, v in h5[UNW].items() if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"})[0]
        prod = h5[f"{UNW}/{pol}/unwrappedPhase"][()].astype(np.float32)
        coh = h5[f"{UNW}/{pol}/coherenceMagnitude"][()].astype(np.float32)
        pcc = h5[f"{UNW}/{pol}/connectedComponents"][()].astype(np.int64)
        md = h5[f"{UNW}/mask"][()]
    valid = (md != 127) & np.isfinite(prod) & np.isfinite(coh)
    igc = np.ascontiguousarray(np.exp(1j * wrap(prod)), np.complex64)
    t0 = time.perf_counter()
    _unw, _cc = ww.unwrap(igc, np.ascontiguousarray(coh, np.float32), 16.0, np.ascontiguousarray(valid, bool))
    unw = np.asarray(_unw, np.float64)
    dt = time.perf_counter() - t0
    reg = valid & (pcc > 0)
    a = np.rint((unw - prod) / TAU)[reg]; a = a - modal(a)
    print(f"A_016 default (robust): {dt:.1f}s match={100*np.mean(np.abs(a)<0.5):.2f}%", flush=True)
    np.savez_compressed(A016OUT / "a016_default_fixed.npz",
                        unw=unw.astype(np.float32), prod=prod, pcc=pcc.astype(np.int32),
                        coh=coh, mask=valid, ig=wrap(prod).astype(np.float32))


if __name__ == "__main__":
    do_nisar()
    do_a016()
    print("done", flush=True)
