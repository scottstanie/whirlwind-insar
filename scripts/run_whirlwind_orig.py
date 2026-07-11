#!/usr/bin/env python3
"""Run the original whirlwind (whole-image Carballo MCF, no tiling) on NISAR GUNW
frames and save results for comparison plots.

Usage:
    uv run python scripts/run_whirlwind_orig.py --frames 005_D_077 005_D_078 006_A_035

Outputs: /Volumes/WD_.../phass_ref/<frame>_wworig.npz
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import h5py
import numpy as np

WD = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")
TWOPI = 2.0 * np.pi
GUNW_DIR = WD / "nisar_gunw"


def wrap_phase(x):
    return (x + np.pi) % TWOPI - np.pi


def gunw_layers(path):
    with h5py.File(path, "r") as h:
        base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
        grp = h[base]
        pols = [
            k
            for k, v in grp.items()
            if isinstance(v, h5py.Group) and k.upper() not in ("MASK", "METADATA")
        ]
        p = sorted(pols)[0]
        prod_unw = h[f"{base}/{p}/unwrappedPhase"][()].astype(np.float32)
        coh = h[f"{base}/{p}/coherenceMagnitude"][()].astype(np.float32)
        prod_cc = h[f"{base}/{p}/connectedComponents"][()].astype(np.int32)
        mask_arr = h[f"{base}/mask"][()] if "mask" in grp else None
    mask = (
        (mask_arr != 255) & ((mask_arr // 100) % 10 == 0)
        if mask_arr is not None
        else np.ones(prod_unw.shape, bool)
    )
    mask &= np.isfinite(prod_unw) & np.isfinite(coh)
    return prod_unw, coh, prod_cc, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames", nargs="+", default=["005_D_077", "005_D_078", "006_A_035"]
    )
    args = ap.parse_args()

    from whirlwind_orig._unwrap import unwrap as ww_orig_unwrap

    out_dir = WD / "phass_ref"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fr in args.frames:
        globs = glob.glob(str(GUNW_DIR / f"*{fr}*.h5"))
        if not globs:
            print(f"{fr}: no h5 found in {GUNW_DIR}")
            continue
        path = globs[0]
        print(f"{fr}: loading {Path(path).name} ...", flush=True)
        prod_unw, coh, prod_cc, mask = gunw_layers(path)
        wrapped = np.where(mask, wrap_phase(prod_unw), 0.0).astype(np.float32)
        ig = np.exp(1j * wrapped).astype(np.complex64)
        coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(
            np.float32
        )

        print(f"  shape={ig.shape}  running whirlwind_orig ...", flush=True)
        t0 = time.perf_counter()
        unw = ww_orig_unwrap(ig, coh_in, 16.0, mask=~mask)
        dt = time.perf_counter() - t0
        unw = np.asarray(unw, np.float32)
        print(f"  done in {dt:.1f}s", flush=True)

        npz = out_dir / f"{fr}_wworig.npz"
        np.savez_compressed(
            npz,
            unw=unw,
            prod_unw=prod_unw,
            prod_cc=prod_cc,
            mask=mask,
            coh=np.nan_to_num(coh).astype(np.float32),
        )
        print(f"  saved -> {npz}")


if __name__ == "__main__":
    main()
