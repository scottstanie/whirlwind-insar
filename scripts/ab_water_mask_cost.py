#!/usr/bin/env python3
"""A/B the GUNW water-masking decision: does masking water cost us runtime?

Masking water severs the valid domain along every river, which leaves the MCF
problem fragmented. This runs the same frame twice -- water masked vs kept --
and reports runtime, component counts, and the solver's stranded-residue state,
so the runtime difference can be attributed rather than guessed at.

Run with WHIRLWIND_DEBUG=1 to see the adaptive-resume / BFS-drain lines that
the fragmented case triggers, and WHIRLWIND_TIMING=1 for the phase/conncomp
split. Sequential on purpose: one heavy NISAR-scale unwrap at a time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

import whirlwind as ww

B = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"


def load(path: Path):
    with h5py.File(path, "r") as f:
        pol = next(k for k in f[B] if k in ("HH", "VV", "HV", "VH"))
        low = f[f"{B}/mask"][()].astype(np.int64) & 0xFF
        unw = f[f"{B}/{pol}/unwrappedPhase"][()].astype(np.float32)
        coh = f[f"{B}/{pol}/coherenceMagnitude"][()].astype(np.float32)
    fin = np.isfinite(unw) & np.isfinite(coh)
    nd = low == 255
    water = ((low // 100) % 10) != 0
    subswath_ok = (((low // 10) % 10) > 0) & ((low % 10) > 0)
    igram = np.exp(1j * np.angle(np.exp(1j * unw))).astype(np.complex64)
    return {
        "igram": igram,
        "coh": np.clip(np.nan_to_num(coh), 0.0, 1.0),
        "subswath": (~nd) & subswath_ok & fin,
        "water": water,
        "pol": pol,
    }


def run_case(name: str, d: dict, mask: np.ndarray, nlooks: float) -> dict:
    lab, _ = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    igram = np.where(mask, d["igram"], 0).astype(np.complex64)
    corr = np.where(mask, d["coh"], 0.0).astype(np.float32)
    print(
        f"\n--- {name}: valid={mask.mean():.3f} "
        f"regions={len(sizes)} (>1000px: {(sizes > 1000).sum()})",
        flush=True,
    )
    t = time.perf_counter()
    unw, cc = ww.unwrap(igram, corr, nlooks=nlooks, mask=mask)
    dt = time.perf_counter() - t
    print(f"--- {name}: {dt:.1f}s  ww_num_cc={len(np.unique(cc)) - 1}", flush=True)
    return {
        "name": name,
        "runtime_s": dt,
        "regions": len(sizes),
        "big_regions": int((sizes > 1000).sum()),
        "unw": unw,
        "cc": cc,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("product", type=Path)
    p.add_argument("--nlooks", type=float, default=16.0)
    args = p.parse_args()

    d = load(args.product)
    keep = run_case("water KEPT (paired samples)", d, d["subswath"], args.nlooks)
    masked = run_case("water MASKED", d, d["subswath"] & ~d["water"], args.nlooks)

    print(f"\n{'case':<28} {'runtime':>9} {'regions':>9} {'big':>5}")
    for r in (keep, masked):
        print(
            f"{r['name']:<28} {r['runtime_s']:>8.1f}s {r['regions']:>9} {r['big_regions']:>5}"
        )
    print(
        f"\nmasking water costs {masked['runtime_s'] / keep['runtime_s']:.2f}x runtime"
    )


if __name__ == "__main__":
    main()
