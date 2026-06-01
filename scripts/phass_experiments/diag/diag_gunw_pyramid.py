"""Does the #28 pyramidal unwrap fix A_016's fragmented-scene offset block?
My linear-cost coarse ANCHOR mis-levels the right half by +2 cycles. The pyramid
uses a reuse/convex base solver (which fixes the linear-cost boundary-stacking),
so its coarse level may level the two halves correctly. Test on saved A_016
arrays vs production comp1 (cc>0). Compares against the known tiled baselines
(512=57.5%, 1024=91%, 2048=97.1%).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

BENCH = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_variants")
A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
# the bench saved arrays under the original out-dir:
ORIG = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_bench") / A016 / "full_arrays.npz"
TAU = 2 * np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def match_cc(unw, wrapped, prod, mask, pcc):
    kw = np.rint((unw - wrapped) / TAU)
    amb = np.rint((unw - prod) / TAU)
    reg = mask & (pcc > 0)
    a = amb[reg]; a = a[np.isfinite(a)]; a = a - modal(a)
    return 100 * np.mean(np.abs(a) < 0.5)


def main() -> None:
    import whirlwind as ww
    d = np.load(ORIG)
    mask = d["mask"]; prod = d["prod_unw"]; pcc = d["prod_cc"]; coh = d["coh"]
    ig = d["ig"].astype(np.float32)
    igc = np.exp(1j * ig).astype(np.complex64); igc[~mask] = 0
    cohw = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    wrapped = ig
    print(f"A_016 {igc.shape} valid={mask.mean():.3f}  (baselines: tiled512=57.5%, 1024=91%, 2048=97.1%)", flush=True)

    configs = [
        ("pyramid base=4 reuse", dict(base_factor=4, solver="reuse", tile_size=0)),
        ("pyramid base=8 reuse", dict(base_factor=8, solver="reuse", tile_size=0)),
        ("pyramid base=8 convex", dict(base_factor=8, solver="convex", tile_size=0)),
        ("pyramid base=8 reuse tile512", dict(base_factor=8, solver="reuse", tile_size=512)),
        ("pyramid base=16 reuse", dict(base_factor=16, solver="reuse", tile_size=0)),
    ]
    for label, kw in configs:
        t0 = time.perf_counter()
        unw = ww.unwrap_pyramid(igc, cohw, nlooks=16.0, mask=mask, **kw)
        dt = time.perf_counter() - t0
        m = match_cc(unw, wrapped, prod, mask, pcc)
        print(f"  {label:32s} {dt:6.1f}s  cc>0 match={m:6.2f}%", flush=True)


if __name__ == "__main__":
    main()
