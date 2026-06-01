"""Atlanta S-1 OPERA frame: tiled anchor+cascade (new default) vs no-anchor,
measured against the OPERA (SNAPHU) reference. Mirrors run_nisar_anchor.py.

Reference: opera.displacement.tif (LOS m) -> phase via 4pi/lambda with the sign
auto-detected by congruence (same as analyze_atlanta.py). Regions:
  mainland = largest OPERA component & mask  (tripwire)
  reliable = (cc>0) & mask
  full     = mask
Runs the two variants SEQUENTIALLY (one heavy unwrap at a time).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
TAU = np.float32(2 * np.pi)
LAMBDA_S1 = 0.05546576
TS, OV, NLOOKS = 512, 64, 50.0


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def match_pct(kw, ks, region):
    d = (kw - ks)[region]; d = d[np.isfinite(d)]; d = d - modal(d)
    n = d.size
    return (float((d == 0).sum())/n*100, float((np.abs(d) == 1).sum())/n*100, float((np.abs(d) >= 2).sum())/n*100)


def main() -> None:
    import whirlwind as ww
    phase = rasterio.open(N / "opera.int.phs.tif").read(1).astype(np.float32)
    coh = rasterio.open(N / "opera.int.cor.tif").read(1).astype(np.float32)
    disp = rasterio.open(N / "opera.displacement.tif").read(1).astype(np.float32)
    cc = rasterio.open(N / "opera.conncomp.tif").read(1).astype(np.int32)

    mask = np.isfinite(phase) & np.isfinite(coh) & np.isfinite(disp) & (coh > 0) & (coh < 1.0)
    wrapped = np.angle(np.exp(1j * np.where(mask, phase, 0.0))).astype(np.float32)
    ig = np.exp(1j * np.where(mask, phase, 0.0)).astype(np.complex64)
    ig[~mask] = 0
    cohw = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)

    # sign-detect displacement -> phase
    best = None
    for s in (+1.0, -1.0):
        pr = s * disp * (4.0 * np.pi / LAMBDA_S1)
        resid = (pr - wrapped) / TAU
        spread = float(np.nanstd((resid - np.round(resid))[mask]))
        if best is None or spread < best[0]:
            best = (spread, s, pr)
    spread, s, phase_ref = best
    k_ref = np.round((phase_ref - wrapped) / TAU)
    k_ref[~mask] = np.nan
    print(f"[atlanta] sign={s:+.0f} congruence-std={spread:.4f}", flush=True)

    labels, counts = np.unique(cc[cc > 0], return_counts=True)
    main_label = int(labels[np.argmax(counts)])
    mainland = mask & (cc == main_label)
    reliable = mask & (cc > 0)
    full = mask
    print(f"shape={phase.shape} valid={mask.sum():,} mainland(cc={main_label})={mainland.sum():,} "
          f"reliable={reliable.sum():,}", flush=True)

    results = {}
    for label, no_anchor in [("anchor", False), ("no_anchor", True)]:
        if no_anchor:
            os.environ["WHIRLWIND_NO_ANCHOR"] = "1"
        else:
            os.environ.pop("WHIRLWIND_NO_ANCHOR", None)
        t0 = time.perf_counter()
        unw, _cc = ww.unwrap(ig, cohw, nlooks=NLOOKS, mask=mask, tile_size=TS, tile_overlap=OV)
        dt = time.perf_counter() - t0
        kw = np.round((unw - wrapped) / TAU); kw[~mask] = np.nan
        results[label] = {"elapsed": dt,
                          "mainland": match_pct(kw, k_ref, mainland),
                          "reliable": match_pct(kw, k_ref, reliable),
                          "full": match_pct(kw, k_ref, full)}
        print(f"\n[{label}] {dt:.1f}s", flush=True)
        for reg in ("mainland", "reliable", "full"):
            m0, m1, m2 = results[label][reg]
            print(f"   {reg:9s} match={m0:6.2f}%  |dK|=1={m1:5.2f}%  |dK|>=2={m2:5.2f}%", flush=True)
        np.save(OUT / f"atlanta_{label}_unw.npy", unw.astype(np.float32))

    np.save(OUT / "atlanta_kref.npy", k_ref.astype(np.float32))
    np.save(OUT / "atlanta_cc.npy", cc)
    np.save(OUT / "atlanta_mask.npy", mask)
    np.save(OUT / "atlanta_wrapped.npy", wrapped)

    print("\n=== ATLANTA SUMMARY (match%) ===", flush=True)
    for reg in ("mainland", "reliable", "full"):
        na = results["no_anchor"][reg][0]; an = results["anchor"][reg][0]
        print(f"{reg:9s} no_anchor={na:6.2f}%  anchor={an:6.2f}%  delta={an-na:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
