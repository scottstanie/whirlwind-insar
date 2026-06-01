"""Fast 5x-Atlanta test of the FIXED convex cost (deviation offset + sound
preload) vs the linear cost and snaphu, on identical input. Whole-image
(unwrap_convex has no tiling yet). The deviation offset needs an absolute-level
source for big ramps, but on this decimated frame the per-arc gradients are
mostly sub-pi so integration carries the ramp; this isolates whether the convex
curvature + wrap-line offset beats the linear cost's integration runaway.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
TAU = np.float32(2 * np.pi)
LAMBDA_S1 = 0.05546576
SUB = 5


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def match(kw, kref, region):
    d = (kw - kref)[region]; d = d[np.isfinite(d)]; d = d - modal(d)
    n = d.size
    return float((d == 0).sum())/n*100, float((np.abs(d) >= 2).sum())/n*100


def main() -> None:
    import whirlwind as ww
    ifg = rasterio.open(N / "opera.int.tif").read(1)[::SUB, ::SUB].astype(np.complex64)
    cor = rasterio.open(N / "opera.int.cor.tif").read(1)[::SUB, ::SUB].astype(np.float32)
    disp = rasterio.open(N / "opera.displacement.tif").read(1)[::SUB, ::SUB].astype(np.float32)
    cc = rasterio.open(N / "opera.conncomp.tif").read(1)[::SUB, ::SUB].astype(np.int32)

    mask = np.isfinite(ifg) & np.isfinite(cor) & (cor > 0) & (cor < 1.0) & (np.abs(ifg) > 0)
    ifg = np.where(mask, ifg, 0).astype(np.complex64)
    cor = np.clip(np.where(mask, cor, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ifg).astype(np.float32)
    best = None
    for s in (+1.0, -1.0):
        pr = s * disp * (4.0 * np.pi / LAMBDA_S1)
        resid = (pr - wrapped) / TAU
        spread = float(np.nanstd((resid - np.round(resid))[mask]))
        if best is None or spread < best[0]:
            best = (spread, s, pr)
    _, s, pr = best
    kref = np.round((pr - wrapped) / TAU); kref[~mask] = np.nan
    labels, counts = np.unique(cc[cc > 0], return_counts=True)
    mainland = mask & (cc == int(labels[np.argmax(counts)]))
    print(f"shape={ifg.shape} valid={mask.sum():,} mainland={mainland.sum():,} sign={s:+.0f}", flush=True)

    runs = {}

    t0 = time.perf_counter()
    uw = ww.unwrap(ifg, cor, nlooks=50.0, mask=mask)
    runs["ww linear whole"] = (time.perf_counter() - t0, ww.unwrap and None)
    kw = np.round((uw - wrapped) / TAU); kw[~mask] = np.nan
    runs["ww linear whole"] = (runs["ww linear whole"][0], match(kw, kref, mainland))

    t0 = time.perf_counter()
    uc = ww.unwrap_convex(ifg, cor, nlooks=50.0, mask=mask)
    dt = time.perf_counter() - t0
    kc = np.round((uc - wrapped) / TAU); kc[~mask] = np.nan
    runs["ww convex whole"] = (dt, match(kc, kref, mainland))
    np.save(OUT / "atlanta5_convex_unw.npy", uc.astype(np.float32))

    print("\n=== ATLANTA 5x: convex-cost fix (match vs OPERA on mainland) ===", flush=True)
    for k, (dt, (m0, m2)) in runs.items():
        print(f"{k:18s} {dt:6.1f}s  match={m0:6.2f}%  |dK|>=2={m2:5.2f}%", flush=True)
    print("(reference: snaphu 3x3 ~97.9%, ww linear tiled256 ~41%)", flush=True)


if __name__ == "__main__":
    main()
