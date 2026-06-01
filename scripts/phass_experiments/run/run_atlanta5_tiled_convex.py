"""Decisive 5x-Atlanta test: convex per-tile cost (WHIRLWIND_TILE_CONVEX=1)
vs linear, BOTH through the full tiled+anchor+cascade pipeline (ts=256, ov=32).
The env toggle is cached per-process (OnceLock), so run ONE variant per
invocation: set WHIRLWIND_TILE_CONVEX in the environment for the convex run.
Reference: snaphu 3x3 ~97.9%, ww linear tiled ~41%, multilook=8 ~97.7%.
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
    convex = "WHIRLWIND_TILE_CONVEX" in os.environ
    label = "convex" if convex else "linear"
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

    t0 = time.perf_counter()
    u, _cc = ww.unwrap(ifg, cor, nlooks=50.0, mask=mask, tile_size=256, tile_overlap=32)
    dt = time.perf_counter() - t0
    k = np.round((u - wrapped) / TAU); k[~mask] = np.nan
    m0, m2 = match(k, kref, mainland)
    np.save(OUT / f"atlanta5_tiled_{label}_unw.npy", u.astype(np.float32))
    print(f"ww tiled256 {label:7s} {dt:6.1f}s  match={m0:6.2f}%  |dK|>=2={m2:5.2f}%  "
          f"(mainland={mainland.sum():,})", flush=True)


if __name__ == "__main__":
    main()
