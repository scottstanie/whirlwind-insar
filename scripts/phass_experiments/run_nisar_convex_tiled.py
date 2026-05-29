"""Does the convex per-tile cost (WHIRLWIND_TILE_CONVEX=1) remove the spurious
col-4032 sliver (a linear-MCF discharge artifact) without regressing the
mainland? ONE heavy NISAR unwrap (tiled512+anchor+cascade, convex per-tile).
Compares against the saved linear shipped field (nisar_cascade_unw.npy).
Set WHIRLWIND_TILE_CONVEX externally for the convex run.
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
TS, OV, NLOOKS = 512, 64, 100.0


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def match(kw, ks, region):
    d = (kw - ks)[region]; d = d[np.isfinite(d)]; d = d - modal(d)
    n = d.size
    return float((d == 0).sum())/n*100, float((np.abs(d) == 1).sum())/n*100, float((np.abs(d) >= 2).sum())/n*100


def main() -> None:
    import whirlwind as ww
    convex = "WHIRLWIND_TILE_CONVEX" in os.environ
    label = "convex" if convex else "linear"

    ig = rasterio.open(N / "20251224_20260117.int.looked.tif").read(1).astype(np.complex64)
    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig[~mask] = 0
    coh = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    sk = np.load(OUT / "nisar_anchor_sk.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    mainland = (scc == 1) & mask
    reliable = (scc > 0) & mask

    t0 = time.perf_counter()
    unw = ww.unwrap(ig, coh, nlooks=NLOOKS, mask=mask, tile_size=TS, tile_overlap=OV)
    dt = time.perf_counter() - t0
    kw = np.round((unw - wrapped) / TAU); kw[~mask] = np.nan
    np.save(OUT / f"nisar_tileconvex_{label}_unw.npy", unw.astype(np.float32))

    print(f"[{label}] {dt:.1f}s", flush=True)
    for reg, name in [(mainland, "mainland"), (reliable, "reliable"), (mask, "full")]:
        m0, m1, m2 = match(kw, sk, reg)
        print(f"   {name:9s} match={m0:6.2f}%  |dK|=1={m1:5.2f}%  |dK|>=2={m2:5.2f}%", flush=True)

    # col-4032 sliver check: ΔK over the strip rows 956..1375, cols 4029..4036
    dk = kw - sk; dk = dk - modal(dk[mainland])
    rows = slice(956, 1376)
    print("   col-4032 sliver (median ΔK on mainland over strip rows):", flush=True)
    cols = []
    for j in range(4029, 4037):
        v = np.where(mainland[rows, j], dk[rows, j], np.nan)
        v = v[np.isfinite(v)]
        cols.append(int(np.median(v)) if v.size else 9)
    print("     cols 4029..4036: " + " ".join(f"{c:+d}" for c in cols)
          + ("   <- 4032/4033 should be 0 if fixed" ), flush=True)
    # count coherent vertical-line error px on mainland (coh>0.45)
    valid = mainland & (coh > 0.45)
    u = np.where(valid, unw, np.nan)
    kl = np.round((u[:, :-2] - u[:, 1:-1]) / TAU); kr = np.round((u[:, 2:] - u[:, 1:-1]) / TAU)
    vc = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
    gh = vc & (kl == kr) & (kl != 0)
    print(f"   1px ghost-line px (coh>0.45 mainland): {int(gh.sum()):,}  worst col: {int(gh.sum(0).max())}", flush=True)


if __name__ == "__main__":
    main()
