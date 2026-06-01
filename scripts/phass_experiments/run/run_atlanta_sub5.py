"""Fast Atlanta iteration on 5x-subsampled data (the user's recipe).

Loads the COMPLEX interferogram opera.int.tif decimated [::5,::5] (like
dolphin subsample_factor=5), coherence likewise, and compares:
  * snaphu.unwrap(ifg, cor, 50, ntiles=(3,3))   -- the user's working recipe
  * whirlwind whole-image  ww.unwrap(...)        -- feasible at 5x
  * whirlwind tiled+anchor+cascade               -- the new default
against each other and the OPERA reference (displacement, decimated).
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
    # reference K from OPERA displacement (sign auto-detect)
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

    results = {}

    # snaphu on identical input
    try:
        import snaphu
        t0 = time.perf_counter()
        sunw, scc = snaphu.unwrap(ifg, cor, nlooks=50.0, cost="smooth",
                                  mask=mask.astype(np.uint8), ntiles=(3, 3), tile_overlap=200)
        dt = time.perf_counter() - t0
        ks = np.round((np.asarray(sunw) - wrapped) / TAU); ks[~mask] = np.nan
        results["snaphu(3x3)"] = (dt, match(ks, kref, mainland), ks)
        np.save(OUT / "atlanta5_snaphu_unw.npy", np.asarray(sunw, np.float32))
    except Exception as e:
        print(f"snaphu skipped: {e}", flush=True)

    # whirlwind whole-image
    t0 = time.perf_counter()
    uw = ww.unwrap(ifg, cor, nlooks=50.0, mask=mask)
    dt = time.perf_counter() - t0
    kw = np.round((uw - wrapped) / TAU); kw[~mask] = np.nan
    results["ww whole"] = (dt, match(kw, kref, mainland), kw)
    np.save(OUT / "atlanta5_wwwhole_unw.npy", uw.astype(np.float32))

    # whirlwind tiled+anchor+cascade
    t0 = time.perf_counter()
    ut = ww.unwrap(ifg, cor, nlooks=50.0, mask=mask, tile_size=256, tile_overlap=32)
    dt = time.perf_counter() - t0
    kt = np.round((ut - wrapped) / TAU); kt[~mask] = np.nan
    results["ww tiled256"] = (dt, match(kt, kref, mainland), kt)
    np.save(OUT / "atlanta5_wwtiled_unw.npy", ut.astype(np.float32))

    np.save(OUT / "atlanta5_kref.npy", kref.astype(np.float32))
    np.save(OUT / "atlanta5_mask.npy", mask)
    np.save(OUT / "atlanta5_wrapped.npy", wrapped)
    np.save(OUT / "atlanta5_cc.npy", cc)

    print("\n=== ATLANTA 5x (match vs OPERA on mainland) ===", flush=True)
    for k, (dt, (m0, m2), _) in results.items():
        print(f"{k:14s} {dt:6.1f}s  match={m0:6.2f}%  |dK|>=2={m2:5.2f}%", flush=True)
    # whirlwind vs snaphu directly (same input)
    if "snaphu(3x3)" in results:
        ks = results["snaphu(3x3)"][2]
        for k in ("ww whole", "ww tiled256"):
            kk = results[k][2]
            d = (kk - ks)[mainland]; d = d[np.isfinite(d)]; d = d - modal(d)
            print(f"{k:14s} vs snaphu: {float((d==0).sum())/d.size*100:.2f}% match", flush=True)


if __name__ == "__main__":
    main()
