"""Produce the current Atlanta report inputs with the new binary: full-res
ww.unwrap(..., multilook=8) (the honest noisy-scene path), plus the reference
arrays. Saves atlanta_rep_{unw,kref,cc,mask,wrapped}.npy. Prints mainland AND
full-image match vs the OPERA reference.
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


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def main() -> None:
    import whirlwind as ww
    ig = rasterio.open(N / "opera.int.tif").read(1).astype(np.complex64)
    coh = rasterio.open(N / "opera.int.cor.tif").read(1).astype(np.float32)
    disp = rasterio.open(N / "opera.displacement.tif").read(1).astype(np.float32)
    cc = rasterio.open(N / "opera.conncomp.tif").read(1).astype(np.int32)
    mask = np.isfinite(ig) & np.isfinite(coh) & np.isfinite(disp) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig = np.where(mask, ig, 0).astype(np.complex64)
    cohw = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
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
    print(f"shape={ig.shape} valid={mask.sum():,} mainland={mainland.sum():,} sign={s:+.0f}", flush=True)

    t0 = time.perf_counter()
    unw, _cc = ww.unwrap(ig, cohw, nlooks=50.0, mask=mask, multilook=8)
    dt = time.perf_counter() - t0
    # multilook upsample (block-replicate) may trim the trailing partial block.
    hh, wwd = unw.shape
    m = mask[:hh, :wwd]; wr = wrapped[:hh, :wwd]; kr = kref[:hh, :wwd]
    reg = mainland[:hh, :wwd]
    kw = np.round((unw - wr) / TAU); kw[~m] = np.nan
    c = modal((kw - kr)[reg])
    dml = (kw - c - kr)[reg]; dml = dml[np.isfinite(dml)]
    dfu = (kw - c - kr)[m]; dfu = dfu[np.isfinite(dfu)]
    m0 = float((np.abs(dml) < 0.5).sum()) / dml.size * 100
    f0 = float((np.abs(dfu) < 0.5).sum()) / dfu.size * 100
    m2 = float((np.abs(dml) >= 2).sum()) / dml.size * 100
    print(f"multilook=8  {dt:5.1f}s  mainland={m0:6.2f}%  |dK|>=2={m2:5.2f}%  full={f0:6.2f}%", flush=True)

    np.save(OUT / "atlanta_rep_unw.npy", unw.astype(np.float32))
    np.save(OUT / "atlanta_rep_kref.npy", kr.astype(np.float32))
    np.save(OUT / "atlanta_rep_cc.npy", cc[:hh, :wwd].astype(np.int32))
    np.save(OUT / "atlanta_rep_mask.npy", m)
    np.save(OUT / "atlanta_rep_wrapped.npy", wr.astype(np.float32))
    print("saved atlanta_rep_* arrays", flush=True)


if __name__ == "__main__":
    main()
