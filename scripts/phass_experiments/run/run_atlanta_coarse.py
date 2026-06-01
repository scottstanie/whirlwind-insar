"""Does a multilooked (coarse) whole-image unwrap of Atlanta match OPERA?
Tests the user's "subsample to constrain large-scale" intuition and whether
the failure is noise (fixable by multilook) or a solver problem (needs convex).

Multilook the complex igram by L, ww.unwrap the coarse whole image, upsample,
compare K to the OPERA reference on the mainland. Tries L in {4,8,16}.
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


def multilook(ig, coh, mask, L):
    h, w = ig.shape
    H, W = h // L, w // L
    igc = np.zeros((H, W), np.complex64)
    cohc = np.zeros((H, W), np.float32)
    mkc = np.zeros((H, W), bool)
    for ci in range(H):
        zr = ig[ci*L:(ci+1)*L]
        cr = coh[ci*L:(ci+1)*L]
        mr = mask[ci*L:(ci+1)*L]
        for cj in range(W):
            sel = mr[:, cj*L:(cj+1)*L]
            n = int(sel.sum())
            if 2 * n >= L * L:
                z = zr[:, cj*L:(cj+1)*L][sel].sum()
                igc[ci, cj] = z / abs(z) if abs(z) > 0 else 0
                cohc[ci, cj] = cr[:, cj*L:(cj+1)*L][sel].mean()
                mkc[ci, cj] = True
    return igc, cohc, mkc


def main() -> None:
    import whirlwind as ww
    phase = rasterio.open(N / "opera.int.phs.tif").read(1).astype(np.float32)
    coh = rasterio.open(N / "opera.int.cor.tif").read(1).astype(np.float32)
    disp = rasterio.open(N / "opera.displacement.tif").read(1).astype(np.float32)
    cc = rasterio.open(N / "opera.conncomp.tif").read(1).astype(np.int32)
    mask = np.isfinite(phase) & np.isfinite(coh) & np.isfinite(disp) & (coh > 0) & (coh < 1.0)
    wrapped = np.angle(np.exp(1j * np.where(mask, phase, 0.0))).astype(np.float32)
    ig = np.exp(1j * np.where(mask, phase, 0.0)).astype(np.complex64); ig[~mask] = 0
    cohw = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    pr = disp * (4.0 * np.pi / LAMBDA_S1)
    kref = np.round((pr - wrapped) / TAU); kref[~mask] = np.nan
    labels, counts = np.unique(cc[cc > 0], return_counts=True)
    mainland = mask & (cc == int(labels[np.argmax(counts)]))

    for L in (4, 8, 16):
        t0 = time.perf_counter()
        igc, cohc, mkc = multilook(ig, cohw, mask, L)
        unc, _cc = ww.unwrap(igc, cohc, nlooks=50.0 * L * L, mask=mkc)
        dt = time.perf_counter() - t0
        # upsample (nearest) and compare on mainland (downsample mainland to coarse for fair compare)
        H, W = unc.shape
        up = np.kron(unc, np.ones((L, L), np.float32))
        hh, wwd = up.shape
        m2 = mask[:hh, :wwd]
        kw = np.round((up - wrapped[:hh, :wwd]) / TAU); kw[~m2] = np.nan
        d = (kw - kref[:hh, :wwd])[mainland[:hh, :wwd]]; d = d[np.isfinite(d)]; d = d - modal(d)
        m0 = float((d == 0).sum())/d.size*100
        m2 = float((np.abs(d) >= 2).sum())/d.size*100
        print(f"L={L:2d}  coarse={H}x{W}  {dt:.1f}s  mainland match={m0:.2f}%  |dK|>=2={m2:.2f}%", flush=True)
        np.save(OUT / f"atlanta_coarseL{L}_unw.npy", up.astype(np.float32))


if __name__ == "__main__":
    main()
