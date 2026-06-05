"""Verify the feather (NISAR seam lines) + the multilook= API (Atlanta one-call).

NISAR: ww.unwrap(.., tile_size=512, tile_overlap=64) - feather+anchor+cascade.
       Check mainland match holds ~99.89% and the per-column vertical-tear count
       at tile-seam columns drops vs the saved no-anchor baseline.
Atlanta: ww.unwrap(.., multilook=8) - the new one-call noisy-scene path; should
       reproduce the multilook-8 + tiled recipe (~97.7%).
Re-saves NISAR/Atlanta arrays for the report figures.
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


def match(kw, ks, region):
    d = (kw - ks)[region]
    d = d[np.isfinite(d)]
    d = d - modal(d)
    return float((d == 0).sum()) / d.size * 100, float(
        (np.abs(d) >= 2).sum()
    ) / d.size * 100


def vtears(unw, mask):
    a = unw.copy()
    a[~mask] = np.nan
    dh = np.abs(a[:, 1:] - a[:, :-1])
    return np.nansum(dh > np.pi, axis=0)  # per-column-edge tear count


def main() -> None:
    import whirlwind as ww

    # ---- NISAR (feather) ----
    ig = (
        rasterio.open(N / "20251224_20260117.int.looked.tif")
        .read(1)
        .astype(np.complex64)
    )
    coh = (
        rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif")
        .read(1)
        .astype(np.float32)
    )
    sk = np.load(OUT / "nisar_anchor_sk.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig[~mask] = 0
    coh = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    mainland = (scc == 1) & mask

    t0 = time.perf_counter()
    unw, _cc = ww.unwrap(
        ig, coh, nlooks=100.0, mask=mask, tile_size=512, tile_overlap=64
    )
    dt = time.perf_counter() - t0
    kw = np.round((unw - wrapped) / TAU)
    kw[~mask] = np.nan
    m0, m2 = match(kw, sk, mainland)
    np.save(OUT / "nisar_cascade_unw.npy", unw.astype(np.float32))

    seam_cols = [1855, 2303, 2751, 3199, 3647, 4095]
    new_t = vtears(unw, mask)
    old = np.load(OUT / "nisar_no_anchor_unw.npy")
    old_t = vtears(old, mask)
    print(
        f"[NISAR feather] {dt:.1f}s  mainland match={m0:.2f}%  |dK|>=2={m2:.2f}%",
        flush=True,
    )
    print(
        f"  total vertical tears: no-anchor={int(old_t.sum()):,}  feather={int(new_t.sum()):,}",
        flush=True,
    )
    print(f"  at tile-seam cols {seam_cols}:", flush=True)
    print(f"    no-anchor: {[int(old_t[c]) for c in seam_cols]}", flush=True)
    print(f"    feather:   {[int(new_t[c]) for c in seam_cols]}", flush=True)
    del ig, coh, unw, old

    # ---- Atlanta (multilook= API) ----
    phase = rasterio.open(N / "opera.int.phs.tif").read(1).astype(np.float32)
    acoh = rasterio.open(N / "opera.int.cor.tif").read(1).astype(np.float32)
    disp = rasterio.open(N / "opera.displacement.tif").read(1).astype(np.float32)
    acc = rasterio.open(N / "opera.conncomp.tif").read(1).astype(np.int32)
    amask = (
        np.isfinite(phase)
        & np.isfinite(acoh)
        & np.isfinite(disp)
        & (acoh > 0)
        & (acoh < 1.0)
    )
    awr = np.angle(np.exp(1j * np.where(amask, phase, 0.0))).astype(np.float32)
    aig = np.exp(1j * np.where(amask, phase, 0.0)).astype(np.complex64)
    aig[~amask] = 0
    acohw = np.clip(np.where(amask, acoh, 0), 0, 1).astype(np.float32)
    pr = disp * (4.0 * np.pi / LAMBDA_S1)
    akref = np.round((pr - awr) / TAU)
    akref[~amask] = np.nan
    labels, counts = np.unique(acc[acc > 0], return_counts=True)
    amain = amask & (acc == int(labels[np.argmax(counts)]))

    t0 = time.perf_counter()
    aunw, _acc = ww.unwrap(aig, acohw, nlooks=50.0, mask=amask, multilook=8)
    adt = time.perf_counter() - t0
    akw = np.round((aunw - awr) / TAU)
    akw[~amask] = np.nan
    am0, am2 = match(akw, akref, amain)
    print(
        f"\n[Atlanta multilook=8 API] {adt:.1f}s  mainland match={am0:.2f}%  |dK|>=2={am2:.2f}%",
        flush=True,
    )
    np.save(OUT / "atlanta_ml8api_unw.npy", aunw.astype(np.float32))


if __name__ == "__main__":
    main()
