"""Is the col-4032 -1 strip born inside ONE tile's per-tile solve, or in the
composite? Re-unwrap the two overlapping tiles STANDALONE (whole, no sub-tiling
-- exactly what unwrap_one_tile_coh does) and check each tile's local strip cols
against SNAPHU.

Seam at col 4032 = 9*448 (step = tile_size 512 - overlap 64). Tile col 8 starts
at 3584 (covers col 4032 at local 448); tile col 9 starts at 4032 (local 0).
Strip rows ~956..1375 live in tile row starting at 896 (896..1408).

If standalone tile-8 shows -1 at local col 448-449 -> the per-tile MCF solve is
the culprit (composite faithfully reproduces it because tile 8 dominates the
taper there). If tile-9 local col 0-1 is correct, a continuity-aware composite
would fix it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
TAU = float(2 * np.pi)
TS, OV = 512, 64
NLOOKS = 100.0


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def axis_starts(total, tile, step):
    if total <= tile:
        return [0]
    starts = [0]
    while True:
        nxt = starts[-1] + step
        if nxt + tile >= total:
            starts.append(total - tile)
            return starts
        starts.append(nxt)


def main():
    import whirlwind as ww

    ig = rasterio.open(N / "20251224_20260117.int.looked.tif").read(1).astype(np.complex64)
    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    sk = np.load(OUT / "nisar_anchor_sk.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    mask_full = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig[~mask_full] = 0
    coh = np.clip(np.where(mask_full, coh, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    m, n = ig.shape
    print(f"image {m}x{n}", flush=True)
    rs = axis_starts(m, TS, TS - OV)
    cs = axis_starts(n, TS, TS - OV)
    print(f"col tile starts include 4032: {4032 in cs}; row starts include 896: {896 in cs or 896 in rs}", flush=True)
    print(f"col starts near 4032: {[c for c in cs if 3500 < c < 4100]}", flush=True)

    r0 = 896
    for cstart, loccol in [(3584, 448), (4032, 0)]:
        sl = (slice(r0, r0 + TS), slice(cstart, cstart + TS))
        sub_ig = np.ascontiguousarray(ig[sl])
        sub_coh = np.ascontiguousarray(coh[sl])
        sub_mask = np.ascontiguousarray(mask_full[sl])
        # standalone whole-tile unwrap == unwrap_one_tile_coh
        u = ww.unwrap(sub_ig, sub_coh, nlooks=NLOOKS, mask=sub_mask)
        kt = np.round((u - np.angle(sub_ig)) / TAU)
        kt[~sub_mask] = np.nan
        skt = sk[sl]
        dk = kt - skt
        ml = (scc[sl] == 1) & sub_mask
        dk_ml = dk[ml]
        dk = dk - modal(dk_ml)
        # local strip rows = global 956..1375 -> local (956-896)..(1375-896) = 60..479
        lr = slice(60, 480)
        print(f"\n=== tile colstart={cstart} (strip at local col {loccol}/{loccol+1}) ===", flush=True)
        cols = range(max(0, loccol - 4), min(TS, loccol + 6))
        meds = []
        for j in cols:
            v = np.where(ml[lr, j], dk[lr, j], np.nan)
            v = v[np.isfinite(v)]
            meds.append((j, int(np.median(v)) if v.size else 9, v.size))
        print("  local col : medianDK (nrows)", flush=True)
        for j, md, ns in meds:
            tag = " <-- strip" if j in (loccol, loccol + 1) else ""
            print(f"   {j:4d} : {md:+d} ({ns}){tag}", flush=True)


if __name__ == "__main__":
    main()
