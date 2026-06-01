"""Validate the staggered double-tiling fix for tile-SEAM artifacts (the +1/+2
strip at col-896 the user flagged).

Idea (user's 'shifting tiles'): every pixel is a tile EDGE in grid-1 is a tile
INTERIOR in a grid shifted by half a step. Run the unwrap twice (grid-1 and a
half-step-shifted grid-2), then merge per-pixel preferring the solution where the
pixel is FARTHER from any seam (more interior => higher per-tile quality, where
the seam-free re-solve was shown to be clean: 89.7%->99.9% on the strip window).

Shift is realized by padding the image (invalid) by tile_step/2 before unwrapping,
then cropping back. Merge: align grid-2 to grid-1 by global modal offset, then
where they AGREE use both; where they DISAGREE pick the more-interior one.

Test: A_016 strip removed + global match; A_013 (clean) no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

import whirlwind as ww

GUNW = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw")
TAU = 2 * np.pi
UNW = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
TS, OV = 512, 64
STEP = TS - OV  # 448
FRAMES = {
    "A_016": "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001.h5",
    "A_013": "NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_20251017T124857_20251029T124836_20251029T124858_X05010_N_P_J_001.h5",
}


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def read(name):
    with h5py.File(GUNW / FRAMES[name]) as h5:
        pol = sorted(k for k, v in h5[UNW].items() if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"})[0]
        prod = h5[f"{UNW}/{pol}/unwrappedPhase"][()].astype(np.float32)
        coh = h5[f"{UNW}/{pol}/coherenceMagnitude"][()].astype(np.float32)
        pcc = h5[f"{UNW}/{pol}/connectedComponents"][()].astype(np.int64)
        md = h5[f"{UNW}/mask"][()]
    valid = (md != 127) & np.isfinite(prod) & np.isfinite(coh)
    return wrap(prod).astype(np.float32), coh, valid, prod, pcc


def seam_dist(shape, shift):
    """Per-pixel distance to the nearest tile seam for a grid whose tiles start
    at `shift + k*STEP`. Approx: distance to nearest tile-start line (rows & cols)."""
    H, W = shape
    starts_r = np.array([shift + k * STEP for k in range(-1, H // STEP + 2)])
    starts_c = np.array([shift + k * STEP for k in range(-1, W // STEP + 2)])
    ri = np.arange(H)[:, None]
    ci = np.arange(W)[None, :]
    dr = np.min(np.abs(ri - starts_r[None, None, :].reshape(1, 1, -1)), axis=2) if False else \
        np.min(np.abs(np.arange(H)[:, None] - starts_r[None, :]), axis=1)
    dc = np.min(np.abs(np.arange(W)[:, None] - starts_c[None, :]), axis=1)
    return np.minimum(dr[:, None], dc[None, :])


def unwrap_shifted(ig, coh, valid, shift):
    """Unwrap with the tile grid effectively shifted by `shift` (pad top/left)."""
    H, W = ig.shape
    igc = np.exp(1j * ig).astype(np.complex64)
    if shift == 0:
        u, _cc = ww.unwrap(np.ascontiguousarray(igc), np.ascontiguousarray(coh, np.float32),
                      16.0, np.ascontiguousarray(valid, bool), tile_size=TS, tile_overlap=OV)
        return np.asarray(u, np.float64)
    pig = np.zeros((H + shift, W + shift), np.complex64)
    pco = np.zeros((H + shift, W + shift), np.float32)
    pmk = np.zeros((H + shift, W + shift), bool)
    pig[shift:, shift:] = igc; pco[shift:, shift:] = coh; pmk[shift:, shift:] = valid
    u, _cc = ww.unwrap(np.ascontiguousarray(pig), np.ascontiguousarray(pco), 16.0,
                  np.ascontiguousarray(pmk), tile_size=TS, tile_overlap=OV)
    return np.asarray(u, np.float64)[shift:, shift:]


def main() -> None:
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(FRAMES)
    SHIFT = STEP // 2  # 224
    for name in names:
        ig, coh, valid, prod, pcc = read(name)
        reg = valid & (pcc > 0) & np.isfinite(prod)

        u1 = unwrap_shifted(ig, coh, valid, 0)
        u2 = unwrap_shifted(ig, coh, valid, SHIFT)
        # align u2 to u1 by global modal integer offset
        off = modal(np.rint((u2 - u1)[valid] / TAU))
        u2 = u2 - off * TAU

        d1 = seam_dist(ig.shape, 0)
        d2 = seam_dist(ig.shape, SHIFT)
        # merge: prefer more-interior; where they agree (same integer) either is fine
        agree = np.rint((u1 - u2) / TAU) == 0
        pick1 = (d1 >= d2)
        merged = np.where(agree, np.where(pick1, u1, u2), np.where(pick1, u1, u2))
        # (when they disagree, still pick the more-interior one)

        def match(u):
            a = np.rint((u - prod) / TAU)[reg]; a = a[np.isfinite(a)]; return 100 * np.mean(np.abs(a - modal(a)) < 0.5)
        # strip box (A_016)
        box = np.zeros_like(valid); box[1110:1376, 860:946] = True
        def stripoff(u):
            a = np.rint((u - prod) / TAU); a = a - modal(a[reg]); return int((box & reg & (np.abs(a) >= 1)).sum())
        print(f"{name}: grid1={match(u1):.2f}%  grid2={match(u2):.2f}%  MERGED={match(merged):.2f}%  "
              f"agree={100*agree[valid].mean():.1f}%", flush=True)
        if name == "A_016":
            print(f"   strip off-by>=1: grid1={stripoff(u1):,}  grid2={stripoff(u2):,}  merged={stripoff(merged):,}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
