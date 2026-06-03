#!/usr/bin/env python3
"""Multilook-anchor + seamless whole-image flatten-refine (issue #65, the
artifact-free path). The anchor is a COARSE multilook unwrap (coherent averaging
suppresses noise AND shrinks the image below the ~256px runaway threshold, so
the coarse solve does not run away); we upsample it, flatten the full-res wrapped
phase by it, unwrap the residual WHOLE-IMAGE (no tiling -> no seams), add back.
Optionally cascade through several scales.

Pure orchestration of ww.unwrap (no Rust changes), for fast iteration before
hardening into the API.

    WHIRLWIND_TILE_SOLVER=reuse env -u CONDA_PREFIX uv run --no-sync \
        python scripts/multilook_anchor_refine.py --size 1024 --looks 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

TWOPI = 2.0 * np.pi
WD = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")


def wrap(x):
    return (x + np.pi) % TWOPI - np.pi


def percomp_match(test, prod_unw, wrapped, prod_cc, valid):
    amb = np.rint((test - wrapped) / TWOPI) - np.rint((prod_unw - wrapped) / TWOPI)
    in_comp = valid & (prod_cc > 0)
    if not in_comp.any():
        return float("nan")
    off = np.zeros(amb.shape)
    for lab in np.unique(prod_cc[in_comp]):
        m = valid & (prod_cc == lab)
        off[m] = np.rint(np.median(amb[m]))
    return float(np.mean((amb - off)[in_comp] == 0))


def block_mean(a, L):
    """Mean-pool by factor L (crop to a multiple of L). Works for complex/float."""
    M, N = a.shape[:2]
    M2, N2 = (M // L) * L, (N // L) * L
    a = a[:M2, :N2]
    return a.reshape(M2 // L, L, N2 // L, L).mean(axis=(1, 3))


def upsample(a, shape):
    """Bilinear upsample to `shape` (smooth, so per-arc gradients are nonzero)."""
    from scipy.ndimage import zoom
    zy, zx = shape[0] / a.shape[0], shape[1] / a.shape[1]
    return zoom(a, (zy, zx), order=1).astype(np.float32)[: shape[0], : shape[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024, help="0 = full frame; else center crop")
    ap.add_argument("--looks", type=int, nargs="+", default=[8], help="multilook factor(s)")
    ap.add_argument("--nlooks", type=float, default=16.0)
    ap.add_argument("--anchor-only", action="store_true", help="skip the (dead) flatten+refine")
    args = ap.parse_args()

    sn = np.load(WD / "snaphu_ref" / "D_077.npz")
    prod_unw = sn["prod_unw"].astype(np.float32)
    prod_cc = sn["prod_cc"].astype(np.int64)
    coh = np.nan_to_num(sn["coh"]).astype(np.float32)
    mask = sn["mask"]
    wrapped = np.nan_to_num(sn["wrapped"]).astype(np.float32)
    M, N = prod_unw.shape
    if args.size:
        s = args.size
        i0, j0 = (M - s) // 2, (N - s) // 2
        sl = (slice(i0, i0 + s), slice(j0, j0 + s))
        prod_unw, prod_cc, coh, mask, wrapped = (a[sl].copy() for a in (prod_unw, prod_cc, coh, mask, wrapped))
        M, N = s, s
    ig = np.exp(1j * wrapped).astype(np.complex64)
    big = max(M, N) + 100
    import whirlwind as ww

    def match(field):
        v = mask & np.isfinite(field)
        return percomp_match(np.asarray(field, np.float32), prod_unw, wrapped, prod_cc, v)

    print(f"D_077 {M}x{N}")
    t0 = time.perf_counter()
    u_tiled, _ = ww.unwrap(ig, coh, args.nlooks, mask)
    print(f"  baseline tiled        : {match(u_tiled)*100:6.2f}%  ({time.perf_counter()-t0:.1f}s)")

    for L in args.looks:
        # --- coarse multilook anchor ---
        t0 = time.perf_counter()
        cig = block_mean(ig, L).astype(np.complex64)        # coherent average
        ccoh = np.clip(block_mean(coh, L), 0, 1).astype(np.float32)
        cmask = block_mean(mask.astype(np.float32), L) > 0.5
        cunw, _ = ww.unwrap(cig, ccoh, args.nlooks * L * L, cmask,
                            tile_size=max(cig.shape) + 50, tile_overlap=0)  # whole-image coarse
        cunw = np.asarray(cunw, np.float32)
        cunw = np.where(np.isfinite(cunw), cunw, 0.0)
        anchor = upsample(cunw, (M, N))
        t_anchor = time.perf_counter() - t0
        coarse_px = max(cig.shape)
        if args.anchor_only:
            print(f"  multilook-{L:>2d} anchor   : coarse {coarse_px}px  match={match(anchor)*100:6.2f}%  ({t_anchor:.1f}s)")
            continue

        # --- flatten by anchor, refine residual whole-image, add back ---
        t1 = time.perf_counter()
        flat = wrap(wrapped - anchor).astype(np.float32)
        ig_flat = np.exp(1j * flat).astype(np.complex64)
        resid, _ = ww.unwrap(ig_flat, coh, args.nlooks, mask, tile_size=big, tile_overlap=0)
        final = np.asarray(resid, np.float32) + anchor
        t_refine = time.perf_counter() - t1
        coarse_px = max(cig.shape)
        print(f"  multilook-{L} anchor    : coarse {coarse_px}px match={match(anchor)*100:6.2f}%  "
              f"-> flatten+refine={match(final)*100:6.2f}%  (anchor {t_anchor:.1f}s + refine {t_refine:.1f}s)")


if __name__ == "__main__":
    main()
