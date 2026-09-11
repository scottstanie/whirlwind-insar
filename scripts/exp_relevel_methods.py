#!/usr/bin/env python3
"""Compare subswath re-leveling rules on cached ``compare_gunw`` output.

On NISAR fixed-PRF frames the mask splits the scene into subswath stripes and
every observed error is one integer cycle offset per stripe, so the leveling
rule is the whole ballgame. Four rules are scored here:

``before``
    whatever the run used -- whirlwind's minimum-jump ``bridge``, which is right
    only when the true cross-gap phase change is under half a cycle.
``extrap``
    extrapolate the phase gradient from both sides of each gap (median over the
    thousands of rows crossing it) and round the mismatch to an integer. Uses a
    64 px window at each edge, so it is a purely *local* predictor.
``surface``
    fit a smooth low-order 2-D polynomial -- the bulk ionospheric field -- to the
    whole frame, alternating with per-stripe integer offsets chosen to minimise
    each stripe's median residual. Every stripe votes on every other through the
    shared surface, so a gap decision uses the whole frame rather than its edges.
``oracle``
    the best any pure re-leveling can do: per stripe, the integer nearest to
    production. Not achievable without truth -- it is the ceiling, and the gap
    between ``oracle`` and 1.0 is error *inside* stripes that no leveling fixes.

Usage::

    python scripts/exp_relevel_methods.py gunw_results/ww-gunw-A_140_010_7700 ww_gunw_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from exp_gap_relevel import _load_compare_gunw, gap_offsets, relevel, short

TWOPI = 2.0 * np.pi


def _poly_design(yy: np.ndarray, xx: np.ndarray, order: int) -> np.ndarray:
    """2-D polynomial terms up to total degree ``order``, on normalised coords."""
    return np.stack(
        [
            yy**p * xx**q
            for d in range(order + 1)
            for p, q in ((d - k, k) for k in range(d + 1))
        ],
        axis=1,
    )


def surface_offsets(
    unw: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    order: int = 3,
    iters: int = 12,
    stride: int = 6,
) -> dict[int, int]:
    """Integer per-stripe offsets that best fit one smooth surface over the frame.

    Alternates a least-squares surface fit against a robust per-stripe integer
    update. A stripe offset is a frame-spanning bar that a low-order polynomial
    cannot mimic, so the surface stays a model of the bulk field rather than
    absorbing the very offsets being solved for.
    """
    m, n = mask.shape
    sub = (slice(None, None, stride), slice(None, None, stride))
    msub, lsub, usub = mask[sub], labels[sub], unw[sub]
    sel = msub & (lsub > 0) & np.isfinite(usub)
    ii, jj = np.mgrid[0:m:stride, 0:n:stride]
    yy = (ii[sel] / m * 2.0 - 1.0).astype(np.float64)
    xx = (jj[sel] / n * 2.0 - 1.0).astype(np.float64)
    phase = usub[sel].astype(np.float64)
    lab = lsub[sel]
    uniq = np.unique(lab)
    A = _poly_design(yy, xx, order)

    k = dict.fromkeys((int(v) for v in uniq), 0)
    for _ in range(iters):
        y = phase + TWOPI * np.array([k[int(v)] for v in lab])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        changed = False
        for v in uniq:
            dk = -int(round(float(np.median(resid[lab == v])) / TWOPI))
            if dk:
                k[int(v)] += dk
                changed = True
        if not changed:
            break
    return {v: kk for v, kk in k.items() if kk}


def apply_offsets(
    unw: np.ndarray, labels: np.ndarray, offs: dict[int, int]
) -> np.ndarray:
    out = unw.copy()
    for lab, kk in offs.items():
        if kk:
            out[labels == lab] += TWOPI * kk
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--min-area-frac", type=float, default=0.002)
    args = p.parse_args()

    cg = _load_compare_gunw()
    rows = []
    hdr = f"{'frame':<16} {'before':>8} {'extrap':>8} {'surface':>8} {'oracle':>8}   surface offsets"
    print(hdr)
    print("-" * len(hdr))
    for d in args.dirs:
        for sub in sorted(d.iterdir()):
            if not (sub / "full_arrays.npz").exists():
                continue
            z = np.load(sub / "full_arrays.npz")
            mask, coh = z["mask"], z["coh"]
            unw = np.where(mask, z["ww_aligned"], np.nan)
            labels, _ = ndimage.label(mask)
            sizes = np.bincount(labels.ravel())
            labels[
                np.isin(labels, np.flatnonzero(sizes < args.min_area_frac * mask.size))
            ] = 0

            def score(u):
                st, *_ = cg.compute_compare_stats(
                    ig=z["ig"],
                    coh=coh,
                    mask=mask,
                    prod_unw=z["prod_unw"],
                    prod_cc=z["prod_cc"],
                    ww_unw=np.where(mask, u, 0.0).astype(np.float32),
                    ww_cc=z["ww_cc"] if z["ww_cc"].size else None,
                    runtime_s=0.0,
                    rss_delta_mb=None,
                )
                return st["ambiguity_match_frac_percomp"]

            before = json.loads((sub / "full.json").read_text())[
                "ambiguity_match_frac_percomp"
            ]
            ex, _ = relevel(unw, labels, gap_offsets(unw, mask, coh, labels))
            so = surface_offsets(unw, mask, labels, order=args.order)
            surf = apply_offsets(unw, labels, so)
            orc = unw.copy()
            for L in (v for v in np.unique(labels) if v):
                sel = labels == L
                orc[sel] += TWOPI * int(
                    round(float(np.median((z["prod_unw"][sel] - unw[sel]) / TWOPI)))
                )
            r = (short(sub.name), before, score(ex), score(surf), score(orc))
            rows.append(r)
            print(
                f"{r[0]:<16} {r[1]:>8.4f} {r[2]:>8.4f} {r[3]:>8.4f} {r[4]:>8.4f}   {so}"
            )

    a = np.array([r[1:] for r in rows])
    print("-" * len(hdr))
    print(
        f"{'mean':<16} {a[:, 0].mean():>8.4f} {a[:, 1].mean():>8.4f} "
        f"{a[:, 2].mean():>8.4f} {a[:, 3].mean():>8.4f}"
    )
    for j, nm in ((1, "extrap"), (2, "surface")):
        d = a[:, j] - a[:, 0]
        print(
            f"  {nm:<8} better {int((d > 1e-3).sum())}/{len(d)}  "
            f"worse {int((d < -1e-3).sum())}/{len(d)}  "
            f"captures {(a[:, j] - a[:, 0]).sum() / max((a[:, 3] - a[:, 0]).sum(), 1e-9):.0%} of the oracle gain"
        )


if __name__ == "__main__":
    main()
