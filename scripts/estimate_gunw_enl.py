#!/usr/bin/env python3
"""Estimate the equivalent number of looks (ENL) of a NISAR GUNW's coherence.

This is the number to pass as `nlooks` to the cost model. It is easy to get
wrong: the unwrap-grid coherence is estimated over
`numberOfRangeLooks x numberOfAzimuthLooks` RSLC samples -- isce3 re-runs
`crossmul` from the RSLCs at those looks rather than multilooking the already
5x6 RIFG -- so the sample count is their **product** (13 x 16 = 208), not
either axis alone, and not the two multilooks composed.

Two independent estimates are printed:

1. **Theoretical.** samples / (range oversample x azimuth oversample), where
   range oversample = (c / 2B) / slantRangeSpacing and azimuth oversample =
   PRF / processedAzimuthBandwidth. Multilooking correlated samples buys fewer
   than `samples` independent looks.

2. **Empirical, from the coherence histogram over decorrelated water.** For true
   coherence 0 and L looks the sample coherence has
   ``p(g) = 2(L-1) g (1-g^2)^(L-2)``, whose mode is ``1/sqrt(2L-3)``, so
   ``L = (mode^-2 + 3)/2``. The mode is used rather than the mean because
   shoreline and partial-water pixels contaminate the upper tail and would bias
   a mean estimate downward in L.

Takes either a GUNW .h5 (both estimates) or a compare_gunw `_arrays.npz`
(empirical only, using the production-dropped region as the water proxy).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

C = 299_792_458.0
B = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
MP = "/science/LSAR/GUNW/metadata/processingInformation/parameters"


def theoretical(path: Path) -> None:
    import h5py

    with h5py.File(path, "r") as f:
        u = f[f"{MP}/unwrappedInterferogram/frequencyA"]
        rg_looks = int(u["numberOfRangeLooks"][()])
        az_looks = int(u["numberOfAzimuthLooks"][()])
        bw = float(u["rangeBandwidth"][()])
        az_bw = float(u["azimuthBandwidth"][()])
        ref = f[f"{MP}/reference/frequencyA"]
        dr = float(ref["slantRangeSpacing"][()])
        dt = float(ref["zeroDopplerTimeSpacing"][()])

    samples = rg_looks * az_looks
    rg_res = C / (2.0 * bw)
    rg_over = rg_res / dr
    prf = 1.0 / dt
    az_over = prf / az_bw
    enl = samples / (rg_over * az_over)

    print("theoretical")
    print(
        f"  looks (from product metadata)  {rg_looks} rg x {az_looks} az = {samples} samples"
    )
    print(
        f"  range   res {rg_res:.2f} m / spacing {dr:.2f} m -> oversample {rg_over:.3f}"
    )
    print(
        f"  azimuth PRF {prf:.1f} Hz / bandwidth {az_bw:.1f} Hz -> oversample {az_over:.3f}"
    )
    print(f"  ENL = {samples} / {rg_over * az_over:.3f} = {enl:.0f}")


def empirical(coh: np.ndarray, water: np.ndarray, label: str) -> None:
    # Take the interior of the largest decorrelated blob: shoreline and
    # partial-water pixels have real coherence and would skew the fit.
    lab, _ = ndimage.label(water)
    if lab.max() == 0:
        print(f"empirical ({label}): no decorrelated region found")
        return
    biggest = 1 + np.argmax(np.bincount(lab.ravel())[1:])
    core = ndimage.binary_erosion(lab == biggest, iterations=3)
    g = coh[core]
    g = g[np.isfinite(g) & (g > 0)]
    if g.size < 5000:
        print(f"empirical ({label}): only {g.size} core px, too few")
        return

    hist, edges = np.histogram(g, bins=np.linspace(0, 0.6, 121))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = centers[np.argmax(hist)]
    L_mode = (mode**-2 + 3) / 2
    L_mean = np.pi / (4 * g.mean() ** 2)

    print(f"\nempirical ({label}, {g.size:,} core px)")
    print(f"  mode  {mode:.3f} -> L = (mode^-2 + 3)/2 = {L_mode:.0f}")
    print(
        f"  mean  {g.mean():.3f} -> L = pi/(4 mean^2)  = {L_mean:.0f}  (contamination-biased, a floor)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="GUNW .h5 or compare_gunw _arrays.npz")
    args = p.parse_args()

    if args.path.suffix == ".npz":
        f = np.load(args.path)
        coh, mask, prod_cc = f["coh"], f["mask"], f["prod_cc"]
        empirical(
            coh, mask & (prod_cc == 0) & (coh < 0.25), "production-dropped region"
        )
        return

    import h5py

    theoretical(args.path)
    with h5py.File(args.path, "r") as f:
        pol = next(k for k in f[B] if k in ("HH", "VV", "HV", "VH"))
        coh = f[f"{B}/{pol}/coherenceMagnitude"][()]
        low = f[f"{B}/mask"][()].astype(np.int64) & 0xFF
    water = (((low // 100) % 10) != 0) & np.isfinite(coh)
    if water.sum() > 5000:
        empirical(coh, water, "water-flagged pixels")
    else:
        empirical(coh, np.isfinite(coh) & (coh < 0.2), "lowest-coherence region")


if __name__ == "__main__":
    main()
