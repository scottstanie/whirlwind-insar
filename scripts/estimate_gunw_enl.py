#!/usr/bin/env python3
"""Inspect plausible look counts for a NISAR GUNW coherence raster.

The cost model needs an effective number of looks, but neither estimate below
is universal. The unwrap-grid coherence is formed from
`numberOfRangeLooks x numberOfAzimuthLooks` RSLC samples -- isce3 re-runs
`crossmul` from the RSLCs at those looks rather than multilooking the already
multilooked RIFG -- so the sample count is their **product** (for example,
13 x 16 = 208), not
either axis alone, and not the two multilooks composed.

Two independent estimates are printed:

1. **Nominal metadata estimate.** samples / (range oversample x azimuth
   oversample), where
   range oversample = (c / 2B) / slantRangeSpacing and azimuth oversample =
   PRF / processedAzimuthBandwidth. Multilooking correlated samples buys fewer
   than `samples` independent looks. This remains an upper-bound model, not the
   value production necessarily passes to SNAPHU.

2. **Zero-coherence model fit over water.** For true coherence 0 and L looks the
   sample coherence has
   ``p(g) = 2(L-1) g (1-g^2)^(L-2)``, whose mode is ``1/sqrt(2L-3)``, so
   ``L = (mode^-2 + 3)/2``. The mode is used rather than the mean because
   shoreline and partial-water pixels contaminate the upper tail and would bias
   a mean estimate downward in L. Water is not guaranteed to have zero true
   coherence, so this is a diagnostic fit rather than a ground-truth ENL.

Takes either a GUNW .h5 (both estimates) or a compare_gunw `_arrays.npz`
(water fit only). New NPZ files store the actual GUNW water and subswath flags;
older files without them must be regenerated or inspected through the HDF5.
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

    print("nominal metadata estimate")
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
    # Take the interior of the largest water blob: shoreline and partial-water
    # pixels are especially unlikely to satisfy the zero-coherence model.
    lab, _ = ndimage.label(water)
    if lab.max() == 0:
        print(f"water fit ({label}): no water region found")
        return
    biggest = 1 + np.argmax(np.bincount(lab.ravel())[1:])
    core = ndimage.binary_erosion(lab == biggest, iterations=3)
    g = coh[core]
    g = g[np.isfinite(g) & (g > 0)]
    if g.size < 5000:
        print(f"water fit ({label}): only {g.size} core px, too few")
        return

    hist, edges = np.histogram(g, bins=np.linspace(0, 0.6, 121))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = centers[np.argmax(hist)]
    L_mode = (mode**-2 + 3) / 2
    L_mean = np.pi / (4 * g.mean() ** 2)

    print(f"\nzero-coherence water fit ({label}, {g.size:,} core px)")
    print(f"  mode  {mode:.3f} -> L = (mode^-2 + 3)/2 = {L_mode:.0f}")
    print(
        f"  mean  {g.mean():.3f} -> L = pi/(4 mean^2)  = {L_mean:.0f}  (contamination-biased, a floor)"
    )
    print(
        "  caveat: this estimate is valid only insofar as the water's true coherence is zero"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="GUNW .h5 or compare_gunw _arrays.npz")
    args = p.parse_args()

    if args.path.suffix == ".npz":
        with np.load(args.path) as f:
            required = {"coh", "water", "subswath_valid"}
            missing = required.difference(f.files)
            if missing:
                raise SystemExit(
                    f"{args.path} lacks {sorted(missing)}; regenerate it with the "
                    "current compare_gunw.py or pass the source GUNW HDF5"
                )
            coh = f["coh"]
            water = f["water"] & f["subswath_valid"]
        empirical(coh, water, "GUNW water flag")
        return

    import h5py

    theoretical(args.path)
    with h5py.File(args.path, "r") as f:
        pol = next(k for k in f[B] if k in ("HH", "VV", "HV", "VH"))
        coh = f[f"{B}/{pol}/coherenceMagnitude"][()]
        low = f[f"{B}/mask"][()].astype(np.int64) & 0xFF
    valid_subswath = (((low // 10) % 10) > 0) & ((low % 10) > 0)
    water = (((low // 100) % 10) != 0) & valid_subswath & np.isfinite(coh)
    empirical(coh, water, "GUNW water flag")


if __name__ == "__main__":
    main()
