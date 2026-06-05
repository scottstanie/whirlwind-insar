#!/usr/bin/env python3
"""Uniform unwrapper comparison harness via tophu: run snaphu / phass / icu
(single-tile, the isce3/NISAR algorithms behind a common API) on NISAR GUNW
frames and score per-connected-component ambiguity match vs the production GUNW
unwrap + runtime + conncomp count. Complements scripts/snaphu_nisar_compare.py
(snaphu-py direct) and the whirlwind bench - together they give a 4-way
whirlwind-vs-snaphu-vs-phass-vs-icu comparison.

Run in an env that has tophu + isce3 (NOT the whirlwind .venv):
    ~/miniforge3/envs/mapping-312/bin/python scripts/tophu_compare.py \
        --local-h5 <WD>/nisar_gunw/*D_077*.h5 --nlooks 16 --unwrappers snaphu phass

tophu callbacks: SnaphuUnwrap(cost='smooth', init_method='mcf'),
PhassUnwrap(good_coherence=0.7, min_region_size=200), ICUUnwrap(...). Each is
called cb(igram, coherence, nlooks, scratchdir) -> (unwrapped, conncomp); there
is no mask arg, so masked (water) pixels are passed with coherence = 0.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

TWOPI = 2.0 * np.pi


def wrap_phase(x):
    return (x + np.pi) % TWOPI - np.pi


def gunw_layers(h5, pol=None):
    base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
    grp = h5[base]
    pols = [
        k
        for k, v in grp.items()
        if isinstance(v, h5py.Group) and k.upper() not in ("MASK", "METADATA")
    ]
    p = pol or sorted(pols)[0]
    prod_unw = h5[f"{base}/{p}/unwrappedPhase"][()].astype(np.float32)
    coh = h5[f"{base}/{p}/coherenceMagnitude"][()].astype(np.float32)
    cc = h5[f"{base}/{p}/connectedComponents"][()].astype(np.int64)
    mask_arr = h5[f"{base}/mask"][()] if "mask" in grp else None
    return p, prod_unw, coh, cc, mask_arr


def water_only_mask(mask_arr, shape):
    if mask_arr is None:
        return np.ones(shape, bool)
    return (mask_arr != 255) & ((mask_arr // 100) % 10 == 0)


def percomp_match(test_unw, prod_unw, wrapped, prod_cc, valid):
    amb = np.rint((test_unw - wrapped) / TWOPI) - np.rint((prod_unw - wrapped) / TWOPI)
    in_comp = valid & (prod_cc > 0)
    if not in_comp.any():
        return float("nan")
    off = np.zeros(amb.shape, np.float64)
    for lab in np.unique(prod_cc[in_comp]):
        m = valid & (prod_cc == lab)
        off[m] = np.rint(np.median(amb[m]))
    return float(np.mean((amb - off)[in_comp] == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-h5", nargs="+", type=Path, required=True)
    ap.add_argument("--nlooks", type=float, default=16.0)
    ap.add_argument(
        "--unwrappers",
        nargs="+",
        default=["snaphu", "phass", "icu"],
        choices=["snaphu", "phass", "icu"],
    )
    ap.add_argument(
        "--save-dir",
        default=None,
        help="save each unwrapper's unw+cc (+prod/wrapped/mask) as npz for plotting",
    )
    args = ap.parse_args()

    import tophu

    def make(name):
        if name == "snaphu":
            return tophu.SnaphuUnwrap(cost="smooth", init_method="mcf")
        if name == "phass":
            return tophu.PhassUnwrap(good_coherence=0.7, min_region_size=200)
        if name == "icu":
            return tophu.ICUUnwrap()
        raise ValueError(name)

    for path in args.local_h5:
        with h5py.File(path, "r") as h:
            pol, prod_unw, coh, prod_cc, mask_arr = gunw_layers(h)
        mask = (
            water_only_mask(mask_arr, prod_unw.shape)
            & np.isfinite(prod_unw)
            & np.isfinite(coh)
        )
        wrapped = wrap_phase(prod_unw).astype(np.float32)
        ig = np.exp(1j * wrapped).astype(np.complex64)
        # tophu callbacks have no mask arg: zero coherence where masked (water/invalid).
        coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(
            np.float32
        )
        frame = path.name.split("_004_4000")[0].split("GUNW_")[-1]
        for name in args.unwrappers:
            cb = make(name)
            with tempfile.TemporaryDirectory() as sd:
                t0 = time.perf_counter()
                unw, cc = cb(ig, coh_in, args.nlooks, Path(sd))
                dt = time.perf_counter() - t0
            unw = np.asarray(unw, np.float32)
            cc = np.asarray(cc).astype(np.int32)
            valid = mask & np.isfinite(unw)
            pc = percomp_match(unw, prod_unw, wrapped, prod_cc, valid)
            print(
                f"{frame}: {name:7s} {dt:6.1f}s  per-comp-match-vs-prod={pc * 100:5.1f}%  "
                f"ncc={int(cc.max())}  shape={ig.shape}",
                flush=True,
            )
            if args.save_dir:
                sd = Path(args.save_dir)
                sd.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    sd / f"{frame}_{name}.npz",
                    unw=unw,
                    cc=cc,
                    prod_unw=prod_unw.astype(np.float32),
                    prod_cc=prod_cc.astype(np.int32),
                    wrapped=wrapped,
                    mask=mask,
                    coh=np.nan_to_num(coh).astype(np.float32),
                )


if __name__ == "__main__":
    main()
