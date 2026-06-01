"""Cost-model A/B on a real NISAR GUNW: linear vs reuse vs convex.

This is the real-scene validation flagged in ``paper/pyramid_aliasing.md``: does
the better default cost (reuse / convex) keep its synthetic advantage — fixing
the linear cost's corner/boundary mis-routing without a speed penalty — on a
real interferogram?

It can't run in the sandbox (Earthdata/ASF are network-blocked and no creds),
so download a GUNW yourself, then:

    uv run python scripts/cost_model_real_ab.py --local-h5 NISAR_..._001.h5 \
        --nlooks 16 --crop 2048

What it does
------------
Reads the production 80 m unwrapped phase, re-wraps it to [-pi, pi) (the same
apples-to-apples convention as ``bench_nisar_gunw.py``), reads coherence + mask,
and runs ``unwrap`` (linear), ``unwrap_reuse``, ``unwrap_convex`` on the SAME
input. Since the production unwrap is the reference, we report per-method
K-agreement with it (fraction of pixels on the same integer 2pi cycle, after a
global offset align) and wall time. The linear cost's corner failure should show
up as lower K-agreement concentrated at steep/boundary regions; reuse/convex
should match production more closely. Treat production as "truth" with the usual
caveat that it is itself an unwrapper (SNAPHU/PHASS-family), not ground truth.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import whirlwind as ww

TWOPI = 2.0 * np.pi


def wrap_phase(x):
    return (x + np.pi) % TWOPI - np.pi


def k_agree(unw, ref, valid):
    d = (unw - ref)[valid]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    d = d - TWOPI * round(float(np.median(d)) / TWOPI)
    return float(np.mean(np.round(d / TWOPI) == 0))


def load_gunw(h5, pol):
    import h5py

    unw_base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
    with h5py.File(h5, "r") as f:
        prod = f[f"{unw_base}/{pol}/unwrappedPhase"][:].astype(np.float32)
        coh = f[f"{unw_base}/{pol}/coherenceMagnitude"][:].astype(np.float32)
        try:
            mask = f[f"{unw_base}/mask"][:].astype(bool)
        except KeyError:
            mask = np.isfinite(prod) & (coh > 0)
    return prod, coh, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-h5", type=Path, required=True)
    ap.add_argument("--pol", default="HH")
    ap.add_argument("--nlooks", type=float, default=16.0)
    ap.add_argument("--crop", type=int, default=0, help="Center-crop to crop x crop (0 = full).")
    args = ap.parse_args()

    prod, coh, mask = load_gunw(args.local_h5, args.pol)
    if args.crop and min(prod.shape) > args.crop:
        m, n = prod.shape
        r0, c0 = (m - args.crop) // 2, (n - args.crop) // 2
        sl = (slice(r0, r0 + args.crop), slice(c0, c0 + args.crop))
        prod, coh, mask = prod[sl], coh[sl], mask[sl]

    prod = np.where(np.isfinite(prod), prod, 0.0).astype(np.float32)
    coh = np.where(np.isfinite(coh), coh, 0.0).astype(np.float32)
    ig = np.exp(1j * wrap_phase(prod)).astype(np.complex64)
    ig[~mask] = 0
    valid = mask & (coh > 0)
    print(f"scene {prod.shape}  valid={valid.mean() * 100:.0f}%  nlooks={args.nlooks}")

    ww.set_num_threads(1)
    solvers = [
        ("linear", lambda: ww.unwrap(ig, coh, nlooks=args.nlooks, mask=mask)[0]),
        ("reuse", lambda: ww.unwrap_reuse(ig, coh, nlooks=args.nlooks, mask=mask)),
        ("convex", lambda: ww.unwrap_convex(ig, coh, nlooks=args.nlooks, mask=mask)),
    ]
    base = None
    for name, fn in solvers:
        t = time.perf_counter()
        unw = fn()
        dt = time.perf_counter() - t
        ka = k_agree(unw, prod, valid)
        base = base or dt
        print(f"  {name:7s} K-agree-vs-production={ka * 100:5.1f}%  {dt * 1000:8.1f}ms ({dt / base:.2f}x)")


if __name__ == "__main__":
    main()
