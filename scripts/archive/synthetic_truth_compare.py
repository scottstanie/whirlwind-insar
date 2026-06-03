#!/usr/bin/env python3
"""Circularity-free decisive test: a noisy steep ramp with KNOWN truth, run
through snaphu (cost=smooth,init=mcf), ww (solver per WHIRLWIND_TILE_SOLVER env,
both whole-image and tiled), and Itoh (flow=0). Compare each to truth. Because
the truth is known (not production=snaphu), this isolates whether snaphu really
recovers the ramp better than ww on identical data, separate from the
"production was made by snaphu" self-match in the GUNW bench.

Run TWICE (convex then reuse), the synthetic is seeded so data is identical:
    WHIRLWIND_TILE_SOLVER=convex env -u CONDA_PREFIX uv run --no-sync \
        python scripts/synthetic_truth_compare.py --tag convex
    WHIRLWIND_TILE_SOLVER=reuse  env -u CONDA_PREFIX uv run --no-sync \
        python scripts/synthetic_truth_compare.py --tag reuse
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

TWOPI = 2.0 * np.pi
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/synthetic_truth")


def simulate(truth, gamma, nlooks, seed):
    """Standard correlated circular-Gaussian InSAR sim, L=nlooks looks.
    Returns (ifg complex64, coherence float32) multilooked per pixel."""
    rng = np.random.default_rng(seed)
    m, n = truth.shape
    L = int(nlooks)
    g = gamma[..., None]  # (m,n,1)
    ph = truth[..., None]
    # a, b iid CN(0,1); master = a, slave = exp(i ph)(g a + sqrt(1-g^2) b)
    a = (rng.standard_normal((m, n, L)) + 1j * rng.standard_normal((m, n, L))) / np.sqrt(2)
    b = (rng.standard_normal((m, n, L)) + 1j * rng.standard_normal((m, n, L))) / np.sqrt(2)
    slave = np.exp(1j * ph) * (g * a + np.sqrt(np.clip(1 - g**2, 0, 1)) * b)
    master = a
    cross = slave * np.conj(master)            # phase ~ +truth
    ifg = cross.mean(axis=2)
    p1 = (np.abs(master) ** 2).mean(axis=2)
    p2 = (np.abs(slave) ** 2).mean(axis=2)
    coh = np.abs(ifg) / np.sqrt(p1 * p2 + 1e-12)
    ifg = (ifg / (np.abs(ifg) + 1e-12)).astype(np.complex64)  # unit-magnitude
    return ifg, np.clip(coh, 0, 1).astype(np.float32)


def quality(unw, truth, valid):
    """frac within 0.1 rad of truth after removing global offset; cycle-RMSE."""
    d = (unw - truth)[valid]
    d = d - np.median(d)
    frac = float(np.mean(np.abs(d) < 0.1))
    rms = float(np.sqrt(np.mean(d**2)))
    # cycle accuracy: fraction at the dominant integer-cycle level
    cyc = np.rint(d / TWOPI)
    vals, cnts = np.unique(cyc, return_counts=True)
    cyc_acc = float(cnts.max() / cyc.size)
    return frac, rms, cyc_acc


def itoh(wrapped):
    """flow=0 integration: cumulative sum of wrapped gradients from (0,0)."""
    dx = np.angle(np.exp(1j * (wrapped[:, 1:] - wrapped[:, :-1])))
    dy = np.angle(np.exp(1j * (wrapped[1:, :] - wrapped[:-1, :])))
    out = np.zeros_like(wrapped, dtype=np.float64)
    out[0, 1:] = np.cumsum(dx[0, :])
    out[1:, 0] = wrapped[0, 0] + np.cumsum(dy[:, 0])
    for i in range(1, out.shape[0]):
        out[i, 1:] = out[i, 0] + np.cumsum(dx[i, :])
    return (out + wrapped[0, 0]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="convex|reuse (sets which ww solver, via env)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nlooks", type=float, default=16.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    solver = os.environ.get("WHIRLWIND_TILE_SOLVER", "reuse")

    import whirlwind as ww
    do_snaphu = args.tag == "convex"  # only need snaphu/itoh once
    if do_snaphu:
        import snaphu

    N = args.size
    # Cases: vary coherence (uniform) + a low-coh stripe ("river") variant.
    cases = [
        # Stress: large frame, low coherence, steep ramps (many residues), to try
        # to reproduce the D_077 whole-image runaway and see if snaphu diverges.
        ("hard_g40_20cyc", 0.40, 20.0, None),
        ("hard_g30_30cyc", 0.30, 30.0, None),
        ("hard_g45_20cyc_river", 0.45, 20.0, "river"),
        ("vhard_g25_40cyc", 0.25, 40.0, None),
    ]
    rows = []
    for name, g0, cycles, special in cases:
        truth = (TWOPI * cycles * (np.add.outer(np.arange(N), np.arange(N)) / N)).astype(np.float32)
        gamma = np.full((N, N), g0, np.float32)
        if special == "river":
            gamma[N // 2 - 3 : N // 2 + 3, :] = 0.05  # near-zero coh horizontal stripe
        ifg, coh = simulate(truth, gamma, args.nlooks, seed=12345)
        wrapped = np.angle(ifg).astype(np.float32)
        valid = np.ones((N, N), bool)
        mask = valid.copy()

        res = {"case": name, "g0": g0, "cycles": cycles, "special": special or ""}

        # ww (solver per env): whole-image AND tiled-default
        big = N + 100
        t0 = time.perf_counter()
        u_whole, _ = ww.unwrap(ifg, coh, args.nlooks, mask, tile_size=big, tile_overlap=0)
        res[f"ww_{solver}_whole"] = quality(np.asarray(u_whole, np.float32), truth, valid)
        res[f"ww_{solver}_whole_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        u_tiled, _ = ww.unwrap(ifg, coh, args.nlooks, mask)  # default tiling
        res[f"ww_{solver}_tiled"] = quality(np.asarray(u_tiled, np.float32), truth, valid)
        res[f"ww_{solver}_tiled_s"] = time.perf_counter() - t0

        if do_snaphu:
            t0 = time.perf_counter()
            u_sn, _ = snaphu.unwrap(ifg, coh, args.nlooks, cost="smooth", init="mcf", mask=mask)
            res["snaphu"] = quality(np.asarray(u_sn, np.float32), truth, valid)
            res["snaphu_s"] = time.perf_counter() - t0
            res["itoh"] = quality(itoh(wrapped), truth, valid)

        rows.append(res)
        print(f"[{name}] " + "  ".join(
            f"{k}=frac{v[0]:.3f}/cyc{v[2]:.3f}" for k, v in res.items()
            if isinstance(v, tuple)), flush=True)

    np.save(OUT / f"results_{args.tag}.npy", np.array(rows, dtype=object), allow_pickle=True)
    print(f"saved -> {OUT}/results_{args.tag}.npy", flush=True)


if __name__ == "__main__":
    main()
