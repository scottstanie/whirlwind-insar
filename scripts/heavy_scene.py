#!/usr/bin/env python3
"""Build a large, realistic noisy interferogram and cache it to disk.

Realistic = a diagonal ramp truth (so we *have* ground truth) modulated by a
spatially-varying coherence map: high coherence in most of the scene plus
several large low-coherence patches mimicking vegetated / shadowed regions of
a typical Sentinel-1 IW frame. Multilook count = 4.

We cache to /tmp/heavy_<H>x<W>.npz so the benchmark runner doesn't pay scene
generation cost — and so multiple library timings see the *same* input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]


def coherence_map(shape, *, low=0.30, high=0.90, n_patches=14, seed=0xC0FFEE,
                  flavor="patchy"):
    """Coherence field. `flavor`:

    - "patchy": smooth field of low-γ blobs over a high-γ background. Roughly
      mimics vegetated/water patches in a Sentinel-1 IW frame.
    - "noisy": uniform low γ everywhere — maximally hard for the MCF solver.
    - "uniform-high": uniform γ=high; produces a clean scene with almost no
      residues (sanity scene).
    """
    if flavor == "uniform-high":
        return np.full(shape, float(high), dtype=np.float32)
    if flavor == "noisy":
        return np.full(shape, float(low), dtype=np.float32)
    if flavor != "patchy":
        raise ValueError(f"unknown flavor {flavor!r}")

    rng = np.random.default_rng(seed)
    H, W = shape
    gamma = np.full(shape, float(high), dtype=np.float32)

    # Place random low-coh blobs.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for _ in range(n_patches):
        cy = rng.uniform(0.1 * H, 0.9 * H)
        cx = rng.uniform(0.1 * W, 0.9 * W)
        # Mix of sizes — some large, some small.
        sigma = rng.uniform(0.06, 0.22) * min(H, W)
        # Soft circular falloff (gaussian); blobs add on top by darkening.
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        falloff = np.exp(-d2 / (2 * sigma * sigma)).astype(np.float32)
        # Darken toward `low` proportional to falloff.
        gamma = gamma - (gamma - low) * falloff

    np.clip(gamma, 0.05, 0.995, out=gamma)
    return gamma


def make_scene(H, W, *, nlooks=4, fringe_density=3.0, seed=0xC0FFEE,
               flavor="patchy", low=0.30, high=0.90):
    """Diagonal-ramp truth + Lee-distributed noisy ifg modulated by gamma map."""
    import whirlwind_rs as ww  # for simulate_ifg

    # Diagonal ramp truth: many fringes so even γ=0.9 still leaves a few residues.
    y, x = np.ogrid[-fringe_density:fringe_density:H * 1j, -fringe_density:fringe_density:W * 1j]
    truth = (np.pi * (x + y)).astype(np.float32)

    gamma = coherence_map((H, W), low=low, high=high, seed=seed, flavor=flavor)
    igram, corr = ww.simulate_ifg(truth, gamma, nlooks=nlooks, seed=seed)
    return igram, corr, gamma, truth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--nlooks", type=int, default=4)
    p.add_argument("--fringes", type=float, default=3.0,
                   help="fringe density across the scene (higher -> more residues)")
    p.add_argument("--flavor", choices=("patchy", "noisy", "uniform-high"),
                   default="patchy",
                   help="coherence field shape (default patchy)")
    p.add_argument("--low", type=float, default=0.30,
                   help="γ for the low-coh regions / uniform γ in 'noisy' flavor")
    p.add_argument("--high", type=float, default=0.90,
                   help="γ for the high-coh background")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--out", type=Path,
                   default=Path("/tmp/heavy_scene.npz"))
    p.add_argument("--summary", action="store_true",
                   help="print residue count / coh stats for the cached scene")
    args = p.parse_args()

    H = W = args.size
    print(f"Building {H}x{W} scene "
          f"(flavor={args.flavor}, nlooks={args.nlooks}, fringes={args.fringes})...",
          flush=True)
    igram, corr, gamma, truth = make_scene(
        H, W, nlooks=args.nlooks, fringe_density=args.fringes, seed=args.seed,
        flavor=args.flavor, low=args.low, high=args.high,
    )

    out = args.out
    if out.suffix != ".npz":
        out = out.with_suffix(".npz")
    np.savez(out, igram=igram, corr=corr, gamma=gamma, truth=truth,
             meta=np.array([H, W, args.nlooks, args.fringes, args.seed], dtype=np.float64))
    sz = out.stat().st_size / (1024 * 1024)
    print(f"Saved {out} ({sz:.0f} MiB)")

    if args.summary:
        import whirlwind_rs as ww
        wrapped = np.angle(igram).astype(np.float32)
        residues = ww.compute_residues(wrapped)
        n_res = int(np.count_nonzero(residues))
        print(f"  residues          : {n_res:>10d}  ({100*n_res/igram.size:.2f}% of pixels)")
        print(f"  γ min/median/mean/max : "
              f"{gamma.min():.3f} / {np.median(gamma):.3f} / "
              f"{gamma.mean():.3f} / {gamma.max():.3f}")
        lowmask = gamma < 0.5
        print(f"  fraction γ<0.5    : {100*lowmask.mean():.1f}%")
        print(f"  corr  median/mean : {np.median(corr):.3f} / {corr.mean():.3f}")


if __name__ == "__main__":
    sys.exit(main())
