"""Sparse / irregular-grid phase unwrapping demo on a NISAR GSLC pair.

Demonstrates ``whirlwind.unwrap_sparse`` on a real NISAR-style scene.
Pipeline:

  1. Load wrapped phase + coherence (produced upstream from the GSLC pair —
     see ``GSLC granules`` below).
  2. Pick the high-coherence subset (γ > ``--gamma-min``, default 0.5) — the
     "spurt-style" workflow of unwrapping <10 % of pixels and skipping the
     rest as too noisy to trust.
  3. Subsample (random, fixed seed) to ``--n-points`` pixels for a fast demo.
  4. Convert per-pixel coherence to per-pixel CRLB variance:
     ``σ²(γ, L) = (1 − γ²) / (2·L·γ²)``  (Lee 1994 multilook approx, valid
     at moderate-to-high γ; degrades near γ → 0 which we mask out).
  5. Build a Delaunay triangulation, compute residues and per-edge costs,
     run MCF, integrate. ``max_edge_length`` is auto-tuned to 3x the median
     nearest-neighbour distance — long convex-hull spans are carved out as
     outer-face boundary edges and integration skips them.
  6. Plot wrapped, sampled subset, unwrapped result. Save NPZ for reuse.

GSLC granules (input to interferogram processing)
-------------------------------------------------

These two NISAR L2 GSLC granules (HH polarisation, 50 m posting) cover a
patch over the Long Beach / Catalina / San Clemente region. They are
distributed via the ASF DAAC NISAR pre-launch / cal-val collection — search
by granule name at ``https://search.asf.alaska.edu/`` once the collection
is open, or obtain via the NISAR Sample Product Suite from JPL.

  NISAR_L2_PR_GSLC_008_114_D_071_4005_DHDH_A_20251224T024856_20251224T024931_X05009_N_F_J_001.h5
  NISAR_L2_PR_GSLC_010_114_D_071_4005_DHDH_A_20260117T024857_20260117T024932_X05010_N_F_J_001.h5

To produce the wrapped-phase / coherence rasters this script consumes, see
the companion ``nisar_gslc_interferogram.py`` (forms the boxcar-multilooked
complex IG and the matched coherence at 10x10 looks → 100-look effective).

Then derive a single-band wrapped-phase TIFF via GDAL:

    gdal_translate DERIVED_SUBDATASET:PHASE:<ig>.int.looked.tif <wrapped>.tif

and (if not already done) clean any out-of-range coherence sentinels:

    gdal_calc.py -A <coh>.tif --calc='where((A>=0)&(A<=1),A,0)' \\
        --outfile <coh>.cleaned.tif

How to run
----------

::

    uv run --with rasterio --with matplotlib --with scipy \\
        python scripts/nisar_sparse_demo.py \\
        --wrapped /path/to/wrapped.tif \\
        --coh     /path/to/coh.cleaned.tif \\
        --out     /tmp/nisar-sparse

(Or activate the project venv and run with ``python`` directly.)

Outputs:
  - ``out/sparse.png``         — 2x2 overview figure
  - ``out/sparse_zoom.png``    — top-right zoom on the mountain region
  - ``out/sparse.npz``         — points, wrapped, unwrapped, gamma, mask
  - ``out/timings.txt``        — wall-clock and basic stats
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.spatial import cKDTree

import whirlwind as ww


def coh_to_variance(gamma: np.ndarray, nlooks: float) -> np.ndarray:
    """Lee 1994 multilook phase variance approximation: σ² = (1−γ²)/(2·L·γ²).

    Valid for moderate-to-high γ; diverges at γ → 0, which is why we mask
    out anything below ``--gamma-min`` before computing it.
    """
    g2 = gamma.astype(np.float32) ** 2
    return ((1.0 - g2) / (2.0 * nlooks * g2)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--wrapped",
        type=Path,
        required=True,
        help="wrapped-phase TIFF (float32, radians)",
    )
    ap.add_argument(
        "--coh", type=Path, required=True, help="coherence TIFF (float32, [0, 1])"
    )
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument(
        "--gamma-min",
        type=float,
        default=0.5,
        help="keep pixels with γ > this (default: 0.5)",
    )
    ap.add_argument(
        "--nlooks",
        type=float,
        default=100.0,
        help="effective number of looks for variance estimate (default: 100)",
    )
    ap.add_argument(
        "--n-points",
        type=int,
        default=100_000,
        help="target number of subsampled points (default: 100k)",
    )
    ap.add_argument(
        "--seed", type=int, default=7, help="random seed for subsampling (default: 7)"
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading wrapped + coherence")
    with rasterio.open(args.wrapped) as src:
        wrapped_full = src.read(1).astype(np.float32)
    with rasterio.open(args.coh) as src:
        coh_full = src.read(1).astype(np.float32)
    assert wrapped_full.shape == coh_full.shape, (wrapped_full.shape, coh_full.shape)

    # Validity mask: finite coherence in (γ_min, 1].
    mask = (
        np.isfinite(coh_full)
        & (coh_full > args.gamma_min)
        & (coh_full <= 1.0)
        & np.isfinite(wrapped_full)
    )
    n_valid = int(mask.sum())
    print(
        f"      γ > {args.gamma_min}: {n_valid:,} of {coh_full.size:,} pixels "
        f"({100 * n_valid / coh_full.size:.2f}%)"
    )
    if n_valid < 1000:
        raise SystemExit(f"only {n_valid} valid pixels — γ-min too high?")

    print(f"[2/4] subsampling to {args.n_points} points (seed={args.seed})")
    rng = np.random.default_rng(args.seed)
    idx_all = np.flatnonzero(mask.ravel())
    if len(idx_all) > args.n_points:
        pick = rng.choice(idx_all, size=args.n_points, replace=False)
    else:
        pick = idx_all
    rows, cols = np.unravel_index(pick, mask.shape)
    points = np.column_stack([rows.astype(np.float64), cols.astype(np.float64)])
    gamma = coh_full.ravel()[pick]
    variance = coh_to_variance(gamma, args.nlooks)
    wrapped = wrapped_full.ravel()[pick].astype(np.float32)

    tree = cKDTree(points)
    nn, _ = tree.query(points, k=2)
    nn = nn[:, 1]
    median_nn = float(np.median(nn))
    p90_nn = float(np.percentile(nn, 90))
    max_edge = max(3.0 * median_nn, 50.0)
    print(f"      median NN dist: {median_nn:.2f} px (90th pct: {p90_nn:.2f})")
    print(f"      max_edge_length: {max_edge:.1f} px")

    print(f"[3/4] unwrap_sparse")
    t0 = time.perf_counter()
    unw = ww.unwrap_sparse(points, wrapped, variance, max_edge_length=max_edge)
    elapsed = time.perf_counter() - t0
    nan_frac = float(np.isnan(unw).mean())
    print(f"      elapsed: {elapsed:.2f}s  output NaN frac: {100 * nan_frac:.2f}%")

    # Congruence: wrap(unw - wrapped) should be ≈ 0 at all finite output pixels.
    tau = 2.0 * np.pi
    finite = np.isfinite(unw)
    if finite.any():
        d = unw[finite] - wrapped[finite]
        r = d - tau * np.round(d / tau)
        congruence_max = float(np.max(np.abs(r)))
        print(f"      max |wrap(unw - wrapped)|: {congruence_max:.2e} rad")
    else:
        congruence_max = float("nan")

    print(f"[4/4] saving + plotting")
    np.savez(
        args.out / "sparse.npz",
        points=points,
        wrapped=wrapped,
        unwrapped=unw,
        gamma=gamma,
        shape=np.array(mask.shape),
        gamma_min=args.gamma_min,
        nlooks=args.nlooks,
        n_target=args.n_points,
        max_edge_length=max_edge,
        median_nn=median_nn,
        elapsed_sec=elapsed,
    )

    # Reference both wrapped and unwrapped to their median over finite pixels
    # so the colormap is comparable; the unwrap is only unique up to a global
    # 2π anchor, and Delaunay seed-based integration gives an arbitrary one.
    unw_show = unw.copy()
    if finite.any():
        unw_show = unw_show - np.nanmedian(unw_show[finite])

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)

    # Wrapped phase, full raster
    wrapped_show = np.where(mask, wrapped_full, np.nan)
    axes[0, 0].imshow(
        wrapped_show, cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="none"
    )
    axes[0, 0].set_title(f"wrapped phase (full raster, γ > {args.gamma_min} masked)")

    # Sampled subset as a scatter on top of the wrapped phase
    axes[0, 1].imshow(
        wrapped_show,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        interpolation="none",
        alpha=0.3,
    )
    axes[0, 1].scatter(
        cols, rows, c=wrapped, s=0.4, cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    axes[0, 1].set_title(f"subsampled subset ({len(pick):,} pts, γ>{args.gamma_min})")

    # Unwrapped sparse result as a scatter
    if finite.any():
        lo, hi = np.nanpercentile(unw_show, [1, 99])
    else:
        lo, hi = -1, 1
    axes[1, 0].scatter(
        cols[finite],
        rows[finite],
        c=unw_show[finite],
        s=0.6,
        cmap="turbo",
        vmin=lo,
        vmax=hi,
    )
    axes[1, 0].set_aspect("equal")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(0, mask.shape[1])
    axes[1, 0].set_ylim(mask.shape[0], 0)
    axes[1, 0].set_title(
        f"unwrap_sparse result ({elapsed:.1f} s, "
        f"{100 * (1 - nan_frac):.1f}% finite)\n"
        f"congruence max |wrap(unw − wrapped)| = {congruence_max:.1e} rad"
    )

    # Density / point distribution
    axes[1, 1].hexbin(cols, rows, gridsize=80, cmap="viridis", mincnt=1)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(0, mask.shape[1])
    axes[1, 1].set_ylim(mask.shape[0], 0)
    axes[1, 1].set_title("sample density (hexbin)")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(args.out / "sparse.png", dpi=140)
    plt.close(fig)
    print(f"      wrote {args.out / 'sparse.png'}")

    # Mountain-region zoom (top-right of the GSLC scene as it appears in
    # imshow; row/col bounds tuned to the Long Beach test scene).
    h, w = mask.shape
    r0, r1 = int(h * 0.05), int(h * 0.55)
    c0, c1 = int(w * 0.55), int(w * 0.99)
    inside = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1) & finite
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    axes[0].imshow(
        wrapped_show[r0:r1, c0:c1],
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        interpolation="none",
    )
    axes[0].set_title("wrapped phase (zoom on mountains)")
    if inside.any():
        axes[1].scatter(
            cols[inside] - c0,
            rows[inside] - r0,
            c=unw_show[inside],
            s=2.0,
            cmap="turbo",
            vmin=lo,
            vmax=hi,
        )
    axes[1].set_aspect("equal")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, c1 - c0)
    axes[1].set_ylim(r1 - r0, 0)
    axes[1].set_title("unwrap_sparse (same zoom)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(args.out / "sparse_zoom.png", dpi=140)
    plt.close(fig)
    print(f"      wrote {args.out / 'sparse_zoom.png'}")

    with open(args.out / "timings.txt", "w") as f:
        f.write(f"gamma_min:        {args.gamma_min}\n")
        f.write(f"nlooks:           {args.nlooks}\n")
        f.write(f"n_target:         {args.n_points}\n")
        f.write(f"n_picked:         {len(pick)}\n")
        f.write(f"raster_shape:     {mask.shape}\n")
        f.write(f"median_nn_dist:   {median_nn:.3f} px\n")
        f.write(f"p90_nn_dist:      {p90_nn:.3f} px\n")
        f.write(f"max_edge_length:  {max_edge:.3f} px\n")
        f.write(f"unwrap_elapsed:   {elapsed:.3f} s\n")
        f.write(f"nan_frac:         {nan_frac:.6f}\n")
        f.write(f"congruence_max:   {congruence_max:.3e} rad\n")
    print(f"      wrote {args.out / 'timings.txt'}")


if __name__ == "__main__":
    main()
