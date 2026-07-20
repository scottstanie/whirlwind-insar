#!/usr/bin/env python3
"""A/B the PHASS arc-cost surface on whirlwind's own capacity-1 linear solver.

Motivated by the NISAR cryosphere stacked-cut frame (009_074_A_137, see
docs/BUG_NISAR_CRYO_STACKED_CUTS.md): the capacity-1 AND the uncapacitated
(`WHIRLWIND_UNWRAP_SOLVER=multi`) linear solves both split the glacier by -3
cycles, while actual isce3 PHASS does not. Preprocessing and arc capacity are
ruled out, so the remaining candidate is the COST SURFACE. This script keeps
whirlwind's exact parity solver (`ww._native.unwrap_linear_ext_costs`) and
swaps in PHASS's arc costs (isce3 `PhassUnwrapper.cc`):

  cost  = uchar(min(gamma_a^2, gamma_b^2) * 100)          # squared coherence
  cost  = 255 if cost > int(good_corr^2 * 100)            # high-coh clamp (49)
  cost  = 0   if wrapped |dphi| >= 1.0 rad (both valid)   # steep-gradient zero

Variants isolate the two special ingredients:
  phass             full recipe
  phass-nogradzero  drop the >=1 rad zeroing (keep squared corr + clamp)
  phass-noclamp     drop the 255 clamp        (keep squared corr + gradzero)

Inputs come from a compare_gunw.py output ``full_arrays.npz`` (ig, coh, mask,
prod_unw, prod_cc), so preprocessing and the agreement metric are byte-for-byte
the benchmark's. Run one heavy frame at a time.

Example:
  PYTHONPATH=python python scripts/phass_cost_ablation.py \
    --npz .../compare/multi/NISAR_..._001/full_arrays.npz \
    --out-dir .../compare/phass-cost-ww --variant phass
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "aws-batch"))

from compare_gunw import compute_compare_stats, plot_result  # noqa: E402

import whirlwind as ww  # noqa: E402
from whirlwind import (  # noqa: E402
    CONNCOMP_RELIABILITY_UNIT,
    bridge_components,
    conncomp_min_coherence_auto,
    conncomp_reliability_from_coherence,
)
from whirlwind._native import components_snaphu  # noqa: E402

PI = np.pi
TWO_PI = 2.0 * np.pi


def phass_costs(
    ig_solver: np.ndarray,
    coh_solver: np.ndarray,
    good_corr: float = 0.7,
    grad_th: float = 1.0,
    clamp: bool = True,
    gradzero: bool = True,
) -> np.ndarray:
    """PHASS arc costs in Rust order [DOWN(n_v), UP(n_v), RIGHT(n_h), LEFT(n_h)].

    Mirrors isce3 PhassUnwrapper.cc: corr is squared, cost_scale=100, the
    uchar cast truncates, mask_th = int(good_corr^2 * 100), and the gradient
    zeroing applies only where both pixels have corr^2 > 1e-9 (masked pixels
    enter with coherence 0, so their arcs stay cost 0 exactly like
    unwrap_linear's zero-cost masked sea).
    """
    corr2 = coh_solver.astype(np.float64) ** 2
    mask_th = int(good_corr**2 * 100)
    small = 1e-9

    def arc_costs(c2_a, c2_b, ph_a, ph_b):
        cost = (np.minimum(c2_a, c2_b) * 100).astype(np.int32)
        if clamp:
            cost[cost > mask_th] = 255
        if gradzero:
            d = np.abs(ph_a - ph_b)
            d = np.where(d > PI, TWO_PI - d, d)
            cost[(d >= grad_th) & (c2_a > small) & (c2_b > small)] = 0
        return cost

    # DOWN/UP arcs cross the pixel edge between (i, j-1) and (i, j).
    cost_dx = arc_costs(
        corr2[:, :-1], corr2[:, 1:], ig_solver[:, :-1], ig_solver[:, 1:]
    )
    # RIGHT/LEFT arcs cross the pixel edge between (i-1, j) and (i, j).
    cost_dy = arc_costs(
        corr2[:-1, :], corr2[1:, :], ig_solver[:-1, :], ig_solver[1:, :]
    )

    dn = np.pad(cost_dx, [(0, 0), (1, 1)]).ravel()
    rt = np.pad(cost_dy, [(1, 1), (0, 0)]).ravel()
    return np.ascontiguousarray(np.concatenate([dn, dn, rt, rt]), dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, required=True, help="full_arrays.npz")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--variant",
        default="phass",
        choices=["phass", "phass-nogradzero", "phass-noclamp", "carballo"],
        help="carballo = built-in Carballo cost via unwrap_linear (the paired "
        "no-bridge baseline; isolates the cost surface as the only variable)",
    )
    ap.add_argument("--good-corr", type=float, default=0.7)
    ap.add_argument("--grad-th", type=float, default=1.0)
    ap.add_argument("--nlooks", type=float, default=16.0, help="carballo only")
    args = ap.parse_args()

    d = np.load(args.npz)
    ig = d["ig"].astype(np.float32)
    coh = d["coh"].astype(np.float32)
    mask = d["mask"].astype(bool)
    prod_unw = d["prod_unw"].astype(np.float32)
    prod_cc = d["prod_cc"]

    # Identical solver-input sanitization to compare_gunw.compare_one.
    ig_solver = np.where(mask, ig, 0.0).astype(np.float32)
    ig_complex = np.exp(1j * ig_solver).astype(np.complex64)
    coh_solver = np.where(mask, np.clip(np.nan_to_num(coh), 0.0, 1.0), 0.0).astype(
        np.float32
    )

    if args.variant == "carballo":
        t0 = time.perf_counter()
        unw = ww._native.unwrap_linear(ig_complex, coh_solver, args.nlooks, mask)
        runtime_s = time.perf_counter() - t0
    else:
        costs = phass_costs(
            ig_solver,
            coh_solver,
            good_corr=args.good_corr,
            grad_th=args.grad_th,
            clamp=args.variant != "phass-noclamp",
            gradzero=args.variant != "phass-nogradzero",
        )
        nz = costs[costs > 0]
        print(
            f"{args.variant}: costs built - zero={np.mean(costs == 0):.3f} "
            f"clamped255={np.mean(costs == 255):.3f} "
            f"median_nonzero={np.median(nz) if nz.size else float('nan'):.0f}",
            flush=True,
        )
        t0 = time.perf_counter()
        unw = ww._native.unwrap_linear_ext_costs(ig_complex, mask, costs)
        runtime_s = time.perf_counter() - t0
    unw = np.asarray(unw, dtype=np.float32)

    # Same post-solve tail as the public ``ww.unwrap`` (bridge, then the
    # SNAPHU-style conncomp grow at its defaults). Both key off the unwrapped
    # phase rather than the solved network, so they compose with the ext-costs
    # diagnostic path - without this the figure's conncomp panels would be
    # empty and the coverage panel would read "prod only" everywhere.
    unw = bridge_components(unw, mask)
    gamma = conncomp_min_coherence_auto(args.nlooks)
    reliability_raw = round(
        conncomp_reliability_from_coherence(gamma, args.nlooks)
        * CONNCOMP_RELIABILITY_UNIT
    )
    cc = np.asarray(
        components_snaphu(
            ig_complex,
            coh_solver,
            args.nlooks,
            unw,
            mask,
            reliability_raw,
            100,  # min_size_px
            1024,  # max_ncomps
            (7, 7),  # phase_grad_window
        )
    )

    stats, ww_aligned, _residual_wrapped, amb_diff = compute_compare_stats(
        ig=ig,
        coh=coh,
        mask=mask,
        prod_unw=prod_unw,
        prod_cc=prod_cc,
        ww_unw=unw,
        ww_cc=cc,
        runtime_s=runtime_s,
        rss_delta_mb=None,
    )
    stats["variant"] = args.variant
    stats["good_corr"] = args.good_corr
    stats["grad_th"] = args.grad_th
    stats["npz"] = str(args.npz)

    out = args.out_dir / args.variant
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    png = out / "full.png"
    plot_result(
        png,
        ig=ig,
        coh=coh,
        prod_unw=prod_unw,
        ww_aligned=ww_aligned,
        prod_cc=prod_cc,
        ww_cc=cc,
        amb_diff=amb_diff,
        valid=mask & np.isfinite(unw),
        title=f"PHASS-cost ablation [{args.variant}] on ww linear solver\n"
        f"{args.npz.parent.name}, runtime={runtime_s:.1f}s",
        stride=4,
    )
    print(
        f"{args.variant}: {runtime_s:.1f}s  "
        f"match={stats['ambiguity_match_frac']:.4f}  "
        f"per-comp={stats['ambiguity_match_frac_percomp']:.4f}",
        flush=True,
    )
    print(f"wrote {out / 'stats.json'}\nwrote {png}", flush=True)


if __name__ == "__main__":
    main()
