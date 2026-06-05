"""Palos Verdes binary-vs-continuous time series comparison (whirlwind-rs).

Runs the existing 3D unwrap pipeline on the same 1024² PV subset used in
``reproduce.sh``, but under three different cost configurations:

  - ``continuous``  - current default. CRLB-weighted continuous cost per
                       IG; no mask applied beyond what the IG-level
                       ``.int.mask.tif`` already encodes.
  - ``binary-0.6``  - mask = (temporal_coherence_average > 0.6). The IG-
                       level mask is intersected. Edges touching bad
                       pixels get cost 0 (cheap-to-cut) - the closest
                       analog to spurt's hard exclusion of pixels with
                       ``temp_coh < threshold``.
  - ``binary-0.9``  - same with threshold 0.9 (very strict).

For each variant the script writes a `(n_dates, m, n)` date-phase cube to
``<out>/<variant>/date_phases.npy`` plus a JSON of metadata. The companion
``binary_vs_continuous_plots.py`` ingests those cubes and produces the
comparison figures.

How to rerun
------------
::

    uv run python scripts/binary_vs_continuous_pv.py \\
        --dolphin /path/to/dolphin \\
        --out     /tmp/binary-vs-continuous/pv \\
        --window  1000 1500 2024 2524 \\
        --max-igs 60 \\
        --thresholds 0.6 0.9

Defaults match ``reproduce.sh`` (1024² window, 60 IGs).
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    print("rasterio not installed; pip install rasterio", file=sys.stderr)
    raise

import whirlwind as ww

# Reuse the existing discovery + reference-resolution helpers from unwrap_stack.
sys.path.insert(0, str(Path(__file__).parent))
from unwrap_stack import (  # type: ignore
    discover_stack,
    _read_complex,
    _read_f32,
    _read_bool,
    _window_from_args,
    _resolve_reference,
)


def find_temp_coh(dolphin: Path) -> Path:
    """Dolphin emits ``temporal_coherence_average_<a>_<b>.tif`` (one file)."""
    candidates = list(
        (dolphin / "interferograms").glob("temporal_coherence_average_*.tif")
    )
    if not candidates:
        # Some dolphin versions stash it under phase_linking/linked_phase/.
        candidates = list(
            (dolphin / "phase_linking" / "linked_phase").glob(
                "temporal_coherence_average_*.tif"
            )
        )
    if not candidates:
        raise FileNotFoundError(
            "no temporal_coherence_average_*.tif in dolphin output; "
            "this script needs the stack-average temp coh that spurt would use."
        )
    if len(candidates) > 1:
        print(f"[temp_coh] {len(candidates)} candidates, using {candidates[0].name}")
    return candidates[0]


def run_variant(
    *,
    label: str,
    out_dir: Path,
    igs: list,
    dates: list[str],
    date_idx: dict[str, int],
    crlb_cube: np.ndarray,
    win: Window | None,
    ig_dir: Path,
    mask_threshold: float | None,
    temp_coh: np.ndarray | None,
    edges_from: np.ndarray,
    edges_to: np.ndarray,
    edge_priority: np.ndarray,
    n_threads: int,
    ref_i: int,
    ref_j: int,
) -> dict:
    """Per-IG 2D unwrap with CRLB cost + variant mask, then closure correct."""
    n_edges = len(igs)
    m, n = crlb_cube.shape[1:]
    print(f"\n[{label}] unwrapping {n_edges} IGs (mask threshold={mask_threshold})")

    # Build the variant mask once (constant across IGs).
    if mask_threshold is not None:
        assert temp_coh is not None, "binary variant needs temp_coh raster"
        variant_mask = (temp_coh > mask_threshold).astype(bool)
        kept = int(variant_mask.sum())
        print(
            f"[{label}]   variant mask: {kept:,} / {variant_mask.size:,} pixels "
            f"({100 * kept / variant_mask.size:.1f}%) kept at temp_coh > {mask_threshold}"
        )
    else:
        variant_mask = None

    t0 = time.perf_counter()
    unw_stack = np.zeros((n_edges, m, n), dtype=np.float32)

    def unwrap_one(idx_e):
        idx, e = idx_e
        igram = _read_complex(e.ig_path, win)
        variance = crlb_cube[date_idx[e.date_a]] + crlb_cube[date_idx[e.date_b]]
        # IG-level mask (from dolphin), ANDed with variant mask.
        ig_mask = None
        if e.mask_path is not None:
            ig_mask = _read_bool(e.mask_path, win)
        if variant_mask is not None:
            ig_mask = variant_mask if ig_mask is None else (ig_mask & variant_mask)
        unw, _cc = ww.unwrap_crlb(igram, variance, ig_mask)
        return idx, unw

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for idx, unw in ex.map(unwrap_one, enumerate(igs)):
            unw_stack[idx] = unw
    print(f"[{label}]   2D unwrap: {time.perf_counter() - t0:.1f}s")

    # Replace NaN (from disconnected mask components) with 0 before closure,
    # so closure doesn't propagate NaN through every cycle. The downstream
    # plot mask records where unwrap had no answer.
    nan_mask_per_ig = np.isnan(unw_stack)
    if nan_mask_per_ig.any():
        n_nan = int(nan_mask_per_ig.sum())
        print(
            f"[{label}]   {n_nan:,} NaN pixels in unwrap stack "
            f"({100 * n_nan / unw_stack.size:.2f}% across IGs); replacing with 0 for closure"
        )
        unw_stack_for_closure = np.where(nan_mask_per_ig, 0.0, unw_stack)
    else:
        unw_stack_for_closure = unw_stack

    # Closure correction over the temporal graph.
    t0 = time.perf_counter()
    closure = ww.closure_correct(
        unw_stack_for_closure,
        edges_from,
        edges_to,
        len(dates),
        0,
        edge_priority,
    )
    print(f"[{label}]   closure: {time.perf_counter() - t0:.1f}s")

    # Re-mark the NaN pixels so they're visibly missing downstream.
    closure["corrected"][nan_mask_per_ig] = np.nan
    # date_phases: NaN any date where any IG involving it was masked at that
    # pixel - conservative.
    bad_per_date = np.zeros((len(dates), m, n), dtype=bool)
    for i, (a, b) in enumerate([(e.date_a, e.date_b) for e in igs]):
        bad_per_date[date_idx[a]] |= nan_mask_per_ig[i]
        bad_per_date[date_idx[b]] |= nan_mask_per_ig[i]
    closure["date_phases"][bad_per_date] = np.nan

    # Reference-pixel anchoring. With a sparse binary mask the reference may
    # land in a different connected component than most pixels, leaving
    # ref_vals NaN; subtracting NaN propagates it to the entire IG. Fall
    # back per-IG to the median of finite values so the surviving
    # connected component still shows meaningful relative phase. This is
    # purely a visualization choice - for cross-variant comparison the
    # absolute offsets aren't directly meaningful anyway.
    ref_vals = closure["corrected"][:, ref_i, ref_j].copy()
    n_ref_nan = int(np.isnan(ref_vals).sum())
    if n_ref_nan:
        print(
            f"[{label}]   reference pixel ({ref_i}, {ref_j}) was NaN in {n_ref_nan}/{n_edges} IGs; "
            f"falling back to per-IG finite-median anchor for those."
        )
        for i in range(n_edges):
            if np.isnan(ref_vals[i]):
                finite = closure["corrected"][i][np.isfinite(closure["corrected"][i])]
                ref_vals[i] = float(np.median(finite)) if finite.size > 0 else 0.0
    closure["corrected"] -= ref_vals[:, None, None]
    ref_dates = closure["date_phases"][:, ref_i, ref_j].copy()
    n_date_nan = int(np.isnan(ref_dates).sum())
    if n_date_nan:
        for k in range(len(dates)):
            if np.isnan(ref_dates[k]):
                finite = closure["date_phases"][k][
                    np.isfinite(closure["date_phases"][k])
                ]
                ref_dates[k] = float(np.median(finite)) if finite.size > 0 else 0.0
    closure["date_phases"] -= ref_dates[:, None, None]

    # Persist.
    vdir = out_dir / label
    vdir.mkdir(parents=True, exist_ok=True)
    np.save(vdir / "date_phases.npy", closure["date_phases"])
    np.save(vdir / "unw_stack.npy", closure["corrected"])
    if variant_mask is not None:
        np.save(vdir / "variant_mask.npy", variant_mask)
    meta = {
        "label": label,
        "mask_threshold": mask_threshold,
        "n_edges": n_edges,
        "n_dates": len(dates),
        "dates": dates,
        "shape": [int(m), int(n)],
        "reference_pixel": [int(ref_i), int(ref_j)],
        "nan_fraction_per_ig": float(nan_mask_per_ig.mean()),
    }
    with (vdir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{label}]   wrote {vdir}/")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dolphin", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--window",
        type=int,
        nargs=4,
        default=None,
        help="i0 j0 i1 j1 - same convention as unwrap_stack.py",
    )
    ap.add_argument("--max-igs", type=int, default=None)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.9])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--reference",
        default="auto",
        help="pixel anchor: auto | dolphin | 'i,j' (window-local)",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[discover] {args.dolphin}")
    igs, crlb_paths = discover_stack(args.dolphin)
    if args.max_igs is not None:
        igs = igs[: args.max_igs]
    dates = sorted({e.date_a for e in igs} | {e.date_b for e in igs})
    date_idx = {d: i for i, d in enumerate(dates)}
    print(f"[discover] {len(igs)} IGs, {len(dates)} dates")

    missing = [d for d in dates if d not in crlb_paths]
    if missing:
        raise RuntimeError(f"missing crlb_<date>.tif for {missing[:3]} ...")

    temp_coh_path = find_temp_coh(args.dolphin)
    print(f"[discover] temp_coh: {temp_coh_path.name}")

    win = _window_from_args(args.window)
    if win is not None:
        print(f"[window] {win}")

    print(f"[crlb] loading {len(dates)} rasters")
    sample = _read_f32(crlb_paths[dates[0]], win)
    m, n = sample.shape
    crlb_cube = np.zeros((len(dates), m, n), dtype=np.float32)
    crlb_cube[0] = sample
    for k, d in enumerate(dates[1:], start=1):
        crlb_cube[k] = _read_f32(crlb_paths[d], win)

    temp_coh = _read_f32(temp_coh_path, win)
    np.save(args.out / "temp_coh.npy", temp_coh)
    print(
        f"[temp_coh] median={np.median(temp_coh):.3f} mean={float(temp_coh.mean()):.3f} "
        f"max={float(temp_coh.max()):.3f}"
    )

    edges_from = np.array([date_idx[e.date_a] for e in igs], dtype=np.uint32)
    edges_to = np.array([date_idx[e.date_b] for e in igs], dtype=np.uint32)

    def median_var(e):
        v = crlb_cube[date_idx[e.date_a]] + crlb_cube[date_idx[e.date_b]]
        valid = v > 0
        return float(np.median(v[valid])) if valid.any() else float(np.inf)

    edge_priority = np.array([median_var(e) for e in igs], dtype=np.float32)

    # The reference pixel must be in the kept set of EVERY variant; otherwise
    # binary variants NaN that pixel and reference subtraction propagates NaN
    # to the entire IG. Pick the pixel with the largest temp_coh in the
    # window - guaranteed to survive every threshold up to that value.
    if args.reference == "auto":
        flat_idx = int(np.argmax(temp_coh))
        ref_i, ref_j = (
            int(flat_idx // temp_coh.shape[1]),
            int(flat_idx % temp_coh.shape[1]),
        )
        ref_mode = f"auto (max temp_coh = {float(temp_coh[ref_i, ref_j]):.3f})"
    else:
        ref_i, ref_j, ref_mode = _resolve_reference(
            args.reference, args.dolphin, crlb_cube, win
        )
    print(f"[reference] ({ref_i}, {ref_j}) - {ref_mode}")
    # Sanity check: reference is above the strictest threshold we'll use.
    max_t = max(args.thresholds) if args.thresholds else 0.0
    if temp_coh[ref_i, ref_j] <= max_t:
        print(
            f"[reference] WARNING: ref temp_coh={float(temp_coh[ref_i, ref_j]):.3f} "
            f"is ≤ max threshold {max_t}; binary variants will NaN it out."
        )

    # Persist a top-level summary so the plot script can find every variant.
    summary = {
        "dolphin": str(args.dolphin),
        "window": args.window,
        "shape": [int(m), int(n)],
        "n_igs": len(igs),
        "n_dates": len(dates),
        "dates": dates,
        "reference_pixel": [int(ref_i), int(ref_j)],
        "temp_coh_file": str(temp_coh_path),
        "variants": [],
    }

    # continuous: no mask threshold
    meta = run_variant(
        label="continuous",
        out_dir=args.out,
        igs=igs,
        dates=dates,
        date_idx=date_idx,
        crlb_cube=crlb_cube,
        win=win,
        ig_dir=args.dolphin / "interferograms",
        mask_threshold=None,
        temp_coh=None,
        edges_from=edges_from,
        edges_to=edges_to,
        edge_priority=edge_priority,
        n_threads=args.threads,
        ref_i=ref_i,
        ref_j=ref_j,
    )
    summary["variants"].append(meta)
    for t in args.thresholds:
        meta = run_variant(
            label=f"binary_T{t:.2f}",
            out_dir=args.out,
            igs=igs,
            dates=dates,
            date_idx=date_idx,
            crlb_cube=crlb_cube,
            win=win,
            ig_dir=args.dolphin / "interferograms",
            mask_threshold=t,
            temp_coh=temp_coh,
            edges_from=edges_from,
            edges_to=edges_to,
            edge_priority=edge_priority,
            n_threads=args.threads,
            ref_i=ref_i,
            ref_j=ref_j,
        )
        summary["variants"].append(meta)

    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
