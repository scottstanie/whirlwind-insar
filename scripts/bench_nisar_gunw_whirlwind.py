#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "h5py",
#   "matplotlib",
#   "pandas",
#   "earthaccess",
#   "psutil",
# ]
# ///
"""Benchmark a phase-unwrapper against NISAR L2 GUNW products.

Default workflow
----------------
For each NISAR GUNW product, this script reads the production 80 m unwrapped
phase, re-wraps it to [-pi, pi), reads the 80 m coherence/mask/connected
components, runs whirlwind, and compares the output to the production GUNW
unwrapped phase, recording runtime, peak-RSS delta, and ambiguity-match stats.

Which solver (`--solver`)
-------------------------
* `linear` (DEFAULT, RECOMMENDED): the VERIFIED single-tile, whole-image
  `unwrap_linear` with fixed Carballo parity costs. On the full D_077 frame it
  matches Python ww-orig at 99.49%% and beats single-tile snaphu on BOTH speed
  (~158 s vs ~588 s) and accuracy (99.49%% vs 99.30%%). This is the path to show
  on NISAR-sized interferograms.
* `tiled`: the `ww.unwrap` tiled path. EXPERIMENTAL — tiling is not yet validated
  on NISAR-scale frames and can produce invalid (fast-but-wrong) results.

Run the cost model at `--nlooks ~16` (NISAR GUNW unwrap looks are ~13x16);
`--nlooks 1` gives a near-flat phase-difference PDF and degenerate routing.

Why re-wrap the production unwrapped phase?
-------------------------------------------
Current beta GUNW products include a 20 m complex wrappedInterferogram, but
some public release notes have warned that this layer may be incorrectly
georeferenced in beta products. Re-wrapping the 80 m unwrappedPhase gives an
apples-to-apples benchmark of the unwrapping algorithm on the same grid as the
production unwrapped product. This is primarily a runtime/regression/ambiguity
consistency benchmark, not a test of the full NISAR wrapped-product geocoding.

Examples
--------
Local file:
    python bench_nisar_gunw_whirlwind.py \
        --local-h5 NISAR_L2_PR_GUNW_..._001.h5 \
        --out-dir ww_bench --nlooks 16 --sizes 1024 2048 full

Search + download a named granule via earthaccess/EDL:
    python bench_nisar_gunw_whirlwind.py \
        --granule NISAR_L2_PR_GUNW_007_164_D_077_010_2000_QD_20251215T140630_20251215T140646_20260120T140632_20260120T140648_X05010_N_P_J_001 \
        --data-dir ./nisar_data --out-dir ww_bench --nlooks 16 --sizes 1024 full

Search by bbox/time:
    python bench_nisar_gunw_whirlwind.py \
        --bbox -124 32 -114 42 --start 2026-01-01 --end 2026-02-01 \
        --count 5 --data-dir ./nisar_data --out-dir ww_bench --nlooks 16
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TWOPI = 2.0 * np.pi
SHORT_NAME = "NISAR_L2_GUNW_BETA_V1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark whirlwind.unwrap against NISAR L2 GUNW products.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("inputs")
    src.add_argument("--local-h5", nargs="*", type=Path, default=[], help="Existing GUNW .h5 files to benchmark.")
    src.add_argument("--granule", nargs="*", default=[], help="Exact NISAR GUNW granule names to search/download with earthaccess.")
    src.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"), help="Bounding box for earthaccess search.")
    src.add_argument("--start", help="Start date for earthaccess search, e.g. 2026-01-01.")
    src.add_argument("--end", help="End date for earthaccess search, e.g. 2026-02-01.")
    src.add_argument("--count", type=int, default=1, help="Max search results for bbox/time search.")
    src.add_argument("--data-dir", type=Path, default=Path("nisar_data"), help="Where earthaccess downloads products.")

    run = p.add_argument_group("benchmark")
    run.add_argument("--out-dir", type=Path, default=Path("ww_gunw_bench"), help="Output directory.")
    run.add_argument("--pol", default=None, help="Polarization group to use, e.g. HH or VV. Default: first available.")
    run.add_argument(
        "--solver",
        choices=["linear", "tiled"],
        default="linear",
        help="Which whirlwind solver to benchmark. 'linear' (default, RECOMMENDED) = the "
        "VERIFIED single-tile whole-image `unwrap_linear` with fixed Carballo parity costs; "
        "matches Python ww-orig at ~99.5%% on D_077 and beats single-tile snaphu on both speed "
        "and accuracy. 'tiled' = the `ww.unwrap` tiled path (EXPERIMENTAL; tiling is not yet "
        "validated on NISAR-scale frames and can produce invalid results).",
    )
    run.add_argument("--nlooks", type=float, default=16.0, help="nlooks for the Carballo cost model (NISAR GUNW unwrap looks ~13x16; nlooks=1 gives a near-flat PDF and degenerate routing — keep ~16).")
    run.add_argument("--tile-size", type=int, default=0, help="tile_size for --solver tiled. 0=auto (512 for >512px). Use a value >= frame dims to force a single whole-image solve. Ignored for --solver linear (always single-tile).")
    run.add_argument("--tile-overlap", type=int, default=0, help="tile_overlap passed to ww.unwrap (0=auto).")
    run.add_argument("--sizes", nargs="*", default=["full"], help="Square center-crop sizes to run, plus optional 'full'.")
    run.add_argument("--crop", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"), help="Explicit crop window. Overrides --sizes.")
    run.add_argument("--coh-threshold", type=float, default=0.0, help="Minimum coherence for unwrapping mask.")
    run.add_argument(
        "--mask-policy",
        choices=["water_only", "nisar_land", "not_127", "zero_is_good", "nonzero_digits", "ignore"],
        default="water_only",
        help="How to convert the GUNW unwrappedInterferogram/mask dataset to a boolean valid mask.",
    )
    run.add_argument(
        "--require-prod-cc-for-stats",
        action="store_true",
        help="Restrict comparison statistics to pixels where production connectedComponents > 0.",
    )
    run.add_argument(
        "--use-product-wrapped",
        action="store_true",
        help="Experimental: use phase(wrappedInterferogram) instead of rewrapping production unwrappedPhase. Requires same shape/grid; otherwise the run is skipped.",
    )
    run.add_argument("--plot-downsample", type=int, default=1, help="Stride for PNG plots only.")
    run.add_argument("--force", action="store_true", help="Rerun even if JSON result exists.")
    return p.parse_args()


def wrap_phase(x: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi), preserving NaNs."""
    return (x + np.pi) % TWOPI - np.pi


def read_array(ds: h5py.Dataset, dtype: np.dtype | type = np.float32) -> np.ndarray:
    arr = ds[()]
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    fill = ds.attrs.get("_FillValue")
    if fill is not None and np.issubdtype(arr.dtype, np.floating):
        fill_value = np.asarray(fill).reshape(-1)[0]
        arr = arr.copy()
        arr[arr == fill_value] = np.nan
    if np.issubdtype(arr.dtype, np.floating):
        # Some sample/tutorial products historically used -9999-style fill values.
        arr = np.where(arr < -1.0e20, np.nan, arr)
    return arr


def choose_pol(h5: h5py.File, base: str, requested: str | None) -> str:
    grp = h5[base]
    pols = [k for k, v in grp.items() if isinstance(v, h5py.Group)]
    pols = [p for p in pols if p.upper() not in {"MASK", "METADATA"}]
    if requested:
        if requested not in pols:
            raise KeyError(f"Requested pol {requested!r} not found under {base}; available={pols}")
        return requested
    if not pols:
        raise KeyError(f"No polarization groups found under {base}")
    return sorted(pols)[0]


def gunw_paths(h5: h5py.File, pol: str | None) -> dict[str, str]:
    unw_base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
    wrap_base = "/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram"
    pol = choose_pol(h5, unw_base, pol)
    return {
        "pol": pol,
        "unw": f"{unw_base}/{pol}/unwrappedPhase",
        "coh_unw": f"{unw_base}/{pol}/coherenceMagnitude",
        "cc": f"{unw_base}/{pol}/connectedComponents",
        "mask": f"{unw_base}/mask",
        "wrapped": f"{wrap_base}/{pol}/wrappedInterferogram",
        "coh_wrapped": f"{wrap_base}/{pol}/coherenceMagnitude",
    }


def mask_to_bool(mask_arr: np.ndarray | None, policy: str, shape: tuple[int, int]) -> np.ndarray:
    if mask_arr is None or policy == "ignore":
        return np.ones(shape, dtype=bool)
    if mask_arr.shape != shape:
        raise ValueError(f"Mask shape {mask_arr.shape} does not match data shape {shape}")
    if policy == "not_127":
        return mask_arr != 127
    if policy == "zero_is_good":
        return mask_arr == 0
    if policy == "nonzero_digits":
        second_digit = (mask_arr // 10) % 10
        third_digit = mask_arr % 10
        return (second_digit > 0) & (third_digit > 0)
    if policy == "water_only":
        # Exclude only the water flag; keep subswath-invalid pixels valid (A/B
        # against nisar_land to isolate whether subswath-seam masking, not water,
        # triggers the tiled vertical-banding artifact).
        return (mask_arr != 255) & ((mask_arr // 100) % 10 == 0)
    if policy == "nisar_land":
        # GUNW `mask` is a 3-digit code [water][subswath_ref][subswath_sec]
        # (water digit: 1=water). Keep NON-water pixels that are valid samples
        # in BOTH RSLC subswaths (both subswath digits > 0). _FillValue is 255.
        water = (mask_arr // 100) % 10
        ref_sub = (mask_arr // 10) % 10
        sec_sub = mask_arr % 10
        return (mask_arr != 255) & (water == 0) & (ref_sub > 0) & (sec_sub > 0)
    raise ValueError(policy)


def center_crop_slices(shape: tuple[int, int], size: int | str) -> tuple[slice, slice, str]:
    ny, nx = shape
    if str(size).lower() == "full":
        return slice(0, ny), slice(0, nx), "full"
    n = int(size)
    if n > ny or n > nx:
        raise ValueError(f"Requested crop size {n} exceeds array shape {shape}")
    y0 = (ny - n) // 2
    x0 = (nx - n) // 2
    return slice(y0, y0 + n), slice(x0, x0 + n), f"{n}x{n}"


def explicit_crop_slices(crop: list[int]) -> tuple[slice, slice, str]:
    y0, y1, x0, x1 = crop
    return slice(y0, y1), slice(x0, x1), f"y{y0}_{y1}_x{x0}_{x1}"


def component_summary(cc: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    vals = cc[valid]
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return {"num_cc": 0, "largest_cc_frac": 0.0, "nonzero_cc_frac": 0.0}
    labels, counts = np.unique(vals.astype(np.int64), return_counts=True)
    return {
        "num_cc": int(labels.size),
        "largest_cc_frac": float(counts.max() / max(1, valid.sum())),
        "nonzero_cc_frac": float(vals.size / max(1, valid.sum())),
    }


def safe_percentiles(x: np.ndarray, q: Iterable[float]) -> list[float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [math.nan for _ in q]
    return [float(v) for v in np.nanpercentile(x, list(q))]


def compute_compare_stats(
    ig: np.ndarray,
    coh: np.ndarray,
    mask: np.ndarray,
    prod_unw: np.ndarray,
    prod_cc: np.ndarray,
    ww_unw: np.ndarray,
    ww_cc: np.ndarray | None,
    runtime_s: float,
    rss_delta_mb: float | None,
    require_prod_cc: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(ig) & np.isfinite(coh) & np.isfinite(prod_unw) & np.isfinite(ww_unw)
    if require_prod_cc:
        valid &= prod_cc > 0
    if valid.sum() == 0:
        raise ValueError("No valid pixels for comparison after masking.")

    # Align a global 2pi offset before measuring ambiguity differences.
    global_cycle_offset = int(np.rint(np.nanmedian((ww_unw[valid] - prod_unw[valid]) / TWOPI)))
    ww_aligned = ww_unw - global_cycle_offset * TWOPI

    # Per-pixel ambiguity integers relative to the exact wrapped input supplied to ww.
    prod_amb = np.rint((prod_unw - ig) / TWOPI).astype(np.float64)
    ww_amb = np.rint((ww_aligned - ig) / TWOPI).astype(np.float64)
    amb_diff = ww_amb - prod_amb

    residual = ww_aligned - prod_unw
    residual_wrapped = wrap_phase(residual)
    wrap_consistency = wrap_phase(ww_unw - ig)

    resid_valid = residual[valid]
    resid_wrap_valid = residual_wrapped[valid]
    amb_valid = amb_diff[valid]
    coh_valid = coh[valid]
    abs_amb = np.abs(amb_valid)
    nonzero_amb = abs_amb > 0

    # Per-(production)-component cycle alignment. The absolute cycle of a region
    # isolated by water / decorrelation is unobservable, so align ww to
    # production WITHIN each production connected component before scoring
    # accuracy — this separates "right shape within a region" from "guessed the
    # same arbitrary inter-region offset". Scored only over prod_cc > 0 pixels.
    in_comp = valid & (prod_cc > 0)
    if in_comp.any():
        off_map = np.zeros(amb_diff.shape, dtype=np.float64)
        for lab in np.unique(prod_cc[in_comp]):
            m = valid & (prod_cc == lab)
            off_map[m] = np.rint(np.median(amb_diff[m]))
        amb_pc = amb_diff - off_map
        match_percomp = float(np.mean(amb_pc[in_comp] == 0))
    else:
        match_percomp = float("nan")

    # Coverage / recall. `mask` already encodes "land + finite coherence +
    # finite production unwrap", i.e. pixels that HAVE data. Of those, what
    # fraction does each unwrapper actually label (conncomp > 0)? This is the
    # precision/recall lens: a high-precision-but-low-recall solver (e.g. ICU)
    # gives up on data it can't trust.
    data = mask
    ndata = int(data.sum())
    recall_prod = float((prod_cc[data] > 0).mean()) if ndata else float("nan")
    recall_ww = (
        float((np.asarray(ww_cc)[data] > 0).mean())
        if (ww_cc is not None and ndata)
        else None
    )

    stats: dict[str, Any] = {
        "runtime_s": float(runtime_s),
        "rss_delta_mb": None if rss_delta_mb is None else float(rss_delta_mb),
        "shape_y": int(ig.shape[0]),
        "shape_x": int(ig.shape[1]),
        "num_pixels": int(ig.size),
        "num_valid": int(valid.sum()),
        "valid_frac": float(valid.mean()),
        "coh_mean_valid": float(np.nanmean(coh_valid)),
        "coh_p05_valid": safe_percentiles(coh_valid, [5])[0],
        "coh_p50_valid": safe_percentiles(coh_valid, [50])[0],
        "coh_p95_valid": safe_percentiles(coh_valid, [95])[0],
        "global_cycle_offset_removed": global_cycle_offset,
        "ambiguity_match_frac": float(np.mean(amb_valid == 0)),
        "ambiguity_match_frac_percomp": match_percomp,
        "ambiguity_nonzero_frac": float(np.mean(nonzero_amb)),
        "data_frac": float(data.mean()),
        "num_data": ndata,
        "prod_unwrapped_recall": recall_prod,
        "ww_unwrapped_recall": recall_ww,
        "ambiguity_abs_mean_cycles": float(np.mean(abs_amb)),
        "ambiguity_abs_p95_cycles": safe_percentiles(abs_amb, [95])[0],
        "residual_mean_rad": float(np.nanmean(resid_valid)),
        "residual_std_rad": float(np.nanstd(resid_valid)),
        "residual_rmse_rad": float(np.sqrt(np.nanmean(resid_valid**2))),
        "residual_wrapped_rmse_rad": float(np.sqrt(np.nanmean(resid_wrap_valid**2))),
        "residual_wrapped_p95_abs_rad": safe_percentiles(np.abs(resid_wrap_valid), [95])[0],
        "ww_wrap_consistency_p95_abs_rad": safe_percentiles(np.abs(wrap_consistency[valid]), [95])[0],
    }
    stats |= {f"prod_{k}": v for k, v in component_summary(prod_cc, valid).items()}
    if ww_cc is not None:
        stats |= {f"ww_{k}": v for k, v in component_summary(np.asarray(ww_cc), valid).items()}
    return stats, ww_aligned, residual_wrapped, amb_diff


def plot_result(
    out_png: Path,
    ig: np.ndarray,
    coh: np.ndarray,
    prod_unw: np.ndarray,
    ww_aligned: np.ndarray,
    ww_cc: np.ndarray | None,
    amb_diff: np.ndarray,
    valid: np.ndarray,
    title: str,
    stride: int = 1,
) -> None:
    s = (slice(None, None, stride), slice(None, None, stride))
    cc_arr = (
        np.asarray(ww_cc)[s]
        if ww_cc is not None and np.asarray(ww_cc).shape == ig.shape
        else np.zeros_like(ig)
    )
    arrays = [ig[s], coh[s], prod_unw[s], ww_aligned[s], cc_arr, amb_diff[s]]
    names = ["wrapped input (rad)", "coherence", "NISAR GUNW unwrapped", "whirlwind aligned", "whirlwind conncomps", "ambiguity diff (cycles)"]
    cmaps = ["twilight", "gray", "viridis", "viridis", "tab20", "RdBu"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(title)
    for ax, arr, name, cmap in zip(axes.ravel(), arrays, names, cmaps, strict=True):
        arrp = np.asarray(arr, dtype=float)
        arrp = np.where(valid[s], arrp, np.nan) if arrp.shape == valid[s].shape else arrp
        if name == "wrapped input (rad)":
            vmin, vmax = -np.pi, np.pi
        elif name == "coherence":
            vmin, vmax = 0.0, 1.0
        elif name == "whirlwind conncomps":
            # 0 = background/dropped → transparent; cycle label colors mod 20.
            arrp = np.where(arrp > 0, ((arrp - 1) % 20) + 1, np.nan)
            vmin, vmax = 0, 20
        elif name == "ambiguity diff (cycles)":
            vmax_abs = np.nanpercentile(np.abs(arrp), 99) if np.isfinite(arrp).any() else 1.0
            vmax_abs = float(max(vmax_abs, 1.0))
            vmin, vmax = -vmax_abs, vmax_abs
        else:
            lo, hi = safe_percentiles(arrp, [2, 98])
            vmin, vmax = lo, hi
        im = ax.imshow(arrp, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.78)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def download_with_earthaccess(args: argparse.Namespace) -> list[Path]:
    try:
        import earthaccess
    except ImportError as e:
        raise SystemExit("Install earthaccess or pass --local-h5 files.") from e

    args.data_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.login()
    results = []

    for granule in args.granule:
        # CMR granule_name supports wildcards; the product file may have .h5 appended.
        r = earthaccess.search_data(short_name=SHORT_NAME, granule_name=f"{granule}*", count=10)
        if not r:
            raise RuntimeError(f"No earthaccess results found for granule_name={granule!r}")
        results.extend(r)

    if args.bbox or args.start or args.end:
        kwargs: dict[str, Any] = {"short_name": SHORT_NAME, "count": args.count}
        if args.bbox:
            kwargs["bounding_box"] = tuple(args.bbox)
        if args.start or args.end:
            if not (args.start and args.end):
                raise SystemExit("Provide both --start and --end for temporal search.")
            kwargs["temporal"] = (args.start, args.end)
        results.extend(earthaccess.search_data(**kwargs))

    if not results:
        return []

    downloaded = [Path(p) for p in earthaccess.download(results, local_path=str(args.data_dir))]
    return [p for p in downloaded if is_main_gunw_h5(p)]


def is_main_gunw_h5(path: Path) -> bool:
    name = path.name
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        return False
    if any(s in name for s in ["QA_STATS", "STATS", "_QA_", "REPORT"]):
        return False
    return "GUNW" in name


def product_id(path: Path) -> str:
    return path.stem.replace(".", "_")


def get_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def run_one_product(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    print(f"\n=== {path} ===", flush=True)
    out_product = args.out_dir / product_id(path)
    out_product.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as h5:
        paths = gunw_paths(h5, args.pol)
        pol = paths["pol"]
        prod_unw_full = read_array(h5[paths["unw"]], np.float32)
        coh_full = read_array(h5[paths["coh_unw"]], np.float32)
        prod_cc_full = h5[paths["cc"]][()].astype(np.int64, copy=False)
        mask_arr_full = h5[paths["mask"]][()] if paths["mask"] in h5 else None

        if args.use_product_wrapped:
            wrapped_complex = h5[paths["wrapped"]][()]
            if wrapped_complex.shape != prod_unw_full.shape:
                print(
                    f"Skipping product wrappedInterferogram for {path.name}: shape {wrapped_complex.shape} != unwrapped shape {prod_unw_full.shape}. "
                    "Use default rewrapped unwrappedPhase for apples-to-apples 80 m tests.",
                    flush=True,
                )
                ig_full = wrap_phase(prod_unw_full)
            else:
                ig_full = np.angle(wrapped_complex).astype(np.float32)
        else:
            ig_full = wrap_phase(prod_unw_full).astype(np.float32)

    base_mask_full = mask_to_bool(mask_arr_full, args.mask_policy, prod_unw_full.shape)
    base_mask_full &= np.isfinite(prod_unw_full) & np.isfinite(coh_full) & (coh_full >= args.coh_threshold)

    if args.crop:
        crop_specs = [explicit_crop_slices(args.crop)]
    else:
        crop_specs = []
        for size in args.sizes:
            try:
                crop_specs.append(center_crop_slices(prod_unw_full.shape, size))
            except ValueError as e:
                print(f"Skipping crop {size}: {e}", flush=True)

    rows: list[dict[str, Any]] = []
    for ys, xs, label in crop_specs:
        result_json = out_product / f"{label}.json"
        if result_json.exists() and not args.force:
            print(f"  {label}: exists, skipping (--force to rerun)", flush=True)
            rows.append(json.loads(result_json.read_text()))
            continue

        ig = np.ascontiguousarray(ig_full[ys, xs])
        coh = np.ascontiguousarray(coh_full[ys, xs])
        prod_unw = np.ascontiguousarray(prod_unw_full[ys, xs])
        prod_cc = np.ascontiguousarray(prod_cc_full[ys, xs])
        mask = np.ascontiguousarray(base_mask_full[ys, xs])

        if mask.sum() == 0:
            print(f"  {label}: no valid pixels, skipping", flush=True)
            continue

        print(f"  {label}: running whirlwind on shape={ig.shape}, valid={mask.mean():.3f}", flush=True)
        import whirlwind as ww  # Delayed import so search/download can work without it.

        gc.collect()
        rss0 = get_rss_mb()
        t0 = time.perf_counter()
        # ww.unwrap takes a COMPLEX interferogram; `ig` here is the real wrapped
        # phase (kept real for the comparison stats below). Convert for the call.
        ig_complex = np.exp(1j * ig).astype(np.complex64)
        if args.solver == "linear":
            # VERIFIED path: single-tile whole-image MCF with fixed Carballo parity
            # costs (matches Python ww-orig). No connected-component labels are
            # returned by this solver, so component-level stats are skipped.
            ww_unw = ww._native.unwrap_linear(ig_complex, coh, float(args.nlooks), mask)
            ww_cc = None
        else:
            ww_unw, ww_cc = ww.unwrap(
                ig_complex, coh, args.nlooks, mask,
                tile_size=args.tile_size, tile_overlap=args.tile_overlap,
            )
        runtime_s = time.perf_counter() - t0
        rss1 = get_rss_mb()
        rss_delta = None if (rss0 is None or rss1 is None) else rss1 - rss0

        ww_unw = np.asarray(ww_unw, dtype=np.float32)
        ww_cc_arr = None if ww_cc is None else np.asarray(ww_cc)
        stats, ww_aligned, residual_wrapped, amb_diff = compute_compare_stats(
            ig=ig,
            coh=coh,
            mask=mask,
            prod_unw=prod_unw,
            prod_cc=prod_cc,
            ww_unw=ww_unw,
            ww_cc=ww_cc_arr,
            runtime_s=runtime_s,
            rss_delta_mb=rss_delta,
            require_prod_cc=args.require_prod_cc_for_stats,
        )
        stats.update(
            {
                "product": path.name,
                "product_path": str(path),
                "crop": label,
                "pol": pol,
                "solver": args.solver,
                "nlooks": args.nlooks,
                "coh_threshold": args.coh_threshold,
                "mask_policy": args.mask_policy,
                "input_phase_source": "phase(wrappedInterferogram)" if args.use_product_wrapped else "wrap(unwrappedPhase)",
            }
        )

        result_json.write_text(json.dumps(stats, indent=2, sort_keys=True))
        np.savez_compressed(
            out_product / f"{label}_arrays.npz",
            ig=ig,
            coh=coh,
            mask=mask,
            prod_unw=prod_unw,
            prod_cc=prod_cc,
            ww_unw=ww_unw,
            ww_aligned=ww_aligned,
            ww_cc=np.asarray([]) if ww_cc_arr is None else ww_cc_arr,
            residual_wrapped=residual_wrapped,
            ambiguity_diff=amb_diff,
        )
        plot_result(
            out_product / f"{label}.png",
            ig=ig,
            coh=coh,
            prod_unw=prod_unw,
            ww_aligned=ww_aligned,
            ww_cc=ww_cc_arr,
            amb_diff=amb_diff,
            valid=mask & np.isfinite(ww_unw),
            title=f"{path.name}\n{label}, solver={args.solver}, pol={pol}, nlooks={args.nlooks}, runtime={runtime_s:.2f}s",
            stride=max(1, args.plot_downsample),
        )
        rec_ww = stats.get("ww_unwrapped_recall")
        rec_ww_s = f"{rec_ww:.3f}" if rec_ww is not None else "n/a"
        print(
            f"  {label}: {runtime_s:.2f}s  match={stats['ambiguity_match_frac']:.3f} "
            f"per-comp={stats['ambiguity_match_frac_percomp']:.3f}  data={stats['data_frac']*100:.0f}% "
            f"recall ww={rec_ww_s}/prod={stats['prod_unwrapped_recall']:.3f}  "
            f"cc ww={stats.get('ww_num_cc')}/prod={stats.get('prod_num_cc')}",
            flush=True,
        )
        rows.append(stats)

        # Free crop arrays before next run.
        del ig, coh, prod_unw, prod_cc, mask, ww_unw, ww_aligned, residual_wrapped, amb_diff
        gc.collect()

    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    h5s = [p for p in args.local_h5 if is_main_gunw_h5(p)]
    # Only hit earthaccess when a remote search was actually requested; otherwise
    # a missing earthaccess install would needlessly abort a local-only run.
    if args.granule or args.bbox or args.start or args.end:
        h5s.extend(download_with_earthaccess(args))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    h5s = [p for p in h5s if not (p.resolve() in seen or seen.add(p.resolve()))]
    if not h5s:
        raise SystemExit("No GUNW .h5 files found. Pass --local-h5 or --granule/--bbox search options.")

    all_rows: list[dict[str, Any]] = []
    for h5 in h5s:
        all_rows.extend(run_one_product(h5, args))

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = args.out_dir / "summary.csv"
        parquet_path = args.out_dir / "summary.parquet"
        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            pass
        print(f"\nWrote {csv_path}")
        # Compact console summary.
        cols = [
            "product",
            "crop",
            "runtime_s",
            "shape_y",
            "shape_x",
            "valid_frac",
            "ambiguity_match_frac",
            "residual_wrapped_p95_abs_rad",
            "prod_num_cc",
            "ww_num_cc",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
