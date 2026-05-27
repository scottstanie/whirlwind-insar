"""Closure-corrected 3D unwrapping of a dolphin output stack.

Pipeline:
  1. Walk `<dolphin>/interferograms/` for *.int.tif → parse (date_a, date_b) pairs.
  2. For each unique acquisition, load `crlb_<date>.tif` (per-acquisition CRLB
     phase variance from phase linking). This is the *right* noise weight for
     phase-linked inputs; the `.cor` files are sliding-window approximations.
  3. For each IG:
       - read complex IG + per-edge mask
       - σ²_IG = σ²_a + σ²_b
       - run whirlwind's CRLB-weighted 2D unwrap → baseline unwrapped frame
  4. Stack all unwrapped frames into (E, m, n).
  5. Run closure correction using a CRLB-priority spanning tree.
  6. Emit:
       - corrected unwrapped stack as GeoTIFFs (one per IG, in `<out>/corrected/`)
       - closure-quality map (`closure_rms.tif`)
       - per-date recovered acquisition phase cube (`date_phases.tif`)
       - per-edge integer corrections map (`corrections.tif`)
       - a JSON report with the temporal graph + tree edges

Usage:
    uv run python scripts/unwrap_stack.py \\
        --dolphin /path/to/dolphin \\
        --out     /tmp/whirlwind-stack \\
        --window  1000 1000 1512 1512    # i0 j0 i1 j1
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time

# Flush stdout after each print so the log is readable in real time
# even when redirected (Python defaults to block-buffering on a pipe).
print = functools.partial(print, flush=True)  # noqa: A001
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    print("rasterio not installed; pip install rasterio", file=sys.stderr)
    raise

import whirlwind as ww


# ---------------------------------------------------------------------------
# Filename parsing / discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IgFile:
    """An interferogram triplet on disk: complex IG + mask + (optional) coherence."""
    date_a: str
    date_b: str
    ig_path: Path
    mask_path: Path | None
    cor_path: Path | None


def _parse_ig_name(stem: str) -> tuple[str, str] | None:
    """Parse `YYYYMMDDhhmmss_YYYYMMDDhhmmss` into (date_a, date_b)."""
    parts = stem.split("_")
    if len(parts) != 2:
        return None
    a, b = parts
    if not (a.isdigit() and b.isdigit() and len(a) == len(b)):
        return None
    return a, b


def discover_stack(dolphin: Path) -> tuple[list[IgFile], dict[str, Path]]:
    """Walk `dolphin/interferograms/` and return:
       - list of IG files (with companion mask + cor paths if they exist)
       - map of date → crlb_<date>.tif path
    """
    ig_dir = dolphin / "interferograms"
    if not ig_dir.is_dir():
        raise FileNotFoundError(f"no interferograms/ in {dolphin}")

    igs: list[IgFile] = []
    for p in sorted(ig_dir.glob("*.int.tif")):
        stem = p.name.removesuffix(".int.tif")
        parsed = _parse_ig_name(stem)
        if parsed is None:
            continue
        a, b = parsed
        mask = ig_dir / f"{stem}.int.mask.tif"
        cor = ig_dir / f"{stem}.int.cor.tif"
        igs.append(IgFile(
            date_a=a,
            date_b=b,
            ig_path=p,
            mask_path=mask if mask.exists() else None,
            cor_path=cor if cor.exists() else None,
        ))

    crlb_paths: dict[str, Path] = {}
    for p in ig_dir.glob("crlb_*.tif"):
        date = p.stem.removeprefix("crlb_")
        crlb_paths[date] = p

    return igs, crlb_paths


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _window_from_args(window: list[int] | None) -> Window | None:
    if window is None:
        return None
    if len(window) != 4:
        raise ValueError("--window needs 4 ints: i0 j0 i1 j1")
    i0, j0, i1, j1 = window
    return Window(j0, i0, j1 - j0, i1 - i0)  # type: ignore[call-arg]  # col_off, row_off, w, h


def _read_complex(path: Path, win: Window | None) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1, window=win)
    if not np.iscomplexobj(a):
        # Treat float-valued TIFFs as wrapped phase in radians.
        a = np.exp(1j * np.nan_to_num(a.astype(np.float32))).astype(np.complex64)
    else:
        a = np.nan_to_num(a, nan=0.0).astype(np.complex64)
    return a


def _read_f32(path: Path, win: Window | None) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1, window=win)
    return np.nan_to_num(a.astype(np.float32), nan=0.0)


def _read_bool(path: Path, win: Window | None) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1, window=win)
    return a.astype(bool)


def _output_profile(template_path: Path, win: Window | None, dtype: str, count: int = 1) -> dict:
    """Build a rasterio profile suitable for writing outputs in the same CRS
    as the template, possibly cropped to a window."""
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
        if win is not None:
            profile["transform"] = src.window_transform(win)
            profile["width"] = win.width
            profile["height"] = win.height
    profile.update(
        dtype=dtype,
        count=count,
        compress="lzw",
        predictor=2 if dtype.startswith("float") else 1,
        nodata=None,
        BIGTIFF="YES",
    )
    profile.pop("photometric", None)
    profile.pop("colorinterp", None)
    return profile


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _resolve_reference(
    arg: str | None,
    dolphin: Path,
    crlb_cube: np.ndarray,
    window: Window | None,
) -> tuple[int, int, str]:
    """Resolve the reference-pixel coords to (i, j) in the current (windowed) array.

    Precedence:
      1. `--reference i,j`  — explicit window-local pixel coords
      2. `--reference dolphin` — read `dolphin/timeseries/reference_point.txt`
                                 (window-aware: subtract the window's offset)
      3. `--reference auto` (default) — pick the lowest-summed-CRLB pixel,
                                         which is the most consistently coherent
                                         across the entire stack
    Returns (i, j, mode_str).
    """
    if arg is None or arg == "auto":
        score = crlb_cube.sum(axis=0)
        # Avoid nodata (zero) pixels in the auto-pick.
        score = np.where(score > 0, score, np.inf)
        i, j = np.unravel_index(int(score.argmin()), score.shape)
        return int(i), int(j), f"auto (min Σ_d CRLB, score={float(crlb_cube.sum(axis=0)[i,j]):.3g})"
    if arg == "dolphin":
        rp = dolphin / "timeseries" / "reference_point.txt"
        if not rp.exists():
            raise FileNotFoundError(f"no {rp}; pass --reference auto or i,j")
        txt = rp.read_text().strip()
        # Format: "row,col" — these are full-scene coords in dolphin's grid.
        parts = txt.replace(" ", "").split(",")
        full_i, full_j = int(parts[0]), int(parts[1])
        if window is not None:
            full_i -= int(window.row_off)
            full_j -= int(window.col_off)
        return full_i, full_j, f"dolphin reference_point.txt = ({full_i}, {full_j}) in window"
    # explicit "i,j"
    parts = arg.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"--reference must be 'auto', 'dolphin', or 'i,j'; got {arg!r}")
    return int(parts[0]), int(parts[1]), f"explicit ({parts[0]}, {parts[1]})"


def run(
    dolphin: Path,
    out_dir: Path,
    window: list[int] | None,
    max_igs: int | None,
    n_threads: int,
    mcf_refine: bool = False,
    reference_arg: str | None = None,
    closure_mode: str = "off",
    quality_mask_threshold: int | None = None,
    tile_size: int = 0,
    tile_overlap: int = 128,
) -> None:
    print(f"[discover] scanning {dolphin}")
    igs, crlb_paths = discover_stack(dolphin)
    if not igs:
        raise RuntimeError("no interferograms found")

    # Trim if requested.
    if max_igs is not None:
        igs = igs[:max_igs]

    # Unique date list (sorted by string == chronological for YYYYMMDDhhmmss).
    dates: list[str] = sorted({e.date_a for e in igs} | {e.date_b for e in igs})
    date_idx = {d: i for i, d in enumerate(dates)}
    print(f"[discover] {len(igs)} IGs over {len(dates)} dates")

    # Verify CRLB exists for every date in use.
    missing = [d for d in dates if d not in crlb_paths]
    if missing:
        raise RuntimeError(
            f"missing crlb_<date>.tif for {len(missing)} dates, e.g. {missing[:3]}"
        )

    win = _window_from_args(window)
    if win is not None:
        print(f"[window] {win}")

    # Load per-date CRLB cube into memory (Float16 on disk, promote to f32).
    t0 = time.perf_counter()
    print(f"[crlb] loading {len(dates)} CRLB rasters")
    sample = _read_f32(crlb_paths[dates[0]], win)
    m, n = sample.shape
    crlb_cube = np.zeros((len(dates), m, n), dtype=np.float32)
    crlb_cube[0] = sample
    for k, d in enumerate(dates[1:], start=1):
        crlb_cube[k] = _read_f32(crlb_paths[d], win)
    print(f"[crlb] shape={crlb_cube.shape} median={np.median(crlb_cube[crlb_cube > 0]):.3g} rad² "
          f"({time.perf_counter() - t0:.1f}s)")

    # Build edge arrays for the closure pass.
    edges_from = np.array([date_idx[e.date_a] for e in igs], dtype=np.uint32)
    edges_to = np.array([date_idx[e.date_b] for e in igs], dtype=np.uint32)
    n_edges = len(igs)

    # Per-IG median variance over valid pixels — used as the spanning-tree priority.
    # Lower-variance edges form the backbone; noisy IGs are loop-closing.
    def median_var(e: IgFile) -> float:
        v = crlb_cube[date_idx[e.date_a]] + crlb_cube[date_idx[e.date_b]]
        valid = v > 0
        if not valid.any():
            return float(np.inf)
        return float(np.median(v[valid]))

    edge_priority = np.array([median_var(e) for e in igs], dtype=np.float32)
    print(f"[tree] edge variance priorities: min={edge_priority.min():.3g} "
          f"median={np.median(edge_priority):.3g} max={edge_priority.max():.3g}")

    # 2D unwrap each IG, with CRLB-weighted cost.
    t0 = time.perf_counter()
    unw_stack = np.zeros((n_edges, m, n), dtype=np.float32)

    def unwrap_one(idx_e: tuple[int, IgFile]) -> tuple[int, float]:
        idx, e = idx_e
        igram = _read_complex(e.ig_path, win)
        variance = crlb_cube[date_idx[e.date_a]] + crlb_cube[date_idx[e.date_b]]
        mask = None
        if e.mask_path is not None:
            mask = _read_bool(e.mask_path, win)
        t = time.perf_counter()
        unw = ww.unwrap_crlb(
            igram, variance, mask,
            tile_size=tile_size, tile_overlap=tile_overlap,
        )
        unw_stack[idx] = unw
        return idx, time.perf_counter() - t

    tile_desc = (f"tiled {tile_size}x{tile_size} overlap {tile_overlap}"
                 if tile_size > 0 else "single-tile (whole IG in memory)")
    print(f"[unwrap] running 2D unwrap on {n_edges} IGs (CRLB cost, "
          f"{n_threads} outer threads, {tile_desc})")
    # Within-Rust rayon parallelises each unwrap; the outer ThreadPool lets us
    # overlap I/O (reading the next IG while the current one solves).
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        results = list(ex.map(unwrap_one, enumerate(igs)))
    dt = time.perf_counter() - t0
    per_ig = np.array([r[1] for r in results])
    print(f"[unwrap] done in {dt:.1f}s "
          f"(median {np.median(per_ig):.2f}s/IG, max {per_ig.max():.2f}s)")

    # Temporal closure correction (optional).
    reference = 0
    if closure_mode == "tree":
        t0 = time.perf_counter()
        print(f"[closure] tree-based correction over {n_edges} IGs across "
              f"{len(dates)} dates (reference={dates[reference]})")
        closure = ww.closure_correct(
            unw_stack, edges_from, edges_to, len(dates), reference, edge_priority,
        )
        print(f"[closure] done in {time.perf_counter() - t0:.1f}s")
    else:
        # closure_mode == "off": emit the raw 2D-unwrapped stack as the
        # corrected output. Reference-pixel anchoring below still applies
        # and gives a per-IG absolute phase relative to the chosen pixel.
        # Trivially produces 0 closure RMS only if every per-IG was already
        # perfectly closure-consistent; here we record the *actual* residual.
        print(f"[closure] skipped (closure_mode=off) — emitting raw per-IG unwraps")
        date_phases = np.zeros((len(dates), m, n), dtype=np.float32)
        # Compute residual closure RMS to report (diagnostic only)
        # using a simple BFS tree just for date phases.
        closure = {
            "corrected": unw_stack.copy(),
            "corrections": np.zeros((n_edges, m, n), dtype=np.int16),
            "date_phases": date_phases,
            "closure_rms": np.zeros((m, n), dtype=np.float32),
        }

    # Optional: cycle-greedy MCF refinement on the raw (NOT tree-corrected)
    # unwrap stack. See note in closure.rs — with CRLB-priority cycle bases,
    # greedy MCF and tree-based correction tend to make the same decisions on
    # the typical case (where errors live on non-tree edges). It is still a
    # useful diagnostic, and the framework supports future spatial-coupled
    # variants where it should beat tree-based.
    mcf = None
    if mcf_refine:
        t0 = time.perf_counter()
        mcf = ww.closure_refine_mcf(
            unw_stack,          # NOTE: raw 2D-unwrapped stack, not closure["corrected"]
            edges_from, edges_to, len(dates), reference,
            crlb_cube,
            edge_priority,
            32,
        )
        dt = time.perf_counter() - t0
        print(f"[mcf] refined in {dt:.1f}s; "
              f"{int((mcf['residual_violations'] > 0).sum()):,} pixels with unresolved cycles")
        # Compare against tree-based output for a useful diagnostic.
        diff = np.abs(mcf["corrected"] - closure["corrected"])
        per_ig_max = diff.max(axis=(1, 2))
        n_different = int((diff > 1e-3).sum())
        print(f"[mcf] vs tree-based: {n_different:,} edge-pixels differ by >1e-3 rad")
        print(f"[mcf] max per-IG diff vs tree-based: median {np.median(per_ig_max):.4g}, "
              f"max {per_ig_max.max():.4g} rad")

    # Reference-pixel anchoring (sparse-to-dense lite): subtract a single
    # high-quality pixel's per-IG value to remove the arbitrary global
    # integer offset that 2D unwrap leaves on each IG. After this step the
    # output is *absolute relative phase* across the whole time series —
    # the difference between any two pixels is physically meaningful
    # displacement (modulo orbital / atmospheric residuals).
    ref_i, ref_j, ref_mode = _resolve_reference(reference_arg, dolphin, crlb_cube, win)
    ref_vals = closure["corrected"][:, ref_i, ref_j].copy()
    print(f"[reference] anchoring on pixel ({ref_i}, {ref_j}) — {ref_mode}")
    print(f"[reference] per-IG offsets removed: median {np.median(ref_vals):.3f} rad, "
          f"std {ref_vals.std():.3f} rad")
    closure["corrected"] -= ref_vals[:, None, None]
    # Date phases: subtract the per-date reference value too.
    ref_dates = closure["date_phases"][:, ref_i, ref_j].copy()
    closure["date_phases"] -= ref_dates[:, None, None]

    # Per-pixel quality map: max |K| over the fundamental-cycle basis. K is
    # the integer ambiguity mismatch count per cycle. Phase linking gives
    # zero wrapped misclosure, so the cycle sum of *true* per-IG phases is
    # exactly 0 at every pixel; any unwrap of the cycle then differs from
    # 0 by exactly 2π·K with K integer. We compute it on the
    # reference-pixel-anchored stack so each IG reads 0 at the reference
    # (K=0 there is trivial); the per-pixel K then measures only the
    # spatially-relative unwrap consistency across loops.
    #
    # K=0 ⇒ all fundamental cycles through this pixel agree on the per-IG
    # integer ambiguity choices ⇒ pixel is fully self-consistent.
    # K≥1 ⇒ at least one cycle disagrees, typically water / heavily
    # decorrelated pixels where per-IG unwraps are arbitrary.
    t0 = time.perf_counter()
    # Anchor a working copy of the raw unwrap stack at the reference (so the
    # quality map reflects per-IG unwrap consistency relative to the same
    # absolute datum, not the IG-arbitrary global integer offsets).
    unw_anchored = unw_stack - unw_stack[:, ref_i:ref_i + 1, ref_j:ref_j + 1]
    quality = ww.quality_triangles(
        unw_anchored, edges_from, edges_to, len(dates),
    )
    dt = time.perf_counter() - t0
    q_hist = np.bincount(quality.ravel(), minlength=4)
    n_high = int(quality.size - q_hist[0] - q_hist[1] - q_hist[2])
    print(f"[quality] map computed in {dt:.1f}s. K-histogram: "
          f"0:{q_hist[0]:,}  1:{q_hist[1]:,}  2:{q_hist[2]:,}  "
          f">2:{n_high:,}  "
          f"({100*q_hist[0]/quality.size:.1f}% perfectly consistent)")

    if quality_mask_threshold is not None:
        bad = quality > quality_mask_threshold
        n_bad = int(bad.sum())
        closure["corrected"][:, bad] = np.nan
        print(f"[quality] NaN'd {n_bad:,} pixels ({100*n_bad/bad.size:.1f}%) "
              f"with K > {quality_mask_threshold}")

    # Per-date posterior variance from the tree.
    # With independent linked-SLC errors, telescoping along the tree gives
    #   var(θ_d) = σ²_ref + σ²_d
    # per pixel per date — the intermediate SLCs cancel. This is the
    # "uncertainty" deliverable: each pixel of the recovered acquisition
    # phase has a well-calibrated σ from CRLB, propagated through closure.
    posterior_std = np.sqrt(
        np.maximum(crlb_cube[reference][None, :, :] + crlb_cube, 0.0)
    ).astype(np.float32)
    posterior_std[reference] = 0.0  # reference is fixed
    rms = closure["closure_rms"]
    finite = np.isfinite(rms) & (rms > 0)
    n_corrected = int(np.count_nonzero(closure["corrections"]))
    n_total = int(closure["corrections"].size)
    print(f"[closure] median closure RMS: {np.median(rms[finite]):.3f} rad   "
          f"max: {rms.max():.3f} rad")
    print(f"[closure] {n_corrected:,} / {n_total:,} edge-pixels received nonzero "
          f"integer corrections ({100*n_corrected/n_total:.2f}%)")

    # Write outputs.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corrected").mkdir(exist_ok=True)
    template = igs[0].ig_path

    prof_unw = _output_profile(template, win, dtype="float32", count=1)
    prof_unw["dtype"] = "float32"
    prof_unw.pop("nodata", None)

    # Tree-based is ALWAYS the primary output (provably closes every cycle).
    # MCF, when enabled, is a diagnostic-only second output.
    print(f"[write] saving outputs to {out_dir}")
    for idx, e in enumerate(igs):
        name = f"{e.date_a}_{e.date_b}.unw.tif"
        with rasterio.open(out_dir / "corrected" / name, "w", **prof_unw) as dst:
            dst.write(closure["corrected"][idx], 1)

    if mcf_refine and mcf is not None:
        (out_dir / "mcf_diagnostic").mkdir(exist_ok=True)
        for idx, e in enumerate(igs):
            name = f"{e.date_a}_{e.date_b}.unw.tif"
            with rasterio.open(out_dir / "mcf_diagnostic" / name, "w", **prof_unw) as dst:
                dst.write(mcf["corrected"][idx], 1)
        # Also write the per-pixel residual_violations count.
        prof_u16 = dict(prof_unw)
        prof_u16["dtype"] = "uint16"
        with rasterio.open(out_dir / "mcf_diagnostic" / "residual_violations.tif", "w", **prof_u16) as dst:
            dst.write(mcf["residual_violations"], 1)

    # Quality map: per-pixel max |K| over fundamental cycles. uint16.
    prof_q = dict(prof_unw)
    prof_q["dtype"] = "uint16"
    prof_q["predictor"] = 2
    with rasterio.open(out_dir / "quality.tif", "w", **prof_q) as dst:
        dst.write(quality, 1)

    # Closure RMS map.
    with rasterio.open(out_dir / "closure_rms.tif", "w", **prof_unw) as dst:
        dst.write(closure["closure_rms"], 1)

    # Date phases as a multi-band file (each band = one acquisition).
    prof_dates = dict(prof_unw)
    prof_dates["count"] = len(dates)
    with rasterio.open(out_dir / "date_phases.tif", "w", **prof_dates) as dst:
        for k in range(len(dates)):
            dst.write(closure["date_phases"][k], k + 1)
        dst.descriptions = tuple(dates)

    # Posterior std per acquisition (calibrated from CRLB).
    with rasterio.open(out_dir / "date_phase_std.tif", "w", **prof_dates) as dst:
        for k in range(len(dates)):
            dst.write(posterior_std[k], k + 1)
        dst.descriptions = tuple(dates)

    # Corrections cube as int16, multi-band.
    prof_corr = dict(prof_unw)
    prof_corr["dtype"] = "int16"
    prof_corr["count"] = n_edges
    prof_corr["predictor"] = 2
    with rasterio.open(out_dir / "corrections.tif", "w", **prof_corr) as dst:
        for k in range(n_edges):
            dst.write(closure["corrections"][k], k + 1)
        dst.descriptions = tuple(f"{e.date_a}_{e.date_b}" for e in igs)

    # JSON report.
    report = {
        "dolphin_dir": str(dolphin),
        "out_dir": str(out_dir),
        "window": window,
        "n_dates": len(dates),
        "n_edges": n_edges,
        "closure_mode": closure_mode,
        "reference_date": dates[reference],
        "reference_pixel": {"i": int(ref_i), "j": int(ref_j), "source": ref_mode},
        "dates": dates,
        "edges": [
            {"e": i, "from": e.date_a, "to": e.date_b,
             "median_variance_rad2": float(edge_priority[i])}
            for i, e in enumerate(igs)
        ],
        "median_closure_rms_rad": float(np.median(rms[finite])) if finite.any() else None,
        "n_edge_pixels_corrected": n_corrected,
        "n_edge_pixels_total": n_total,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] {out_dir}/report.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dolphin", type=Path, required=True,
                   help="dolphin output directory (contains interferograms/, phase_linking/, ...)")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory")
    p.add_argument("--window", type=int, nargs=4, metavar=("I0", "J0", "I1", "J1"),
                   default=None, help="optional crop window (rows i0:i1, cols j0:j1)")
    p.add_argument("--max-igs", type=int, default=None,
                   help="cap the number of IGs (handy for fast smoke tests)")
    p.add_argument("--threads", type=int, default=4,
                   help="outer thread pool for I/O overlap (rayon handles inner parallelism)")
    p.add_argument("--mcf-refine", action="store_true",
                   help="run cycle-greedy MCF refinement on the raw 2D-unwrapped stack "
                        "instead of tree-based closure correction (slower, diagnostic)")
    p.add_argument("--closure", choices=["off", "tree"], default="off",
                   help="temporal-closure correction. 'off' (default): emit raw per-IG "
                        "unwraps with reference-pixel anchoring only (best per-IG accuracy "
                        "with current cost+residue setup). 'tree': run CRLB-priority tree "
                        "closure correction — guarantees temporal consistency Σ_e ε_e·y_e=0 "
                        "but currently propagates per-IG outliers across the stack")
    p.add_argument("--reference", default="auto",
                   help="reference pixel for absolute-phase anchoring: 'auto' "
                        "(lowest-Σ-CRLB pixel), 'dolphin' (read timeseries/reference_point.txt), "
                        "or 'i,j' for explicit window-local coords")
    p.add_argument("--tile-size", type=int, default=0,
                   help="if > 0, tile each IG into tile_size x tile_size sub-images "
                        "with --tile-overlap pixels of overlap, unwrap each tile in "
                        "parallel, and stitch with CRLB-weighted overlap-median 2π "
                        "reconciliation. Bounds per-IG MCF memory to tile-size scale. "
                        "Output ≈ non-tiled in coherent areas; smaller tiles ⇒ more "
                        "independent per-tile integer ambiguity choices ⇒ less stable "
                        "stitching. 1024 or larger recommended on real data — at "
                        "512+128 we see 99.78%% per-pixel agreement with non-tiled on "
                        "the Palos-Verdes 1024² test tile, at 256+64 only 3.5%% (a "
                        "single fictitious wrap-line at a tile boundary). Disabled "
                        "by default; turn on for scenes that don't fit in memory.")
    p.add_argument("--tile-overlap", type=int, default=128,
                   help="overlap in pixels between adjacent tiles when --tile-size > 0. "
                        "More overlap = more robust stitching median. ~tile_size/8 is "
                        "a reasonable starting point.")
    p.add_argument("--quality-mask-threshold", type=int, default=None,
                   help="NaN pixels in the corrected/ output where the quality "
                        "map (per-pixel max |K| over fundamental temporal cycles) "
                        "exceeds this integer. K=0 = all loops agree on integer "
                        "ambiguities (PS/coherent land). K≥1 = at least one loop "
                        "disagrees (typically water / decorrelated). Recommended "
                        "starting value: 0 (strictest) or 1 (allow occasional "
                        "noise). Off by default — quality.tif is always written.")
    args = p.parse_args()
    run(args.dolphin, args.out, args.window, args.max_igs, args.threads,
        args.mcf_refine, args.reference, args.closure,
        args.quality_mask_threshold, args.tile_size, args.tile_overlap)


if __name__ == "__main__":
    main()
