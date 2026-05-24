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

import whirlwind_rs as ww


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

def run(
    dolphin: Path,
    out_dir: Path,
    window: list[int] | None,
    max_igs: int | None,
    n_threads: int,
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
        unw = ww.unwrap_crlb(igram, variance, mask)
        unw_stack[idx] = unw
        return idx, time.perf_counter() - t

    print(f"[unwrap] running 2D unwrap on {n_edges} IGs (CRLB cost, {n_threads} threads)")
    # Within-Rust rayon parallelises each unwrap; the outer ThreadPool lets us
    # overlap I/O (reading the next IG while the current one solves).
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        results = list(ex.map(unwrap_one, enumerate(igs)))
    dt = time.perf_counter() - t0
    per_ig = np.array([r[1] for r in results])
    print(f"[unwrap] done in {dt:.1f}s "
          f"(median {np.median(per_ig):.2f}s/IG, max {per_ig.max():.2f}s)")

    # Closure correction.
    t0 = time.perf_counter()
    reference = 0
    print(f"[closure] correcting {n_edges} IGs across {len(dates)} dates "
          f"(reference={dates[reference]})")
    closure = ww.closure_correct(
        unw_stack,
        edges_from,
        edges_to,
        len(dates),
        reference,
        edge_priority,
    )
    dt = time.perf_counter() - t0
    print(f"[closure] done in {dt:.1f}s")

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

    print(f"[write] saving outputs to {out_dir}")
    for idx, e in enumerate(igs):
        name = f"{e.date_a}_{e.date_b}.unw.tif"
        with rasterio.open(out_dir / "corrected" / name, "w", **prof_unw) as dst:
            dst.write(closure["corrected"][idx], 1)

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
        "reference_date": dates[reference],
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
    args = p.parse_args()
    run(args.dolphin, args.out, args.window, args.max_igs, args.threads)


if __name__ == "__main__":
    main()
