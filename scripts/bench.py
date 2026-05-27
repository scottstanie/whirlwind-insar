#!/usr/bin/env python3
"""Reproducible cross-library phase-unwrap benchmark.

Compares whirlwind-rs against snaphu and (when installable) kamui's PUMA on a
fixed battery of synthetic + real interferograms. Writes a markdown table to
stdout and to scripts/out/BENCH_RESULTS.md so it can be diffed after every
perf change.

Run with:
    python scripts/bench.py                     # default battery
    python scripts/bench.py --sizes 256,1024    # custom sizes
    python scripts/bench.py --skip palos        # skip a scene by substring
    python scripts/bench.py --quick             # smallest cases only
    python scripts/bench.py --no-snaphu         # disable snaphu
    python scripts/bench.py --no-kamui          # disable PUMA / kamui
    python scripts/bench.py --real              # include real data scenes too
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

warnings.filterwarnings("ignore")

# kamui needs np.float_ which was removed in numpy 2.0; shim before importing.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "int_"):
    np.int_ = np.int64  # type: ignore[attr-defined]

try:
    import whirlwind as ww  # type: ignore
    HAS_WW = True
except Exception as e:  # noqa: BLE001
    print(f"warning: whirlwind not importable: {e}", file=sys.stderr)
    HAS_WW = False

try:
    import snaphu  # type: ignore
    HAS_SNAPHU = True
except Exception:
    HAS_SNAPHU = False

try:
    import kamui  # type: ignore
    HAS_KAMUI = True
except Exception:
    HAS_KAMUI = False

try:
    import rasterio  # type: ignore
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF)
    raw = r.ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def fmt_bytes(b: int) -> str:
    f = float(b)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if f < 1024 or unit == "GiB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{b} B"


@dataclass
class Row:
    scene: str
    library: str
    size: str
    pixels: int
    seconds: float
    peak_rss_bytes: int
    delta_rss_bytes: int
    note: str = ""

    def mpx_per_s(self) -> float:
        return (self.pixels / 1e6) / self.seconds if self.seconds > 0 else 0.0


def call_whirlwind(igram, corr, nlooks, mask):
    return ww.unwrap(igram, corr, float(nlooks), mask)


def call_snaphu(igram, corr, nlooks, mask):
    unw, _ = snaphu.unwrap(igram, corr, nlooks=float(nlooks), cost="smooth", mask=mask)
    return unw.astype(np.float32)


def call_kamui(igram, corr, nlooks, mask):
    phase = np.angle(igram).astype(np.float64)
    weights = corr.astype(np.float64) if corr is not None else None
    unw = kamui.unwrap_dimensional(phase, weights=weights)
    return None if unw is None else unw.astype(np.float32)


LIBS: dict[str, Callable] = {}
if HAS_WW:
    LIBS["whirlwind-unwrap"] = call_whirlwind
if HAS_SNAPHU:
    LIBS["snaphu"] = call_snaphu
if HAS_KAMUI:
    LIBS["kamui (PUMA)"] = call_kamui


def synthetic_diagonal_ramp(size, *, gamma=0.99, nlooks=1, seed=42):
    y, x = np.ogrid[-3:3:size * 1j, -3:3:size * 1j]
    truth = (np.pi * (x + y)).astype(np.float32)
    if gamma >= 0.99 and nlooks == 1:
        igram = np.exp(1j * truth).astype(np.complex64)
        corr = np.full(truth.shape, float(gamma), dtype=np.float32)
    else:
        if not HAS_WW:
            raise RuntimeError("whirlwind needed to synthesize noisy data")
        g = np.full(truth.shape, float(gamma), dtype=np.float32)
        igram, corr = ww.simulate_ifg(truth, g, nlooks=nlooks, seed=seed)
    return igram, corr, truth


def real_palos_pair():
    if not HAS_RASTERIO:
        return None
    d = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes/opera-cslc-s1/Palos_Verdes_Landslides_D071/dolphin/interferograms")
    if not d.exists():
        return None
    pair = next(d.glob("*_*.int.tif"), None)
    if not pair:
        return None
    stem = pair.stem.replace(".int", "")
    cor = d / f"{stem}.int.cor.tif"
    mask = d / f"{stem}.int.mask.tif"
    with rasterio.open(pair) as src:
        igram = src.read(1).astype(np.complex64)
    with rasterio.open(cor) as src:
        corr = np.nan_to_num(src.read(1).astype(np.float32), nan=0.0).clip(0, 0.999)
    m = None
    if mask.exists():
        with rasterio.open(mask) as src:
            m = src.read(1).astype(np.bool_)
    return np.nan_to_num(igram, nan=0.0), corr, m, f"palos_{stem}"


def real_rosamond_pair():
    if not HAS_RASTERIO:
        return None
    d = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/rosamond/Rosamond_C13_RO43_eSM/insar_output/network_output/20250807_20250810")
    if not d.exists():
        return None
    phs = next(d.glob("*PHS*.tif"), None)
    coh = next(d.glob("*COH*.tif"), None)
    if not (phs and coh):
        return None
    with rasterio.open(phs) as src:
        phase = np.nan_to_num(src.read(1)[1000:1000 + 512, 1000:1000 + 512].astype(np.float32), nan=0.0)
    with rasterio.open(coh) as src:
        corr = np.nan_to_num(src.read(1)[1000:1000 + 512, 1000:1000 + 512].astype(np.float32), nan=0.0).clip(0, 0.999)
    igram = np.exp(1j * phase).astype(np.complex64)
    return igram, corr, None, "rosamond_512x512"


def bench_one(scene, igram, corr, nlooks, mask, lib_name, fn) -> Row:
    gc.collect()
    rss_before = peak_rss_bytes()
    t0 = time.perf_counter()
    try:
        out = fn(igram, corr, nlooks, mask)
        note = "" if out is not None else "no output"
    except Exception as e:  # noqa: BLE001
        return Row(scene=scene, library=lib_name, size=f"{igram.shape[0]}x{igram.shape[1]}",
                   pixels=int(igram.size), seconds=float("nan"),
                   peak_rss_bytes=0, delta_rss_bytes=0, note=f"FAILED: {e}")
    dt = time.perf_counter() - t0
    rss_after = peak_rss_bytes()
    return Row(
        scene=scene, library=lib_name,
        size=f"{igram.shape[0]}x{igram.shape[1]}",
        pixels=int(igram.size),
        seconds=dt,
        peak_rss_bytes=rss_after,
        delta_rss_bytes=max(0, rss_after - rss_before),
        note=note,
    )


def render_markdown(rows: list[Row]) -> str:
    by_scene: dict[str, list[Row]] = {}
    for r in rows:
        by_scene.setdefault(r.scene, []).append(r)
    lib_order = list(LIBS.keys()) or ["whirlwind-unwrap"]

    lines: list[str] = []
    lines.append("# whirlwind-rs cross-library benchmark\n")
    lines.append("Auto-generated by `scripts/bench.py`. Re-run after any perf change.\n")
    lines.append(f"Host: {sys.platform}, Python "
                 f"{sys.version_info.major}.{sys.version_info.minor}, libraries: "
                 + ", ".join(lib_order) + "\n")

    lines.append("## Wall-time (seconds)\n")
    header = "| Scene | size | Mpx | " + " | ".join(lib_order) + " | snaphu / ww |"
    sep = "|" + "|".join(["---"] * (3 + len(lib_order) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for scene, scene_rows in by_scene.items():
        first = scene_rows[0]
        row_by_lib = {r.library: r for r in scene_rows}
        cells = [scene, first.size, f"{first.pixels / 1e6:.2f}"]
        snaphu_s, ww_s = None, None
        for lib in lib_order:
            r = row_by_lib.get(lib)
            if r is None or not np.isfinite(r.seconds):
                cells.append("n/a")
            else:
                cells.append(f"{r.seconds:.3f}")
                if lib == "snaphu":
                    snaphu_s = r.seconds
                if lib == "whirlwind-unwrap":
                    ww_s = r.seconds
        cells.append(f"{snaphu_s / ww_s:.2f}x" if (snaphu_s and ww_s) else "n/a")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Throughput (Mpx/s)\n")
    header = "| Scene | size | " + " | ".join(lib_order) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(lib_order))) + "|"
    lines.append(header)
    lines.append(sep)
    for scene, scene_rows in by_scene.items():
        first = scene_rows[0]
        row_by_lib = {r.library: r for r in scene_rows}
        cells = [scene, first.size]
        for lib in lib_order:
            r = row_by_lib.get(lib)
            cells.append(f"{r.mpx_per_s():.2f}" if (r and np.isfinite(r.seconds)) else "n/a")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Peak RSS delta (working-set per call)\n")
    header = "| Scene | size | " + " | ".join(lib_order) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(lib_order))) + "|"
    lines.append(header)
    lines.append(sep)
    for scene, scene_rows in by_scene.items():
        first = scene_rows[0]
        row_by_lib = {r.library: r for r in scene_rows}
        cells = [scene, first.size]
        for lib in lib_order:
            r = row_by_lib.get(lib)
            cells.append(fmt_bytes(r.delta_rss_bytes) if r else "n/a")
        lines.append("| " + " | ".join(cells) + " |")

    failures = [r for r in rows if r.note and "FAIL" in r.note]
    if failures:
        lines.append("\n## Failures / notes\n")
        for r in failures:
            lines.append(f"- **{r.scene}** / {r.library}: {r.note}")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="256,512,1024,2048",
                   help="comma-separated synthetic image sizes")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip", action="append", default=[])
    p.add_argument("--no-snaphu", action="store_true")
    p.add_argument("--no-kamui", action="store_true")
    p.add_argument("--no-ww", action="store_true")
    p.add_argument("--real", action="store_true")
    p.add_argument("--very-noisy-max-size", type=int, default=1024,
                   help="cap on size for γ=0.3 (otherwise minutes for some libs)")
    p.add_argument("--nlooks", type=int, default=0,
                   help="If > 0, override the per-scene multilook count and use "
                   "this value everywhere. Default 0 = per-scene defaults: clean "
                   "ramp uses 1 (single-look matches γ=0.99 by construction), "
                   "γ=0.7 uses 10, γ=0.3 uses 4. Note: snaphu's smooth-cost "
                   "init time depends on both γ and nlooks in non-obvious ways, "
                   "so snaphu's clean runs can take *longer* than its noisy "
                   "runs — that's a real snaphu artifact, not a bench bug.")
    args = p.parse_args()

    libs = dict(LIBS)
    if args.no_snaphu:
        libs.pop("snaphu", None)
    if args.no_kamui:
        libs.pop("kamui (PUMA)", None)
    if args.no_ww:
        libs.pop("whirlwind-unwrap", None)
    LIBS.clear()
    LIBS.update(libs)

    sizes = [int(s) for s in args.sizes.split(",") if s]
    if args.quick:
        sizes = [256]

    rows: list[Row] = []

    def add_scene(name, igram, corr, nlooks, mask):
        if any(s in name for s in args.skip):
            return
        print(f"\n=== {name} ({igram.shape}, {len(libs)} libs) ===")
        for lib_name, fn in libs.items():
            print(f"  {lib_name:14s} ...", flush=True, end="")
            r = bench_one(name, igram, corr, nlooks, mask, lib_name, fn)
            rows.append(r)
            tag = (f"{r.seconds:7.3f}s  {r.mpx_per_s():5.2f} Mpx/s"
                   if np.isfinite(r.seconds) else r.note)
            print(f" {tag}")

    def nlooks_for(default):
        return args.nlooks if args.nlooks > 0 else default

    for size in sizes:
        nl = nlooks_for(1)
        ig, co, _ = synthetic_diagonal_ramp(size, gamma=0.99, nlooks=nl)
        add_scene(f"clean ramp {size}x{size}", ig, co, float(nl), None)

    for size in sizes:
        nl = nlooks_for(10)
        ig, co, _ = synthetic_diagonal_ramp(size, gamma=0.7, nlooks=nl)
        add_scene(f"noisy ramp γ=0.7 {size}x{size}", ig, co, float(nl), None)

    for size in sizes:
        if size > args.very_noisy_max_size:
            continue
        nl = nlooks_for(4)
        ig, co, _ = synthetic_diagonal_ramp(size, gamma=0.3, nlooks=nl)
        add_scene(f"very noisy ramp γ=0.3 {size}x{size}", ig, co, float(nl), None)

    if args.real:
        ros = real_rosamond_pair()
        if ros:
            ig, co, m, name = ros
            add_scene(name, ig, co, 4.0, m)
        palos = real_palos_pair()
        if palos:
            ig, co, m, name = palos
            add_scene(name, ig, co, 15.0, m)

    md = render_markdown(rows)
    out_md = OUT_DIR / "BENCH_RESULTS.md"
    out_json = OUT_DIR / "BENCH_RESULTS.json"
    out_md.write_text(md)
    out_json.write_text(json.dumps([asdict(r) for r in rows], indent=2))

    print("\n" + md)
    print(f"\nWrote {out_md} and {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
