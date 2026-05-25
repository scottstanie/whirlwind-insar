"""Memory + wall-clock probe: non-tiled vs tiled CRLB unwrap on one IG.

Used to generate the memory-profile plot in docs/figures/. Sized for a
2048² window since that's where tiling actually buys memory; the 1024²
test tile is too small for the per-IG residue/cost/network footprint to
matter compared to the in-process IG read buffers.

Usage:
    uv run --with memory-profiler mprof run --include-children \\
        scripts/bench_tile_memory.py [non-tiled|tiled]
    uv run --with memory-profiler mprof plot --output \\
        docs/figures/fig_tile_memory.png mprofile_*.dat
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

import whirlwind_rs as ww


DOLPHIN = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/"
    "palos-verdes/Palos_Verdes_C13_RO23_SP/e2e_output_20260519/dolphin"
)
IG_NAME = "20251226090359_20251229080043"
# 2048x2048 window covering city + water — typical scene.
WIN = Window(800, 0, 2048, 2048)


def load_ig_and_var() -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(DOLPHIN / "interferograms" / f"{IG_NAME}.int.tif") as src:
        ig = np.nan_to_num(src.read(1, window=WIN).astype(np.complex64), nan=0.0)
    da, db = IG_NAME.split("_")
    var = np.zeros(ig.shape, dtype=np.float32)
    for d in (da, db):
        with rasterio.open(DOLPHIN / "interferograms" / f"crlb_{d}.tif") as src:
            var += np.nan_to_num(src.read(1, window=WIN).astype(np.float32), nan=0.0)
    return ig, var


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "non-tiled"
    if mode not in ("non-tiled", "tiled"):
        raise SystemExit(f"unknown mode {mode!r}; expected 'non-tiled' or 'tiled'")
    print(f"[bench] loading 2048² window for IG {IG_NAME}")
    ig, var = load_ig_and_var()
    print(f"[bench] running {mode} unwrap on {ig.shape}")
    t0 = time.perf_counter()
    if mode == "non-tiled":
        unw = ww.unwrap_crlb(ig, var, None)
    else:
        unw = ww.unwrap_crlb(ig, var, None, tile_size=512, tile_overlap=128)
    print(f"[bench] {mode}: {time.perf_counter() - t0:.2f}s  "
          f"range [{unw.min():.2f}, {unw.max():.2f}]")


if __name__ == "__main__":
    main()
