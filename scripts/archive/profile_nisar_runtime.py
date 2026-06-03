"""Profile #52: where does the NISAR `unwrap` time go, and was the ~18 min a
debug-build artifact?

RELEASE build + `WHIRLWIND_TIMING=1` per-stage timing, over a center-crop sweep
(1024 / 2048 / 4096 / full). Each crop runs one `ww.unwrap` (Goldstein off);
the Rust stage timers print to stderr. One heavy unwrap at a time.

Run:
    WHIRLWIND_TIMING=1 env -u CONDA_PREFIX uv run --with rasterio \
        python scripts/profile_nisar_runtime.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

NISAR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
IG = NISAR / "20251224_20260117.int.looked.tif"
COH = NISAR / "20251224_20260117.int.coh.looked.cleaned.tif"
SIZES = [1024, 2048, 4096, "full"]
NLOOKS = 100.0


def center_crop(a: np.ndarray, n):
    if n == "full":
        return a
    m0 = (a.shape[0] - n) // 2
    c0 = (a.shape[1] - n) // 2
    return a[m0:m0 + n, c0:c0 + n]


def main() -> None:
    import rasterio
    import whirlwind as ww

    with rasterio.open(IG) as s:
        ig_full = s.read(1).astype(np.complex64)
    with rasterio.open(COH) as s:
        coh_full = s.read(1).astype(np.float32)

    rows = []
    for sz in SIZES:
        ig = np.ascontiguousarray(center_crop(ig_full, sz))
        coh = np.ascontiguousarray(center_crop(coh_full, sz))
        mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
        ig = ig.copy()
        ig[~mask] = 0
        coh = np.clip(np.where(mask, coh, 0.0), 0.0, 1.0).astype(np.float32)
        mpx = ig.size / 1e6
        print(f"\n========== crop {sz} ({ig.shape[0]}x{ig.shape[1]}, "
              f"{mpx:.1f} Mpx, valid {mask.mean() * 100:.0f}%) ==========", flush=True)
        t0 = time.perf_counter()
        unw, cc = ww.unwrap(ig, coh, NLOOKS, mask=mask, goldstein_alpha=0)
        dt = time.perf_counter() - t0
        print(f"[total] {sz}: {dt:.2f}s  ({mpx / dt:.2f} Mpx/s)  n_cc={int(cc.max())}",
              flush=True)
        rows.append((str(sz), mpx, dt))

    print("\n=== scaling summary (release) ===")
    print(f"{'crop':>6} {'Mpx':>7} {'sec':>8} {'Mpx/s':>8}")
    for sz, mpx, dt in rows:
        print(f"{sz:>6} {mpx:7.1f} {dt:8.2f} {mpx / dt:8.2f}")


if __name__ == "__main__":
    main()
