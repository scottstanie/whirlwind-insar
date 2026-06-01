"""NISAR K-match gate (was the #50 reuse-vs-linear gate; `linear` was removed in
#50). Runs ww.unwrap (Goldstein off) with the process's WHIRLWIND_TILE_SOLVER
and K-matches vs SNAPHU 9x9 on the cc=1 mainland. `WHIRLWIND_TILE_SOLVER` now
accepts only `reuse` (default) / `convex` (research); any other value falls
through to reuse. Invoke once per solver (the flag is read once per process):

    env -u CONDA_PREFIX uv run --with rasterio python scripts/bench_reuse_vs_linear.py
    WHIRLWIND_TILE_SOLVER=convex env -u CONDA_PREFIX uv run --with rasterio python scripts/bench_reuse_vs_linear.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from report_goldstein_ab import IG, COH, SNAPHU_UNW, SNAPHU_CC, NLOOKS, TAU, kmatch  # noqa: E402


def rd(p, dt):
    import rasterio
    with rasterio.open(p) as s:
        return s.read(1).astype(dt)


def main():
    import whirlwind as ww
    solver = os.environ.get("WHIRLWIND_TILE_SOLVER", "reuse (default)")
    ig = rd(IG, np.complex64)
    coh = rd(COH, np.float32)
    su = rd(SNAPHU_UNW, np.float32)
    scc = rd(SNAPHU_CC, np.uint32)
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig = ig.copy(); ig[~mask] = 0
    coh = np.clip(np.where(mask, coh, 0.0), 0.0, 1.0).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    sk = np.round((su - wrapped) / TAU).astype(np.int64)
    main_mask = (scc == 1) & mask

    t0 = time.perf_counter()
    unw, cc = ww.unwrap(ig, coh, NLOOKS, mask=mask, goldstein_alpha=0)
    dt = time.perf_counter() - t0
    common = main_mask & (cc > 0) & np.isfinite(unw)
    km = kmatch(unw, wrapped, sk, common)
    print(f"solver={solver:18}  {dt:5.1f}s  cov={(cc>0).mean()*100:5.2f}%  "
          f"K-match={km['match_pct']:.3f}%  |dK|=1={km['dk1_pct']:.3f}%  |dK|>=2={km['dk2plus_pct']:.3f}%")


if __name__ == "__main__":
    main()
