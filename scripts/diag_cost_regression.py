"""Bit-exactness guard for the cost-env-var purge (#36): unwrap a fixed NISAR
center crop (Goldstein off, default cost) and save (unw, cc). Run BEFORE the
purge and AFTER; the two .npz must be bit-identical (the purge only deletes
default-off branches, so the default path must not change).

    env -u CONDA_PREFIX uv run --with rasterio python scripts/diag_cost_regression.py before
    # ... edit + maturin develop --release ...
    env -u CONDA_PREFIX uv run --with rasterio python scripts/diag_cost_regression.py after
    # then compare (the script prints the verdict when given two existing labels)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NISAR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
IG = NISAR / "20251224_20260117.int.looked.tif"
COH = NISAR / "20251224_20260117.int.coh.looked.cleaned.tif"
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/cost_purge")
N = 2048
NLOOKS = 100.0


def run(label: str) -> None:
    import rasterio
    import whirlwind as ww

    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(IG) as s:
        ig = s.read(1).astype(np.complex64)
    with rasterio.open(COH) as s:
        coh = s.read(1).astype(np.float32)
    m0 = (ig.shape[0] - N) // 2
    c0 = (ig.shape[1] - N) // 2
    ig = np.ascontiguousarray(ig[m0:m0 + N, c0:c0 + N])
    coh = np.ascontiguousarray(coh[m0:m0 + N, c0:c0 + N])
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig = ig.copy()
    ig[~mask] = 0
    coh = np.clip(np.where(mask, coh, 0.0), 0.0, 1.0).astype(np.float32)
    unw, cc = ww.unwrap(ig, coh, NLOOKS, mask=mask, goldstein_alpha=0)
    np.savez(OUT / f"{label}.npz", unw=unw, cc=cc)
    print(f"saved {OUT / f'{label}.npz'}  unw[finite]={np.isfinite(unw).sum():,}  n_cc={int(cc.max())}")


def compare() -> None:
    a = np.load(OUT / "before.npz")
    b = np.load(OUT / "after.npz")
    unw_eq = np.array_equal(a["unw"], b["unw"], equal_nan=True)
    cc_eq = np.array_equal(a["cc"], b["cc"])
    print(f"unw bit-identical: {unw_eq}")
    print(f"cc  bit-identical: {cc_eq}")
    print("VERDICT:", "PASS (default cost path unchanged)" if (unw_eq and cc_eq) else "FAIL (default path changed!)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "before"
    if arg == "compare":
        compare()
    else:
        run(arg)
