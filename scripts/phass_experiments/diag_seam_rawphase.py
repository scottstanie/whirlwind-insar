"""What is whirlwind reacting to at the col-4032 sliver? Show, over the strip
rows (956..1375), per-column-edge stats around cols 4030..4035:
  - mean |raw wrapped horizontal gradient| (the true dpsi the MCF sees)
  - mean 7x7-box-smoothed |gradient| (what the cost actually uses as alpha)
  - mean coherence
The hypothesis: a thin 2px feature has large RAW gradient at its two edges
(cost gamma*max(0,pi-alpha) ~ 0 -> cheap to cut), which a statistical cost would
judge as noise given the coherence/looks. If smoothing collapses it, that is the
offset-erasure mechanism.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
TAU = float(2 * np.pi)


def wrap(x):
    return np.angle(np.exp(1j * x))


def main():
    ig = rasterio.open(N / "20251224_20260117.int.looked.tif").read(1).astype(np.complex64)
    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    psi = np.angle(ig).astype(np.float32)

    rows = slice(956, 1376)
    # raw horizontal gradient at edge between col j and j+1: wrap(psi[:,j+1]-psi[:,j])
    dx_raw = wrap(psi[:, 1:] - psi[:, :-1])  # edge j is between col j,j+1
    # 7x7 box-smoothed gradient (as cost path does, via box_filter_2d on the wrapped grad)
    dx_sm = uniform_filter(dx_raw, size=7, mode="nearest")

    print("col-edge j|j+1 :  mean|raw dpsi|   mean|smoothed|   mean coh(col j)", flush=True)
    for j in range(4028, 4037):
        rr = dx_raw[rows, j]
        ss = dx_sm[rows, j]
        cc = coh[rows, j]
        mm = mask[rows, j]
        print(f"   {j:4d}|{j+1:<4d}    {np.mean(np.abs(rr[mm])):7.3f}        "
              f"{np.mean(np.abs(ss[mm])):7.3f}        {np.mean(cc[mm]):.3f}", flush=True)

    # also: is there a thin amplitude/coherence feature? show coh profile across cols
    print("\ncoherence profile across cols (mean over strip rows):", flush=True)
    for j in range(4028, 4037):
        cc = coh[rows, j]; mm = mask[rows, j]
        print(f"   col {j}: coh={np.mean(cc[mm]):.3f}  validfrac={mm.mean():.2f}", flush=True)

    # raw phase values: do cols 4032-4033 sit ~1 cycle off in the WRAPPED phase
    # relative to neighbors (i.e., is the data itself suggesting a jump)?
    print("\nmean wrapped phase per col over strip rows (rad):", flush=True)
    for j in range(4029, 4037):
        v = psi[rows, j][mask[rows, j]]
        print(f"   col {j}: <psi>={np.mean(v):+.3f}  std={np.std(v):.3f}", flush=True)


if __name__ == "__main__":
    main()
