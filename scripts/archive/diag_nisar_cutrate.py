"""Offline: is the multi-shift gate firing on the NISAR scene? (#52)

Recomputes `coherent_cut_rate` (the Rust gate metric in tile.rs) in numpy from
the SAVED goldstein_0.0 unwrap — no 18-min re-unwrap. If the rate exceeds
COH_CUT_FLOOR=1.5e-3, `unwrap_tiled_robust` would fire the multi-shift re-solve
(~4x cost), which is the leading suspect for the ~18 min runtime.

Run:
    env -u CONDA_PREFIX uv run --with rasterio python scripts/diag_nisar_cutrate.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NISAR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
IG = NISAR / "20251224_20260117.int.looked.tif"
COH = NISAR / "20251224_20260117.int.coh.looked.cleaned.tif"
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/goldstein_ab")
COH_CUT_THR = 0.7
COH_CUT_FLOOR = 1.5e-3
TAU = 2.0 * np.pi


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % TAU - np.pi


def coherent_cut_rate(unw, wrapped, coh, valid):
    """Coherence-weighted count of branch cuts through coh>thr edges, per valid
    pixel — the numpy twin of tile.rs::coherent_cut_rate."""
    nvalid = int(valid.sum())
    total = 0.0
    # right edges (i,j)-(i,j+1)
    cR = np.minimum(coh[:, :-1], coh[:, 1:])
    vR = valid[:, :-1] & valid[:, 1:] & (cR > COH_CUT_THR)
    flowR = np.abs(np.round((unw[:, 1:] - unw[:, :-1] - wrap_to_pi(wrapped[:, 1:] - wrapped[:, :-1])) / TAU))
    total += float(np.where(vR & (flowR > 0), flowR * cR, 0.0).sum())
    # down edges (i,j)-(i+1,j)
    cD = np.minimum(coh[:-1, :], coh[1:, :])
    vD = valid[:-1, :] & valid[1:, :] & (cD > COH_CUT_THR)
    flowD = np.abs(np.round((unw[1:, :] - unw[:-1, :] - wrap_to_pi(wrapped[1:, :] - wrapped[:-1, :])) / TAU))
    total += float(np.where(vD & (flowD > 0), flowD * cD, 0.0).sum())
    return total / max(nvalid, 1)


def main() -> None:
    import rasterio

    with rasterio.open(IG) as s:
        ig = s.read(1).astype(np.complex64)
    with rasterio.open(COH) as s:
        coh = s.read(1).astype(np.float32)
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    coh = np.clip(np.where(mask, coh, 0.0), 0.0, 1.0)
    wrapped = np.angle(np.where(mask, ig, 0)).astype(np.float32)

    unw = np.load(OUT / "goldstein_0.0.npz")["unw"]
    valid = mask & np.isfinite(unw)
    rate = coherent_cut_rate(unw, wrapped, coh, valid)
    fires = rate > COH_CUT_FLOOR
    print(f"shape={ig.shape}  valid={int(valid.sum()):,}")
    print(f"coherent_cut_rate(final goldstein_0.0) = {rate:.3e}")
    print(f"COH_CUT_FLOOR = {COH_CUT_FLOOR:.1e}  ->  gate would {'FIRE (multi-shift, ~4x)' if fires else 'NOT fire (1x)'}")
    print("note: this is the FINAL (post-shift) result; if it fired, the base "
          "rate was >= this. A low value here means the result is clean but does "
          "not by itself prove the base didn't fire.")


if __name__ == "__main__":
    main()
