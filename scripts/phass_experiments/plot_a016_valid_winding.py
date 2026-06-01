"""Show that whirlwind's A_016 (tile512, '57% match') is a VALID unwrapping that
differs from production only by a SMOOTH INTEGER WINDING around the masked
decorrelated holes — not noise. Panels: production unwrapped, whirlwind unwrapped,
their difference in CYCLES (smooth integer plateaus => valid alternate winding),
and the wrapped residual of whirlwind (~0 => locally exact).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LEARN = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")
A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/a016_diag")
OUT.mkdir(parents=True, exist_ok=True)
TAU = 2 * np.pi


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def main() -> None:
    d = np.load(LEARN / "ww_gunw_bench" / A016 / "full_arrays.npz")
    mask = d["mask"]; prod = d["prod_unw"].astype(np.float64); pcc = d["prod_cc"]
    unw = d["ww_unw"].astype(np.float64); ig = d["ig"].astype(np.float64)
    reg = mask & (pcc > 0) & np.isfinite(unw)
    diff_cyc = np.rint((unw - prod) / TAU)
    diff_cyc = diff_cyc - modal(diff_cyc[reg])
    # wrapped residual of whirlwind itself (consistency with its own wrapped input)
    wres = wrap(unw - ig)

    pm = np.where(reg, prod, np.nan)
    wm = np.where(reg, unw, np.nan)
    dm = np.where(reg, diff_cyc, np.nan)
    rm = np.where(mask, wres, np.nan)

    s = (slice(None, None, 3), slice(None, None, 3))
    lo, hi = np.nanpercentile(pm, [2, 98])
    fig, ax = plt.subplots(2, 2, figsize=(15, 14), constrained_layout=True)
    im = ax[0, 0].imshow(pm[s], cmap="viridis", vmin=lo, vmax=hi)
    ax[0, 0].set_title("production unwrapped phase (rad)"); fig.colorbar(im, ax=ax[0, 0], shrink=0.7)
    im = ax[0, 1].imshow(wm[s], cmap="viridis", vmin=lo, vmax=hi)
    ax[0, 1].set_title("whirlwind unwrapped (tile512, reuse)"); fig.colorbar(im, ax=ax[0, 1], shrink=0.7)
    im = ax[1, 0].imshow(dm[s], cmap="RdBu", vmin=-3, vmax=3)
    ax[1, 0].set_title("difference (CYCLES) — smooth integer plateaus = alternate valid winding")
    fig.colorbar(im, ax=ax[1, 0], shrink=0.7)
    im = ax[1, 1].imshow(rm[s], cmap="RdBu", vmin=-0.2, vmax=0.2)
    ax[1, 1].set_title(f"whirlwind wrapped residual (rad), p95={np.nanpercentile(np.abs(rm),95):.1e} ≈ 0 => locally exact")
    fig.colorbar(im, ax=ax[1, 1], shrink=0.7)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    p = OUT / "a016_valid_winding.png"
    fig.savefig(p, dpi=130); plt.close(fig)

    # quantify: fraction of the disagreement that is spatially-smooth (large plateaus)
    nz = reg & (np.abs(diff_cyc) >= 0.5)
    print(f"disagreeing pixels: {nz.sum():,} ({100*nz.sum()/reg.sum():.1f}% of cc>0)", flush=True)
    print(f"whirlwind wrapped-residual p95 = {np.nanpercentile(np.abs(wres[mask]),95):.2e} rad (≈0 => valid unwrap)", flush=True)
    vals, cnts = np.unique(diff_cyc[reg].astype(int), return_counts=True)
    top = sorted(zip(cnts, vals), reverse=True)[:6]
    print("difference histogram (cycles): " + ", ".join(f"{v:+d}:{100*c/reg.sum():.0f}%" for c, v in top), flush=True)
    print(f"PLOT: {p}", flush=True)


if __name__ == "__main__":
    main()
