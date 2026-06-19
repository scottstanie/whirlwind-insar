"""Visual comparison of unwrapped results: production vs whirlwind vs isce2-ICU,
per NISAR GUNW frame. Renders a 2x3 panel (coherence, production, whirlwind,
ICU, and ambiguity-diff maps for whirlwind & ICU vs production) so an interesting
score-table row can always be eyeballed. Also caches whirlwind (unw, cc) + the
input arrays to npz, so downstream work (bridging prototype) reuses the single
heavy unwrap instead of recomputing.

Runs whirlwind once per frame (heavy) -- SEQUENTIAL, one at a time. ICU phase is
read from the already-written icu_scratch/<frame>.unw rasters (no recompute).

Usage (base miniforge3 env -- has whirlwind + matplotlib):
    /Users/staniewi/miniforge3/bin/python scripts/plot_unwrap_compare.py [A_025 A_016 D_074 D_077]
"""

import sys
import glob
import os

import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase, percomp_match

import whirlwind as ww

TWOPI = 2.0 * np.pi
ICU_DIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/icu_scratch"
OUT_DIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_sweep"
CACHE_DIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/bridge_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

frames = sys.argv[1:] or ["A_025", "A_016", "D_074", "D_077"]


def global_align(u, prod, valid):
    """Remove a single global integer-2pi offset vs production (unobservable)."""
    g = np.rint(np.nanmedian(np.rint((u[valid] - prod[valid]) / TWOPI)))
    return u - TWOPI * g


def amb_diff(u, prod, valid):
    """Per-pixel integer ambiguity diff vs production, global-median centered."""
    out = np.full(prod.shape, np.nan, np.float32)
    out[valid] = np.rint((u[valid] - prod[valid]) / TWOPI)
    out -= np.rint(np.nanmedian(out[valid]))
    return out


for frame in frames:
    h5path = glob.glob(
        f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5"
    )[0]
    with h5py.File(h5path, "r") as h:
        pol, prod_unw, coh, prod_cc, mask_arr = gunw_layers(h)
    mask = (
        water_only_mask(mask_arr, prod_unw.shape)
        & np.isfinite(prod_unw)
        & np.isfinite(coh)
    )
    wrapped = np.where(mask, wrap_phase(prod_unw), 0.0).astype(np.float32)
    ig = np.exp(1j * wrapped).astype(np.complex64)
    coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)
    length, width = prod_unw.shape

    # Whirlwind (single heavy unwrap) + cache.
    unw, cc = ww.unwrap(ig, coh_in, 16.0, mask)
    unw = np.asarray(unw, np.float32)
    cc = np.asarray(cc).astype(np.int32)
    np.savez_compressed(
        f"{CACHE_DIR}/{frame}.npz",
        unw=unw,
        cc=cc,
        coh=coh_in,
        prod=prod_unw.astype(np.float32),
        prod_cc=prod_cc.astype(np.int32),
        mask=mask,
        wrapped=wrapped,
    )

    # ICU phase from the already-written raster (BIL 2-band: band1 = phase).
    icu_path = f"{ICU_DIR}/{frame}.unw"
    if os.path.exists(icu_path):
        raw = np.fromfile(icu_path, dtype=np.float32).reshape(length, 2, width)
        icu_amp, icu_unw = raw[:, 0, :], raw[:, 1, :]
        icu_done = icu_amp != 0.0
    else:
        icu_unw = np.full(prod_unw.shape, np.nan, np.float32)
        icu_done = np.zeros(prod_unw.shape, bool)

    v_ww = mask & np.isfinite(unw)
    v_icu = mask & np.isfinite(icu_unw) & icu_done
    pc_ww = percomp_match(unw, prod_unw, wrapped, prod_cc, v_ww)
    pc_icu = percomp_match(icu_unw, prod_unw, wrapped, prod_cc, v_icu)
    cov_icu = v_icu.sum() / max(mask.sum(), 1)

    # Aligned phase maps for display.
    ww_a = global_align(unw, prod_unw, v_ww)
    icu_a = global_align(icu_unw, prod_unw, v_icu)
    pu = np.where(mask, prod_unw, np.nan)
    vlo, vhi = np.nanpercentile(pu, [2, 98])

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    def show(a, arr, title, valid=None, **kw):
        d = np.where(valid if valid is not None else mask, arr, np.nan)
        im = a.imshow(d, **kw)
        a.set_title(title, fontsize=11)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)

    show(ax[0, 0], coh_in, "coherence", cmap="gray", vmin=0, vmax=1)
    show(
        ax[0, 1],
        pu,
        f"production unwrap ({int(prod_cc.max())} cc)",
        valid=mask,
        cmap="viridis",
        vmin=vlo,
        vmax=vhi,
    )
    show(
        ax[0, 2],
        ww_a,
        f"whirlwind  per-comp={pc_ww*100:.1f}%",
        valid=v_ww,
        cmap="viridis",
        vmin=vlo,
        vmax=vhi,
    )
    show(
        ax[1, 0],
        icu_a,
        f"isce2 ICU  per-comp={pc_icu*100:.1f}%  cov={cov_icu*100:.0f}%",
        valid=v_icu,
        cmap="viridis",
        vmin=vlo,
        vmax=vhi,
    )
    show(
        ax[1, 1],
        amb_diff(unw, prod_unw, v_ww),
        "whirlwind ambiguity diff (cyc)",
        valid=v_ww,
        cmap="RdBu",
        vmin=-2,
        vmax=2,
    )
    show(
        ax[1, 2],
        amb_diff(icu_unw, prod_unw, v_icu),
        "ICU ambiguity diff (cyc)",
        valid=v_icu,
        cmap="RdBu",
        vmin=-2,
        vmax=2,
    )

    fig.suptitle(
        f"{frame}: production vs whirlwind ({pc_ww*100:.1f}%) vs isce2-ICU ({pc_icu*100:.1f}%)  -- shape {prod_unw.shape}",
        fontsize=13,
    )
    fig.tight_layout()
    out = f"{OUT_DIR}/{frame}_icu_vs_ww.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(
        f"{frame}: ww={pc_ww*100:.1f}%  icu={pc_icu*100:.1f}% (cov {cov_icu*100:.0f}%)  -> {out}",
        flush=True,
    )
