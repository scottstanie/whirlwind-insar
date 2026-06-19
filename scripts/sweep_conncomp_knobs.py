"""Demonstrate whirlwind's connected-component (conncomp) tuning knobs on one
NISAR frame (default A_018, which has a large decorrelated area peppered with
small, low-coherence components).

It runs ``ww.unwrap`` under a handful of knob settings and renders a panel grid
plus a summary table, so a new user can SEE what each knob does:

  cost_threshold       raw cost units; an edge is a component boundary when its
                       statistical cost <= threshold. Higher => more boundaries
                       => smaller, safer components. Prefer the physical knobs.
  conncomp_sigma       sets cost_threshold from a Gaussian-equivalent noise
                       level (~3.5 == the default 50). Higher => stricter.
  conncomp_cycle_prob  sets it from a target per-edge one-cycle probability
                       (~2.4e-4 == the default). Lower => stricter.
  min_size_px          drop components smaller than this many pixels.
  max_ncomps           keep only the N largest components.

Usage (base miniforge3 env): python scripts/sweep_conncomp_knobs.py [FRAME=A_018]
Output: docs/figures/conncomp_knobs_<frame>.png + a printed table.
One HEAVY unwrap per config (sequential; ~5 configs).
"""

import glob
import sys

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase

import whirlwind as ww

H5DIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw"
REPO_FIG = "docs/figures"

# (panel title, ww.unwrap kwargs beyond the inputs). bridge defaults on; it does
# not affect labelling, only the phase level.
CONFIGS = [
    ("default\n(cost_threshold=50, min_size_px=100)", {}),
    ("looser boundaries\n(conncomp_sigma=2.5)", {"conncomp_sigma": 2.5}),
    ("stricter boundaries\n(conncomp_sigma=4.5)", {"conncomp_sigma": 4.5}),
    (
        "looser via cycle_prob\n(conncomp_cycle_prob=1e-2)",
        {"conncomp_cycle_prob": 1e-2},
    ),
    ("drop small comps\n(min_size_px=2000)", {"min_size_px": 2000}),
    ("keep 5 largest\n(max_ncomps=5)", {"max_ncomps": 5}),
]


def cc_stats(cc, valid):
    cc = np.asarray(cc)
    labs, cnts = np.unique(cc[(cc > 0) & valid], return_counts=True)
    return {
        "ncc": int(labs.size),
        "labeled_frac": float((cc[valid] > 0).mean()) if valid.any() else 0.0,
        "smallest": int(cnts.min()) if cnts.size else 0,
        "n_small_2k": int((cnts < 2000).sum()),
    }


def show_labels(ax, cc, valid, title):
    """Render conncomp labels: background grey, each label a cycled tab20 color."""
    arr = np.asarray(cc).astype(float)
    arr = np.where((arr > 0) & valid, ((arr - 1) % 20) + 1, np.nan)
    ax.imshow(
        np.where(valid, 0.0, np.nan), cmap="gray", vmin=-1, vmax=1
    )  # valid extent
    ax.imshow(arr, cmap="tab20", vmin=0, vmax=20, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    frame = sys.argv[1] if len(sys.argv) > 1 else "A_018"
    h5 = glob.glob(f"{H5DIR}/*_{frame}_*.h5")[0]
    with h5py.File(h5, "r") as h:
        pol, prod_unw, coh, prod_cc, mask_arr = gunw_layers(h)
    mask = (
        water_only_mask(mask_arr, prod_unw.shape)
        & np.isfinite(prod_unw)
        & np.isfinite(coh)
    )
    wrapped = np.where(mask, wrap_phase(prod_unw), 0.0).astype(np.float32)
    ig = np.exp(1j * wrapped).astype(np.complex64)
    coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)
    valid = mask

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    ax = axes.ravel()
    # Panel 0: coherence (where the decorrelated area is).
    im = ax[0].imshow(np.where(valid, coh_in, np.nan), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("coherence", fontsize=10)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    fig.colorbar(im, ax=ax[0], shrink=0.7)
    # Panel 1: production conncomps (reference).
    show_labels(
        ax[1],
        prod_cc,
        valid,
        f"NISAR GUNW conncomps\n(n={int(np.unique(prod_cc[prod_cc>0]).size)})",
    )

    rows = []
    for k, (title, kw) in enumerate(CONFIGS):
        _, cc = ww.unwrap(ig, coh_in, 16.0, mask, **kw)
        st = cc_stats(cc, valid)
        rows.append((title.replace("\n", " "), kw, st))
        show_labels(
            ax[2 + k],
            cc,
            valid,
            f"{title}\nncc={st['ncc']}, labeled={st['labeled_frac']*100:.0f}%",
        )
        print(
            f"{title.splitlines()[0]:22s} {str(kw):55s} ncc={st['ncc']:4d} "
            f"labeled={st['labeled_frac']*100:5.1f}% small(<2k)={st['n_small_2k']:3d} "
            f"smallest={st['smallest']}",
            flush=True,
        )

    fig.suptitle(
        f"{frame}: whirlwind connected-component tuning knobs (pol={pol}, nlooks=16)",
        fontsize=14,
    )
    out = f"{REPO_FIG}/conncomp_knobs_{frame}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure -> {out}", flush=True)
    # Also drop a copy next to the other sweep outputs for convenience.
    fig_alt = f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final/conncomp_knobs_{frame}.png"
    import shutil

    shutil.copy(out, fig_alt)
    print(f"copy   -> {fig_alt}", flush=True)


if __name__ == "__main__":
    main()
