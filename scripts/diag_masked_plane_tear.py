"""Visualize the masked-plane tear (a known limitation of the default solver).

A clean diagonal plane phase = pi*(x+y) (well sampled, zero noise, only benign
boundary residues), masked to a horizontal band, is torn into a spurious 2pi step
by the DEFAULT (single-tile linear) solver but unwrapped exactly by the reuse
solver.

ROOT CAUSE (2026-06-09): the masked region is filled with phase 0, and the linear
path's ww-orig "soft mask" pipeline (mask-blind residues + free masked arcs +
whole-image integration) invents same-sign charge along the ramp/zero boundary;
balancing it routes a 2pi cut across the band. Making the linear path mask-aware
fixes this synthetic case but COLLAPSES real masked NISAR frames (D_077: 99.5% ->
~6%), because a 47%-masked frame genuinely needs to integrate through the masked
fill - so the soft-mask pipeline is load-bearing and there is no localized fix.
The reuse solver (fully mask-aware network) is the working alternative.

The figure shows the wrapped fringes, the truth, the torn linear result, and the
integer-cycle error map (an entire sub-region of the band one full cycle low).

Saves a PNG next to this script and prints its absolute path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import whirlwind as ww

TAU = 2.0 * np.pi


def build():
    y, x = np.ogrid[-3:3:256j, -3:3:256j]
    phase = (np.pi * (x + y)).astype(np.float32)
    igram = np.exp(1j * phase).astype(np.complex64)
    corr = np.ones(igram.shape, np.float32) * 0.999
    mask = np.zeros(igram.shape, bool)
    mask[64:-64] = True  # valid = middle horizontal band; top/bottom masked
    igram[~mask] = np.nan + 1j * np.nan
    corr[~mask] = 0.0
    return igram, corr, mask, phase


def aligned_cycle_err(unw, phase, mask):
    d = unw - phase
    off = TAU * round(float(np.mean(d[mask])) / TAU)
    cyc = np.rint((d - off) / TAU)
    return np.where(mask, cyc, np.nan)


def show(ax, arr, mask, title, mask_it=True, **kw):
    a = np.where(mask, arr, np.nan) if mask_it else arr
    im = ax.imshow(a, origin="lower", **kw)
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([])
    ax.set_yticks([])
    # outline the valid band
    ax.axhline(64, color="k", lw=0.6, ls=":")
    ax.axhline(192, color="k", lw=0.6, ls=":")
    return im


def main():
    igram, corr, mask, phase = build()
    wrapped = np.angle(np.exp(1j * phase)).astype(np.float32)

    unw_lin = ww.unwrap(igram.copy(), corr.copy(), nlooks=1.0, mask=mask,
                        goldstein_alpha=0)[0]
    cyc_lin = aligned_cycle_err(unw_lin, phase, mask)

    pmin = float(np.nanmin(np.where(mask, phase, np.nan)))
    pmax = float(np.nanmax(np.where(mask, phase, np.nan)))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))

    # Top row: the inputs / what unwrapping has to do.
    im_w = show(axes[0, 0], wrapped, mask, mask_it=False,
                title="WRAPPED phase (what the solver sees)\n"
                      "steep ramp -> many diagonal fringes",
                cmap="twilight", vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im_w, ax=axes[0, 0], shrink=0.6, label="rad")

    im_t = show(axes[0, 1], phase, mask,
                "TRUTH unwrapped (masked to band)\n= add the right # of cycles",
                cmap="viridis", vmin=pmin, vmax=pmax)
    fig.colorbar(im_t, ax=axes[0, 1], shrink=0.6, label="rad")

    im_l = show(axes[0, 2], unw_lin, mask,
                "DEFAULT (linear) unwrap -> TORN\n"
                "lower triangle is one cycle low",
                cmap="viridis", vmin=pmin, vmax=pmax)
    fig.colorbar(im_l, ax=axes[0, 2], shrink=0.6, label="rad")

    # Bottom row: the error and the explanation.
    im_e = show(axes[1, 0], cyc_lin, mask,
                "integer-cycle ERROR (linear)\nblue = -1 cycle (the tear)",
                cmap="coolwarm", vmin=-1.5, vmax=1.5)
    fig.colorbar(im_e, ax=axes[1, 0], shrink=0.6, label="cycles")

    # zoom on the tear boundary in the band
    z = cyc_lin[64:192, 110:190]
    axes[1, 1].imshow(z, origin="lower", cmap="coolwarm", vmin=-1.5, vmax=1.5,
                      aspect="auto")
    axes[1, 1].set_title("zoom on the cut line\n(0 | -1 along a diagonal)",
                         fontsize=10.5)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    ax = axes[1, 2]
    ax.axis("off")
    n_tear = int(np.nansum(cyc_lin == -1))
    n_valid = int(mask.sum())
    ax.text(
        0.0, 1.0,
        "Why the default solver tears it\n"
        "-------------------------------\n"
        "* No noise, well sampled (~85px/\n"
        "  fringe), only benign boundary\n"
        "  residues. Full plane: exact.\n"
        "* But the masked band is filled\n"
        "  with phase 0, and the linear\n"
        "  path's soft-mask pipeline (mask-\n"
        "  blind residues + whole-grid\n"
        "  integrate) invents same-sign\n"
        "  charge along the ramp/zero edge\n"
        "  and routes a 2pi cut across the\n"
        "  band.\n\n"
        f"  linear: {n_tear:,}/{n_valid:,} px "
        f"({100*n_tear/n_valid:.0f}%) at -1.\n"
        "  reuse solver: 0 px off (exact).\n\n"
        "Making linear mask-aware fixes\n"
        "this but collapses real masked\n"
        "frames (D_077 99.5%->~6%), so the\n"
        "soft-mask path stays the default.\n"
        "One region -> bridge is a no-op.",
        va="top", ha="left", fontsize=10, family="monospace",
    )

    fig.suptitle(
        "Masked-plane tear: a straight mask edge across steep diagonal fringes "
        "trips the linear solver",
        fontsize=12.5,
    )
    out = Path(__file__).resolve().parent / "diag_masked_plane_tear.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
