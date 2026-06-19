"""Visualize the masked-plane tear (a FIXED limitation of the default solver).

A clean diagonal plane phase = pi*(x+y) (well sampled, zero noise, only benign
boundary residues), masked to a horizontal band, used to be torn into a spurious
2pi step by the DEFAULT (single-tile linear) solver.

ROOT CAUSE (2026-06-09, revised): the capacity-1 GUTTER-STACKING limitation -
not a soft-mask problem. The masked region is filled with phase 0; the mask-blind
residues deposit one +/- charge pair per fringe on the opposite band edges, and
because the band spans the full image width (disconnecting the top/bottom seas),
each pair needs one unit of flow ACROSS the band. The only zero-cost,
integration-invisible crossings are the two image-edge "gutter" columns -
capacity 1 each - so a 3-fringe ramp forced one cut through the band interior.
See scripts/diag_tear_capacity_hypothesis.py for the controlled confirmation
(<=2 fringes never tore; a sea corridor never tore; reuse never tore because its
arcs are multi-unit, not because it is mask-aware).

FIX: the gutter ring (vertical arcs in the first/last residue columns,
horizontal in the first/last rows) is multi-unit - those arcs cost 0 and are
never read by integrate(), so unlimited flow on them is pure gauge. All
crossings now ride the gutter for free and the unwrap is EXACT; quality on the
13-frame NISAR GUNW bench is unchanged (identical per-component match).

Earlier fix attempts that made the linear path mask-aware DID fix this synthetic
but collapsed real masked NISAR frames (D_077 99.5% -> ~6%): mask-blind residues
and whole-grid integration are a matched pair, which is why the fix had to be
capacity-side, not mask-side.

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

    unw_lin = ww.unwrap(
        igram.copy(), corr.copy(), nlooks=1.0, mask=mask, goldstein_alpha=0
    )[0]
    cyc_lin = aligned_cycle_err(unw_lin, phase, mask)

    pmin = float(np.nanmin(np.where(mask, phase, np.nan)))
    pmax = float(np.nanmax(np.where(mask, phase, np.nan)))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))

    # Top row: the inputs / what unwrapping has to do.
    im_w = show(
        axes[0, 0],
        wrapped,
        mask,
        mask_it=False,
        title="WRAPPED phase (what the solver sees)\n"
        "steep ramp -> many diagonal fringes",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    fig.colorbar(im_w, ax=axes[0, 0], shrink=0.6, label="rad")

    im_t = show(
        axes[0, 1],
        phase,
        mask,
        "TRUTH unwrapped (masked to band)\n= add the right # of cycles",
        cmap="viridis",
        vmin=pmin,
        vmax=pmax,
    )
    fig.colorbar(im_t, ax=axes[0, 1], shrink=0.6, label="rad")

    im_l = show(
        axes[0, 2],
        unw_lin,
        mask,
        "DEFAULT (linear) unwrap -> EXACT\n" "(multi-unit gutter ring fix)",
        cmap="viridis",
        vmin=pmin,
        vmax=pmax,
    )
    fig.colorbar(im_l, ax=axes[0, 2], shrink=0.6, label="rad")

    # Bottom row: the error map and the explanation.
    im_e = show(
        axes[1, 0],
        cyc_lin,
        mask,
        "integer-cycle ERROR (linear)\nall zero post-fix",
        cmap="coolwarm",
        vmin=-1.5,
        vmax=1.5,
    )
    fig.colorbar(im_e, ax=axes[1, 0], shrink=0.6, label="cycles")

    # the formerly-torn region in the band
    z = cyc_lin[64:192, 110:190]
    axes[1, 1].imshow(
        z, origin="lower", cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto"
    )
    axes[1, 1].set_title(
        "formerly-torn region\n(pre-fix: 0 | -1 cut line)", fontsize=10.5
    )
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    ax = axes[1, 2]
    ax.axis("off")
    n_tear = int(np.nansum(np.abs(np.nan_to_num(cyc_lin)) > 0))
    n_valid = int(mask.sum())
    ax.text(
        0.0,
        1.0,
        "Why the default solver tore it\n"
        "------------------------------\n"
        "* The full-width band disconnects\n"
        "  the top/bottom 0-fill seas; each\n"
        "  fringe needs one unit of flow\n"
        "  ACROSS the band.\n"
        "* Only the 2 image-edge gutter\n"
        "  columns cross for free - and\n"
        "  each had CAPACITY 1, so fringe\n"
        "  #3 cut through the band interior\n"
        "  (capacity-1 gutter stacking).\n\n"
        "FIX: gutter ring is multi-unit\n"
        "(cost-0, integration-invisible =\n"
        "pure gauge). All crossings ride\n"
        "the gutter; the unwrap is exact.\n\n"
        f"  linear: {n_tear:,}/{n_valid:,} px off "
        f"({100*n_tear/max(n_valid,1):.0f}%).\n"
        "  (pre-fix: 13,696 px / 42% at -1)\n\n"
        "13-frame NISAR bench: identical\n"
        "per-component match - the fix is\n"
        "gauge-only on real frames.",
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle(
        "Masked-plane tear (FIXED): capacity-1 gutter stacking, "
        "resolved by the multi-unit gutter ring",
        fontsize=12.5,
    )
    out = Path(__file__).resolve().parent / "diag_masked_plane_tear.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
