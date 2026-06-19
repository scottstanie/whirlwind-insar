"""Test the CAPACITY hypothesis for the masked-plane tear.

Claim under test: the tear in diag_masked_plane_tear.py is the documented
capacity-1 boundary-stacking limitation, not a fundamental soft-mask problem.

Mechanism: the valid band spans the FULL image width, so the top/bottom seas
are disconnected and every fringe termination charge on the top band edge must
send one unit of flow ACROSS the band to its partner on the bottom edge. The
only integration-invisible crossings are the two image-boundary "gutter"
columns (their vertical residue arcs are never read by integrate() and carry
cost 0), but each gutter arc has CAPACITY 1 in the linear solver - so with C
fringe crossings, max(0, C - 2) units are forced through the band interior,
each one a 2pi tear line. Reuse mode doesn't tear because used arcs become
free multi-unit (all C units share one gutter).

Predictions (3-cycle band ramp tears at baseline):
  A. <= 2 crossings  -> NO tear   (2 gutters suffice)
  B. 3 crossings but band NOT spanning full width -> NO tear
     (sea corridor connects top/bottom seas at zero cost)
  C. more crossings -> tear GROWS roughly as (C - 2) cut lines
  D. reuse solver   -> never tears (known, sanity)

Usage: python scripts/diag_tear_capacity_hypothesis.py
Saves diag_tear_capacity_hypothesis.png next to this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import whirlwind as ww
from whirlwind import unwrap_reuse

TAU = 2.0 * np.pi


def build(cycles: float, corridor: bool):
    """Band-masked diagonal ramp with `cycles` full fringes along the band.

    corridor=True also masks a vertical strip so the band does NOT span the
    full width - the top and bottom seas become connected.
    """
    n = 256
    y, x = np.ogrid[-0.5 : 0.5 : complex(n), -0.5 : 0.5 : complex(n)]
    phase = (TAU * cycles * (x + y)).astype(np.float32)
    igram = np.exp(1j * phase).astype(np.complex64)
    corr = np.ones(igram.shape, np.float32) * 0.999
    mask = np.zeros(igram.shape, bool)
    mask[64:-64] = True
    if corridor:
        mask[:, 200:216] = False  # cut the band -> sea wraps around its end
    igram[~mask] = 0
    corr[~mask] = 0.0
    return igram, corr, mask, phase


def cycle_err(unw, phase, mask):
    """Integer-cycle error, aligned PER connected valid region (the relative
    2pi offset between disconnected regions is unobservable - gauge, not
    error). NaN outside the mask."""
    from whirlwind import label_components

    d = unw - phase
    labels, n = label_components(mask)
    labels = np.asarray(labels)
    cyc = np.full(d.shape, np.nan, np.float64)
    for lbl in range(1, n + 1):
        sel = labels == lbl
        off = TAU * np.round(float(np.mean(d[sel])) / TAU)
        cyc[sel] = np.rint((d[sel] - off) / TAU)
    return cyc


def run_case(label, cycles, corridor=False, solver="linear"):
    igram, corr, mask, phase = build(cycles, corridor)
    if solver == "reuse":
        unw = np.asarray(unwrap_reuse(igram, corr, 1.0, mask), np.float32)
    else:
        unw = ww.unwrap(igram, corr, nlooks=1.0, mask=mask, goldstein_alpha=0)[0]
    cyc = cycle_err(unw, phase, mask)
    bad = int(((cyc != 0) & np.isfinite(cyc)).sum())
    tot = int(mask.sum())
    print(f"{label:55s} {bad:7,}/{tot:,} px off ({100 * bad / tot:5.1f}%)")
    return cyc, bad, tot


def main():
    cases = [
        ("A: 1-cycle ramp, full-width band (predict NO tear)", 1.0, False, "linear"),
        ("A: 2-cycle ramp, full-width band (predict NO tear)", 2.0, False, "linear"),
        ("baseline: 3-cycle ramp, full-width band (TEARS)", 3.0, False, "linear"),
        ("B: 3-cycle ramp + sea corridor (predict NO tear)", 3.0, True, "linear"),
        ("C: 5-cycle ramp, full-width band (predict MORE tear)", 5.0, False, "linear"),
        ("D: 3-cycle ramp, reuse solver (predict NO tear)", 3.0, False, "reuse"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (label, cycles, corridor, solver) in zip(axes.flat, cases):
        cyc, bad, tot = run_case(label, cycles, corridor, solver)
        ax.imshow(cyc, origin="lower", cmap="coolwarm", vmin=-2.5, vmax=2.5)
        ax.set_title(f"{label}\n{bad:,} px off ({100 * bad / tot:.1f}%)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Masked-plane tear vs the capacity-1 gutter hypothesis "
        "(integer-cycle error maps; white=masked)",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "diag_tear_capacity_hypothesis.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
