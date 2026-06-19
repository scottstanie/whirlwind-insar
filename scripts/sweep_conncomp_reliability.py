#!/usr/bin/env python3
"""Sweep the `conncomp_reliability` knob and plot what it does, so a useful value
is guessable instead of mysterious.

The knob is a threshold on each edge's convex-cost *reliability*
``min(poscost, negcost)``. A clean edge at coherence ``gamma`` and ``L`` looks has
reliability ``~ COST_SCALE * nshortcycle**2 / sigma2(gamma) = 1e6 / sigma2``, with
``sigma2(gamma) = (1 - gamma**2) / (2*L*gamma**2)`` (Just/Bamler). That is why
meaningful thresholds are large (~1e6): they are ``1e6 / sigma2``. Inverting gives
a guessable rule — cutting at ``threshold`` drops edges below coherence

    gamma_min(threshold) = sqrt( 1 / (1 + 2*L * 1e6 / threshold) ).

This script sweeps the threshold on the cached NISAR frames and plots:
  (left)  labeled fraction vs threshold, with each frame's production-SNAPHU
          labeled fraction as a dashed target line;
  (right) component count vs threshold;
the top axis is annotated with the coherence-equivalent gamma_min so you can read
off "to keep only coherence > 0.3, use ~3e6". It also overlays, as crosses, the
median coherence of the pixels each step newly drops -- a check that the knob is
really acting as a coherence cut.

Outputs PNG + CSV into ``./nisar-pngs/<date>/``.

Usage: .venv/bin/python scripts/sweep_conncomp_reliability.py [FRAMES...]
"""
from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import whirlwind as ww

CACHE_DIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final"
NLOOKS = 16.0
COST_SCALE = 100.0
NSHORTCYCLE = 100.0
RELIABILITY_AT_UNIT_VAR = COST_SCALE * NSHORTCYCLE**2  # = 1e6 (clean-edge reliability at sigma2=1)

# 0 plus a log ramp through the useful range.
THRESHOLDS = [0] + [int(t) for t in np.logspace(4, 7.5, 12)]
FRAMES = ["D_077", "A_025", "D_075", "A_016", "A_030"]


def sigma2(gamma: np.ndarray | float, nlooks: float) -> np.ndarray | float:
    g = np.clip(gamma, 1e-3, 0.999)
    return (1.0 - g * g) / (2.0 * nlooks * g * g)


def gamma_min_for_threshold(thr: float, nlooks: float) -> float:
    """Coherence below which a clean edge's reliability falls under `thr`."""
    if thr <= 0:
        return 0.0
    s2 = RELIABILITY_AT_UNIT_VAR / thr  # sigma2 whose clean reliability == thr
    return float(np.sqrt(1.0 / (1.0 + 2.0 * nlooks * s2)))


def labeled_fraction(cc: np.ndarray, valid: np.ndarray) -> float:
    v = cc[valid]
    return float((v > 0).mean())


def ncc(cc: np.ndarray, valid: np.ndarray) -> int:
    v = cc[valid]
    return int(np.unique(v[v > 0]).size)


def main():
    frames = [a for a in sys.argv[1:] if not a.startswith("--")] or FRAMES
    out_dir = Path("nisar-pngs") / dt.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_frame = {}  # frame -> dict of arrays
    for fr in frames:
        d = np.load(f"{CACHE_DIR}/{fr}_panels.npz")
        wrapped = d["wrapped"]
        coh = d["coh"]
        mask = d["mask"].astype(bool)
        prod_cc = d["prod_cc"]
        ww_unw = d["ww_unw"].astype(np.float32)
        valid = mask & np.isfinite(ww_unw)
        ig = np.exp(1j * wrapped).astype(np.complex64)
        corr = np.clip(np.nan_to_num(coh), 0, 1).astype(np.float32)
        prod_lab = labeled_fraction(prod_cc, valid)

        labeled, ncomps, drop_coh = [], [], []
        prev_labeled_mask = None
        for thr in THRESHOLDS:
            cc = ww._native.components_snaphu(
                ig, corr, NLOOKS, ww_unw, mask, int(thr), 100, 4096
            )
            lab = labeled_fraction(cc, valid)
            labeled.append(lab * 100)
            ncomps.append(ncc(cc, valid))
            # median coherence of pixels this step newly dropped (label>0 -> 0).
            this_lab = valid & (cc > 0)
            if prev_labeled_mask is not None:
                newly = prev_labeled_mask & ~this_lab
                drop_coh.append(float(np.median(corr[newly])) if newly.any() else np.nan)
            else:
                drop_coh.append(np.nan)
            prev_labeled_mask = this_lab
            rows.append(
                {
                    "frame": fr,
                    "threshold": int(thr),
                    "gamma_min_equiv": round(gamma_min_for_threshold(thr, NLOOKS), 3),
                    "labeled_pct": round(lab * 100, 3),
                    "n_components": ncc(cc, valid),
                    "prod_labeled_pct": round(prod_lab * 100, 3),
                }
            )
        per_frame[fr] = dict(
            labeled=labeled, ncomps=ncomps, drop_coh=drop_coh, prod_lab=prod_lab * 100
        )
        print(f"{fr}: prod labeled%={prod_lab*100:.1f}; swept {len(THRESHOLDS)} thresholds", flush=True)

    # Plot. x = threshold (log, with 0 shown as a small floor for the log axis).
    xfloor = THRESHOLDS[1] / 3
    xs = [xfloor if t == 0 else t for t in THRESHOLDS]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    for fr in frames:
        pf = per_frame[fr]
        (line,) = axL.plot(xs, pf["labeled"], "o-", label=fr)
        axL.axhline(pf["prod_lab"], ls="--", lw=1, color=line.get_color(), alpha=0.6)
    axL.set_xscale("log")
    axL.set_xlabel("conncomp_reliability threshold  (0 shown at left)")
    axL.set_ylabel("labeled fraction (%)")
    axL.set_title(
        "Coverage vs reliability knob\n(dashed = that frame's production-SNAPHU labeled %)"
    )
    axL.legend(fontsize=9)
    axL.grid(True, which="both", alpha=0.25)
    # Top axis: coherence-equivalent gamma_min.
    axT = axL.secondary_xaxis("top")
    ticks = [t for t in THRESHOLDS if t > 0]
    axT.set_xticks(ticks)
    axT.set_xticklabels([f"{gamma_min_for_threshold(t, NLOOKS):.2f}" for t in ticks], fontsize=8)
    axT.set_xlabel("coherence-equivalent gamma_min  (clean-edge cutoff)")

    for fr in frames:
        axR.plot(xs, per_frame[fr]["ncomps"], "o-", label=fr)
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel("conncomp_reliability threshold  (0 shown at left)")
    axR.set_ylabel("number of components")
    axR.set_title("Fragmentation vs reliability knob")
    axR.legend(fontsize=9)
    axR.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        "conncomp_reliability sweep — raise to label fewer (lower-coherence) pixels.  "
        f"nlooks={NLOOKS:.0f};  threshold ≈ 1e6 / sigma²(gamma_min)",
        fontsize=13,
    )
    png = out_dir / "conncomp_reliability_sweep.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_dir / "conncomp_reliability_sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {png}\nWrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
