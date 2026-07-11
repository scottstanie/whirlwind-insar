#!/usr/bin/env python3
"""Sweep the `conncomp_reliability` knob and SHOW what it does, so a useful value
is guessable instead of mysterious.

`conncomp_reliability` is in inverse-variance (`1/sigma2`) units: an edge of
coherence `gamma` (and `L` looks) is cut roughly when the value exceeds
`1/sigma2(gamma)`, with `sigma2(gamma) = (1 - gamma**2) / (2*L*gamma**2)`
(Just/Bamler). So the knob reads as a "minimum coherence to keep" via
`whirlwind.conncomp_reliability_from_coherence(gamma, nlooks)`. This script sweeps
a set of target coherences and writes, into `./nisar-pngs/<date>/`:

  * conncomp_reliability_sweep.png  -- two line plots: labeled fraction and
    component count vs the target coherence (with each frame's production-SNAPHU
    labeled fraction as a dashed target line), plus the `conncomp_reliability`
    value annotated on top.
  * <frame>_conncomp_sweep_images.png -- the actual conncomp label IMAGES across
    the sweep (coherence + production conncomp for reference, then ww conncomps
    at increasing thresholds), so you can see components shrink/fragment.
  * conncomp_reliability_sweep.csv -- the numbers.

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
# Target minimum coherences to sweep (0 -> threshold 0, label everything).
GAMMAS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
# Subset shown as conncomp images per frame.
IMAGE_GAMMAS = [0.0, 0.15, 0.2, 0.25, 0.3, 0.4]
FRAMES = ["005_D_077", "005_A_025", "005_D_075", "005_A_016", "005_A_030"]
IMAGE_FRAMES = ["005_D_077", "005_A_025", "005_A_030"]


def reliability_for_gamma(gamma: float) -> float:
    """Public `conncomp_reliability` value (1/sigma2 units) for a target coherence."""
    return 0.0 if gamma <= 0 else ww.conncomp_reliability_from_coherence(gamma, NLOOKS)


def labels_for_show(cc: np.ndarray, valid: np.ndarray) -> np.ndarray:
    cc = np.where(valid, np.asarray(cc).astype(np.int64), 0)
    return np.where(cc > 0, ((cc - 1) % 20) + 1, np.nan).astype(float)


def labeled_fraction(cc: np.ndarray, valid: np.ndarray) -> float:
    v = cc[valid]
    return float((v > 0).mean())


def ncc(cc: np.ndarray, valid: np.ndarray) -> int:
    v = cc[valid]
    return int(np.unique(v[v > 0]).size)


def load(fr: str) -> dict:
    d = np.load(f"{CACHE_DIR}/{fr}_panels.npz")
    mask = d["mask"].astype(bool)
    ww_unw = d["ww_unw"].astype(np.float32)
    valid = mask & np.isfinite(ww_unw)
    return {
        "coh": d["coh"],
        "mask": mask,
        "valid": valid,
        "prod_cc": d["prod_cc"],
        "ww_unw": ww_unw,
        "ig": np.exp(1j * d["wrapped"]).astype(np.complex64),
        "corr": np.clip(np.nan_to_num(d["coh"]), 0, 1).astype(np.float32),
    }


def grow(a: dict, gamma: float) -> np.ndarray:
    # Drive the public knob through ww.unwrap's exact conversion: raw threshold =
    # conncomp_reliability * COST_SCALE * nshortcycle**2.
    raw = round(reliability_for_gamma(gamma) * ww.CONNCOMP_RELIABILITY_UNIT)
    return ww._native.components_snaphu(
        a["ig"], a["corr"], NLOOKS, a["ww_unw"], a["mask"], int(raw), 100, 4096
    ).astype(np.int64)


def line_plots(per_frame: dict, frames: list[str], out_dir: Path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    for fr in frames:
        pf = per_frame[fr]
        (line,) = axL.plot(GAMMAS, pf["labeled"], "o-", label=fr)
        axL.axhline(pf["prod_lab"], ls="--", lw=1, color=line.get_color(), alpha=0.6)
        axR.plot(GAMMAS, pf["ncomps"], "o-", label=fr)
    axL.set_xlabel("target minimum coherence  (--conncomp-min-coherence)")
    axL.set_ylabel("labeled fraction (%)")
    axL.set_title(
        "Coverage vs the knob\n(dashed = that frame's production-SNAPHU labeled %)"
    )
    axL.legend(fontsize=9)
    axL.grid(True, alpha=0.25)
    axT = axL.secondary_xaxis("top")
    axT.set_xticks(GAMMAS)
    axT.set_xticklabels(
        ["0"] + [f"{reliability_for_gamma(g):.1f}" for g in GAMMAS[1:]], fontsize=8
    )
    axT.set_xlabel("conncomp_reliability value (1/σ² units)")
    axR.set_xlabel("target minimum coherence  (--conncomp-min-coherence)")
    axR.set_ylabel("number of components")
    axR.set_yscale("log")
    axR.set_title("Fragmentation vs the knob")
    axR.legend(fontsize=9)
    axR.grid(True, which="both", alpha=0.25)
    fig.suptitle(
        "conncomp_reliability sweep — raise to label fewer (lower-coherence) pixels.  "
        f"nlooks={NLOOKS:.0f}",
        fontsize=13,
    )
    png = out_dir / "conncomp_reliability_sweep.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}", flush=True)


def image_grid(a: dict, fr: str, out_dir: Path):
    valid = a["valid"]
    panels = [
        (np.where(valid, a["coh"], np.nan), "coherence", "gray", 0.0, 1.0),
        (
            labels_for_show(a["prod_cc"], valid),
            f"production SNAPHU conncomp (n={ncc(a['prod_cc'], valid)})",
            "tab20",
            0,
            20,
        ),
    ]
    for g in IMAGE_GAMMAS:
        cc = grow(a, g)
        rel = reliability_for_gamma(g)
        title = (
            f"reliability=0 (label all)"
            if g == 0
            else f"min-coh {g:.2f}  (reliability={rel:.1f})"
        )
        panels.append(
            (
                labels_for_show(cc, valid),
                f"{title}\nn={ncc(cc, valid)}, labeled={labeled_fraction(cc, valid)*100:.0f}%",
                "tab20",
                0,
                20,
            )
        )
    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows), constrained_layout=True
    )
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.ravel(), panels, strict=False):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if cmap == "gray":
            fig.colorbar(im, ax=ax, shrink=0.75)
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    fig.suptitle(
        f"{fr}: connected components as `conncomp_reliability` rises "
        f"(production SNAPHU labels {labeled_fraction(a['prod_cc'], valid)*100:.0f}%)",
        fontsize=13,
    )
    png = out_dir / f"{fr}_conncomp_sweep_images.png"
    fig.savefig(png, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}", flush=True)


def main():
    frames = [a for a in sys.argv[1:] if not a.startswith("--")] or FRAMES
    out_dir = Path("nisar-pngs") / dt.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, per_frame = [], {}
    for fr in frames:
        a = load(fr)
        prod_lab = labeled_fraction(a["prod_cc"], a["valid"])
        labeled, ncomps = [], []
        for g in GAMMAS:
            cc = grow(a, g)
            labeled.append(labeled_fraction(cc, a["valid"]) * 100)
            ncomps.append(ncc(cc, a["valid"]))
            rows.append(
                {
                    "frame": fr,
                    "min_coherence": g,
                    "conncomp_reliability": round(reliability_for_gamma(g), 3),
                    "labeled_pct": round(labeled[-1], 3),
                    "n_components": ncomps[-1],
                    "prod_labeled_pct": round(prod_lab * 100, 3),
                }
            )
        per_frame[fr] = dict(labeled=labeled, ncomps=ncomps, prod_lab=prod_lab * 100)
        print(
            f"{fr}: prod labeled%={prod_lab*100:.1f}; swept {len(GAMMAS)} coherences",
            flush=True,
        )
        if fr in IMAGE_FRAMES:
            image_grid(a, fr, out_dir)

    line_plots(per_frame, frames, out_dir)
    csv_path = out_dir / "conncomp_reliability_sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}\nDONE -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
