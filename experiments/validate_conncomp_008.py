#!/usr/bin/env python3
"""Check a connected-component coherence floor across the NISAR frames.

Re-labels the cached ``ww.unwrap`` phase (``ww_4way_final/<frame>_panels.npz``)
with the native conncomp grow at a few coherence floors. There is no re-unwrap,
so it is fast and runs one frame at a time. For each frame it records the labeled
fraction and component count at min-coherence 0.0, 0.08, and 0.10, and writes an
eight-panel comparison figure.

The question it answers: does a 0.08 floor hold the component count near the
no-floor baseline, or does it fragment the map the way 0.15-0.20 does?
"""

from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

import whirlwind as ww

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aws-batch"))
import compare_gunw as cg  # noqa: E402

CACHE = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final")
OUT = Path("gunw_results/conncomp_default_0p08")
NLOOKS = 16.0
FRAMES = [
    "005_A_013",
    "005_A_016",
    "005_A_018",
    "005_A_020",
    "005_A_022",
    "005_A_025",
    "005_A_028",
    "005_A_030",
    "006_A_035",
    "005_D_074",
    "005_D_075",
    "005_D_077",
    "005_D_078",
]
GAMMAS = [("0.00", 0.0), ("0.08", 0.08), ("0.10", 0.10)]


def grow(ig, corr, unw, mask, gamma: float) -> np.ndarray:
    raw = (
        0
        if gamma == 0.0
        else round(
            ww.conncomp_reliability_from_coherence(gamma, NLOOKS)
            * ww.CONNCOMP_RELIABILITY_UNIT
        )
    )
    return np.asarray(
        ww._native.components_snaphu(ig, corr, NLOOKS, unw, mask, int(raw), 100, 4096)
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for fr in FRAMES:
        p = CACHE / f"{fr}_panels.npz"
        if not p.exists():
            print(f"{fr}: MISSING {p}", flush=True)
            continue
        d = np.load(p)
        wrapped = d["wrapped"].astype(np.float32)
        coh = d["coh"].astype(np.float32)
        mask = d["mask"].astype(bool)
        prod_unw = d["prod_unw"].astype(np.float32)
        prod_cc = d["prod_cc"]
        ww_unw = d["ww_unw"].astype(np.float32)
        d.close()

        valid = mask & np.isfinite(ww_unw)
        ig = np.exp(1j * wrapped).astype(np.complex64)
        corr = np.clip(np.nan_to_num(coh), 0, 1).astype(np.float32)
        unw_grow = np.where(mask, ww_unw, np.nan).astype(np.float32)

        row: dict = {"frame": fr}
        cc08 = None
        for label, g in GAMMAS:
            cc = grow(ig, corr, unw_grow, mask, g)
            v = cc[valid]
            row[f"labeled_{label}"] = round(100 * float((v > 0).mean()), 1)
            row[f"ncomps_{label}"] = int(np.unique(v[v > 0]).size)
            if label == "0.08":
                cc08 = cc.copy()
            del cc, v
        rows.append(row)

        stats, ww_aligned, _resid, amb = cg.compute_compare_stats(
            ig=wrapped,
            coh=coh,
            mask=mask,
            prod_unw=prod_unw,
            prod_cc=prod_cc,
            ww_unw=ww_unw,
            ww_cc=cc08,
            runtime_s=0.0,
            rss_delta_mb=None,
        )
        cg.plot_result(
            OUT / f"{fr}_8panel.png",
            ig=wrapped,
            coh=coh,
            prod_unw=prod_unw,
            ww_aligned=ww_aligned,
            prod_cc=prod_cc,
            ww_cc=cc08,
            amb_diff=amb,
            valid=valid,
            title=f"NISAR {fr} - whirlwind (conncomp min_coh=0.08) vs GUNW",
            stride=2,
        )
        print(
            f"{fr}: 0.0={row['labeled_0.00']}%/{row['ncomps_0.00']}c  "
            f"0.08={row['labeled_0.08']}%/{row['ncomps_0.08']}c  "
            f"0.10={row['labeled_0.10']}%/{row['ncomps_0.10']}c  "
            f"phase-match%={100 * stats['ambiguity_match_frac_percomp']:.1f}",
            flush=True,
        )
        del ig, corr, unw_grow, cc08, wrapped, coh, mask, prod_unw, prod_cc
        del ww_unw, ww_aligned, amb, valid
        gc.collect()

    with open(OUT / "conncomp_0p08_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT / 'conncomp_0p08_validation.csv'} and {len(rows)} figures.")


if __name__ == "__main__":
    main()
