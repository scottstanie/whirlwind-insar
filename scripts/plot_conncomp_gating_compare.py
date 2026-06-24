#!/usr/bin/env python3
"""Show WHY a high coherence cutoff shatters whirlwind's conncomp.

For a few NISAR frames, plot the conncomp label map under different gating:
production (GUNW), our gentle default (edge-cut min_coh=0.08), a SNAPHU-like
cutoff via edge-cutting (0.28), and the same cutoff via pixel-masking (0.28).
Both 0.28 variants fragment badly; this makes that visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import whirlwind as ww

CACHE = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final")
OUT = Path("gunw_results/conncomp_gating_compare")
NLOOKS = 16.0
FRAMES = sys.argv[1:] or ["A_035", "D_077"]


def load(fr: str) -> dict:
    d = np.load(CACHE / f"{fr}_panels.npz")
    mask = d["mask"].astype(bool)
    ww_unw = d["ww_unw"].astype(np.float32)
    return dict(
        mask=mask,
        valid=mask & np.isfinite(ww_unw),
        coh=d["coh"],
        prod_cc=d["prod_cc"],
        ig=np.exp(1j * d["wrapped"]).astype(np.complex64),
        corr=np.clip(np.nan_to_num(d["coh"]), 0, 1).astype(np.float32),
        unw=np.where(mask, ww_unw, np.nan).astype(np.float32),
    )


def edgecut(a: dict, gamma: float) -> np.ndarray:
    raw = round(
        ww.conncomp_reliability_from_coherence(gamma, NLOOKS)
        * ww.CONNCOMP_RELIABILITY_UNIT
    )
    return np.asarray(
        ww._native.components_snaphu(
            a["ig"], a["corr"], NLOOKS, a["unw"], a["mask"], int(raw), 100, 4096
        )
    )


def pixelmask(a: dict, rho0: float) -> np.ndarray:
    m2 = a["mask"] & (a["corr"] >= rho0)
    unw2 = np.where(m2, a["unw"], np.nan).astype(np.float32)
    return np.asarray(
        ww._native.components_snaphu(a["ig"], a["corr"], NLOOKS, unw2, m2, 0, 100, 4096)
    )


def ncomp(cc, valid):
    v = cc[valid]
    return int(np.unique(v[v > 0]).size)


def cc_panel(ax, cc, valid, title):
    arr = np.where(
        (cc > 0) & valid, ((cc.astype(np.int64) - 1) % 20) + 1, np.nan
    ).astype(float)
    ax.imshow(arr[::4, ::4], cmap="tab20", vmin=0, vmax=20, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rho0 = 1.25 * (1.3 / 16 + 0.14)  # SNAPHU zero-corr cutoff at L=16 ~ 0.28
    for fr in FRAMES:
        a = load(fr)
        v = a["valid"]
        fig, axes = plt.subplots(1, 5, figsize=(22, 5), constrained_layout=True)
        im = axes[0].imshow(
            np.where(v, a["coh"], np.nan)[::4, ::4], cmap="gray", vmin=0, vmax=1
        )
        axes[0].set_title("coherence", fontsize=11)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(im, ax=axes[0], shrink=0.7)
        cc_panel(
            axes[1], a["prod_cc"], v, f"production GUNW\n{ncomp(a['prod_cc'], v)} comps"
        )
        e08 = edgecut(a, 0.08)
        cc_panel(
            axes[2], e08, v, f"whirlwind default\nedge-cut 0.08: {ncomp(e08, v)} comps"
        )
        e28 = edgecut(a, 0.28)
        cc_panel(axes[3], e28, v, f"edge-cut 0.28\n{ncomp(e28, v)} comps (shattered)")
        pm = pixelmask(a, rho0)
        cc_panel(axes[4], pm, v, f"pixel-mask 0.28\n{ncomp(pm, v)} comps (shattered)")
        fig.suptitle(f"NISAR {fr} - conncomp gating comparison (L=16)", fontsize=13)
        out = OUT / f"{fr}_gating.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
