"""Goldstein on-vs-off A/B for the report - NISAR 40 MHz mainland scene.

Runs ``whirlwind.unwrap`` with ``goldstein_alpha=0.0`` and ``=0.7`` on the same
NISAR interferogram, SEQUENTIALLY (one heavy unwrap at a time - laptop limit),
and reports K-match vs the SNAPHU 9x9 reference on the cc=1 mainland. This is
the comparison that decides whether Goldstein-off should stay the default.

Outputs (per-variant .npz, summary.json, comparison PNG) under OUT.

Run:
    env -u CONDA_PREFIX uv run --with rasterio --with matplotlib \
        python scripts/report_goldstein_ab.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

NISAR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
IG = NISAR / "20251224_20260117.int.looked.tif"
COH = NISAR / "20251224_20260117.int.coh.looked.cleaned.tif"
SNAPHU_UNW = NISAR / "20251224_20260117.snaphu_9x9.unw.tif"
SNAPHU_CC = NISAR / "20251224_20260117.snaphu_9x9.cc.tif"
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/goldstein_ab")
NLOOKS = 100.0  # 10 range x 10 az boxcar looks
ALPHAS = (0.0, 0.7)
TAU = np.float32(2 * np.pi)


def kmatch(
    ww_unw: np.ndarray, wrapped: np.ndarray, snaphu_k: np.ndarray, common: np.ndarray
) -> dict:
    """K-agreement vs SNAPHU on the common mask, after removing a global cycle
    offset (the modal dK)."""
    # Compute K only on `common` (ww_unw is NaN off-coverage; casting NaN warns).
    ww_k = np.round((ww_unw[common] - wrapped[common]) / TAU).astype(np.int64)
    dk = (ww_k - snaphu_k[common]).astype(np.int64)
    center = int(np.bincount(dk - dk.min()).argmax() + dk.min())
    dk_c = dk - center
    n = max(dk_c.size, 1)
    return {
        "match_pct": float((dk_c == 0).sum()) / n * 100.0,
        "dk1_pct": float((np.abs(dk_c) == 1).sum()) / n * 100.0,
        "dk2plus_pct": float((np.abs(dk_c) >= 2).sum()) / n * 100.0,
        "global_offset": center,
    }


def main() -> None:
    import rasterio  # noqa: F401  (delayed; heavy deps)
    import whirlwind as ww

    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(IG) as s:
        ig = s.read(1).astype(np.complex64)
    with rasterio.open(COH) as s:
        coh = s.read(1).astype(np.float32)
    with rasterio.open(SNAPHU_UNW) as s:
        snaphu_unw = s.read(1).astype(np.float32)
    with rasterio.open(SNAPHU_CC) as s:
        snaphu_cc = s.read(1).astype(np.uint32)

    assert ig.shape == coh.shape == snaphu_unw.shape, "shape mismatch in inputs"
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig = ig.copy()
    ig[~mask] = 0
    coh = np.clip(np.where(mask, coh, 0.0), 0.0, 1.0).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    snaphu_k = np.round((snaphu_unw - wrapped) / TAU).astype(np.int64)
    main_mask = (snaphu_cc == 1) & mask
    print(
        f"shape={ig.shape}  valid={mask.mean() * 100:.1f}%  "
        f"snaphu cc=1 mainland={int(main_mask.sum()):,} px",
        flush=True,
    )

    rows: list[dict] = []
    kfields: dict[str, np.ndarray] = {}
    for alpha in ALPHAS:
        tag = f"goldstein_{alpha:.1f}"
        print(f"\n=== {tag} (one heavy unwrap; sequential) ===", flush=True)
        t0 = time.perf_counter()
        unw, cc = ww.unwrap(ig, coh, NLOOKS, mask=mask, goldstein_alpha=alpha)
        dt = time.perf_counter() - t0
        common = main_mask & (cc > 0) & np.isfinite(unw)
        km = kmatch(unw, wrapped, snaphu_k, common)
        row = {
            "tag": tag,
            "alpha": alpha,
            "runtime_s": round(dt, 1),
            "coverage_pct": float((cc > 0).mean() * 100.0),
            "n_components": int(cc.max()),
            "n_common_px": int(common.sum()),
            **km,
        }
        rows.append(row)
        kfields[tag] = np.where(common, np.round((unw - wrapped) / TAU), np.nan)
        np.savez_compressed(OUT / f"{tag}.npz", unw=unw, cc=cc)
        print(json.dumps(row, indent=2), flush=True)
        del unw, cc

    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))

    print("\n=== Goldstein A/B vs SNAPHU 9x9 (NISAR cc=1 mainland) ===")
    hdr = f"{'method':16}{'runtime':>9}{'cov%':>8}{'#cc':>7}{'K-match%':>10}{'|dK|=1%':>9}{'|dK|>=2%':>10}"
    print(hdr)
    for r in rows:
        print(
            f"{r['tag']:16}{r['runtime_s']:8.1f}s{r['coverage_pct']:8.2f}"
            f"{r['n_components']:7d}{r['match_pct']:10.3f}{r['dk1_pct']:9.3f}"
            f"{r['dk2plus_pct']:10.3f}"
        )

    # Comparison figure: SNAPHU K and each variant's dK-vs-SNAPHU.
    import matplotlib.pyplot as plt

    snaphu_kf = np.where(main_mask, snaphu_k, np.nan)
    panels = [("SNAPHU 9x9 K", snaphu_kf, "viridis", None)]
    for tag in kfields:
        dkf = kfields[tag] - snaphu_kf
        dkf = dkf - np.nanmedian(dkf)
        panels.append((f"{tag} − SNAPHU (dK)", dkf, "RdBu", 2.0))

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6 * len(panels), 6), constrained_layout=True
    )
    for ax, (name, arr, cmap, vlim) in zip(np.atleast_1d(axes), panels):
        kw = dict(cmap=cmap, interpolation="nearest")
        if vlim is not None:
            kw.update(vmin=-vlim, vmax=vlim)
        im = ax.imshow(arr, **kw)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    png = OUT / "goldstein_ab_nisar.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"\nplot: {png}")
    print(f"summary: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
