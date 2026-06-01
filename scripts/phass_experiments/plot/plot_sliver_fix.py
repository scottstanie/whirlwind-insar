"""HONEST before/after of the bounded sliver cleanup at the REAL sliver column
(4032), heal OFF vs ON in the SAME binary (two sequential unwraps, one process).
Fixes the prior figure which cropped col 4071 and labelled an unhealed 2px line
"gone". Crops the actual 2px/-1 strip (cols 4032-4033, rows ~956-1375).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = float(2 * np.pi)
TS, OV, NLOOKS = 512, 64, 100.0


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import whirlwind as ww

    ig = rasterio.open(N / "20251224_20260117.int.looked.tif").read(1).astype(np.complex64)
    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig[~mask] = 0
    coh = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    sk = np.load(OUT / "nisar_anchor_sk.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    mainland = (scc == 1) & mask

    os.environ["WHIRLWIND_NO_HEAL"] = "1"
    off, _cc = ww.unwrap(ig, coh, nlooks=NLOOKS, mask=mask, tile_size=TS, tile_overlap=OV)
    os.environ.pop("WHIRLWIND_NO_HEAL", None)
    on, _cc = ww.unwrap(ig, coh, nlooks=NLOOKS, mask=mask, tile_size=TS, tile_overlap=OV)

    def col4032(u, label):
        k = np.round((u - wrapped) / TAU); k[~mask] = np.nan
        dk = k - sk; dk = dk - modal(dk[mainland])
        rows = slice(956, 1376)
        meds = []
        for j in range(4029, 4037):
            v = np.where(mainland[rows, j], dk[rows, j], np.nan); v = v[np.isfinite(v)]
            meds.append(int(np.median(v)) if v.size else 9)
        m0 = float((dk[mainland] == 0).sum()) / np.isfinite(dk[mainland]).sum() * 100
        print(f"  {label}: col 4029..4036 ΔK = {meds}  mainland={m0:.3f}%", flush=True)
        return meds

    print("col-4032 sliver:", flush=True)
    col4032(off, "heal OFF")
    col4032(on, "heal ON ")

    rc, j = 1165, 4032
    r0, r1, c0, c1 = rc - 180, rc + 180, j - 90, j + 90
    cr = lambda a: a[r0:r1, c0:c1]; mk = cr(mainland)
    med = np.nanmedian(off[mainland])
    b = np.where(mk, cr(off) - med, np.nan)
    a = np.where(mk, cr(on) - med, np.nan)
    vlo, vhi = np.nanpercentile(b[np.isfinite(b)], [2, 98])
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    axes[0].imshow(b, cmap="twilight", vmin=vlo, vmax=vhi, interpolation="nearest")
    axes[0].set_title("cleanup OFF — 2-px −1 sliver at col 4032", fontsize=13)
    axes[1].imshow(a, cmap="twilight", vmin=vlo, vmax=vhi, interpolation="nearest")
    axes[1].set_title("cleanup ON — gone", fontsize=13)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("NISAR bounded sliver cleanup @ col 4032 (heal OFF vs ON, same binary)", fontsize=14)
    fig.tight_layout()
    outp = PLOTS / "nisar_sliver_fix_4032.png"
    fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
