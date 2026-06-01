"""Clean isolated before/after of the heal: run the FULL pipeline
(anchor+cascade+feather) with the heal OFF then ON, so only the heal differs.
Crop a smooth high-coherence ghost column. Report match + ghost count.
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
    sk = np.load(OUT / "nisar_anchor_sk.npy"); scc = np.load(OUT / "nisar_anchor_scc.npy")
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig[~mask] = 0; coh = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    mainland = (scc == 1) & mask

    os.environ["WHIRLWIND_NO_HEAL"] = "1"
    off, _cc = ww.unwrap(ig, coh, nlooks=100.0, mask=mask, tile_size=512, tile_overlap=64)
    os.environ.pop("WHIRLWIND_NO_HEAL", None)
    on, _cc = ww.unwrap(ig, coh, nlooks=100.0, mask=mask, tile_size=512, tile_overlap=64)

    def mp(u):
        k = np.round((u - wrapped) / TAU); k[~mask] = np.nan
        d = (k - sk)[mainland]; d = d[np.isfinite(d)]; d = d - modal(d)
        return float((d == 0).sum())/d.size*100
    print(f"mainland match: heal OFF={mp(off):.3f}%  ON={mp(on):.3f}%", flush=True)

    # smooth ghost column in `off`
    valid = mainland & (coh > 0.45)
    u = np.where(valid, off, np.nan)
    kl = np.round((u[:, :-2] - u[:, 1:-1]) / TAU); kr = np.round((u[:, 2:] - u[:, 1:-1]) / TAU)
    vc = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
    gh = vc & (kl == kr) & (kl != 0)
    gcol = gh.sum(axis=0)
    smooth = np.zeros(gcol.shape)
    for jj in np.argsort(gcol)[::-1][:40]:
        j = jj + 1
        rows = np.where(gh[:, jj])[0]
        if rows.size < 30:
            continue
        band = off[rows.min():rows.max(), max(0, j-30):j-2]
        bm = mainland[rows.min():rows.max(), max(0, j-30):j-2]
        if bm.sum() < 50:
            continue
        g = np.abs(np.diff(np.where(bm, band, np.nan), axis=1))
        smooth[jj] = gcol[jj] / (1 + 5 * np.nanmean(g[np.isfinite(g)]))
    jj = int(np.argmax(smooth)); j = jj + 1
    rows = np.where(gh[:, jj])[0]; rc = int(np.median(rows))
    print(f"smooth ghost col {j}, {int(gcol[jj])} px, rows {rows.min()}..{rows.max()}", flush=True)
    r0 = max(0, rc-180); r1 = min(off.shape[0], rc+180); c0 = max(0, j-90); c1 = min(off.shape[1], j+90)
    cr = lambda a: a[r0:r1, c0:c1]; mk = cr(mainland)
    b = np.where(mk, cr(off) - np.nanmedian(off[mainland]), np.nan)
    a = np.where(mk, cr(on) - np.nanmedian(on[mainland]), np.nan)
    vlo, vhi = np.nanpercentile(b[np.isfinite(b)], [2, 98])
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    axes[0].imshow(b, cmap="twilight", vmin=vlo, vmax=vhi, interpolation="nearest")
    axes[0].set_title("heal OFF — 1-px ghost line", fontsize=13)
    axes[1].imshow(a, cmap="twilight", vmin=vlo, vmax=vhi, interpolation="nearest")
    axes[1].set_title("heal ON — gone", fontsize=13)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"NISAR ghost-line heal (full pipeline, only heal toggled) @ col {j}", fontsize=14)
    fig.tight_layout()
    out = PLOTS / "nisar_heal_beforeafter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
