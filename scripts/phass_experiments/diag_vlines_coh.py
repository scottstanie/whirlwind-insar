"""Characterize the vertical lines that matter: vertical 2pi tears in the
HIGH-COHERENCE mainland only (cc==1). Are they tile-seam-aligned (448 step)?
clean 2pi? present before the anchor/cascade/feather (per-tile origin) or
introduced by the pipeline?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = float(2 * np.pi)


def col_tears(unw, region):
    """Per column-edge: count of |d(unw)| in (pi, 3pi) tears within the region
    (both pixels in region). Returns (count[n-1], mean_mag[n-1])."""
    a = unw.copy().astype(np.float64); a[~region] = np.nan
    d = np.abs(a[:, 1:] - a[:, :-1])
    both = region[:, 1:] & region[:, :-1]
    tear = (d > np.pi) & both
    cnt = tear.sum(axis=0)
    with np.errstate(invalid="ignore"):
        mag = np.where(cnt > 0, np.nansum(np.where(tear, d, 0), axis=0) / np.maximum(cnt, 1), np.nan)
    return cnt, mag


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    mask = np.load(OUT / "nisar_anchor_mask.npy")
    wrapped = np.load(OUT / "nisar_anchor_wrapped.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    unw = np.load(OUT / "nisar_cascade_unw.npy")       # feathered default
    old = np.load(OUT / "nisar_no_anchor_unw.npy")     # pre-anchor/feather
    mainland = (scc == 1) & mask                        # high-coherence land SNAPHU trusts

    cnt, mag = col_tears(unw, mainland)
    cnt_old, _ = col_tears(old, mainland)
    step = 512 - 64
    top = np.argsort(cnt)[::-1][:15]
    print("col   mainland_tears  seam?(448)  meancoh_col   tear_mag/2pi   old_tears", flush=True)
    cohcol = np.where(mainland, coh, np.nan)
    for j in sorted(top):
        seam = "SEAM" if (j % step) in (0, 1, step - 1) or ((j + 1) % step) in (0, 1) else ""
        # distance to nearest multiple of step
        near = min(j % step, step - (j % step))
        mc = np.nanmean(cohcol[:, j])
        print(f"{j:5d}  {int(cnt[j]):13d}  d2seam={near:3d} {seam:4s}  {mc:.3f}        "
              f"{mag[j]/TAU:5.2f}        {int(cnt_old[j])}", flush=True)

    print(f"\ntotal mainland vertical tears: feather={int(cnt.sum()):,}  pre-anchor={int(cnt_old.sum()):,}", flush=True)
    # how many worst-15 are near a tile seam (within 2px)?
    nearseam = sum(1 for j in top if min(j % step, step - (j % step)) <= 2)
    print(f"of the 15 worst high-coh tear columns, {nearseam} are within 2px of a tile seam (step {step})", flush=True)

    # Zoom the worst one
    j = int(top[np.argmax(cnt[top])])
    rows = np.where(mainland[:, j])[0]
    r0 = max(0, rows[len(rows)//3]); r1 = min(unw.shape[0], r0 + 400)
    c0 = max(0, j - 60); c1 = min(unw.shape[1], j + 60)
    cr = lambda a: a[r0:r1, c0:c1]
    mk = cr(mainland)
    fig, axes = plt.subplots(1, 3, figsize=(15, 9))
    p_old = np.where(mk, cr(old) - np.nanmedian(old[mainland]), np.nan)
    p_new = np.where(mk, cr(unw) - np.nanmedian(unw[mainland]), np.nan)
    vlo, vhi = np.nanpercentile(p_new[np.isfinite(p_new)], [2, 98])
    axes[0].imshow(cr(np.where(mainland, coh, np.nan)), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"coherence @ col {j} (d2seam={min(j%step, step-(j%step))})", fontsize=11)
    axes[1].imshow(p_old, cmap="twilight", vmin=vlo, vmax=vhi, interpolation="nearest")
    axes[1].set_title("pre-anchor unwrapped", fontsize=11)
    axes[2].imshow(p_new, cmap="twilight", vmin=vlo, vmax=vhi, interpolation="nearest")
    axes[2].set_title("feather default unwrapped", fontsize=11)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"NISAR worst HIGH-COH vertical line @ col {j}", fontsize=13)
    fig.tight_layout()
    out = PLOTS / "nisar_vline_coh_diag.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
