"""Before/after of the ghost-line fix in a SMOOTH high-coherence region.
before = pre-fix per-tile result (nisar_no_anchor); after = shipped healed
output (nisar_cascade). Auto-picks a ghost column sitting in a locally-smooth,
coherent area (matching the user's view) so the line is clearly visible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = float(2 * np.pi)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    mask = np.load(OUT / "nisar_anchor_mask.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    before = np.load(OUT / "nisar_no_anchor_unw.npy")
    after = np.load(OUT / "nisar_cascade_unw.npy")
    mainland = (scc == 1) & mask
    valid = mainland & (coh > 0.45)

    # ghost columns in `before`: both horiz neighbors agree on same nonzero k
    u = np.where(valid, before, np.nan)
    kl = np.round((u[:, :-2] - u[:, 1:-1]) / TAU)
    kr = np.round((u[:, 2:] - u[:, 1:-1]) / TAU)
    vc = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
    gh = vc & (kl == kr) & (kl != 0)
    gcol = gh.sum(axis=0)
    # prefer columns in smooth surroundings: penalize local horizontal gradient
    smooth = np.zeros(gcol.shape)
    for jj in np.argsort(gcol)[::-1][:40]:
        j = jj + 1
        rows = np.where(gh[:, jj])[0]
        if rows.size < 30:
            continue
        r0, r1 = rows.min(), rows.max()
        band = before[r0:r1, max(0, j - 30):j - 2]
        bm = mainland[r0:r1, max(0, j - 30):j - 2]
        if bm.sum() < 50:
            continue
        g = np.abs(np.diff(np.where(bm, band, np.nan), axis=1))
        smooth[jj] = gcol[jj] / (1.0 + 5 * np.nanmean(g[np.isfinite(g)]))
    jj = int(np.argmax(smooth)); j = jj + 1
    rows = np.where(gh[:, jj])[0]
    rc = int(np.median(rows))
    r0 = max(0, rc - 180); r1 = min(before.shape[0], rc + 180)
    c0 = max(0, j - 90); c1 = min(before.shape[1], j + 90)
    print(f"smooth ghost line at col {j}, rows {rows.min()}..{rows.max()} ({gcol[jj]} px)", flush=True)

    cr = lambda a: a[r0:r1, c0:c1]
    mk = cr(mainland)
    b = np.where(mk, cr(before) - np.nanmedian(before[mainland]), np.nan)
    a = np.where(mk, cr(after) - np.nanmedian(after[mainland]), np.nan)
    vlo, vhi = np.nanpercentile(b[np.isfinite(b)], [2, 98])
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    axes[0].imshow(b, cmap="twilight", vmin=vlo, vmax=vhi, interpolation="nearest")
    axes[0].set_title("before (per-tile MCF) — ghost line", fontsize=13)
    axes[1].imshow(a, cmap="twilight", vmin=vlo, vmax=vhi, interpolation="nearest")
    axes[1].set_title("after (heal) — line gone", fontsize=13)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"NISAR ghost-line heal, smooth high-coh region @ col {j}", fontsize=14)
    fig.tight_layout()
    out = PLOTS / "nisar_line_beforeafter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
