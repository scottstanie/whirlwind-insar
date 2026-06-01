"""Find the residual thin vertical lines in the NISAR unwrap and attribute them:
are they low-coherence / invalid COLUMNS in the input (data artifact, present in
SNAPHU too) or whirlwind-introduced seam tears?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = float(2 * np.pi)


def vtear_score(unw, mask):
    """Per-column count of vertical tears (|d row| > pi) — high = a vertical line."""
    a = unw.copy(); a[~mask] = np.nan
    dh = np.abs(a[:, 1:] - a[:, :-1])  # horizontal neighbor diff -> vertical-line edges
    tears = np.nansum(dh > np.pi, axis=0)
    return tears  # length n-1, index j ~ edge between col j and j+1


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    mask = np.load(OUT / "nisar_anchor_mask.npy")
    wrapped = np.load(OUT / "nisar_anchor_wrapped.npy")
    sk = np.load(OUT / "nisar_anchor_sk.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    unw = np.load(OUT / "nisar_cascade_unw.npy")
    sunw = sk * TAU + wrapped
    region = (scc == 1) & mask

    ww_t = vtear_score(unw, mask)
    sn_t = vtear_score(sunw, mask)
    valid_per_col = mask.sum(axis=0)
    coh_per_col = np.where(mask, coh, np.nan)
    mean_coh_col = np.nanmean(coh_per_col, axis=0)

    # top whirlwind vertical-line columns
    top = np.argsort(ww_t)[::-1][:12]
    print("col   ww_tears  snaphu_tears  valid_frac  mean_coh", flush=True)
    for j in sorted(top):
        print(f"{j:5d}  {ww_t[j]:8d}  {sn_t[j]:12d}  "
              f"{valid_per_col[j]/mask.shape[0]:.3f}      {mean_coh_col[j]:.3f}", flush=True)

    # how many of whirlwind's worst lines coincide with SNAPHU tears or low-valid cols?
    worst = np.argsort(ww_t)[::-1][:30]
    snaphu_also = int((sn_t[worst] > ww_t[worst] * 0.3).sum())
    lowvalid = int((valid_per_col[worst] < mask.shape[0] * 0.5).sum())
    print(f"\nof whirlwind's 30 worst vertical-line columns: {snaphu_also} also tear in SNAPHU; "
          f"{lowvalid} are <50% valid (data gaps)", flush=True)

    # Zoom on the single worst whirlwind line
    j = int(top[np.argmax(ww_t[top])])
    rows = np.where(mask[:, j])[0]
    r0 = rows[len(rows)//4] if rows.size else 0
    r0 = max(0, r0); r1 = min(unw.shape[0], r0 + 1200)
    c0 = max(0, j - 120); c1 = min(unw.shape[1], j + 120)
    cr = lambda a: a[r0:r1, c0:c1]
    mk = cr(mask)
    fig, axes = plt.subplots(1, 4, figsize=(22, 8))
    panels = [("input coherence", cr(np.where(mask, coh, np.nan)), "viridis", 0, 1),
              ("input wrapped", np.where(mk, cr(wrapped), np.nan), "twilight", -np.pi, np.pi),
              ("SNAPHU unwrapped", np.where(mk, cr(sunw) - np.nanmedian(sunw[region]), np.nan), "twilight", None, None),
              ("whirlwind unwrapped", np.where(mk, cr(unw) - np.nanmedian(unw[region]), np.nan), "twilight", None, None)]
    vlo, vhi = np.nanpercentile(panels[2][1], [1, 99])
    for ax, (t, im_, cmap, a, b) in zip(axes, panels):
        kw = dict(cmap=cmap, interpolation="nearest")
        if a is not None:
            kw.update(vmin=a, vmax=b)
        elif t.endswith("unwrapped"):
            kw.update(vmin=vlo, vmax=vhi)
        im = ax.imshow(im_, **kw)
        ax.set_title(f"{t}\n(col {j})", fontsize=12); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"NISAR thin vertical line @ col {j}: data artifact or whirlwind seam?", fontsize=13)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "nisar_vline_diag.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
