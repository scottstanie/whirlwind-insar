"""4-up unwrapped-phase comparison: SNAPHU / no-anchor / anchor-single /
anchor-cascade, full frame hi-res, plus (cascade - single) to check the
cascade only refines boundaries (clean) rather than adding low-coh speckle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = float(2 * np.pi)
S = 3


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sk = np.load(OUT / "nisar_anchor_sk.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    mask = np.load(OUT / "nisar_anchor_mask.npy")
    wrapped = np.load(OUT / "nisar_anchor_wrapped.npy")
    unn = np.load(OUT / "nisar_no_anchor_unw.npy")
    una = np.load(OUT / "nisar_anchor_unw.npy")
    unc = np.load(OUT / "nisar_cascade_unw.npy")
    sunw = sk * TAU + wrapped
    mainland = (scc == 1) & mask

    def show(unw):
        u = unw.astype(np.float32).copy(); u[~mask] = np.nan
        return u - np.nanmedian(unw[mainland])

    panels = [("SNAPHU 9x9", show(sunw)), ("no-anchor 99.21%", show(unn)),
              ("anchor 99.63%", show(una)), ("anchor+cascade 99.89%", show(unc))]
    lo, hi = np.nanpercentile(show(sunw)[mainland], [1, 99])
    ds = lambda a: a[::S, ::S]

    fig, axes = plt.subplots(1, 4, figsize=(30, 9))
    for ax, (t, u) in zip(axes, panels):
        im = ax.imshow(ds(u), vmin=lo, vmax=hi, cmap="twilight", interpolation="nearest")
        ax.set_title(t, fontsize=14); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("NISAR unwrapped phase: SNAPHU vs whirlwind variants (full frame)", fontsize=15)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "nisar_variants.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")

    # cascade - single (what the cascade refined)
    kc = np.round((unc - wrapped) / TAU); kc[~mask] = np.nan
    ka = np.round((una - wrapped) / TAU); ka[~mask] = np.nan
    d = kc - ka
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    im = ax.imshow(ds(d), vmin=-2, vmax=2, cmap="RdBu_r", interpolation="nearest")
    ax.set_title("cascade_K - anchor_K (clean coherent edges = good; salt&pepper = bad)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    out2 = PLOTS / "nisar_cascade_vs_anchor.png"
    fig.savefig(out2, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
