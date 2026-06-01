"""What did the anchor actually change, and are coherent blocks gone? Shows
(anchor_K - no_anchor_K) full frame + a high-res lower-left zoom of all three.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = np.float32(2 * np.pi)


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
    una = np.load(OUT / "nisar_anchor_unw.npy")
    unn = np.load(OUT / "nisar_no_anchor_unw.npy")
    mainland = (scc == 1) & mask

    def kf(unw):
        k = np.round((unw - wrapped) / TAU)
        k[~mask] = np.nan
        d = (k - sk)[mainland]
        return k - modal(d[np.isfinite(d)])

    ka, kn = kf(una), kf(unn)
    diff = (ka - kn)
    diff[~mask] = np.nan
    nchanged = int(np.nansum(np.abs(diff) > 0.5))
    print(f"anchor changed {nchanged:,} px ({100*nchanged/mask.sum():.2f}% of valid)")

    sd = sk.astype(np.float32).copy(); sd[~mask] = np.nan
    lo, hi = np.nanpercentile(sd[mask], [1, 99])
    S = 6
    ds = lambda a: a[::S, ::S]

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    im = ax.imshow(ds(diff), vmin=-3, vmax=3, cmap="RdBu_r", interpolation="nearest")
    ax.set_title(f"anchor_K - no_anchor_K  (coherent rectangles = blocks the anchor flipped)\n{nchanged:,} px changed", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    out = PLOTS / "nisar_anchor_whatchanged.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")

    # High-res zoom (stride 2) of the lower-left block region.
    h, w = sd.shape
    r0, r1, c0, c1 = int(h * 0.50), int(h * 0.95), 0, int(w * 0.42)
    cz = lambda a: a[r0:r1:2, c0:c1:2]
    fig, axes = plt.subplots(1, 3, figsize=(20, 9))
    for ax, (t, k) in zip(axes, [("SNAPHU 9x9", sd), ("no-anchor (old)", kn), ("anchor (new)", ka)]):
        im = ax.imshow(cz(k), vmin=lo, vmax=hi, cmap="twilight", interpolation="nearest")
        ax.set_title(t, fontsize=13); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Lower-left low-coh zoom (hi-res): rectangular block discontinuities?", fontsize=14)
    fig.tight_layout()
    out2 = PLOTS / "nisar_anchor_zoom_hires.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
