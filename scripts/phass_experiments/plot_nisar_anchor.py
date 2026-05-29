"""Diagnostic: did the global coarse anchor remove the visible rectangular
blocks? Loads saved arrays from run_nisar_anchor.py and compares SNAPHU /
no-anchor / anchor as unwrapped phase + |dK| error, full frame and a low-coh
zoom.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = np.float32(2 * np.pi)
S = 6  # display stride


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
    sd = sk.astype(np.float32).copy()
    sd[~mask] = np.nan

    lo, hi = np.nanpercentile(sd[mask], [1, 99])
    ds = lambda a: a[::S, ::S]

    # Full-frame K fields (row 0) + |dK| vs SNAPHU over the FULL mask (row 1).
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (t, k) in zip(axes[0], [("SNAPHU 9x9", sd), ("no-anchor (old)", kn), ("anchor (new)", ka)]):
        im = ax.imshow(ds(k), vmin=lo, vmax=hi, cmap="twilight", interpolation="nearest")
        ax.set_title(t, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    def err(k):
        e = np.abs(k - sk).astype(np.float32)
        e[~mask] = np.nan
        return e
    axes[1, 0].axis("off")
    for ax, (t, k) in zip(axes[1, 1:], [("|dK| no-anchor (full frame)", kn), ("|dK| anchor (full frame)", ka)]):
        im = ax.imshow(ds(err(k)), vmin=0, vmax=3, cmap="inferno", interpolation="nearest")
        ax.set_title(t, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("NISAR: global coarse anchor vs no-anchor vs SNAPHU (full frame, not just cc=1)", fontsize=13)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "nisar_anchor_compare.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)

    # Zoom into the lower-left low-coherence region where the blocks were.
    h, w = sd.shape
    r0, r1, c0, c1 = int(h * 0.55), h, 0, int(w * 0.45)
    cz = lambda a: a[r0:r1, c0:c1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    for ax, (t, k) in zip(axes, [("SNAPHU 9x9", sd), ("no-anchor (old)", kn), ("anchor (new)", ka)]):
        im = ax.imshow(cz(k), vmin=lo, vmax=hi, cmap="twilight", interpolation="nearest")
        ax.set_title(t, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("NISAR lower-left low-coherence zoom: are the rectangular blocks gone?", fontsize=13)
    fig.tight_layout()
    out2 = PLOTS / "nisar_anchor_zoom.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"wrote {out2}")
    plt.close(fig)


if __name__ == "__main__":
    main()
