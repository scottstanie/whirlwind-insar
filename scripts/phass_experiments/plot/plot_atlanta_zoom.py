"""Atlanta failure diagnosis: a full-res crop comparing wrapped input,
reference unwrapped, and whirlwind unwrapped — to see if whirlwind is leaving
wrapped-like stripes (solver not carrying cycles) vs runaway.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = float(2 * np.pi)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kref = np.load(OUT / "atlanta_kref.npy")
    mask = np.load(OUT / "atlanta_mask.npy")
    wrapped = np.load(OUT / "atlanta_wrapped.npy")
    wwu = np.load(OUT / "atlanta_anchor_unw.npy")
    refu = kref * TAU + wrapped

    # central high-coh crop
    h, w = mask.shape
    r0, c0 = int(h * 0.35), int(w * 0.40)
    sz = 700
    cr = lambda a: a[r0:r0+sz, c0:c0+sz]
    mk = cr(mask)

    def ref_show():
        u = cr(refu).astype(np.float32); u[~mk] = np.nan
        return u - np.nanmedian(u[mk])

    def ww_show():
        u = cr(wwu).astype(np.float32); u[~mk] = np.nan
        return u - np.nanmedian(u[mk])

    wp = cr(wrapped).astype(np.float32); wp[~mk] = np.nan
    pr, pw = ref_show(), ww_show()
    lo, hi = np.nanpercentile(pr[mk], [1, 99])

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(wp, cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    axes[0].set_title("wrapped input phase (crop)", fontsize=13)
    axes[1].imshow(pr, cmap="twilight", vmin=lo, vmax=hi, interpolation="nearest")
    axes[1].set_title("OPERA/SNAPHU unwrapped", fontsize=13)
    axes[2].imshow(pw, cmap="twilight", vmin=lo, vmax=hi, interpolation="nearest")
    axes[2].set_title("whirlwind unwrapped (anchor+cascade)", fontsize=13)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Atlanta crop [{r0}:{r0+sz}, {c0}:{c0+sz}] — is whirlwind carrying cycles?", fontsize=14)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "atlanta_zoom_diag.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
