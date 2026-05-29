"""High-quality full-frame unwrapped-PHASE comparison: SNAPHU / no-anchor /
anchor. Phase (not K) so blocks show as color steps exactly as the eye sees.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
S = 3


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sk = np.load(OUT / "nisar_anchor_sk.npy")           # SNAPHU K (proxy for phase shape)
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    mask = np.load(OUT / "nisar_anchor_mask.npy")
    wrapped = np.load(OUT / "nisar_anchor_wrapped.npy")
    una = np.load(OUT / "nisar_anchor_unw.npy")
    unn = np.load(OUT / "nisar_no_anchor_unw.npy")
    sunw = sk * float(2 * np.pi) + wrapped  # reconstruct SNAPHU unwrapped phase
    mainland = (scc == 1) & mask

    def show(unw):
        u = unw.astype(np.float32).copy()
        u[~mask] = np.nan
        u = u - np.nanmedian(unw[mainland])
        return u

    panels = [("SNAPHU 9x9", show(sunw)), ("whirlwind no-anchor (old)", show(unn)),
              ("whirlwind anchor (new)", show(una))]
    allv = show(sunw)[mainland]
    lo, hi = np.nanpercentile(allv, [1, 99])
    ds = lambda a: a[::S, ::S]

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    for ax, (t, u) in zip(axes, panels):
        im = ax.imshow(ds(u), vmin=lo, vmax=hi, cmap="twilight", interpolation="nearest")
        ax.set_title(t, fontsize=14); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("NISAR unwrapped phase (full frame, hi-res): does the anchor remove visible rectangular blocks?", fontsize=15)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "nisar_phase_hires.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
