"""A_016 report panel: BEFORE (tile512 default, 55%, +winding+seam-strip) vs
AFTER (gated multi-shift default, 97%, clean). Shows the ambiguity-vs-production
for both and the coherence/conncomp context. Demonstrates the seam-strip artifact
is gone and the winding is resolved.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LEARN = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")
A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
OUT = LEARN / "a016_diag"
TAU = 2 * np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def main() -> None:
    before = np.load(LEARN / "ww_gunw_reuse" / A016 / "full_arrays.npz")
    after = np.load(OUT / "a016_default_fixed.npz")
    prod = before["prod_unw"].astype(np.float64)
    pcc = before["prod_cc"]
    mask = before["mask"]
    coh = before["coh"]
    reg = mask & (pcc > 0) & np.isfinite(prod)

    def amb(unw):
        a = np.rint((unw.astype(np.float64) - prod) / TAU)
        return a - modal(a[reg])

    def m(unw):
        a = amb(unw)[reg]
        a = a[np.isfinite(a)]
        return 100 * np.mean(np.abs(a) < 0.5)

    ab = np.where(reg, amb(before["ww_unw"]), np.nan)
    aa = np.where(reg, amb(after["unw"]), np.nan)
    s = (slice(None, None, 3), slice(None, None, 3))
    fig, ax = plt.subplots(2, 2, figsize=(15, 14), constrained_layout=True)
    im = ax[0, 0].imshow(np.where(mask, coh, np.nan)[s], cmap="gray", vmin=0, vmax=1)
    ax[0, 0].set_title("coherence (large decorrelated body)")
    fig.colorbar(im, ax=ax[0, 0], shrink=0.7)
    im = ax[0, 1].imshow(np.where(reg, pcc, np.nan)[s], cmap="tab10")
    ax[0, 1].set_title("production connectedComponents")
    fig.colorbar(im, ax=ax[0, 1], shrink=0.7)
    im = ax[1, 0].imshow(ab[s], cmap="RdBu", vmin=-2, vmax=2)
    ax[1, 0].set_title(
        f"BEFORE: tile512 default ambiguity diff ({m(before['ww_unw']):.1f}% - winding + seam strip)"
    )
    fig.colorbar(im, ax=ax[1, 0], shrink=0.7)
    im = ax[1, 1].imshow(aa[s], cmap="RdBu", vmin=-2, vmax=2)
    ax[1, 1].set_title(
        f"AFTER: gated multi-shift default ({m(after['unw']):.1f}% - strip gone, winding fixed)"
    )
    fig.colorbar(im, ax=ax[1, 1], shrink=0.7)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    p = OUT / "a016_before_after.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(
        f"A_016 before={m(before['ww_unw']):.2f}%  after={m(after['unw']):.2f}%",
        flush=True,
    )
    print(f"PLOT: {p}", flush=True)


if __name__ == "__main__":
    main()
