"""Visualize WHERE A_016 fails: ambiguity_diff for tile512 (fails, 57%) vs
tile2048 (works, 97%), beside coherence and the production conncomps. Confirms
whether the error is one drifted region (a neck the small tiles can't span) or
scattered, and shows the decorrelation neck geometry.

Outputs a PNG; path printed at the end.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LEARN = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")
A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/a016_diag")
OUT.mkdir(parents=True, exist_ok=True)


def load(variant_dir: Path):
    d = np.load(variant_dir / A016 / "full_arrays.npz")
    return d


def main() -> None:
    t512 = load(LEARN / "ww_gunw_bench")  # tile512 linear, 57%
    t2048 = load(LEARN / "ww_gunw_variants/tile2048")  # tile2048, 97%

    coh = t512["coh"]
    mask = t512["mask"]
    pcc = t512["prod_cc"]
    amb512 = np.where(mask, t512["ambiguity_diff"], np.nan)
    amb2048 = np.where(mask, t2048["ambiguity_diff"], np.nan)
    cohm = np.where(mask, coh, np.nan)
    pccm = np.where(mask, pcc, np.nan)

    s = (slice(None, None, 3), slice(None, None, 3))
    fig, ax = plt.subplots(2, 2, figsize=(15, 14), constrained_layout=True)
    im0 = ax[0, 0].imshow(cohm[s], cmap="gray", vmin=0, vmax=1)
    ax[0, 0].set_title("coherence (dark = decorrelation neck)")
    fig.colorbar(im0, ax=ax[0, 0], shrink=0.7)
    im1 = ax[0, 1].imshow(pccm[s], cmap="tab10")
    ax[0, 1].set_title(f"production connectedComponents (n={int(np.nanmax(pccm))})")
    fig.colorbar(im1, ax=ax[0, 1], shrink=0.7)
    im2 = ax[1, 0].imshow(amb512[s], cmap="RdBu", vmin=-2, vmax=2)
    ax[1, 0].set_title("ambiguity diff: tile512 (57% match) - RED/BLUE = wrong integer")
    fig.colorbar(im2, ax=ax[1, 0], shrink=0.7)
    im3 = ax[1, 1].imshow(amb2048[s], cmap="RdBu", vmin=-2, vmax=2)
    ax[1, 1].set_title("ambiguity diff: tile2048 (97% match)")
    fig.colorbar(im3, ax=ax[1, 1], shrink=0.7)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    p = OUT / "a016_failure_geometry.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)

    # quantify: per-column fraction wrong at tile512, to locate the drift frontier
    wrong = np.abs(amb512) >= 0.5
    colwrong = np.nansum(np.where(mask, wrong, 0), axis=0) / np.maximum(
        1, mask.sum(axis=0)
    )
    # report the column ranges where >50% of valid pixels are wrong
    bad_cols = np.where(colwrong > 0.5)[0]
    if bad_cols.size:
        print(
            f"tile512: columns with >50% wrong: {bad_cols.min()}..{bad_cols.max()} "
            f"({bad_cols.size} cols of {mask.shape[1]})",
            flush=True,
        )
    # coherence valley: per-column mean coherence
    colcoh = np.nansum(np.where(mask, coh, 0), axis=0) / np.maximum(1, mask.sum(axis=0))
    valley = np.argsort(colcoh)[:10]
    print(
        f"lowest-coherence columns (neck candidates): {sorted(valley.tolist())}",
        flush=True,
    )
    print(f"PLOT: {p}", flush=True)


if __name__ == "__main__":
    main()
