#!/usr/bin/env python3
"""Show where whirlwind labels conncomps that production drops, and what a
higher `conncomp_min_coherence` would do about it.

With the paired-sample `subswath` mask policy, water is solved through rather
than cut out -- which is what kept the tested frame one component. The
downside is that whirlwind then *labels* water as valid conncomp where
production reports 0. That is a label-side problem with a label-side knob:
the conncomp coherence floor. This plots the disagreement against coherence so
the floor can be picked from the data instead of guessed.

Reads the `<crop>_arrays.npz` that compare_gunw.py writes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

FLOORS = (0.08, 0.15, 0.20, 0.25, 0.30)
NLOOKS = 50.0  # compare_gunw.py's default
AUTO_FLOOR = min(max(0.32 / NLOOKS**0.5, 0.02), 0.30)  # whirlwind's "auto"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="<crop>_arrays.npz from compare_gunw.py")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--downsample", type=int, default=2)
    args = p.parse_args()

    f = np.load(args.npz)
    coh, mask = f["coh"], f["mask"]
    ww_cc, prod_cc = f["ww_cc"], f["prod_cc"]

    disagree = mask & (prod_cc == 0) & (ww_cc > 0)
    agree_lab = mask & (prod_cc > 0) & (ww_cc > 0)

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.4))
    s = slice(None, None, args.downsample)

    axes[0].imshow(np.where(mask, coh, np.nan)[s, s], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("coherence (valid domain)")

    cat = np.zeros(mask.shape, np.uint8)
    cat[agree_lab] = 1
    cat[disagree] = 2
    cmap = ListedColormap(["white", "0.82", "crimson"])
    axes[1].imshow(cat[s, s], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axes[1].set_title(
        f"crimson = ww labels, production drops\n"
        f"{disagree.sum():,} px ({disagree.sum() / mask.sum():.1%} of valid)"
    )

    bins = np.linspace(0, 1, 60)
    axes[2].hist(
        coh[agree_lab],
        bins=bins,
        density=True,
        alpha=0.65,
        label="production keeps (cc>0)",
        color="0.35",
    )
    axes[2].hist(
        coh[disagree],
        bins=bins,
        density=True,
        alpha=0.65,
        label="production drops (cc=0)",
        color="crimson",
    )
    axes[2].set_xlabel("coherence")
    axes[2].set_ylabel("density")
    axes[2].set_title("the two populations separate cleanly")
    axes[2].legend(fontsize=9)

    kept_prod0 = [(disagree & (coh >= t)).sum() / disagree.sum() for t in FLOORS]
    cost_valid = [1 - (mask & (coh >= t)).sum() / mask.sum() for t in FLOORS]
    axes[3].plot(
        FLOORS,
        [100 * k for k in kept_prod0],
        "o-",
        color="crimson",
        label="of the disagreement, still labeled",
    )
    axes[3].plot(
        FLOORS,
        [100 * c for c in cost_valid],
        "s-",
        color="0.35",
        label="of the whole valid domain, dropped",
    )
    axes[3].set_xlabel("conncomp_min_coherence floor")
    axes[3].set_ylabel("percent")
    axes[3].set_title(
        "floor picks off the disagreement\nfaster than it costs good data"
    )
    axes[3].grid(alpha=0.3)
    axes[3].legend(fontsize=9)
    axes[3].axvline(AUTO_FLOOR, ls=":", c="k")
    axes[3].annotate(
        f"'auto' at {NLOOKS:g} looks\n(0.32/sqrt(nlooks) = {AUTO_FLOOR:.3f})",
        (AUTO_FLOOR, 60),
        fontsize=8,
        ha="left",
        va="center",
    )

    for a in axes[:2]:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(args.npz.parent.name[:80], fontsize=9)
    fig.tight_layout()
    out = args.out or args.npz.parent / "conncomp_water_floor.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")

    print(
        f"\n{'floor':>7} {'disagreement still labeled':>28} {'valid domain dropped':>22}"
    )
    for t, k, c in zip(FLOORS, kept_prod0, cost_valid):
        print(f"{t:>7.2f} {100 * k:>27.1f}% {100 * c:>21.1f}%")


if __name__ == "__main__":
    main()
