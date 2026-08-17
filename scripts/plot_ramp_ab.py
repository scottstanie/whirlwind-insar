#!/usr/bin/env python3
"""A/B the ``remove_ramp`` pre-pass (PR #99) on NISAR GUNW frames.

Reads two ``compare_gunw.py`` output trees produced from the SAME inputs with
the same whirlwind build -- one with ``--remove-ramp`` and one without -- and
reports what the pre-pass changed: the per-component ambiguity agreement
against the production unwrap, the component count, the runtime, and the size
of the ramp that was fitted.

Usage::

    python scripts/plot_ramp_ab.py ww_gunw_ctrl ww_gunw_ramp \
        --out nisar-pngs/ramp_ab.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TWOPI = 2.0 * np.pi


def load_tree(root: Path) -> dict[str, dict]:
    """Map product id -> the ``full.json`` metrics under an out-dir."""
    out = {}
    for js in sorted(root.glob("*/full.json")):
        out[js.parent.name] = json.loads(js.read_text())
    return out


def short(product_id: str) -> str:
    """``NISAR_L2_PR_GUNW_023_076_A_024_...`` -> ``023_076_A_024``."""
    parts = product_id.split("_")
    return "_".join(parts[4:8]) if len(parts) > 8 else product_id[:20]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ctrl_dir", type=Path, help="Baseline out-dir (no --remove-ramp).")
    p.add_argument("ramp_dir", type=Path, help="Out-dir run with --remove-ramp.")
    p.add_argument("--out", type=Path, default=Path("nisar-pngs/ramp_ab.png"))
    p.add_argument(
        "--stride", type=int, default=4, help="Decimation for the raster panels."
    )
    args = p.parse_args()

    ctrl = load_tree(args.ctrl_dir)
    ramp = load_tree(args.ramp_dir)
    common = sorted(set(ctrl) & set(ramp), key=short)
    if not common:
        raise SystemExit(f"No products in both {args.ctrl_dir} and {args.ramp_dir}")

    hdr = (
        f"{'frame':<16} {'fringes':>8} {'per-comp off':>13} {'per-comp on':>12} "
        f"{'delta':>7} {'cc off':>7} {'cc on':>6} {'s off':>7} {'s on':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for pid in common:
        a, b = ctrl[pid], ramp[pid]
        pc_a = a["ambiguity_match_frac_percomp"]
        pc_b = b["ambiguity_match_frac_percomp"]
        rows.append((pid, a, b))
        print(
            f"{short(pid):<16} {a.get('ramp_fringes_across_frame', float('nan')):>8.1f} "
            f"{pc_a:>13.4f} {pc_b:>12.4f} {pc_b - pc_a:>+7.4f} "
            f"{a['ww_num_cc']:>7} {b['ww_num_cc']:>6} "
            f"{a['runtime_s']:>7.1f} {b['runtime_s']:>7.1f}"
        )
    d = np.array(
        [
            r[2]["ambiguity_match_frac_percomp"] - r[1]["ambiguity_match_frac_percomp"]
            for r in rows
        ]
    )
    print("-" * len(hdr))
    print(
        f"mean delta {d.mean():+.4f}   median {np.median(d):+.4f}   "
        f"better {int((d > 1e-4).sum())}/{len(d)}  worse {int((d < -1e-4).sum())}/{len(d)}"
    )

    # One row per frame: production unwrap, then the ambiguity-difference map
    # for each arm (0 = agrees with production on the 2pi integer).
    s = args.stride
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(13, 4.0 * len(rows)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for r, (pid, a, b) in enumerate(rows):
        za = np.load(args.ctrl_dir / pid / "full_arrays.npz")
        zb = np.load(args.ramp_dir / pid / "full_arrays.npz")
        valid = za["mask"][::s, ::s]
        prod = np.where(valid, za["prod_unw"][::s, ::s], np.nan)
        axes[r, 0].imshow(prod, cmap="RdBu_r", interpolation="nearest")
        axes[r, 0].set_title(
            f"{short(pid)}  production unwrap\n"
            f"{a.get('ramp_fringes_across_frame', float('nan')):.1f} fringes of ramp",
            fontsize=9,
        )
        for c, (z, arm, stats) in enumerate(
            ((za, "remove_ramp OFF", a), (zb, "remove_ramp ON", b)), start=1
        ):
            ad = np.where(valid, z["ambiguity_diff"][::s, ::s], np.nan)
            lim = max(1.0, float(np.nanpercentile(np.abs(ad), 99)))
            im = axes[r, c].imshow(
                ad, cmap="RdBu", vmin=-lim, vmax=lim, interpolation="nearest"
            )
            axes[r, c].set_title(
                f"{arm}: ambiguity diff (cycles)\n"
                f"per-comp={stats['ambiguity_match_frac_percomp']:.3f}, "
                f"cc={stats['ww_num_cc']}, {stats['runtime_s']:.0f}s",
                fontsize=9,
            )
            fig.colorbar(im, ax=axes[r, c], shrink=0.8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "remove_ramp pre-pass (PR #99) vs baseline -- same build, same inputs",
        fontsize=13,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nWrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
