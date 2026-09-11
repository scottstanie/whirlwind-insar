#!/usr/bin/env python3
"""View the ``remove_ramp`` A/B (PR #99) as phase, not as agreement-with-production.

Agreement with the production GUNW is only a proxy, and it is worthless on
frames where the production unwrap is itself wrong -- production IS SNAPHU, and
on fixed-PRF frames split into subswaths it mislevels them too. So this script
shows the unwrapped phase for both arms next to production and scores them with
a **reference-free** measure that needs no truth: how continuous the unwrapped
field is across the subswath gaps.

Cross-gap continuity
--------------------
Every error seen on these frames is a whole-subswath integer offset, so the
question is only "are neighbouring subswaths on the same level?". For each row,
each pair of consecutive valid runs is crossed by robustly extrapolating both
sides into the gap (median-of-differences slope, median intercept) and taking
the mismatch in cycles. A correctly leveled pair extrapolates to ~0 mismatch.

Two caveats, stated plainly:

1. It rewards the same continuity the bridge pass tries to achieve, so it is not
   independent of the algorithm under test. It is independent of *production*,
   which is the point.
2. **It has a large noise floor on these frames and cannot resolve small
   differences.** The NISAR fixed-PRF gaps here are all ~140 px (~11 km) wide --
   there are no narrow gaps to fall back on -- so both sides must be
   extrapolated ~70 px through data that does not exist. Production itself
   scores 0.33-0.46 cycles by this measure, and that is the floor, not a defect
   of production. Read the ``prod`` column as the per-frame control: only
   differences well outside it mean anything. Subtracting the fitted plane first
   (done below) does not help, which was checked -- the floor comes from the gap
   width, not from gradient steepness.

Usage::

    python scripts/plot_ramp_phase_view.py ww_gunw_ctrl ww_gunw_ramp \
        --skip 023_098 --out nisar-pngs/ramp_phase_view.png
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


def short(product_id: str) -> str:
    parts = product_id.split("_")
    return "_".join(parts[4:8]) if len(parts) > 8 else product_id[:20]


def _runs(row_valid: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    """Half-open [start, stop) spans of consecutive True at least min_run long."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], row_valid.view(np.int8), [0]))))
    return [(int(a), int(b)) for a, b in zip(idx[::2], idx[1::2]) if b - a >= min_run]


def _edge_fit(y: np.ndarray, x0: float) -> float:
    """Robustly extrapolate a 1-D phase segment to position ``x0``.

    Slope = median first difference, intercept = median residual. Median-based
    so a few decorrelated pixels in the edge window cannot swing the estimate.
    """
    if y.size < 4:
        return float("nan")
    slope = float(np.median(np.diff(y)))
    x = np.arange(y.size, dtype=np.float64)
    intercept = float(np.median(y - slope * x))
    return intercept + slope * x0


def cross_gap_mismatch(
    unw: np.ndarray,
    mask: np.ndarray,
    edge: int = 48,
    min_run: int = 96,
    max_gap: int = 400,
    row_stride: int = 8,
) -> np.ndarray:
    """Per-crossing |discontinuity| in cycles across every subswath gap."""
    out: list[float] = []
    for i in range(0, mask.shape[0], row_stride):
        runs = _runs(mask[i], min_run)
        for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
            gap = a2 - b1
            if not (2 <= gap <= max_gap):
                continue
            left = unw[i, max(a1, b1 - edge) : b1]
            right = unw[i, a2 : min(b2, a2 + edge)]
            if not (np.isfinite(left).all() and np.isfinite(right).all()):
                continue
            mid = gap / 2.0
            l_ex = _edge_fit(left, left.size - 1 + mid)
            r_ex = _edge_fit(right, -mid)
            if np.isfinite(l_ex) and np.isfinite(r_ex):
                out.append(abs(r_ex - l_ex) / TWOPI)
    return np.asarray(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ctrl_dir", type=Path)
    p.add_argument("ramp_dir", type=Path)
    p.add_argument("--out", type=Path, default=Path("nisar-pngs/ramp_phase_view.png"))
    p.add_argument(
        "--skip", nargs="*", default=[], help="Substrings of product ids to drop."
    )
    p.add_argument("--stride", type=int, default=4)
    args = p.parse_args()

    pids = sorted(
        (
            d.name
            for d in args.ctrl_dir.iterdir()
            if (d / "full.json").exists()
            and (args.ramp_dir / d.name / "full.json").exists()
            and not any(s in d.name for s in args.skip)
        ),
        key=short,
    )
    if not pids:
        raise SystemExit("No products to compare.")

    s = args.stride
    fig, axes = plt.subplots(
        len(pids), 3, figsize=(13.5, 4.3 * len(pids)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    hdr = (
        f"{'frame':<16} {'fringes':>8} | {'cross-gap discontinuity (cycles)':^38} | "
        f"{'vs prod (per-comp)':^20}"
    )
    print(hdr)
    print(
        f"{'':<16} {'':>8} | {'prod':>10} {'ramp OFF':>12} {'ramp ON':>12} | "
        f"{'OFF':>9} {'ON':>9}"
    )
    print("-" * len(hdr))

    for r, pid in enumerate(pids):
        a = json.loads((args.ctrl_dir / pid / "full.json").read_text())
        b = json.loads((args.ramp_dir / pid / "full.json").read_text())
        za = np.load(args.ctrl_dir / pid / "full_arrays.npz")
        zb = np.load(args.ramp_dir / pid / "full_arrays.npz")
        mask = za["mask"]
        prod = np.where(mask, za["prod_unw"], np.nan)
        off = np.where(mask, za["ww_aligned"], np.nan)
        on = np.where(mask, zb["ww_aligned"], np.nan)

        # Score continuity on the DE-RAMPED field: extrapolating across a gap
        # accumulates error in proportion to the local gradient, so a steeper
        # (but correct) field is unfairly penalised. Subtracting the same fitted
        # plane from all three candidates removes that bias without changing any
        # relative subswath level, which is the only thing being scored.
        ii, jj = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
        plane = (
            a.get("ramp_row_slope_rad_px", 0.0) * ii
            + a.get("ramp_col_slope_rad_px", 0.0) * jj
        )
        scores = {
            k: cross_gap_mismatch(v - plane, mask)
            for k, v in (("prod", prod), ("off", off), ("on", on))
        }
        med = {
            k: float(np.median(v)) if v.size else float("nan")
            for k, v in scores.items()
        }
        print(
            f"{short(pid):<16} {a.get('ramp_fringes_across_frame', float('nan')):>8.1f} | "
            f"{med['prod']:>10.3f} {med['off']:>12.3f} {med['on']:>12.3f} | "
            f"{a['ambiguity_match_frac_percomp']:>9.3f} "
            f"{b['ambiguity_match_frac_percomp']:>9.3f}"
        )

        # Shared robust color scale across the row, in cycles, after removing a
        # common median so the three panels are directly comparable by eye.
        panels = [
            (prod, f"production (SNAPHU)\ncross-gap {med['prod']:.2f} cyc"),
            (
                off,
                f"whirlwind, remove_ramp OFF\ncross-gap {med['off']:.2f} cyc, "
                f"per-comp vs prod {a['ambiguity_match_frac_percomp']:.3f}",
            ),
            (
                on,
                f"whirlwind, remove_ramp ON\ncross-gap {med['on']:.2f} cyc, "
                f"per-comp vs prod {b['ambiguity_match_frac_percomp']:.3f}",
            ),
        ]
        cyc = [np.asarray(v)[::s, ::s] / TWOPI for v, _ in panels]
        ref = np.nanmedian(cyc[0])
        cyc = [c - ref for c in cyc]
        lo, hi = np.nanpercentile(
            np.concatenate([c[np.isfinite(c)] for c in cyc]), [1, 99]
        )
        for c, (ax, (_, title)) in enumerate(zip(axes[r], panels)):
            im = ax.imshow(
                cyc[c], cmap="RdBu_r", vmin=lo, vmax=hi, interpolation="nearest"
            )
            ax.set_title((f"{short(pid)}   " if c == 0 else "") + title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8, label="cycles" if c == 2 else None)

    fig.suptitle(
        "remove_ramp (PR #99): unwrapped phase, and cross-gap continuity "
        "(reference-free)\nsame build, same inputs -- lower cross-gap = better leveled",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nWrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
