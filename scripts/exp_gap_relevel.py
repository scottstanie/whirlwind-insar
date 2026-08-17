#!/usr/bin/env python3
"""Experiment: re-level NISAR subswaths by gradient extrapolation, not minimum jump.

Motivation
----------
On NISAR fixed-PRF frames the valid mask splits the scene into subswath stripes
separated by ~140 px (~11 km) gaps. Each stripe unwraps independently, so the
only thing left to decide is one integer cycle offset per stripe -- and every
error seen on these frames is exactly that.

Two existing answers:

* **whirlwind** bridges: pick the offset that minimises the jump across the gap.
  That is right only when the true cross-gap phase change is under half a cycle,
  i.e. when the bulk (ionospheric) phase is nearly flat.
* **ISCE3 production** never bridges: it distance-fills the invalid pixels before
  SNAPHU, so SNAPHU integrates through a connected frame.

Note that *naive* gap interpolation is not a third answer -- filling between the
two edge values and integrating through recovers the short-arc difference, which
is the same integer minimum-jump already picks. To do better the fill has to
**continue the fringe gradient** across the gap.

What this tries
---------------
For every row crossing a gap, extrapolate the unwrapped phase from both sides
into the gap (robust median-of-differences slope) and take the mismatch in
cycles. Per row that estimate is noisy -- measured at ~0.4 cycles, comparable to
the half-cycle decision boundary -- but a gap spans thousands of rows, so the
per-gap median should be far tighter. Round the per-gap median to an integer,
then propagate offsets over a spanning tree rooted at the largest stripe.

This is a pure post-process on cached ``compare_gunw`` output: no re-unwrapping,
so it isolates the leveling rule from everything else.

Usage::

    python scripts/exp_gap_relevel.py ww_gunw_ctrl ww_gunw_ramp \
        --out nisar-pngs/gap_relevel.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

TWOPI = 2.0 * np.pi
REPO = Path(__file__).resolve().parent.parent


def _load_compare_gunw():
    """Import the metric definitions from the (hyphenated-dir) harness."""
    spec = importlib.util.spec_from_file_location(
        "compare_gunw", REPO / "aws-batch" / "compare_gunw.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def short(product_id: str) -> str:
    parts = product_id.split("_")
    return "_".join(parts[4:8]) if len(parts) > 8 else product_id[:20]


def _runs(row_valid: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.concatenate(([0], row_valid.view(np.int8), [0]))))
    return [(int(a), int(b)) for a, b in zip(idx[::2], idx[1::2])]


def _extrap(y: np.ndarray, x0: float) -> float:
    """Robustly extrapolate a 1-D phase segment to ``x0`` (median slope/intercept)."""
    slope = float(np.median(np.diff(y)))
    x = np.arange(y.size, dtype=np.float64)
    return float(np.median(y - slope * x)) + slope * x0


def gap_offsets(
    unw: np.ndarray,
    mask: np.ndarray,
    coh: np.ndarray,
    labels: np.ndarray,
    edge: int = 64,
    max_gap: int = 300,
    min_coh: float = 0.3,
) -> dict[tuple[int, int], tuple[float, int]]:
    """Median cross-gap mismatch (cycles) for every adjacent stripe pair."""
    acc: dict[tuple[int, int], list[float]] = defaultdict(list)
    for i in range(mask.shape[0]):
        runs = _runs(mask[i])
        for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
            gap = a2 - b1
            if not (2 <= gap <= max_gap):
                continue
            ll, lr = int(labels[i, b1 - 1]), int(labels[i, a2])
            if ll == lr or ll == 0 or lr == 0:
                continue
            left = unw[i, max(a1, b1 - edge) : b1]
            right = unw[i, a2 : min(b2, a2 + edge)]
            if left.size < 16 or right.size < 16:
                continue
            # Only trust rows whose edge windows are actually coherent; a
            # decorrelated edge gives a meaningless slope.
            if (
                np.mean(coh[i, max(a1, b1 - edge) : b1]) < min_coh
                or np.mean(coh[i, a2 : min(b2, a2 + edge)]) < min_coh
            ):
                continue
            if not (np.isfinite(left).all() and np.isfinite(right).all()):
                continue
            mid = gap / 2.0
            mismatch = (
                _extrap(right, -mid) - _extrap(left, left.size - 1 + mid)
            ) / TWOPI
            key = (ll, lr) if ll < lr else (lr, ll)
            acc[key].append(mismatch if ll < lr else -mismatch)
    return {k: (float(np.median(v)), len(v)) for k, v in acc.items() if len(v) >= 32}


def relevel(
    unw: np.ndarray, labels: np.ndarray, edges: dict[tuple[int, int], tuple[float, int]]
) -> tuple[np.ndarray, dict[int, int]]:
    """Propagate integer offsets over a spanning tree rooted at the largest stripe."""
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    root = int(np.argmax(sizes))
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (a, b), (med, _n) in edges.items():
        adj[a].append((b, med))  # phase(b) - phase(a) = med cycles
        adj[b].append((a, -med))
    offsets = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v, med in adj[u]:
            if v in offsets:
                continue
            # v currently sits `med` cycles off where u's level implies; move it
            # by the nearest integer so the extrapolated fringes line up.
            offsets[v] = offsets[u] - int(round(med))
            q.append(v)
    out = unw.copy()
    for lab, k in offsets.items():
        if k:
            out[labels == lab] += TWOPI * k
    return out, offsets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dirs", nargs="+", type=Path, help="compare_gunw out-dirs to fix up."
    )
    p.add_argument("--out", type=Path, default=Path("nisar-pngs/gap_relevel.png"))
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--stride", type=int, default=4)
    p.add_argument(
        "--min-area-frac",
        type=float,
        default=0.002,
        help="Ignore mask islands smaller than this fraction of the frame.",
    )
    args = p.parse_args()

    cg = _load_compare_gunw()
    panels = []
    hdr = f"{'arm':<10} {'frame':<16} {'per-comp before':>16} {'per-comp after':>15} {'delta':>8}  offsets"
    print(hdr)
    print("-" * len(hdr))

    for d in args.dirs:
        for sub in sorted(d.iterdir(), key=lambda q: q.name):
            if not (sub / "full_arrays.npz").exists():
                continue
            if any(s in sub.name for s in args.skip):
                continue
            z = np.load(sub / "full_arrays.npz")
            stats0 = json.loads((sub / "full.json").read_text())
            mask, coh = z["mask"], z["coh"]
            unw = np.where(mask, z["ww_aligned"], np.nan)

            labels, _n = ndimage.label(mask)
            sizes = np.bincount(labels.ravel())
            small = np.flatnonzero(sizes < args.min_area_frac * mask.size)
            labels[np.isin(labels, small)] = 0

            edges = gap_offsets(unw, mask, coh, labels)
            fixed, offs = relevel(unw, labels, edges)

            new = np.where(mask, fixed, 0.0).astype(np.float32)
            stats1, aligned1, _resid, amb1 = cg.compute_compare_stats(
                ig=z["ig"],
                coh=coh,
                mask=mask,
                prod_unw=z["prod_unw"],
                prod_cc=z["prod_cc"],
                ww_unw=new,
                ww_cc=z["ww_cc"] if z["ww_cc"].size else None,
                runtime_s=0.0,
                rss_delta_mb=None,
            )
            before = stats0["ambiguity_match_frac_percomp"]
            after = stats1["ambiguity_match_frac_percomp"]
            arm = "ramp ON" if "ramp" in d.name else "ramp OFF"
            nz = {k: v for k, v in sorted(offs.items()) if v}
            print(
                f"{arm:<10} {short(sub.name):<16} {before:>16.4f} {after:>15.4f} "
                f"{after - before:>+8.4f}  {nz if nz else 'none'}"
            )
            panels.append(
                (arm, short(sub.name), before, after, z["ambiguity_diff"], amb1, mask)
            )

    s = args.stride
    fig, axes = plt.subplots(
        len(panels), 2, figsize=(9.5, 4.2 * len(panels)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for r, (arm, name, before, after, amb0, amb1, mask) in enumerate(panels):
        for c, (amb, tag, val) in enumerate(
            (
                (amb0, "bridge (minimum jump)", before),
                (amb1, "gradient re-level", after),
            )
        ):
            a = np.where(mask[::s, ::s], amb[::s, ::s], np.nan)
            lim = max(1.0, float(np.nanpercentile(np.abs(a), 99)))
            im = axes[r, c].imshow(a, cmap="RdBu", vmin=-lim, vmax=lim)
            axes[r, c].set_title(
                f"{name} [{arm}]\n{tag}: per-comp {val:.3f}", fontsize=9
            )
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            fig.colorbar(im, ax=axes[r, c], shrink=0.8)
    fig.suptitle(
        "Subswath leveling: minimum-jump bridge vs cross-gap gradient extrapolation\n"
        "(post-process on cached arrays; 0 = agrees with production)",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nWrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
