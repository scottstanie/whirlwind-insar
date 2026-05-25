"""Validate a whirlwind-rs unwrap_stack run against dolphin's `timeseries/` outputs.

dolphin/timeseries/ contains dolphin's *closure-corrected, anchored* time series
— their full 3D-style output, which is the proper baseline for our 3D pipeline
(not the per-IG `unwrapped/` files which are SNAPHU outputs with no temporal
closure or anchoring applied).

For every IG we have in common, this script:
  - reads ours (already anchored to a reference pixel in the orchestrator)
  - reads dolphin's, anchors it to *our* reference pixel for fairness
  - computes per-pixel abs diff and mod-2π diff
  - reports per-IG and aggregate statistics

Usage:
    uv run python scripts/compare_to_dolphin_timeseries.py \\
        --ours    /path/to/whirlwind/out \\
        --dolphin /path/to/dolphin
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

print = functools.partial(print, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, required=True,
                   help="whirlwind-rs unwrap_stack output dir (with report.json)")
    p.add_argument("--dolphin", type=Path, required=True,
                   help="dolphin output dir (with timeseries/)")
    args = p.parse_args()

    report = json.loads((args.ours / "report.json").read_text())
    edges = report["edges"]
    ref = report["reference_pixel"]
    win = report.get("window")
    if win is not None:
        crop = Window(win[1], win[0], win[3] - win[1], win[2] - win[0])  # type: ignore[call-arg]
    else:
        crop = None

    print(f"Comparing {len(edges)} IGs")
    print(f"Our reference pixel: ({ref['i']}, {ref['j']}) — {ref['source']}")

    n_compared = 0
    pct_within_pi2 = []
    abs_rms = []
    abs_max = []
    n_per_ig = []

    ts_dir = args.dolphin / "timeseries"
    if not ts_dir.is_dir():
        raise FileNotFoundError(f"no {ts_dir}")

    for e in edges:
        name = f"{e['from']}_{e['to']}.tif"           # timeseries/
        alt  = f"{e['from']}_{e['to']}.unw.tif"       # corrected/
        ours_path = args.ours / "corrected" / alt
        theirs_path = ts_dir / name
        if not (ours_path.exists() and theirs_path.exists()):
            continue

        with rasterio.open(ours_path) as src:
            ours = src.read(1)
        with rasterio.open(theirs_path) as src:
            theirs = src.read(1, window=crop)
        if ours.shape != theirs.shape:
            print(f"  shape mismatch on {name}: ours {ours.shape}, theirs {theirs.shape}")
            continue

        theirs_anchored = theirs - theirs[ref["i"], ref["j"]]
        valid = np.isfinite(ours) & np.isfinite(theirs_anchored) & (theirs != 0)
        if not valid.any():
            continue

        diff = ours[valid] - theirs_anchored[valid]
        diff_mod = np.angle(np.exp(1j * diff))
        pct = float(100 * np.mean(np.abs(diff_mod) < np.pi / 2))
        rms = float(np.sqrt(np.mean(diff ** 2)))
        mx = float(np.abs(diff).max())

        pct_within_pi2.append(pct)
        abs_rms.append(rms)
        abs_max.append(mx)
        n_per_ig.append(int(valid.sum()))
        n_compared += 1

    if not n_compared:
        print("\nNo IGs in common — check that timeseries/ contains the same date pairs.")
        return

    print(f"\nIGs compared: {n_compared}")
    print(f"Median % within π/2 (mod 2π):     {np.median(pct_within_pi2):.2f}%")
    print(f"Min    % within π/2 across IGs:   {np.min(pct_within_pi2):.2f}%")
    print(f"Median ABSOLUTE RMS diff:         {np.median(abs_rms):.4g} rad")
    print(f"Median ABSOLUTE max diff:         {np.median(abs_max):.4g} rad")
    print(f"Median valid pixels per IG:       {int(np.median(n_per_ig)):,}")
    n_strong = sum(1 for d in abs_rms if d < 1.0)
    print(f"IGs with absolute RMS < 1 rad:    {n_strong}/{n_compared}")


if __name__ == "__main__":
    main()
