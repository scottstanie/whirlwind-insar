"""Validate a whirlwind-rs unwrap_stack run against dolphin's per-IG SNAPHU.

dolphin/unwrapped/*.unw.tif are dolphin's per-IG SNAPHU outputs (no temporal
closure correction, no anchoring) — the right apples-to-apples baseline for
our per-IG closure-corrected unwrap. NOTE: dolphin/timeseries/*.tif is a
different object (SBAS-inverted displacement in METERS), not directly
comparable to per-IG phase outputs.

For every IG we have in common, this script:
  - reads ours (already anchored to a reference pixel in the orchestrator)
  - reads dolphin's SNAPHU output, anchors it to *our* reference pixel
  - computes per-pixel abs diff and mod-2π diff
  - reports per-IG and aggregate statistics
  - (optionally) writes a JSON results file for downstream table-making

Usage:
    uv run python scripts/compare_to_dolphin_unwrapped.py \\
        --ours    /path/to/whirlwind/out \\
        --dolphin /path/to/dolphin \\
        [--json   /path/to/results.json]
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
                   help="dolphin output dir (with unwrapped/)")
    p.add_argument("--json", type=Path, default=None,
                   help="optional path to write per-IG metrics as JSON")
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

    unw_dir = args.dolphin / "unwrapped"
    if not unw_dir.is_dir():
        raise FileNotFoundError(f"no {unw_dir}")

    per_ig: list[dict] = []
    for e in edges:
        ig_name = f"{e['from']}_{e['to']}"
        ours_path = args.ours / "corrected" / f"{ig_name}.unw.tif"
        theirs_path = unw_dir / f"{ig_name}.unw.tif"
        if not (ours_path.exists() and theirs_path.exists()):
            continue

        with rasterio.open(ours_path) as src:
            ours = src.read(1)
        with rasterio.open(theirs_path) as src:
            theirs = src.read(1, window=crop)
        if ours.shape != theirs.shape:
            print(f"  shape mismatch on {ig_name}: ours {ours.shape}, theirs {theirs.shape}")
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
        per_ig.append({
            "ig": ig_name, "pct_within_pi2": pct,
            "abs_rms_rad": rms, "abs_max_rad": mx,
            "valid_pixels": int(valid.sum()),
        })
        n_compared += 1

    if not n_compared:
        print("\nNo IGs in common — check that unwrapped/ contains the same date pairs.")
        return

    summary = {
        "n_compared":               n_compared,
        "median_pct_within_pi2":    float(np.median(pct_within_pi2)),
        "min_pct_within_pi2":       float(np.min(pct_within_pi2)),
        "median_abs_rms_rad":       float(np.median(abs_rms)),
        "median_abs_max_rad":       float(np.median(abs_max)),
        "median_valid_pixels":      int(np.median(n_per_ig)),
        "n_igs_with_rms_lt_1rad":   sum(1 for d in abs_rms if d < 1.0),
    }
    print(f"\nIGs compared: {n_compared}")
    print(f"Median % within π/2 (mod 2π):     {summary['median_pct_within_pi2']:.2f}%")
    print(f"Min    % within π/2 across IGs:   {summary['min_pct_within_pi2']:.2f}%")
    print(f"Median ABSOLUTE RMS diff:         {summary['median_abs_rms_rad']:.4g} rad")
    print(f"Median ABSOLUTE max diff:         {summary['median_abs_max_rad']:.4g} rad")
    print(f"Median valid pixels per IG:       {summary['median_valid_pixels']:,}")
    print(f"IGs with absolute RMS < 1 rad:    {summary['n_igs_with_rms_lt_1rad']}/{n_compared}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "per_ig": per_ig}, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
