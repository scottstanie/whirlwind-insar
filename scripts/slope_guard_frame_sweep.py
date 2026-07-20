#!/usr/bin/env python3
"""Sweep the Carballo aliased-gradient guard across many cached frames.

Each arm is a separate subprocess because `cost::slope_guard()` caches its
configuration in a `OnceLock` - one process can only ever see one setting.
Every arm runs the same default solver and pipeline via
`phass_cost_ablation.py --variant carballo`; only the guard env differs, so a
score change is attributable to the guard alone.

Frames are `compare_gunw.py` output directories containing `full_arrays.npz`,
so the inputs and the agreement metric are exactly the benchmark's.

  PYTHONPATH=python python scripts/slope_guard_frame_sweep.py \
    --compare-dir /Volumes/.../nisar_gunw_hardest/compare \
    --out-dir /Volumes/.../slope-guard-frames --arms baseline zerocost-1.0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


BUDGET_FLOOR_RAD = "1.0"


def arm_env(arm: str) -> dict[str, str]:
    """Arm name -> guard env vars.

    `baseline` | `zerocost-<rad>` | `zeroslope-<rad>` | `budget-<frac>[-<floor>]`

    The budget arm picks the threshold per frame as the `1 - frac` quantile of
    the raw |dphi| distribution, floored at `<floor>` radians (default 1.0), so
    the guard frees at most `frac` of the valid edges.
    """
    if arm == "baseline":
        return {}
    if arm.startswith("budget-"):
        parts = arm.split("-")
        frac = parts[1]
        floor = parts[2] if len(parts) > 2 else BUDGET_FLOOR_RAD
        return {
            "WHIRLWIND_SLOPE_GUARD_BUDGET": frac,
            "WHIRLWIND_SLOPE_GUARD_RAD": floor,
            "WHIRLWIND_SLOPE_GUARD_MODE": "zerocost",
        }
    mode, _, rad = arm.partition("-")
    if mode not in {"zerocost", "zeroslope"} or not rad:
        raise SystemExit(
            f"bad arm {arm!r}: use baseline, zerocost-<rad>, zeroslope-<rad>, "
            "budget-<frac>[-<floor>]"
        )
    return {"WHIRLWIND_SLOPE_GUARD_RAD": rad, "WHIRLWIND_SLOPE_GUARD_MODE": mode}


def short_name(product_dir: Path) -> str:
    """NISAR_L2_PR_GUNW_009_074_A_137_... -> 074_A_137."""
    parts = product_dir.name.split("_")
    return "_".join(parts[5:8]) if len(parts) > 8 else product_dir.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["baseline", "zerocost-1.0"])
    ap.add_argument("--nlooks", type=float, default=16.0)
    ap.add_argument("--frames", nargs="*", help="substring filter on product dir names")
    args = ap.parse_args()

    npzs = sorted(args.compare_dir.glob("*/full_arrays.npz"))
    if args.frames:
        npzs = [p for p in npzs if any(f in p.parent.name for f in args.frames)]
    if not npzs:
        raise SystemExit(f"no */full_arrays.npz under {args.compare_dir}")
    print(f"{len(npzs)} frames x {len(args.arms)} arms", flush=True)

    results: dict[str, dict[str, dict]] = {}
    for npz in npzs:
        name = short_name(npz.parent)
        results[name] = {}
        for arm in args.arms:
            out = args.out_dir / arm / npz.parent.name
            env = {**os.environ, "PYTHONPATH": "python", **arm_env(arm)}
            cmd = [
                sys.executable,
                str(REPO / "scripts" / "phass_cost_ablation.py"),
                "--npz",
                str(npz),
                "--out-dir",
                str(out),
                "--variant",
                "carballo",
                "--nlooks",
                str(args.nlooks),
            ]
            proc = subprocess.run(
                cmd, cwd=REPO, env=env, capture_output=True, text=True
            )
            if proc.returncode != 0:
                print(
                    f"  {name:<14} {arm:<16} FAILED\n{proc.stderr[-2000:]}", flush=True
                )
                continue
            stats = json.loads((out / "carballo" / "stats.json").read_text())
            results[name][arm] = stats
            print(
                f"  {name:<14} {arm:<16} per-comp={stats['ambiguity_match_frac_percomp']:.4f}"
                f"  match={stats['ambiguity_match_frac']:.4f}"
                f"  {stats['runtime_s']:.0f}s",
                flush=True,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sweep.json").write_text(
        json.dumps(results, indent=2, sort_keys=True)
    )

    base = args.arms[0]
    lines = [
        "| frame | " + " | ".join(args.arms) + " | delta vs " + base + " |",
        "| --- | " + " | ".join("---:" for _ in args.arms) + " | ---: |",
    ]
    deltas = []
    for name, per_arm in results.items():
        cells = []
        for arm in args.arms:
            s = per_arm.get(arm)
            cells.append(
                f"{s['ambiguity_match_frac_percomp'] * 100:.2f}%" if s else "-"
            )
        if base in per_arm and args.arms[-1] in per_arm:
            d = (
                per_arm[args.arms[-1]]["ambiguity_match_frac_percomp"]
                - per_arm[base]["ambiguity_match_frac_percomp"]
            ) * 100
            deltas.append(d)
            cells.append(f"{d:+.2f}")
        else:
            cells.append("-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if deltas:
        worst = min(deltas)
        lines += [
            "",
            f"frames improved: {sum(d > 0.01 for d in deltas)}, "
            f"unchanged: {sum(abs(d) <= 0.01 for d in deltas)}, "
            f"regressed: {sum(d < -0.01 for d in deltas)}; "
            f"worst regression {worst:+.2f} pp, mean {sum(deltas) / len(deltas):+.2f} pp",
        ]
    table = "\n".join(lines)
    (args.out_dir / "sweep.md").write_text(table + "\n")
    print("\n" + table, flush=True)
    print(f"\nwrote {args.out_dir / 'sweep.md'}", flush=True)


if __name__ == "__main__":
    main()
