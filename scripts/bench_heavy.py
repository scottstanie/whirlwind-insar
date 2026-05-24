#!/usr/bin/env python3
"""Single-scene heavy benchmark with multiple whirlwind-rs backends + snaphu.

Loads the .npz produced by `heavy_scene.py`, runs each requested configuration
once, and prints a wall-time / throughput / peak-RSS table.

Each whirlwind-rs configuration runs in its own subprocess so the
`WHIRLWIND_DIJKSTRA` OnceLock isn't pinned to the first value seen.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]


def peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def fmt_bytes(b: int) -> str:
    f = float(b)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if f < 1024 or unit == "GiB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{b} B"


@dataclass
class Row:
    label: str
    seconds: float
    peak_rss_bytes: int
    note: str = ""

    @property
    def mpx_per_s(self):
        return getattr(self, "_mpx", 0.0) / self.seconds if self.seconds > 0 else 0.0


def run_subprocess(scene_path, label, env_override=None, lib="ww"):
    """Spawn a child to time one configuration with a fresh process."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_override:
        env.update(env_override)
    code = f"""
import json, os, resource, sys, time
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64

z = np.load({str(scene_path)!r})
igram = z['igram']
corr  = z['corr']
mask = z['mask'] if 'mask' in z.files else None
nlooks = float(z['meta'][2])

t0 = time.perf_counter()
if {lib!r} == 'ww':
    import whirlwind_rs as ww
    unw = ww.unwrap(igram, corr, nlooks, mask)
elif {lib!r} == 'snaphu':
    import snaphu
    unw, _ = snaphu.unwrap(igram, corr, nlooks=nlooks, cost='smooth', mask=mask)
else:
    raise SystemExit('unknown lib')
elapsed = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != 'darwin':
    rss *= 1024
print(json.dumps({{'seconds': elapsed, 'rss': rss, 'shape': list(igram.shape)}}))
"""
    t_wall = time.perf_counter()
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - t_wall
    if res.returncode != 0:
        print(f"  {label}: FAILED rc={res.returncode}")
        print("    stderr:", res.stderr[-500:])
        return Row(label=label, seconds=float("nan"), peak_rss_bytes=0,
                   note=f"FAILED rc={res.returncode}: {res.stderr[-200:].strip()}")
    try:
        data = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return Row(label=label, seconds=float("nan"), peak_rss_bytes=0,
                   note=f"BAD STDOUT: {e}")
    row = Row(label=label, seconds=data["seconds"], peak_rss_bytes=data["rss"])
    print(f"  {label:30s}  {row.seconds:8.2f} s   peak {fmt_bytes(row.peak_rss_bytes)}  "
          f"(child wall {wall:.1f} s)")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=Path("/tmp/heavy_scene.npz"))
    p.add_argument("--no-snaphu", action="store_true")
    p.add_argument("--only", default=None,
                   help="comma-separated labels to run (substring match)")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).parent / "out" / "BENCH_HEAVY.md")
    args = p.parse_args()

    if not args.scene.exists():
        print(f"scene not found: {args.scene}\nrun scripts/heavy_scene.py first")
        return 2

    z = np.load(args.scene)
    H, W = z["igram"].shape
    nlooks = int(z["meta"][2])
    pixels = H * W
    print(f"\nScene: {args.scene}  shape={H}x{W}  nlooks={nlooks}  Mpx={pixels/1e6:.2f}\n")

    configs = [
        # (label, env_override, lib)
        ("whirlwind-rs (dial serial)", {"WHIRLWIND_DIJKSTRA": "dial"}, "ww"),
        ("whirlwind-rs (dial parallel)", {"WHIRLWIND_DIJKSTRA": "dial-par"}, "ww"),
        ("whirlwind-rs (heap)", {"WHIRLWIND_DIJKSTRA": "heap"}, "ww"),
    ]
    if not args.no_snaphu:
        configs.append(("snaphu (smooth cost)", None, "snaphu"))

    if args.only:
        wants = [s.strip().lower() for s in args.only.split(",")]
        configs = [c for c in configs
                   if any(w in c[0].lower() for w in wants)]

    rows: list[Row] = []
    for label, env, lib in configs:
        print(f"  starting: {label}", flush=True)
        r = run_subprocess(args.scene, label, env, lib=lib)
        r._mpx = pixels / 1e6  # type: ignore[attr-defined]
        rows.append(r)

    # Markdown summary.
    lines = [
        f"# whirlwind-rs heavy benchmark ({H}×{W}, nlooks={nlooks})\n",
        f"Scene: `{args.scene}`. Generated by `scripts/heavy_scene.py`. "
        f"Each row runs in a fresh subprocess.\n",
        "| config | wall-time (s) | Mpx/s | peak RSS |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        if not np.isfinite(r.seconds):
            lines.append(f"| {r.label} | n/a | n/a | n/a |  ({r.note})")
            continue
        lines.append(
            f"| {r.label} | {r.seconds:.2f} | {r.mpx_per_s:.2f} | "
            f"{fmt_bytes(r.peak_rss_bytes)} |"
        )

    out_md = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_md)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(
        [{**asdict(r), "mpx_per_s": r.mpx_per_s} for r in rows],
        indent=2,
    ))
    print("\n" + out_md)
    print(f"Wrote {args.out} and {json_path}\n")


if __name__ == "__main__":
    sys.exit(main())
