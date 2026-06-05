"""Sequential driver for the PHASS-cost experiments.

Runs (always one job at a time on this laptop, see
~/.claude/projects/.../memory/concurrency_limit.md):

  1. SNAPHU PV reference            (~13s)
  2. PV {baseline, hard_cut, phass_cost, phass_full}    (~1-2 s each)
  3. NISAR {baseline, hard_cut, phass_cost, phass_full} (~85 s each)
  4. analyze.py - tables + plots

Writes a combined stdout log to <OUT>/run.log.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PY = "/Users/staniewi/miniforge3/envs/mapping-312/bin/python"
ROOT = Path("/Users/staniewi/repos/whirlwind-insar/scripts/phass_experiments")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
LOG = OUT.parent / "run.log"

JOBS: list[tuple[str, list[str]]] = [
    ("snaphu_pv", [PY, str(ROOT / "run" / "run_snaphu_pv.py")]),
    # TODO: find another, but for now, let's skip PV
    # ("pv/baseline",      [PY, str(ROOT / "run" / "run_one.py"), "pv",   "baseline"]),
    # ("pv/hard_cut",      [PY, str(ROOT / "run" / "run_one.py"), "pv",   "hard_cut"]),
    # ("pv/phass_cost",    [PY, str(ROOT / "run" / "run_one.py"), "pv",   "phass_cost"]),
    # ("pv/phass_full",    [PY, str(ROOT / "run" / "run_one.py"), "pv",   "phass_full"]),
    ("nisar/baseline", [PY, str(ROOT / "run" / "run_one.py"), "nisar", "baseline"]),
    ("nisar/hard_cut", [PY, str(ROOT / "run" / "run_one.py"), "nisar", "hard_cut"]),
    ("nisar/phass_cost", [PY, str(ROOT / "run" / "run_one.py"), "nisar", "phass_cost"]),
    ("nisar/phass_full", [PY, str(ROOT / "run" / "run_one.py"), "nisar", "phass_full"]),
    ("analyze", [PY, str(ROOT / "analyze" / "analyze.py")]),
]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w") as log:
        for tag, cmd in JOBS:
            banner = f"\n========== {tag} ==========\n"
            print(banner, end="", flush=True)
            log.write(banner)
            log.flush()
            t0 = time.perf_counter()
            res = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.perf_counter() - t0
            log.write(res.stdout)
            if res.stderr:
                log.write("--- stderr ---\n")
                log.write(res.stderr)
            log.flush()
            # Print a compact summary line so the user can follow progress.
            tail = (res.stdout.strip().splitlines() or [""])[-1]
            print(
                f"[{tag}] {dt:5.1f}s  rc={res.returncode}  | {tail[:120]}", flush=True
            )
            if res.returncode != 0:
                print(f"[{tag}] FAILED - full stderr above; aborting.", flush=True)
                return res.returncode
    print(f"\nlog: {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
