#!/usr/bin/env bash
# Sweep the Carballo aliased-gradient validity guard (see cost::SlopeGuard) on
# one frame's cached compare_gunw arrays.
#
# Each arm runs the SAME default solver and pipeline through
# scripts/phass_cost_ablation.py --variant carballo; only the guard env vars
# differ, so any score change is attributable to the guard alone.
#
#   scripts/run_slope_guard_sweep.sh <full_arrays.npz> <out-dir>
set -euo pipefail

NPZ="${1:?usage: run_slope_guard_sweep.sh <full_arrays.npz> <out-dir>}"
OUT="${2:?usage: run_slope_guard_sweep.sh <full_arrays.npz> <out-dir>}"
cd "$(dirname "$0")/.."

run() { # <label> <extra env assignments...>
  local label="$1"; shift
  printf '>>> %s\n' "$label"
  env PYTHONPATH=python "$@" \
    python scripts/phass_cost_ablation.py \
      --npz "$NPZ" --out-dir "$OUT/$label" --variant carballo 2>&1 |
    grep -E '^carballo:'
}

run baseline
for th in 0.8 1.0 1.4 2.0; do
  run "zerocost-${th}" WHIRLWIND_SLOPE_GUARD_RAD="$th" WHIRLWIND_SLOPE_GUARD_MODE=zerocost
done
# Diagnostic arm: predicted to be WORSE than baseline (a coherent flat-slope
# cut is expensive, so this makes aliased edges harder to cut, not easier).
run zeroslope-1.0 WHIRLWIND_SLOPE_GUARD_RAD=1.0 WHIRLWIND_SLOPE_GUARD_MODE=zeroslope
