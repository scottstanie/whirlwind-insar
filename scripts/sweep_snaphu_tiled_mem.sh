#!/usr/bin/env bash
# Tree-sampled SNAPHU tiled memory + runtime sweep (default 3x3, overlap set in
# snaphu_one.py). /usr/bin/time undercounts the concurrent tile workers, so this
# uses scripts/peak_rss_tree.py (whole-tree summed RSS).
#
# Primary: all 13 frames at default nproc (cpu_count -> all 9 tiles parallel).
# Plus: 005_A_025 capped at nproc=4 as the "report both" low-memory example.
#
# STRICTLY SEQUENTIAL (one heavy unwrap at a time). Resume-friendly.
set -uo pipefail
PY=/Users/staniewi/miniforge3/bin/python
TILES=${TILES:-3}
H5DIR=/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw
OUT=/Users/staniewi/Documents/Learning/snaphu_${TILES}x${TILES}_recheck
mkdir -p "$OUT"
CSV="$OUT/snaphu_tiled.csv"
[ -f "$CSV" ] || echo "frame,engine,tree_peak_bytes,peak_nproc,runtime_s" > "$CSV"
cleanup() { pkill -9 -f -- "snaphu_one.py" 2>/dev/null; true; }
trap cleanup EXIT INT TERM
cd /Users/staniewi/repos/whirlwind-insar

run() {  # frame engine h5 nproc(optional)
  local fr="$1" eng="$2" h5="$3" np="${4:-}"
  grep -q "^$fr,$eng," "$CSV" 2>/dev/null && { echo ">>> $fr $eng (skip)"; return; }
  echo ">>> $fr $eng  ($(date +%H:%M:%S))"
  local log="$OUT/$fr.$eng.log"
  if [ -n "$np" ]; then
    "$PY" scripts/peak_rss_tree.py -- "$PY" scripts/snaphu_one.py "$h5" "$TILES" "$np" > "$log" 2>&1
  else
    "$PY" scripts/peak_rss_tree.py -- "$PY" scripts/snaphu_one.py "$h5" "$TILES" > "$log" 2>&1
  fi
  local b n rt
  b=$(grep -oE '\([0-9]+ bytes\)' "$log" | grep -oE '[0-9]+')
  n=$(grep -oE 'up to [0-9]+ process' "$log" | grep -oE '[0-9]+')
  rt=$(grep -oE '[0-9]+\.[0-9]+s  per-comp' "$log" | grep -oE '[0-9.]+' | head -1)
  echo "$fr,$eng,${b:-?},${n:-?},${rt:-?}" >> "$CSV"
  echo "    $(awk "BEGIN{printf \"%.2f\",${b:-0}/1e9}") GB  nproc=${n:-?}  rt=${rt:-?}s"
}

# Capped-concurrency example on 005_A_025.
A025=$(ls "$H5DIR"/*_A_025_*.h5 2>/dev/null | head -1)
[ -n "$A025" ] && run 005_A_025 snaphu_np4 "$A025" 4

# All frames, all tiles parallel (default nproc = cpu_count).
for h5 in "$H5DIR"/*.h5; do
  fr=$(basename "$h5" | sed -E 's/.*_([AD]_[0-9]{3})_.*/\1/')
  run "$fr" snaphu_par "$h5"
done
echo "DONE -> $CSV"
