#!/usr/bin/env bash
# Re-measure SNAPHU peak memory with a TREE-AWARE summed-RSS sampler, because
# /usr/bin/time -l undercounts SNAPHU's concurrent forked tile workers (9x9).
#
# STRICTLY SEQUENTIAL: one heavy unwrap at a time (this laptop crashes with >=3
# concurrent NISAR-scale unwraps). Resume-friendly: frames already in the CSV are
# skipped. Run in the background; tail the log to watch progress.
#
#   9x9  -> all 13 frames (the suspect, concurrent-fork config)
#   1x1  -> 005_D_077 only, as a same-frame cold-single-tile comparison + a check
#           that the tree sampler agrees with /usr/bin/time for a single process
set -uo pipefail

PY=/Users/staniewi/miniforge3/bin/python
REPO=/Users/staniewi/repos/whirlwind-insar
H5DIR=/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw
OUT=/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/snaphu_mem_recheck
mkdir -p "$OUT"
CSV="$OUT/snaphu_mem.csv"
[ -f "$CSV" ] || echo "frame,engine,tree_peak_bytes,peak_nproc,runtime_s" > "$CSV"

cleanup() { pkill -9 -f -- "snaphu_one.py" 2>/dev/null; true; }
trap cleanup EXIT INT TERM

cd "$REPO"

measure() {  # frame  engine  ntiles  h5  [--plot path]
  local frame="$1" engine="$2" ntiles="$3" h5="$4"; shift 4
  if grep -q "^$frame,$engine," "$CSV" 2>/dev/null; then
    echo ">>> $frame $engine (done, skip)"; return
  fi
  echo ">>> $frame $engine  ($(date +%H:%M:%S))"
  local log="$OUT/$frame.$engine.log"
  # peak_rss_tree runs ONE snaphu_one (which itself is one logical unwrap).
  "$PY" scripts/peak_rss_tree.py "$@" -- "$PY" scripts/snaphu_one.py "$h5" "$ntiles" \
    > "$log" 2>&1 || { echo "    FAILED (see $log)"; return; }
  local bytes nproc rt
  bytes=$(grep -oE 'tree peak RSS = [0-9.]+ GB \([0-9]+ bytes\)' "$log" | grep -oE '\([0-9]+' | tr -d '(')
  nproc=$(grep -oE 'up to [0-9]+ process' "$log" | grep -oE '[0-9]+')
  rt=$(grep -oE '[0-9]+\.[0-9]+s  per-comp' "$log" | grep -oE '[0-9.]+' | head -1)
  echo "$frame,$engine,${bytes:-?},${nproc:-?},${rt:-?}" >> "$CSV"
  echo "    tree_peak=$(awk "BEGIN{printf \"%.2f\",${bytes:-0}/1e9}") GB  nproc=${nproc:-?}  rt=${rt:-?}s"
}

# Same-frame cold-vs-reoptimize comparison on 005_D_077 (with RSS(t) plots).
D077=$(ls "$H5DIR"/*_D_077_*.h5 2>/dev/null | head -1)
if [ -n "$D077" ]; then
  measure 005_D_077 snaphu1_tree   1 "$D077" --plot "$OUT/005_D_077_snaphu1_rss.png"
  measure 005_D_077 snaphu9x9_tree 9 "$D077" --plot "$OUT/005_D_077_snaphu9x9_rss.png"
fi

# 9x9 across all 13 frames for the chart.
for h5 in "$H5DIR"/*.h5; do
  frame=$(basename "$h5" | sed -E 's/.*_([AD]_[0-9]{3})_.*/\1/')
  measure "$frame" snaphu9x9_tree 9 "$h5"
done

echo "DONE -> $CSV"
