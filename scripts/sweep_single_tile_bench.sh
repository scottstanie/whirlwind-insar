#!/usr/bin/env bash
# Single-tile (verified) benchmark sweep over the NISAR GUNW frames.
#
# Runs the VERIFIED single-tile path (`bench_nisar_gunw_whirlwind.py --solver
# linear --nlooks 16`) on each frame in its OWN process, ONE AT A TIME (single-
# tile whole-image is ~6.6 GB/frame — never run these concurrently; see the
# concurrency note in memory). Captures per-frame:
#   * runtime + ambiguity match/percomp + a 3-panel plot   (from the bench)
#   * peak RSS                                              (from /usr/bin/time -l)
#
# Output: $OUT/<frame>/ (bench out-dir incl. summary.csv + PNG) and
#         $OUT/<frame>.time (the /usr/bin/time -l block). Combined table at the
# end via the companion summariser.
set -euo pipefail

H5DIR=${H5DIR:-/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw}
OUT=${OUT:-/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_singletile_sweep}
NLOOKS=${NLOOKS:-16}
mkdir -p "$OUT"

i=0
for h5 in "$H5DIR"/*.h5; do
  i=$((i + 1))
  frame=$(basename "$h5" | sed -E 's/.*_([AD]_[0-9]{3})_.*/\1/')
  printf '>>> [%d] %s\n' "$i" "$frame"
  /usr/bin/time -l python scripts/bench_nisar_gunw_whirlwind.py \
    --local-h5 "$h5" --solver linear --nlooks "$NLOOKS" \
    --out-dir "$OUT/$frame" --force \
    > "$OUT/$frame.stdout" 2> "$OUT/$frame.time" || {
      printf '    FRAME FAILED (exit %d) — see %s\n' "$?" "$OUT/$frame.time"
      continue
    }
  rss=$(grep "maximum resident set size" "$OUT/$frame.time" | grep -oE '[0-9]+' | head -1 || echo 0)
  swaps=$(grep -E '[0-9]+[[:space:]]+swaps' "$OUT/$frame.time" | grep -oE '^[[:space:]]*[0-9]+' | tr -d ' ' || echo '?')
  line=$(grep 'match=' "$OUT/$frame.stdout" | tail -1 || echo '(no match line)')
  printf '    peak_rss=%s bytes  swaps=%s  | %s\n' "$rss" "$swaps" "$line"
done
printf 'SWEEP DONE -> %s\n' "$OUT"
