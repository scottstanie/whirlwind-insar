#!/usr/bin/env bash
# 4-way single-tile sweep over the NISAR GUNW frames:
#   whirlwind (public unwrap, = single-tile linear adaptive default)
#   ww-orig   (Python reference)
#   PHASS     (isce3/tophu)
#   ICU       (isce3/tophu)
# Each engine runs in its OWN process under `/usr/bin/time -l` so runtime AND
# peak RSS are measured separately. ONE heavy unwrap at a time (see memory).
# snaphu is NOT re-run (production GUNW unwrap IS snaphu; snaphu_ref.log has the
# offline single-tile timing). Resume-friendly: rows already in results.csv are
# skipped. Run FOREGROUND in batches (≤10 min tool timeout); re-run to continue.
set -uo pipefail

H5DIR=${H5DIR:-/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw}
OUT=${OUT:-/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_sweep}
NLOOKS=${NLOOKS:-16}
MINIFORGE=/Users/staniewi/miniforge3/bin/python
ISCE3PY=/Users/staniewi/miniforge3/envs/mapping-312/bin/python
ISCE2PY=/Users/staniewi/miniforge3/envs/test-isce2/bin/python
REPO=/Users/staniewi/repos/whirlwind-insar
mkdir -p "$OUT"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "frame,engine,runtime_s,peak_rss_bytes,percomp,ncc,shape" > "$CSV"

# Never orphan a child if this script is killed mid-run.
cleanup() { pkill -9 -f -- "run_native_one.py" 2>/dev/null; pkill -9 -f -- "tophu_compare.py" 2>/dev/null; true; }
trap cleanup EXIT INT TERM

run_one() {  # frame  engine  python  cmd...
  local frame="$1" engine="$2" py="$3"; shift 3
  if grep -q "^$frame,$engine," "$CSV" 2>/dev/null; then
    printf '    %-9s (done, skip)\n' "$engine"; return
  fi
  local base="$OUT/$frame.$engine"
  printf '    %-9s ... ' "$engine"
  if /usr/bin/time -l "$py" "$@" > "$base.out" 2> "$base.time"; then :; else
    printf 'FAILED (exit %d)\n' "$?"; echo "$frame,$engine,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$CSV"; return
  fi
  # runtime + per-comp from the engine's stdout line; RSS from /usr/bin/time -l.
  local line rt pc ncc shp rss
  line=$(grep "per-comp-match-vs-prod" "$base.out" | tail -1)
  rt=$(echo "$line"  | grep -oE '[0-9]+\.[0-9]+s' | head -1 | tr -d 's')
  pc=$(echo "$line"  | grep -oE 'per-comp-match-vs-prod=[ ]*[0-9.]+%' | grep -oE '[0-9.]+')
  ncc=$(echo "$line" | grep -oE 'ncc=[0-9]+' | grep -oE '[0-9]+')
  shp=$(echo "$line" | grep -oE 'shape=\([0-9]+, [0-9]+\)' | tr -d ' ')
  rss=$(grep 'maximum resident set size' "$base.time" | grep -oE '[0-9]+' | head -1)
  echo "$frame,$engine,${rt:-?},${rss:-?},${pc:-?},${ncc:-?},${shp:-?}" >> "$CSV"
  printf '%ss  rss=%sB  percomp=%s%%  ncc=%s\n' "${rt:-?}" "${rss:-?}" "${pc:-?}" "${ncc:-?}"
}

cd "$REPO"
i=0
for h5 in "$H5DIR"/*.h5; do
  i=$((i + 1))
  frame=$(basename "$h5" | sed -E 's/.*_([AD]_[0-9]{3})_.*/\1/')
  printf '>>> [%d] %s\n' "$i" "$frame"
  run_one "$frame" whirlwind "$MINIFORGE" scripts/run_native_one.py "$h5" whirlwind
  run_one "$frame" wworig    "$MINIFORGE" scripts/run_native_one.py "$h5" wworig
  run_one "$frame" phass     "$ISCE3PY"  scripts/tophu_compare.py --local-h5 "$h5" --nlooks "$NLOOKS" --unwrappers phass
  # Single-tile SNAPHU is slow (~10 min/frame); opt-in with SNAPHU=1 to record
  # real per-frame runtimes (per-comp is its self-vs-production-snaphu match).
  if [[ "${SNAPHU:-0}" == "1" ]]; then
    run_one "$frame" snaphu     "$MINIFORGE" scripts/snaphu_one.py "$h5" 1
    run_one "$frame" snaphu9x9  "$MINIFORGE" scripts/snaphu_one.py "$h5" 9
  fi
  # isce2 mroipac ICU (the fast published classic) - opt-in with ICU2=1.
  if [[ "${ICU2:-0}" == "1" ]]; then
    run_one "$frame" icu        "$ISCE2PY"  scripts/icu_isce2_run.py "$frame"
  fi
  # ICU is ~35x slower (~9 min on the EASY frame); only run it on a representative
  # subset (`ICU_FRAMES`) rather than all 13 - set ICU_FRAMES="" to skip entirely.
  if [[ " ${ICU_FRAMES:-A_013 D_074} " == *" $frame "* ]]; then
    run_one "$frame" icu     "$ISCE3PY"  scripts/tophu_compare.py --local-h5 "$h5" --nlooks "$NLOOKS" --unwrappers icu
  fi
done
printf 'SWEEP DONE -> %s\n' "$CSV"
