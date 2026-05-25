#!/usr/bin/env bash
# Reproduce the whirlwind-rs 3D unwrap + validation + figures on a Palos Verdes tile.
#
# Single entry point that runs the entire pipeline end-to-end. Pass --full to
# run on the full scene (takes ~2-4 hours on a laptop, ~25 GB peak RAM).
#
#   ./scripts/reproduce.sh            # 1024^2 tile, ~30 s, ~5 GB RAM
#   ./scripts/reproduce.sh --full     # 4065x3802 full scene, ~2-4 h, ~25 GB
#
# Assumes:
#   - uv sync && uv run maturin develop --release  (done once)
#   - The dolphin output dir at $DOLPHIN_DIR below (override via env var)

set -euo pipefail

DOLPHIN_DIR="${DOLPHIN_DIR:-/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes/Palos_Verdes_C13_RO23_SP/e2e_output_20260519/dolphin}"
OUT_BASE="${OUT_BASE:-/tmp/whirlwind-repro}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$DOLPHIN_DIR" ]]; then
    echo "ERROR: dolphin directory not found: $DOLPHIN_DIR" >&2
    echo "Set DOLPHIN_DIR=/path/to/dolphin to override." >&2
    exit 1
fi

# uv + maturin sometimes trip on CONDA_PREFIX being set too. Unset to be safe.
unset CONDA_PREFIX

FULL=0
if [[ "${1:-}" == "--full" ]]; then
    FULL=1
fi

if [[ $FULL -eq 1 ]]; then
    OUT="$OUT_BASE/full-scene"
    WINDOW_ARG=()
    MAX_IGS_ARG=()
    THREADS=2
    REF=dolphin
    echo "[reproduce] FULL SCENE — this takes 2-4 hours and ~25 GB peak RAM"
else
    OUT="$OUT_BASE/tile-1024"
    WINDOW_ARG=(--window 1000 1500 2024 2524)
    MAX_IGS_ARG=(--max-igs 60)
    THREADS=4
    REF=auto
    echo "[reproduce] 1024^2 tile, 60 IGs — should finish in ~30 s"
fi

mkdir -p "$OUT"

echo "[reproduce] step 1/3: unwrap_stack"
uv run --project "$PROJECT_DIR" python "$PROJECT_DIR/scripts/unwrap_stack.py" \
    --dolphin "$DOLPHIN_DIR" \
    --out "$OUT" \
    --threads "$THREADS" \
    --reference "$REF" \
    "${WINDOW_ARG[@]}" \
    "${MAX_IGS_ARG[@]}"

echo "[reproduce] step 2/3: validate vs dolphin SNAPHU per-IG"
uv run --project "$PROJECT_DIR" python "$PROJECT_DIR/scripts/compare_to_dolphin_unwrapped.py" \
    --ours "$OUT" \
    --dolphin "$DOLPHIN_DIR"

echo "[reproduce] step 3/3: figures"
uv run --project "$PROJECT_DIR" python "$PROJECT_DIR/scripts/make_figures.py" \
    --ours "$OUT" \
    --dolphin "$DOLPHIN_DIR" \
    --out "$OUT/figures" \
    --name palos_verdes

echo ""
echo "[reproduce] done."
echo "  outputs:  $OUT"
echo "  figures:  $OUT/figures"
echo "  report:   $OUT/report.json"
