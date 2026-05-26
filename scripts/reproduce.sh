#!/usr/bin/env bash
# Reproduce the whirlwind-rs 3D unwrap + validation + figures on a Palos Verdes
# dolphin output stack.
#
# Single entry point that runs the entire pipeline end-to-end:
#   unwrap_stack.py  →  compare_to_dolphin_unwrapped.py  →  make_figures.py
#
#   ./scripts/reproduce.sh                  # 1024^2 tile, ~30 s, ~5 GB RAM
#   ./scripts/reproduce.sh --full           # 4065x3802 full scene, single-piece, ~2-4 h, ~25 GB
#   ./scripts/reproduce.sh --full --tile 1500 [--overlap 192]
#                                           # full scene, tiled MCF; produces the
#                                           # "fig_palos_verdes_full_tiled_<size>_*" figures
#
# Assumes:
#   - uv sync && uv run maturin develop --release  (done once)
#   - The dolphin output dir at $DOLPHIN_DIR below (override via env var).
#     Public users without the Capella stack can't reproduce verbatim — the
#     pipeline itself runs on any dolphin output dir with the same layout.

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
TILE_SIZE=0
TILE_OVERLAP=0   # 0 ⇒ let unwrap_stack default kick in (currently 128)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)        FULL=1;            shift ;;
        --tile)        TILE_SIZE="$2";    shift 2 ;;
        --overlap)     TILE_OVERLAP="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0 ;;
        *)
            echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

TILE_ARGS=()
NAME_SUFFIX=""
if [[ $TILE_SIZE -gt 0 ]]; then
    TILE_ARGS+=(--tile-size "$TILE_SIZE")
    if [[ $TILE_OVERLAP -gt 0 ]]; then
        TILE_ARGS+=(--tile-overlap "$TILE_OVERLAP")
    fi
    NAME_SUFFIX="_tiled_${TILE_SIZE}"
fi

if [[ $FULL -eq 1 ]]; then
    OUT="$OUT_BASE/full-scene${NAME_SUFFIX}"
    WINDOW_ARG=()
    MAX_IGS_ARG=()
    THREADS=2
    REF=dolphin
    if [[ $TILE_SIZE -gt 0 ]]; then
        echo "[reproduce] FULL SCENE, tiled ${TILE_SIZE} — caps per-IG memory; wall clock comparable to non-tiled"
    else
        echo "[reproduce] FULL SCENE, single-piece — this takes 2-4 hours and ~25 GB peak RAM"
    fi
else
    OUT="$OUT_BASE/tile-1024${NAME_SUFFIX}"
    WINDOW_ARG=(--window 1000 1500 2024 2524)
    MAX_IGS_ARG=(--max-igs 60)
    THREADS=4
    REF=auto
    echo "[reproduce] 1024^2 tile, 60 IGs — should finish in ~30 s"
fi

NAME="palos_verdes"
if [[ $FULL -eq 1 && $TILE_SIZE -gt 0 ]]; then
    # Match the existing fig_palos_verdes_full_tiled_<size>_*.png naming.
    NAME="palos_verdes_full_tiled_${TILE_SIZE}"
fi

mkdir -p "$OUT"

echo "[reproduce] step 1/3: unwrap_stack"
uv run --project "$PROJECT_DIR" python "$PROJECT_DIR/scripts/unwrap_stack.py" \
    --dolphin "$DOLPHIN_DIR" \
    --out "$OUT" \
    --threads "$THREADS" \
    --reference "$REF" \
    "${WINDOW_ARG[@]}" \
    "${MAX_IGS_ARG[@]}" \
    "${TILE_ARGS[@]}"

echo "[reproduce] step 2/3: validate vs dolphin SNAPHU per-IG"
uv run --project "$PROJECT_DIR" python "$PROJECT_DIR/scripts/compare_to_dolphin_unwrapped.py" \
    --ours "$OUT" \
    --dolphin "$DOLPHIN_DIR"

echo "[reproduce] step 3/3: figures"
uv run --project "$PROJECT_DIR" python "$PROJECT_DIR/scripts/make_figures.py" \
    --ours "$OUT" \
    --dolphin "$DOLPHIN_DIR" \
    --out "$OUT/figures" \
    --name "$NAME"

echo ""
echo "[reproduce] done."
echo "  outputs:  $OUT"
echo "  figures:  $OUT/figures"
echo "  report:   $OUT/report.json"
