# PHASS-inspired cost experiments

Tests three modifications to whirlwind's MCF arc cost, inspired by reading the
PHASS C++ source at `~/repos/isce3/cxx/isce3/unwrap/phass`. Goal: figure out
how much of SNAPHU's no-Goldstein robustness can be recovered by *cost-shape*
changes alone, without touching the SSP solver.

See `paper/different_vs_snaphu_costs.md` for the writeup of the underlying
diagnosis.

## Cost variants (all live in `crates/whirlwind-core/src/cost/mod.rs`, env-var-toggled)

| mode         | env                                                | what it does |
|--------------|----------------------------------------------------|--------------|
| `baseline`   | (none)                                             | Default Carballo: `γ_edge · (π − α_smooth)` |
| `hard_cut`   | `WHIRLWIND_HARD_CUT_THRESH=1.0`                    | Plus: zero-cost cuts wherever `|wrap(Δphase_raw)| ≥ 1.0 rad`. Mirrors PHASS's `phase_diff_th`. |
| `phass_cost` | `WHIRLWIND_PHASS_COST=0.5`                         | Replace cost with `γ_edge² · π` saturated at `0.5² · π`. PHASS coh-only cost, no `α` term, no cuts. |
| `phass_full` | both env vars set                                  | PHASS cost + hard cuts. Closest single-pass emulation of PHASS in whirlwind. |

The deviation-cost experiment (`WHIRLWIND_DEVIATION_COST=1`) is also still
plumbed but is a documented **negative** result — see the cost-doc and the
cost/mod.rs docstring for `deviation_cost_enabled`.

## Scenes

* **NISAR**  `20251224_20260117` HH 50 m, 10×10 boxcar looks
  (`/Volumes/.../Learning/nisar/`).
* **Palos Verdes**  Capella C13_SP `20251129_20251205` HH
  (`/Volumes/.../Learning/capella/palos-verdes/.../network_output/20251129_20251205/`).

## Reproduction

```bash
PY=/Users/staniewi/miniforge3/envs/mapping-312/bin/python
DOLPHIN=/Users/staniewi/miniforge3/envs/mapping-312/bin/dolphin

# (one-time) rebuild the editable Rust extension after editing cost/mod.rs:
cd /Users/staniewi/repos/whirlwind-insar
maturin develop --release

# (one-time) SNAPHU reference for PV — NISAR's already lives next to its TIFF
$PY scripts/phass_experiments/run_snaphu_pv.py

# All modes × 2 scenes, **sequentially** (never run > 1 heavy job in parallel
# on this laptop — see ~/.claude/projects/.../memory/concurrency_limit.md).
# phass_full is parametrised but ~hours on NISAR; not in the routine sweep.
for scene in pv nisar; do
  for mode in baseline hard_cut phass_cost; do
    $PY scripts/phass_experiments/run_one.py "$scene" "$mode"
  done
done

# Tables + side-by-side K-field plots
$PY scripts/phass_experiments/analyze.py

# dolphin PHASS (the actual ISCE3 PHASS via dolphin's binding) on NISAR.
# Headline result: 97.93% K-match with SNAPHU 9x9 at α=0.
NISAR=/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar
EXP=/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments
$DOLPHIN unwrap \
  --ifg-filenames "$NISAR/20251224_20260117.int.looked.tif" \
  --cor-filenames "$NISAR/20251224_20260117.int.coh.looked.cleaned.tif" \
  --output-path "$EXP/dolphin_phass" \
  --nlooks 100 \
  --unwrap-options.unwrap-method PHASS
$PY scripts/phass_experiments/compare_dolphin_phass.py
```

Outputs land under
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/{outputs,plots}/`.

## Conventions

* `outputs/<scene>_<mode>.npz` has keys `unw`, `cc`, `k`, `elapsed` (`k` is the
  integer-cycle field `round((unw − wrapped)/2π)`).
* `outputs/pv_snaphu.npz` is the SNAPHU smooth-cost reference on PV.
* NISAR's SNAPHU 9×9 reference is read from the `.snaphu_9x9.{unw,cc}.tif`
  files in the input directory.
