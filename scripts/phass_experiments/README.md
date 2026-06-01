# Experiment & diagnostic scripts — index

One-off research scripts (reproducers, diagnostics, prototypes, figure builders)
accumulated while getting whirlwind to SNAPHU quality without Goldstein. They are
**kept as a research record** — many are superseded by later work but document
what was tried and why. Most read saved arrays from
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/...` (absolute paths) and write
there; they have **no inter-script imports**, so each runs standalone with
`/Users/staniewi/miniforge3/envs/mapping-312/bin/python <path>`.

## Layout

| folder | purpose | n |
|--------|---------|---|
| [`diag/`](diag) | investigate a specific failure mode | 12 |
| [`proto/`](proto) | algorithm prototypes / design validation (Python, before Rust) | 17 |
| [`plot/`](plot) | generate figures | 15 |
| [`report/`](report) | assemble multi-figure reports (+ refresh their saved arrays) | 3 |
| [`run/`](run) | drive the unwrapper on a scene (variant sweeps) | 20 |
| [`bench/`](bench) | benchmarks / quantitative sweeps vs a reference | 3 |
| [`analyze/`](analyze) | tabulate / summarize results | 2 |

History note: the **current** A_016 / fragmented-scene story lives in `diag/diag_a016_*`,
`proto/proto_shift_selector.py`, `proto/proto_seam_repair.py`, `proto/conncomp_*`,
and `report/`; older `run_atlanta*`, `run_nisar_anchor/cascade`, and the early
`*_proto.py` files are earlier rungs of the same ladder (kept, not current best).

## Index

### diag/ — failure diagnostics
- `diag_a016_connectivity.py` — how A_016's correct-left / drifted-right halves connect
- `diag_a016_geometry.py` — where A_016 fails (winding map)
- `diag_anchor_lk_sweep.py` — does a finer coarse anchor level A_016 correctly?
- `diag_d074_regression.py` — why clean D_074 craters 98→81% at tile≥1024
- `diag_gunw_anchor.py` — why the coarse anchor doesn't bridge A_016's halves
- `diag_gunw_edge.py` — why coarse 2π-offset blocks appear
- `diag_gunw_pyramid.py` — does the pyramid fix A_016's offset block? (no)
- `diag_nisar_vlines.py` — find residual thin vertical lines in the NISAR unwrap
- `diag_seam_4032.py` — localize the NISAR col-4032 seam strip across stages
- `diag_seam_rawphase.py` — what whirlwind reacts to at the col-4032 sliver
- `diag_seam_tiles.py` — is the col-4032 −1 strip born inside one tile's solve?
- `diag_vlines_coh.py` — characterize the 2π tears that matter

### proto/ — algorithm prototypes / validation
- `proto_shift_selector.py` — multi-shift + min-high-coherence-cut selector (A_016 fix; shipped)
- `proto_seam_repair.py` — seam-repair of residual high-coh cut blocks (shipped)
- `proto_staggered_tiles.py` — staggered double-tiling for seam artifacts
- `proto_block_reconcile.py` — full-boundary reconcile validation
- `proto_coh_relevel.py` — coherence-confident block re-leveling
- `proto_region_secondary.py` — SNAPHU-style region-graph secondary post-pass
- `region_reopt_proto.py` — early "reoptimize at end" region reconciliation
- `bigtile_anchor_proto.py` — tile-512 detail + coarse anchor best-of-both
- `whole_image_test.py` — whole-image single-tile baseline (falsified "single-tile=best")
- `reuse_tile_sweep.py` — reuse-solver tile-size sweep over 5 GUNW frames
- `cost_corner_test.py` — 6π steep-ramp corner-bug test under tiling
- `conncomp_size_distribution.py` — no-2π-tear component size distribution
- `conncomp_minsize_analysis.py` — empirical min-size keep/drop analysis
- `conncomp_noise_vs_island.py` — distinguish noise fragments from coherent islands
- `conncomp_gap_coherence.py` — the size gap, drilled by coherence
- `conncomp_spatial_check.py` — are mid-size components compact islands or scattered?
- `prod_island_check.py` — what production's kept components look like

### plot/ — figures
- `plot_a016_fixed.py` — A_016 before/after the fix
- `plot_a016_cut_comparison.py` — A_016 whirlwind-vs-production cut/cost story
- `plot_a016_valid_winding.py` — A_016 output is a valid (differently-wound) unwrap
- `plot_nisar_tiled.py` — headline NISAR: whole-image vs tiled vs SNAPHU 9×9
- `plot_nisar_variants.py` / `plot_nisar_anchor.py` / `plot_nisar_anchor_diff.py` / `plot_nisar_phase_hires.py` — NISAR anchor/cascade comparisons
- `plot_atlanta.py` / `plot_atlanta_zoom.py` — Atlanta S-1 vs OPERA
- `plot_convex.py` / `plot_reuse.py` — solver-variant K-fields
- `plot_heal_beforeafter.py` / `plot_line_beforeafter.py` / `plot_sliver_fix.py` — thin-line/sliver heal before/after

### report/ — assembled reports
- `make_report_figures.py` — NISAR + Atlanta 3-row report (K / phase / conncomp)
- `make_atlanta_report.py` — Atlanta failure+fix panels
- `refresh_report_arrays.py` — re-run NISAR/A_016 on the current build, refresh saved arrays

### run/ — scene runners (20)
Drivers that unwrap a scene under a chosen mode. Key ones: `run_one.py`
(cost-variant runner), `run_all.py` (sequential orchestrator), `run_snaphu_pv.py`
(SNAPHU reference), `run_nisar_{anchor,cascade,convex_tiled}.py`,
`run_atlanta*.py` (8 Atlanta variants, lineage base→anchor→cascade→ml8→report),
`run_{convex,reuse,reuse_hardcut,grounded}.py`, `run_final_verify.py`.

### bench/ — benchmarks
- `bench_nisar_gunw.py` — benchmark whirlwind vs NISAR L2 GUNW products (the readiness gate)
- `bench_default_robust.py` — bench the actual default path (auto-512 gated)
- `compare_dolphin_phass.py` — K-agreement: dolphin PHASS vs SNAPHU 9×9 on NISAR

### analyze/ — summaries
- `analyze.py` — tabulate + plot PHASS-cost experiment results
- `analyze_atlanta.py` — Atlanta-vs-OPERA K-match (global + per-component offsets)

---

## PHASS-inspired cost experiments (the original framework)

Tests three modifications to whirlwind's MCF arc cost, inspired by the PHASS C++
source (`~/repos/isce3/cxx/isce3/unwrap/phass`). See
`paper/different_vs_snaphu_costs.md` for the diagnosis writeup.

Cost variants (in `crates/whirlwind-core/src/cost/mod.rs`, env-var-toggled):

| mode         | env                              | what it does |
|--------------|----------------------------------|--------------|
| `baseline`   | (none)                           | Default Carballo: `γ_edge · (π − α_smooth)` |
| `hard_cut`   | `WHIRLWIND_HARD_CUT_THRESH=1.0`  | + zero-cost cuts where `\|wrap(Δphase_raw)\| ≥ 1.0` rad (PHASS `phase_diff_th`) |
| `phass_cost` | `WHIRLWIND_PHASS_COST=0.5`       | cost `γ_edge² · π` saturated at `0.5²·π` (coh-only, no `α`, no cuts) |
| `phass_full` | both                             | PHASS cost + hard cuts |

Reproduction (always **one heavy job at a time** on this laptop — see
`memory/concurrency_limit.md`):

```bash
PY=/Users/staniewi/miniforge3/envs/mapping-312/bin/python
cd /Users/staniewi/repos/whirlwind-insar
maturin develop --release                      # after editing cost/mod.rs
$PY scripts/phass_experiments/run/run_snaphu_pv.py     # SNAPHU reference (PV)
for scene in pv nisar; do
  for mode in baseline hard_cut phass_cost; do
    $PY scripts/phass_experiments/run/run_one.py "$scene" "$mode"
  done
done
$PY scripts/phass_experiments/analyze/analyze.py        # tables + K-field plots
$PY scripts/phass_experiments/bench/compare_dolphin_phass.py
```

Outputs land under `/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/{outputs,plots}/`.
`outputs/<scene>_<mode>.npz` keys: `unw`, `cc`, `k`, `elapsed` (`k = round((unw−wrapped)/2π)`).
