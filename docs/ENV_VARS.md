# Environment variables

All optional; defaults are sensible for production use. Most are
research/diagnostic knobs and should not be needed by normal users.

## User-facing

| Var | Default | Effect |
|---|---|---|
| `WHIRLWIND_DEBUG` | unset | If set, primal-dual prints per-iteration state (Dijkstra source count, excess, augmented flow) to stderr. Verbose. Useful when an unwrap looks wrong. |
| `WHIRLWIND_TIMING` | unset | If set, the tiled `unwrap`/`unwrap_crlb` path prints a per-stage wall-clock breakdown to stderr (per-tile solve, seam reconcile, feather composite, anchor+cascade, heal, multi-shift gate + whether it fired, conncomp build). Cheap, one line per stage — unlike `WHIRLWIND_DEBUG` it does not flood per-iteration. Use it to profile runtime. **Build `--release`** — debug builds are ~35× slower (a debug full-frame NISAR unwrap is ~18 min vs ~30 s release). |
| `WW_MAX_ITER` | `50` | Primal-dual max iterations before SSP fall-back. The library default (50) is what `unwrap()` / `unwrap_crlb()` use; this var is read by `examples/bench_scale.rs` only. |

## Research / internal

These exist for benchmarking and should not be used in production. They
are read once via a `OnceLock` on first call, so changing them inside a
single Python process has no effect after the first unwrap.

| Var | Default | Effect |
|---|---|---|
| `WHIRLWIND_DIJKSTRA` | `dial` | Select the multi-source Dijkstra backend: `dial` (serial Dial's bucket queue, default and fastest), `heap` (binary heap, reference implementation), `dial-par` (rayon-parallel Dial — slower than serial on every workload we measured; kept for the explanation in [`PERFORMANCE.md`](PERFORMANCE.md#on-parallelizing-the-multi-source-dijkstra)). |
| `WHIRLWIND_NO_ANCHOR` | unset | If set, disables the global coarse anchor **and** the multi-scale cascade in the tiled coherence path (`unwrap(..., tile_size, tile_overlap)`), reverting to the single-f=8 anchorless region vote. The default (unset) is the production path that reaches SNAPHU quality (NISAR 99.79 % K-match). For before/after comparison only — see [`report_anchor_cascade.md`](https://github.com/scottstanie/whirlwind-insar/blob/main/paper/report_anchor_cascade.md). |
| `WHIRLWIND_NO_HEAL` | unset | If set, disables the bounded thin-sliver healing pass in the tiled coherence path. Diagnostic before/after only. |
| `WHIRLWIND_TILE_SOLVER` | `reuse` | Per-tile / whole-image base solver: `reuse` (PHASS flow-reuse, corner-safe, default) or `convex` (SNAPHU-style quadratic, research-only). Any other value falls through to `reuse`. The old `linear` unit-capacity solver was removed in #50 (capacity-1 boundary-stacking bug on steep ramps). |
| `WHIRLWIND_TILE_CONVEX` | unset | Legacy alias selecting the convex (quadratic) solver. Research-only — sound but not a general win (Atlanta +4%, regresses NISAR, ~20× slower); see [`convex_cost_design.md`](https://github.com/scottstanie/whirlwind-insar/blob/main/paper/convex_cost_design.md). |
| `WHIRLWIND_CARBALLO_LUT_DIR` | unset | If set, `unwrap_linear` loads Carballo probability-table blobs from this directory instead of the embedded `ww-orig` parity tables. The directory must contain `carballo_grid_phase.bin`, `carballo_grid_corr.bin`, `carballo_grid_nlooks.bin`, `carballo_p0.bin`, and `carballo_p1.bin`, as written by `scripts/generate_carballo_tables.py --write-rust-bins`. Read once via `OnceLock`; start a fresh process between A/B runs. |

Note: there is no env var for the noisy-scene multilook path or for tiling —
those are proper function arguments: `unwrap(..., tile_size=512,
tile_overlap=64, multilook=8)`. The earlier `WHIRLWIND_LLR_COST` /
`WHIRLWIND_CONVEX_OFFSET_*` / `WHIRLWIND_PHASS_COST` / `WHIRLWIND_DEVIATION_COST`
/ `WHIRLWIND_HARD_CUT_THRESH` / `WHIRLWIND_COH_BIAS_CORRECT` cost-experiment
knobs have been **removed** (all were dead or proven-worse — see
`paper/phass_experiments.md` / `convex_cost_design.md` for the negative results).

For the CRLB cost path (`unwrap_crlb`, `unwrap_crlb_grounded`,
`compute_crlb_costs`) the cost function is fixed — no env-var switch.
