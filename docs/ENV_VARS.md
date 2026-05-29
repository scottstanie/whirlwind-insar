# Environment variables

All optional; defaults are sensible for production use. Most are
research/diagnostic knobs and should not be needed by normal users.

## User-facing

| Var | Default | Effect |
|---|---|---|
| `WHIRLWIND_DEBUG` | unset | If set, primal-dual prints per-iteration state (Dijkstra source count, excess, augmented flow) to stderr. Useful when an unwrap looks wrong. |
| `WW_MAX_ITER` | `50` | Primal-dual max iterations before SSP fall-back. The library default (50) is what `unwrap()` / `unwrap_crlb()` use; this var is read by `examples/bench_scale.rs` only. |

## Research / internal

These exist for benchmarking and should not be used in production. They
are read once via a `OnceLock` on first call, so changing them inside a
single Python process has no effect after the first unwrap.

| Var | Default | Effect |
|---|---|---|
| `WHIRLWIND_DIJKSTRA` | `dial` | Select the multi-source Dijkstra backend: `dial` (serial Dial's bucket queue, default and fastest), `heap` (binary heap, reference implementation), `dial-par` (rayon-parallel Dial — slower than serial on every workload we measured; kept for the explanation in [`PERFORMANCE.md`](PERFORMANCE.md#on-parallelizing-the-multi-source-dijkstra)). |
| `WHIRLWIND_LLR_COST` | unset | Switch the coherence-cost path to the Carballo log-likelihood-ratio cost rather than the SNAPHU-style non-negative cost. Negative-cost-tolerant and currently requires Bellman-Ford preprocessing to seed initial potentials (not yet wired up), so do not use for production unwraps. |
| `WHIRLWIND_NO_ANCHOR` | unset | If set, disables the global coarse anchor **and** the multi-scale cascade in the tiled coherence path (`unwrap(..., tile_size, tile_overlap)`), reverting to the single-f=8 anchorless region vote. The default (unset) is the production path that reaches SNAPHU quality (NISAR 99.79 % K-match). For before/after comparison only — see [`report_anchor_cascade.md`](../paper/report_anchor_cascade.md). |

Note: there is no env var for the noisy-scene multilook path or for tiling —
those are proper function arguments: `unwrap(..., tile_size=512,
tile_overlap=64, multilook=8)`. A set of `WHIRLWIND_CONVEX_*` / `WHIRLWIND_PHASS_COST`
/ `WHIRLWIND_DEVIATION_COST` / `WHIRLWIND_HARD_CUT_THRESH` / `WHIRLWIND_COH_BIAS_CORRECT`
knobs also exist in `cost/mod.rs` for the convex-cost prototype; they are
research-only (the convex solver is unsound — see [`tiling.md`](../paper/tiling.md)
"Corrections" #2) and should not be used for production.

For the CRLB cost path (`unwrap_crlb`, `unwrap_crlb_grounded`,
`compute_crlb_costs`) the cost function is fixed — no env-var switch.
