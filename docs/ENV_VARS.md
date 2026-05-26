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

For the CRLB cost path (`unwrap_crlb`, `unwrap_crlb_grounded`,
`compute_crlb_costs`) the cost function is fixed — no env-var switch.
