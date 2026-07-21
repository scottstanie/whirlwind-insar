# Environment variables

Most users do not need any environment variables. Prefer normal function arguments such as `mask=`, `downsample=`, and `goldstein_alpha=` when they apply.

These variables are mainly for debugging, benchmarking, and reproducing internal experiments.

| Variable                     | Default  | Use                                                                                                                                                                                                                |
| ---------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `WHIRLWIND_DEBUG`            | unset    | Print detailed MCF-solver (PD/SSP) progress to stderr. This is verbose and intended for debugging a suspect unwrap.                                                                                                 |
| `WHIRLWIND_UNWRAP_SOLVER`    | `linear` | Select the solver behind `unwrap()`. The default is the supported 2D path. Other values such as `tiled`, `reuse`, and `convex` are research/debug paths, not recommended for normal use.                           |
| `WHIRLWIND_DIJKSTRA`         | `dial`   | Select the shortest-path backend for benchmarking: `dial`, `heap`, or `dial-par`. The default `dial` backend is fastest in current tests.                                                                          |
| `WHIRLWIND_SLOPE_GUARD`      | on       | Set to `off` to restore the unguarded Carballo cost field for an A/B comparison.                                                                                                                                    |
| `WHIRLWIND_SLOPE_GUARD_RAD`  | `1.0`    | Radian floor for the aliased-gradient robustness guard. With a zero budget, this is a fixed threshold.                                                                                                             |
| `WHIRLWIND_SLOPE_GUARD_BUDGET` | `0.03` | Maximum fraction of valid edges the guard may make free to cut. Set to `0` when testing a fixed radian threshold.                                                                                                  |
| `WHIRLWIND_SLOPE_GUARD_MODE` | `zerocost` | Guard action. `zeroslope` is a diagnostic arm, not a production alternative.                                                                                                                                     |

Environment variables that select solver internals are read once on first use in a Python process. Start a new process between A/B runs.
