# Environment variables

Most users do not need any environment variables. Prefer normal function arguments such as `mask=`, `downsample=`, and `goldstein_alpha=` when they apply.

These variables are mainly for debugging, benchmarking, and reproducing internal experiments.

| Variable                     | Default  | Use                                                                                                                                                                                                                |
| ---------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `WHIRLWIND_DEBUG`            | unset    | Print detailed primal-dual solver progress to stderr. This is verbose and intended for debugging a suspect unwrap.                                                                                                 |
| `WHIRLWIND_UNWRAP_SOLVER`    | `linear` | Select the solver behind `unwrap()`. The default is the supported 2D path. Other values such as `tiled`, `reuse`, and `convex` are research/debug paths, not recommended for normal use.                           |
| `WHIRLWIND_DIJKSTRA`         | `dial`   | Select the shortest-path backend for benchmarking: `dial`, `heap`, or `dial-par`. The default `dial` backend is fastest in current tests.                                                                          |

Environment variables that select solver internals are read once on first use in a Python process. Start a new process between A/B runs.
