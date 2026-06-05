# Environment variables

Most users do not need any environment variables. Prefer normal function arguments such as `mask=`, `multilook=`, and `goldstein_alpha=` when they apply.

These variables are mainly for debugging, benchmarking, and reproducing internal experiments.

| Variable | Default | Use |
|---|---|---|
| `WHIRLWIND_DEBUG` | unset | Print detailed primal-dual solver progress to stderr. This is verbose and intended for debugging a suspect unwrap. |
| `WHIRLWIND_NO_BRIDGE` | unset | Disable the bridge post-pass that sets relative 2pi offsets between disconnected valid-mask regions. Use this only for before/after diagnostics. |
| `WHIRLWIND_UNWRAP_SOLVER` | `linear` | Select the solver behind `unwrap()`. The default is the supported 2D path. Other values such as `tiled`, `reuse`, and `convex` are research/debug paths, not recommended for normal use. |
| `WHIRLWIND_DIJKSTRA` | `dial` | Select the shortest-path backend for benchmarking: `dial`, `heap`, or `dial-par`. The default `dial` backend is fastest in current tests. |
| `WHIRLWIND_CARBALLO_LUT_DIR` | unset | Load alternate Carballo probability-table blobs from a directory for cost-model experiments. The directory must contain the five `.bin` files produced by `scripts/generate_carballo_tables.py --write-rust-bins`. |

Environment variables that select solver internals are read once on first use in a Python process. Start a new process between A/B runs.
