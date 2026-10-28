# whirlwind scripts

## Reproduce the NISAR GUNW benchmark

Compare **whirlwind, ww-orig, PHASS, and ICU** on a single tile. The sweep
measures per-component agreement with the production GUNW unwrap (snaphu),
runtime, and peak RSS:

```bash
# one heavy unwrap at a time; resume-friendly; writes results.csv + per-engine logs
bash scripts/sweep_all_unwrappers.sh
```

It uses two runners, timing and measuring each engine in a separate process:

| script                                                             | engines                                             | env                                       |
| ------------------------------------------------------------------ | --------------------------------------------------- | ----------------------------------------- |
| `run_native_one.py <h5> {whirlwind,wworig}`                        | whirlwind (public `unwrap` default), Python ww-orig | the whirlwind env                         |
| `tophu_compare.py --local-h5 <h5> --unwrappers {phass,icu,snaphu}` | PHASS / ICU / snaphu via isce3+tophu                | an isce3 + tophu env (e.g. `mapping-312`) |

Whirlwind-only sweep (no reference unwrappers):
`bash scripts/sweep_single_tile_bench.sh`, or a single frame with
`python scripts/bench_nisar_gunw_whirlwind.py --nlooks 16 --local-h5 <h5>`.

Scripts that import `whirlwind_orig` (the old pure-Python reference
implementation) need it installed separately - it's not a project dependency,
so a fresh clone/pixi/uv env won't pull it in:
`uv pip install -e /path/to/whirlwind` (or add it to your own venv).

## Diagnostics (masked-frame comparison against the original Python whirlwind)

| script                       | what it shows                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `diag_divergence.py FRAME`   | stage-by-stage bisection: residues, cost, solver - is a frame a Rust↔ww-orig divergence or genuinely hard? |
| `diag_cost_compare.py FRAME` | MCF objective (total cost) + balance for ww-orig vs Rust                                                   |
| `diag_pd_only.py FRAME`      | PD-vs-SSP split, via ww-orig `primal_dual(maxiter=0)`                                                      |
| `run_whirlwind_orig.py`      | run the Python ww-orig reference and save its output                                                       |

## Infrastructure

- `generate_carballo_tables.py` - regenerate the embedded Carballo cost tables from the analytic model (default), or re-export the historical ww-orig tables with `--source-table-dir`.
