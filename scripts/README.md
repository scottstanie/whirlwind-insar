# whirlwind scripts

## Reproduce the NISAR GUNW benchmark (the §9.6 table)

The 4-way single-tile comparison - **whirlwind vs ww-orig vs PHASS vs ICU**,
per-component match vs the production GUNW unwrap (= snaphu), runtime, peak RSS:

```bash
# one heavy unwrap at a time; resume-friendly; writes results.csv + per-engine logs
bash scripts/sweep_all_unwrappers.sh
```

It drives two runners (each engine timed + memory-measured in its own process):

| script                                                             | engines                                             | env                                       |
| ------------------------------------------------------------------ | --------------------------------------------------- | ----------------------------------------- |
| `run_native_one.py <h5> {whirlwind,wworig}`                        | whirlwind (public `unwrap` default), Python ww-orig | the whirlwind env                         |
| `tophu_compare.py --local-h5 <h5> --unwrappers {phass,icu,snaphu}` | PHASS / ICU / snaphu via isce3+tophu                | an isce3 + tophu env (e.g. `mapping-312`) |

Whirlwind-only sweep (no reference unwrappers):
`bash scripts/sweep_single_tile_bench.sh`, or a single frame with
`python scripts/bench_nisar_gunw_whirlwind.py --solver linear --nlooks 16 --local-h5 <h5>`.

## Diagnostics (the masked-frame parity investigation, ATBD §7.6.1)

| script                       | what it shows                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `diag_divergence.py FRAME`   | stage-by-stage bisection: residues, cost, solver - is a frame a Rust↔ww-orig divergence or genuinely hard? |
| `diag_cost_compare.py FRAME` | MCF objective (total cost) + balance for ww-orig vs Rust                                                   |
| `diag_pd_only.py FRAME`      | PD-vs-SSP split, via ww-orig `primal_dual(maxiter=0)`                                                      |
| `run_whirlwind_orig.py`      | run the Python ww-orig reference and save its output                                                       |

## Infrastructure

- `generate_carballo_tables.py` - regenerate the embedded Carballo cost LUT blobs.
