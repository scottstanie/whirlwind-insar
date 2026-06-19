# NISAR 2D unwrapping comparison

This page compares Whirlwind with the unwrapping used in NISAR GUNW products and with a few other 2D unwrappers. The metric is agreement with the production NISAR unwrapped phase after re-wrapping it to create the input wrapped phase.

The comparison uses 13 HH NISAR GUNW frames with `nlooks=16`. Runtimes and memory are from one Apple M-series laptop, so treat them as relative numbers rather than universal benchmarks.

## Summary

- Whirlwind agrees with the production SNAPHU unwrap on at least 98.8 percent of pixels on 12 of 13 frames (99 percent or better on 11 of them).
- The remaining frame, D_075, is the outlier. The sweep's own SNAPHU runs also score 88.2 percent against the production reference there, so this appears to be a configuration mismatch with the production unwrap rather than a Whirlwind-specific failure; PHASS agrees on 48.4 percent.
- Runtime is 10-27 seconds per frame for Whirlwind, compared with 465-1242 seconds for single-tile SNAPHU and about 100-200 seconds for SNAPHU 3x3 tiled (9 tiles in parallel) plus reoptimization.
- Peak memory is about 2.5 GB per NISAR frame for Whirlwind (2.2-2.8 GB), compared with about 8 GB for single-tile SNAPHU and about 6-13 GB for 3x3 tiled SNAPHU. The tiled peak is not intrinsic: it is dominated by the parallel tile phase, so it scales with how many tiles unwrap at once (`nproc`) -- capping concurrency roughly halves it (see the note under the table).

## Metric

The quality number is per-connected-component 2pi ambiguity agreement with the production GUNW unwrap, after median alignment within each component. This checks whether the integer cycle field agrees with the production result while avoiding a single global reference-pixel offset dominating the score.

## Results

| Frame | Whirlwind vs production SNAPHU | PHASS vs production SNAPHU | Note                                    |
| ----- | -----------------------------: | -------------------------: | --------------------------------------- |
| A_013 |                          100.0 |                       99.3 |                                         |
| A_016 |                          100.0 |                       99.6 |                                         |
| A_018 |                          100.0 |                       85.7 |                                         |
| A_020 |                           99.8 |                       99.4 |                                         |
| A_022 |                          100.0 |                       99.4 |                                         |
| A_025 |                          100.0 |                       67.0 | low-coherence river                     |
| A_028 |                          100.0 |                       92.9 |                                         |
| A_030 |                          100.0 |                       75.4 |                                         |
| D_074 |                           98.8 |                       91.2 |                                         |
| D_075 |                           88.2 |                       48.4 | sweep SNAPHU also scores 88.2           |
| D_077 |                           99.5 |                       94.7 |                                         |
| D_078 |                           99.8 |                       96.9 |                                         |
| A_035 |                          100.0 |                       94.6 |                                         |

![13-frame NISAR GUNW comparison](figures/nisar_summary.png)

The full per-frame table with runtime and memory is in [nisar_4way_results.csv](nisar_4way_results.csv).

## Connected components

The phase agreement above is the headline; the connected-component (conncomp)
labels are a separate question that the NISAR team asked us to tighten up. Three
points, with the per-frame numbers in the table below.

**Provenance of the "SNAPHU" conncomps.** The reference conncomp we plot is the
GUNW `connectedComponents` dataset read straight out of the product HDF5. NISAR
production unwrapping *is* SNAPHU (`cost=smooth`, `init=mcf`), so that layer is
the authoritative SNAPHU result — it is **not** a re-run of SNAPHU on our side.
(We do have a separate tophu/SNAPHU re-run for the speed/memory comparison, in
`scripts/tophu_compare.py --save-dir`, but it never feeds the comparison
figures.)

**Water is masked before the conncomp grows.** The GUNW `mask` water flag is
folded into the valid mask (`water_only_mask`), every conncomp edge that touches
a masked pixel is cut, and masked pixels stay label 0. So water never joins or
splinters a component.

**A SNAPHU-faithful conncomp path that stops the splintering.** whirlwind's
original conncomp (`components_only`) cuts an edge by its raw linear coherence
cost, which over-segments low-coherence interiors into many tiny components. The
new `components_snaphu` reproduces SNAPHU's `GrowConnCompsMask` directly: it
recovers each edge's achieved 2π ambiguity from the *unwrapped phase output* and
cuts only where a ±1-cycle "wiggle" against the convex smooth cost is no more
expensive than the achieved flow (`min(poscost, negcost) <= threshold`). It needs
only correlation + the unwrapped output, so it composes with any phase path. The
calibration-free default (`reliability_threshold=0`, `min_size_px=100`) tracks the
production component count closely; the threshold barely moves the partition from
0 to 5e4, so no per-scene tuning is needed.

| Frame | Track | per-comp match % | production SNAPHU | whirlwind old (linear) | whirlwind new (SNAPHU-faithful) |
| ----- | ----: | ---------------: | ----------------: | ---------------------: | ------------------------------: |
| A_013 |     5 |            100.0 |                 1 |                      1 |                               1 |
| A_016 |     5 |            100.0 |                 3 |                      8 |                               8 |
| A_018 |     5 |            100.0 |                 1 |                     69 |                               3 |
| A_020 |     5 |             99.8 |                 1 |                      1 |                               1 |
| A_022 |     5 |            100.0 |                 1 |                      2 |                               1 |
| A_025 |     5 |            100.0 |                 2 |                     41 |                               3 |
| A_028 |     5 |            100.0 |                 1 |                     36 |                               2 |
| A_030 |     5 |            100.0 |                 3 |                    230 |                               3 |
| A_035 |     6 |            100.0 |                 2 |                    119 |                               5 |
| D_074 |     5 |             98.8 |                 1 |                     45 |                               2 |
| D_075 |     5 |             88.2 |                 1 |                     64 |                               3 |
| D_077 |     5 |             99.5 |                 2 |                     46 |                               1 |
| D_078 |     5 |             99.8 |                 1 |                      4 |                               1 |

The new path collapses the splinter (A_030 230→3, A_035 119→5, A_018 69→3,
D_075 64→3, D_077 46→1) to component counts close to production SNAPHU, with the
phase match unchanged. The one residual is A_016, a heavily water-fragmented
coastal scene where genuinely disconnected islands stay separate components
(8 vs production's 3); the phase still matches at 100%.

The full SNAPHU `GrowConnCompsMask` would additionally re-level and re-grow
components across thin masked gaps; matching that bridging of *labels* (so
water-separated slabs share a component id, as on A_016) is the remaining gap and
is tracked as future work.

**Tuning coverage.** `components_snaphu` is the default in `ww.unwrap`
(`conncomp_algorithm="snaphu"`) and in the CLI (`--conncomp-algorithm snaphu`).
By default it labels essentially every reliably unwrapped pixel; production
SNAPHU leaves more low-coherence pixels as background (label 0). To match that,
raise `conncomp_reliability` (CLI `--conncomp-reliability`). It is in
inverse-variance (`1/σ²`) units, so values are small: an edge of coherence γ is
cut roughly when the value exceeds `1/σ²(γ)`. The guessable way to set it is by a
target minimum coherence — `whirlwind.conncomp_reliability_from_coherence(γ, nlooks)`
(CLI `--conncomp-min-coherence`), e.g. `γ=0.3 → ~3.2`.
`scripts/sweep_conncomp_reliability.py` sweeps the knob and plots labeled fraction
and component count against it (with a coherence-equivalent top axis), and writes
per-frame conncomp label images across the sweep (figures + CSV under
`nisar-pngs/<date>/`).

## Runtime and memory

| Engine                         |    Runtime | Peak memory | Notes                                        |
| ------------------------------ | ---------: | ----------: | -------------------------------------------- |
| Whirlwind                      |    10-27 s |     ~2.5 GB | Rust-backed 2D MCF path                      |
| SNAPHU, single tile            | 465-1242 s |       ~8 GB | quality reference, slowest configuration     |
| SNAPHU, 3x3 tiled + reoptimize |   97-201 s |     6-13 GB | 9 tiles; peak set by concurrency (`nproc`)   |
| PHASS                          |   5.5-23 s |  1.7-2.4 GB | faster, lower agreement on several frames    |
| isce2 ICU                      |  109-204 s |  1.5-2.8 GB | leaves some low-coherence areas disconnected |

Memory note: the SNAPHU tiled numbers are peak RSS summed over the whole process tree (`scripts/peak_rss_tree.py`), because SNAPHU forks one worker per concurrent tile and a per-process measure such as `/usr/bin/time` undercounts them. SNAPHU's tiled peak is dominated by the parallel tile phase, not the final reoptimize, so it scales with concurrency: on A_025 the 3x3 tiling peaks at about 12 GB with all 9 tiles unwrapping at once, but about 6 GB capped at `nproc=4` (at roughly +45% runtime). The single-process engines (Whirlwind, PHASS, ICU, single-tile SNAPHU) are one process, so their `nisar_4way_results.csv` figures are unaffected by this distinction.

## Reproduce

The NISAR comparison runs in two stages. **Inputs:** the 13 GUNW `.h5` products
(`H5DIR` in the scripts); **stage-1 cache:** per-frame `<frame>_panels.npz`
arrays (`CACHE_DIR`); **outputs:** the figures below.

1. **Stage 1 — unwrap each frame (heavy, once).** `scripts/plot_nisar_per_frame.py`
   runs the default `ww.unwrap` on every GUNW frame, scores per-component match,
   writes a 6-panel figure, and caches `<frame>_panels.npz`
   (`wrapped, coh, mask, prod_unw, prod_cc, ww_unw, ww_cc`) for reuse. Heavy
   unwraps run strictly one at a time (laptop memory limit).
2. **Stage 2 — conncomp comparison (fast, re-runnable).**
   `scripts/nisar_conncomp_compare.py` is the entry point for the figures the
   NISAR team reviews. It reads the stage-1 cache, computes the new
   SNAPHU-faithful conncomp (`components_snaphu`), and writes one 8-panel
   figure per frame plus a `conncomp_summary.csv` into
   `./nisar-pngs/<YYYY-MM-DD>/`. No re-unwrapping needed; pass `--reunwrap` to
   force stage 1 for a frame whose cache is missing.

   ```
   .venv/bin/python scripts/nisar_conncomp_compare.py            # all 13 frames
   .venv/bin/python scripts/nisar_conncomp_compare.py A_016 D_077 # a subset
   ```

   Needs a whirlwind build that exports `components_snaphu`:
   ```
   RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
       cargo build --release -p whirlwind-py
   cp target/release/lib_native.dylib python/whirlwind/_native.abi3.so
   ```

Other scripts:

- 4-way sweep (whirlwind / SNAPHU / PHASS / ICU, for the speed/memory table):
  `scripts/sweep_all_unwrappers.sh` (uses `scripts/tophu_compare.py`).
- Headline summary figure: `scripts/plot_nisar_summary.py`.

See [Algorithm notes](ALGORITHM.md) for how the unwrapper works,
[Why SNAPHU/PHASS differ](SNAPHU_PHASS_SPEED.md) for the runtime interpretation,
and [Memory and scaling notes](MEMORY_AND_SCALING.md) for rough memory planning.
