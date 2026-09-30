# Performance

Whirlwind solves the whole interferogram as a single tile. Its runtime and memory grow roughly linearly with pixel count, so there is no tiling to configure. The numbers below are from [the Whirlwind paper](https://arxiv.org/abs/2609.36267). Absolute times depend on the machine; the ratios between unwrappers are what carry over.

## What to expect

A 74-Mpixel Sentinel-1 interferogram of the 2019 Ridgecrest earthquakes (7825 × 9509):

| Unwrapper                          | Runtime | Peak memory |
| ---------------------------------- | ------: | ----------: |
| Whirlwind                          |   209 s |      8.7 GB |
| SNAPHU, single tile                |  5823 s |     21.2 GB |
| SNAPHU, 5 × 5 tiles + reoptimize   |  1166 s |     10.2 GB |

Thirteen NISAR GUNW frames, about 18–21 Mpixel each:

| Unwrapper                             | Runtime (s) | Peak memory (GB) |
| ------------------------------------- | ----------: | ---------------: |
| Whirlwind                             |        6–24 |          2.7–3.5 |
| SNAPHU, single tile                   |    466–1242 |          6.2–8.1 |
| SNAPHU, 3 × 3 tiles (4 processes)     |     100–268 |          4.4–5.6 |
| ISCE3 PHASS                           |      5.5–23 |          1.6–2.2 |
| ICU                                   |     109–204 |          1.4–2.6 |

For memory planning, budget about 0.1–0.2 GB per million pixels.

## What drives runtime

- **Residue density.** Clean scenes are dominated by per-pixel work (costs, residues, integration, components), which is parallel and fast. Noisy scenes have more residues to pair, so more time goes to the shortest-path solve.
- **Masking.** Pass a `mask` for water, nodata, and other invalid pixels. Unmasked garbage phase creates residues the solver has to route, and they can dominate the runtime.
- **Threads.** Whirlwind uses all logical CPUs by default. To limit it, set `WHIRLWIND_NUM_THREADS` or call `whirlwind.set_num_threads(n)` before the first unwrap. This is useful when unwrapping several interferograms at once.
- **Preprocessing.** `interpolate`, `goldstein_alpha`, and `downsample` (see [Options and recipes](RECIPES.md)) can reduce the residue count on noisy scenes, at the cost of their own runtime.

## Why Whirlwind is faster than SNAPHU

Both follow the same basic recipe: compute residues, assign statistical edge costs, solve a minimum-cost-flow problem, and integrate. The difference is the shape of the cost:

- SNAPHU's costs depend on the current flow, so SNAPHU re-evaluates them as the solution changes and has to optimize iteratively.
- Whirlwind's costs are fixed and linear in the flow, and each interior arc carries at most one unit. This is a standard linear minimum-cost-flow problem, so it can be solved with successive shortest paths.
- Because the costs are small integers, the shortest-path searches use Dial's bucket queue instead of a binary heap.

The tradeoff: Whirlwind has no topography mode and no explicit discontinuity model like SNAPHU's deformation mode. It targets surface-deformation interferograms.

## Why PHASS is sometimes faster than Whirlwind

ISCE3 PHASS gets its speed from approximations on both sides of the flow solve:

1. **Zero-cost corridors.** An edge detector and phase-gradient thresholds zero the arc costs along likely discontinuities. Residues drain along those corridors almost for free.
2. **Free reuse of flowed arcs.** Once an arc carries flow, later paths use it at zero cost. Each iteration gets cheaper, but the result is no longer a minimum-cost flow for the stated costs.
3. **Region-level repair.** After the flow step, PHASS flood-fills the regions bounded by flow lines, picks each region's 2π level by a histogram vote, and drops small regions.

Whirlwind solves the fixed-cost problem exactly over the whole frame and pairs every residue. That costs more shortest-path work on residue-heavy scenes, but avoids the region voting and small-region discards. In the paper's NISAR comparison, those heuristics are where PHASS most often disagreed with SNAPHU.

## Profiling

Set `WHIRLWIND_DEBUG=1` to print the solver's progress and timings to stderr. See [Environment variables](ENV_VARS.md).
