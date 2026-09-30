# Connected components

`whirlwind.unwrap` returns two arrays: the unwrapped phase and a connected-component label map, similar to SNAPHU's.

```python
unw, conncomp = whirlwind.unwrap(igram, corr, nlooks, mask=mask)
```

- Each positive label marks a region that Whirlwind considers reliably unwrapped. Within one component, the 2π levels are consistent with each other.
- Two different components may be offset from each other by a whole number of cycles.
- `0` means the pixel is masked, was judged unreliable, or belongs to a component that was too small to keep.

A common use is to treat `conncomp > 0` as a "trust this pixel" mask, or to reference each component separately.

## How the labels are grown

The default algorithm (`conncomp_algorithm="snaphu"`) follows SNAPHU's rule:

1. For each pixel edge, ask how much it would cost to shift the unwrapped solution there by ±1 cycle, using SNAPHU's smooth statistical cost.
2. If that cost is below the `conncomp_reliability` margin, the solution is not confidently pinned on that edge, so the edge is cut.
3. Cut regions are thickened sideways (`conncomp_thicken`), so a one-pixel reliable thread cannot join two sides of a wide unreliable area.
4. Pixels are grouped over the remaining edges. Groups smaller than `min_size_px` are dropped, and only the `max_ncomps` largest are kept.

The labels only affect `conncomp`. The unwrapped phase is the same whatever these settings are.

## Options

All of these are keyword arguments to `unwrap`. The CLI uses the same names with dashes, e.g. `--conncomp-reliability`.

| Option                   | Default    | Effect                                                                                                                                     |
| ------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `conncomp_reliability`   | `0.5`      | Cut margin. Higher → more pixels labeled `0`, usually more components. `0` labels nearly every pixel.                                      |
| `conncomp_min_coherence` | `None`     | Set a coherence cutoff instead of a reliability margin (a float, or `"auto"`). When set, it overrides `conncomp_reliability`. CLI: `off`.  |
| `conncomp_thicken`       | `True`     | SNAPHU-style thickening of cut regions. CLI: `--no-conncomp-thicken` to disable.                                                           |
| `min_size_px`            | `100`      | Drop components smaller than this many pixels.                                                                                             |
| `max_ncomps`             | `1024`     | Keep only the N largest components.                                                                                                        |
| `conncomp_algorithm`     | `"snaphu"` | `"linear"` selects an older coherence-cost grow. Its knobs are `cost_threshold`, `conncomp_sigma`, and `conncomp_cycle_prob`.              |

### What `conncomp_reliability` means in coherence terms

The margin is in the solver's own units: inverse phase variance, `1/σ²`. As a rough guide, an edge is cut when its coherence γ satisfies `2·nlooks·γ² / (1 − γ²) < conncomp_reliability`. So the same margin gives a different coherence cutoff at different `nlooks`:

| `conncomp_reliability` | ≈ cutoff at 4 looks | ≈ cutoff at 16 looks | ≈ cutoff at 50 looks |
| ---------------------: | ------------------: | -------------------: | -------------------: |
|                  `0.5` |                0.24 |                 0.12 |                 0.07 |
|                  `1.0` |                0.33 |                 0.17 |                 0.10 |

The default `0.5` drops decorrelated water and near-noise while keeping real low-coherence land. The old default, `0`, labeled decorrelated ocean as one large, confident component. Production SNAPHU also uses a positive margin.

To pick a margin from a target coherence, use `whirlwind.conncomp_reliability_from_coherence(gamma, nlooks)`.

## Recipes

- **Keep the defaults** for most scenes.
- **Water or decorrelated areas are still labeled:** raise `conncomp_reliability`, e.g. to `1.0`.
- **Cut at a coherence you choose:** `conncomp_min_coherence=0.2`, or `conncomp_reliability=ww.conncomp_reliability_from_coherence(0.2, nlooks)`.
- **Label everything that was unwrapped:** `conncomp_reliability=0`.
- **Too many tiny components:** raise `min_size_px` or lower `max_ncomps`.
- **Hard coherence floor on top of the labels:** post-process, e.g. `conncomp[corr < 0.3] = 0`.
- **Skip the labels in the CLI:** `--no-conncomp`.
