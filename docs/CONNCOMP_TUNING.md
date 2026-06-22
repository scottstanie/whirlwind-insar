# Tuning connected components

`whirlwind.unwrap` returns two arrays: the unwrapped phase and a connected-component (conncomp) label map, analogous to SNAPHU's. Each positive integer marks a region the solver believes is unwrapped self-consistently; `0` is background or dropped.

The default in 0.3.0 is the SNAPHU-faithful ambiguity-wiggle grow:
`conncomp_algorithm="snaphu"` in Python and `--conncomp-algorithm snaphu` in the CLI. It recovers each pixel edge's achieved integer ambiguity from the final unwrapped phase and cuts an edge where a +/-1 cycle "wiggle" against SNAPHU's convex smooth cost is no more expensive than the achieved output.

If you only remember one thing: the default `conncomp_reliability=0` is the calibration-free SNAPHU wiggle test. Raise it only when you want a more conservative coverage mask that drops low-coherence pixels to label `0`.

## Default Knobs

All Python names below are keyword arguments to [`unwrap`](ALGORITHM.md). CLI names use dashes.

| Knob                                                                                           | What it does                                                                                                                           | Direction                                                            |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `conncomp_algorithm` (default `"snaphu"`)                                                      | Selects the conncomp grower. `"snaphu"` is the default ambiguity-wiggle path; `"linear"` opts into the legacy raw coherence-cost grow. | use `"linear"` only for old behavior                                 |
| `conncomp_reliability` (default `0.0`)                                                         | SNAPHU path only. Threshold in inverse-variance (`1/sigma^2`) units; the native raw threshold is this value times `1_000_000`.         | higher -> fewer labeled low-coherence pixels, usually more fragments |
| `conncomp_min_coherence` (CLI) / `conncomp_reliability_from_coherence(gamma, nlooks)` (Python) | Convenience mapping from a target minimum coherence to `conncomp_reliability`. For `nlooks=16`, `gamma=0.3` maps to about `3.2`.       | higher gamma -> stricter coverage                                    |
| `min_size_px` (default `100`)                                                                  | Discard components smaller than this many pixels.                                                                                      | higher -> fewer tiny components                                      |
| `max_ncomps` (default `1024`)                                                                  | Keep only the N largest components.                                                                                                    | lower -> fewer labels                                                |

The `snaphu` algorithm ignores `cost_threshold`, `conncomp_sigma`, and `conncomp_cycle_prob`. Those are kept for `conncomp_algorithm="linear"` only.

## Why Reliability 0 Is The Default

The 2026-06-19 NISAR sweep under `nisar-pngs/2026-06-19/` compared the old linear conncomp, the SNAPHU ambiguity-wiggle conncomp at `conncomp_reliability=0`, and reliability thresholds expressed as target minimum coherence.

The SNAPHU default removed the main linear-path splintering while keeping component counts close to production SNAPHU:

| Frame | Production SNAPHU comps | Old linear comps | New SNAPHU comps |
| ----- | ----------------------: | ---------------: | ---------------: |
| A_018 |                       1 |               69 |                3 |
| A_025 |                       2 |               41 |                3 |
| A_030 |                       3 |              230 |                3 |
| A_035 |                       2 |              119 |                5 |
| D_075 |                       1 |               64 |                3 |
| D_077 |                       2 |               46 |                1 |

The reliability sweep showed why a positive threshold should not be the package default. It can make labeled percentage closer to production on some frames, but it often fragments the map:

| Frame | Production labeled % / comps | Default `0.0` labeled % / comps | Threshold example    |     Result |
| ----- | ---------------------------: | ------------------------------: | -------------------- | ---------: |
| D_077 |                     82.8 / 2 |                       100.0 / 1 | `min_coherence=0.20` |  81.3 / 18 |
| A_025 |                     92.0 / 2 |                       100.0 / 3 | `min_coherence=0.15` |  90.2 / 15 |
| D_075 |                     64.4 / 1 |                        99.9 / 3 | `min_coherence=0.20` |  62.3 / 28 |
| A_030 |                     81.8 / 3 |                       100.0 / 3 | `min_coherence=0.15` | 72.1 / 125 |

So the default is intentionally `0`: it gives the stable SNAPHU-style partition. Use a positive `conncomp_reliability` when downstream processing needs conservative coverage more than compact component labels.

## Recipes

- Keep the 0.3.0 default: use `conncomp_algorithm="snaphu"` and `conncomp_reliability=0.0`.
- Want fewer low-coherence pixels labeled: raise `conncomp_reliability`, or in the CLI use `--conncomp-min-coherence 0.15` to `0.3`.
- Too many tiny labels after raising reliability: raise `min_size_px` or lower `max_ncomps`.
- Need the old 0.2.x behavior for comparison: set `conncomp_algorithm="linear"` / `--conncomp-algorithm linear`.
- Want a hard coherence floor: post-process the labels, e.g. `cc[corr < 0.3] = 0`.

## Legacy Linear Knobs

These only apply with `conncomp_algorithm="linear"`:

| Knob                            | What it does                                                                                                         | Direction                 |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `cost_threshold` (default `50`) | Raw Carballo threshold: an edge becomes a boundary when its cost is `<= cost_threshold`.                             | higher -> more boundaries |
| `conncomp_sigma`                | Set `cost_threshold` from a Gaussian-equivalent noise level. `~3.5` maps to `cost_threshold=50`.                     | higher -> stricter        |
| `conncomp_cycle_prob`           | Set `cost_threshold` from a target per-edge one-cycle-correction probability. `~2.4e-4` maps to `cost_threshold=50`. | lower -> stricter         |

Prefer the SNAPHU path for normal use. The linear knobs remain useful for reproducing older runs and for debugging the raw Carballo component grow.

## Reproduce

```bash
python scripts/nisar_conncomp_compare.py
python scripts/sweep_conncomp_reliability.py
```

The first script writes per-frame comparison PNGs and `conncomp_summary.csv`. The second writes `conncomp_reliability_sweep.csv`, `conncomp_reliability_sweep.png`, and per-frame reliability-sweep label images under `nisar-pngs/<date>/`.
