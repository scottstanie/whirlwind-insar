# Tuning connected components

`whirlwind.unwrap` returns two arrays: the unwrapped phase and a connected-component (conncomp) label map, analogous to SNAPHU's. Each positive integer marks a region the solver believes is unwrapped self-consistently; `0` is background or dropped.

The default in 0.3.0 is the SNAPHU-faithful ambiguity-wiggle grow:
`conncomp_algorithm="snaphu"` in Python and `--conncomp-algorithm snaphu` in the CLI. It recovers each pixel edge's achieved integer ambiguity from the final unwrapped phase and cuts an edge where a +/-1 cycle "wiggle" against SNAPHU's convex smooth cost is no more expensive than the achieved output.

The default `conncomp_min_coherence="auto"` labels `0` any pixel whose coherence is below `conncomp_min_coherence_auto(nlooks) = 0.32/sqrt(nlooks)` (`0.08` at 16 looks), so `conncomp == 0` is a coherence cutoff. Set `conncomp_min_coherence=None` for the older behavior (`conncomp_reliability=0`, labels every reliably unwrapped pixel), pass a float for a fixed cutoff, or raise it toward `0.1-0.15` to drop more low-coherence pixels (with more components — see below).

## Default Knobs

All Python names below are keyword arguments to [`unwrap`](ALGORITHM.md). CLI names use dashes.

| Knob                                                                                           | What it does                                                                                                                           | Direction                                                            |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `conncomp_algorithm` (default `"snaphu"`)                                                      | Selects the conncomp grower. `"snaphu"` is the default ambiguity-wiggle path; `"linear"` opts into the legacy raw coherence-cost grow. | use `"linear"` only for old behavior                                 |
| `conncomp_min_coherence` (default `"auto"`, Python + CLI)                                      | SNAPHU path only. Drop (label `0`) pixels roughly below this coherence, so `conncomp > 0` is a reliability mask. `"auto"` = looks-aware `0.32/sqrt(nlooks)` (0.08 @ 16 looks); a float = fixed cutoff; `None` (Python) / `off` (CLI) uses `conncomp_reliability` instead. Maps to reliability via `conncomp_reliability_from_coherence(gamma, nlooks)`. | higher gamma -> stricter coverage, more fragments above ~0.1 |
| `conncomp_reliability` (default `0.0`)                                                         | SNAPHU path, used only when `conncomp_min_coherence` is `None`/`<=0`. Threshold in inverse-variance (`1/sigma^2`) units; native raw threshold is this times `1_000_000`. `0` labels every reliably unwrapped pixel.                                                                                       | higher -> fewer labeled low-coherence pixels, usually more fragments |
| `min_size_px` (default `100`)                                                                  | Discard components smaller than this many pixels.                                                                                      | higher -> fewer tiny components                                      |
| `max_ncomps` (default `1024`)                                                                  | Keep only the N largest components.                                                                                                    | lower -> fewer labels                                                |

The `snaphu` algorithm ignores `cost_threshold`, `conncomp_sigma`, and `conncomp_cycle_prob`. Those are kept for `conncomp_algorithm="linear"` only.

## Why the default floor is gentle and looks-aware

Comparing the connected-component algorithms on a set of NISAR frames, the SNAPHU ambiguity-wiggle grow removes the splintering of the old linear grow while keeping component counts close to production SNAPHU:

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

Those examples use a floor of 0.15-0.20. At 0.08 (16 looks) the component count stays at the no-floor baseline across the frames tested, dropping under 1% of pixels; the count starts to climb above about 0.10.

The default is `conncomp_min_coherence="auto"`, the floor `conncomp_min_coherence_auto(nlooks) = 0.32/sqrt(nlooks)` -- about a third of the sample-coherence noise floor, which falls as `1/sqrt(nlooks)`. It is `0.08` at 16 looks and scales with looks, so the cutoff tracks the coherence a decorrelated pixel actually estimates at that look count rather than a fixed number.

The floor only removes the lowest-coherence pixels; it does not reproduce production SNAPHU's overall conservatism, which is also gated by tile cost thresholds and minimum region sizes, so whirlwind labels more than production on many frames. That is by design: whirlwind's components mark unwrap self-consistency, not a coherence mask. For the older behavior that labels every reliably unwrapped pixel, set `conncomp_min_coherence=None` (with `conncomp_reliability=0`); for production-like coverage, raise the floor toward 0.1-0.15 and expect more fragments.

## Recipes

- Keep the default: `conncomp_algorithm="snaphu"` with `conncomp_min_coherence="auto"` (looks-aware floor; drops only decorrelated pixels; `conncomp == 0` is a basic mask).
- Want the old label-everything behavior: `conncomp_min_coherence=None` (so `conncomp_reliability=0`), or in the CLI `--conncomp-min-coherence off`.
- Want production-SNAPHU-like coverage: raise `conncomp_min_coherence` toward `0.1-0.15` (expect more fragments above ~0.1).
- Too many tiny labels after raising it: raise `min_size_px` or lower `max_ncomps`.
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
