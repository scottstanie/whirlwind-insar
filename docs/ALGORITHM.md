# Unwrapping algorithm in brief

Whirlwind is a 2D minimum-cost-flow phase unwrapper for InSAR interferograms. The public Python entry point is:

```python
unw, conncomp = whirlwind.unwrap(igram, corr, nlooks, mask=mask)
```

The input interferogram is complex. Whirlwind unwraps `angle(igram)` and uses `corr`, `nlooks`, and the optional valid-pixel mask to choose integer $2\pi$ cycle corrections.

## Overview

Wrapped gradients (differences between adjacent pixels) can be locally inconsistent because of noise. Those inconsistencies appear as residues. A valid unwrapped phase needs those residues to be neutralized by integer cycle corrections. Minimum-cost flow chooses a set of correction paths that balances the residues while preferring low-coherence or likely wrap-line edges.

1. Compute wrapped phase from the complex interferogram.
2. Compute residues from 2x2 wrapped-gradient loops.
3. Build coherence-based edge costs using the Carballo/Lee statistical model.
4. Apply an aliased-gradient robustness guard that makes at most the steepest 3% of valid edges free to cut, with a 1 radian floor.
5. Solve a minimum-cost-flow problem that pairs positive and negative residues through low-cost paths.
6. Integrate the corrected gradients through the valid mask.
7. If the valid mask has disconnected regions, apply a bridge post-pass to set their relative $2\pi$ offsets from local phase at the region boundaries (a size-monotone tree in which large regions anchor smaller ones).
8. Grow SNAPHU-faithful connected-component labels from the final unwrapped phase by the ambiguity-wiggle reliability test.

The cost model gives the flow solve a precise statistical meaning: minimizing total cost maximizes a posterior probability over the integer cycle corrections. See [Why it's Bayesian](BAYESIAN.md) for the formulation and how it differs from SNAPHU's cost model.

## Inputs

| Input    | Meaning                                                |
| -------- | ------------------------------------------------------ |
| `igram`  | Complex wrapped interferogram.                         |
| `corr`   | Coherence or correlation in `[0, 1]`.                  |
| `nlooks` | Effective number of looks for the coherence estimator. |
| `mask`   | Optional boolean array where `True` means valid.       |

Pass a mask for real scenes with water, shadow, layover, nodata, or other invalid pixels. Without a mask, filled invalid pixels can create artificial residues along boundaries.

## Outputs

| Output     | Meaning                                                                    |
| ---------- | -------------------------------------------------------------------------- |
| `unw`      | Float32 unwrapped phase in radians.                                        |
| `conncomp` | Uint32 connected-component labels. `0` means background or dropped pixels. |

The unwrapped phase is congruent with the wrapped input modulo $2\pi$. Connected components are useful when comparing against SNAPHU-style products or when downstream code needs component labels. The default component grower matches SNAPHU's ambiguity-wiggle rule; the older linear coherence-cost grow is still available as an opt-out.

## Aliased-gradient robustness guard

The Carballo/Lee cost conditions on one locally expected slope. At shear margins and rupture edges, several unwrapped gradients can produce the same wrapped observation, so that single-branch model can assign an overconfident cost. Whirlwind robustifies the cost by treating at most the steepest 3% of valid edges as uninformative, subject to a 1 radian floor. This is an empirical approximation to a missing heavy-tailed or discontinuity component, not a new likelihood derivation. Set `WHIRLWIND_SLOPE_GUARD=off` to reproduce the unguarded cost field.

## Bridge post-pass

If a valid mask splits the scene into disconnected regions, the relative $2\pi$ offset between those regions is not observed directly from the wrapped phase. Whirlwind unwraps each region, then visits regions from largest to smallest and anchors each one to the nearest region already processed. It reads the relative level from the unwrapped phase in a small box at each bridge endpoint, rounding it to an integer number of cycles. The size ordering keeps tiny islands from setting the level of large land regions.

Our implementation is a port of isce3's NISAR GUNW bridging algorithm, which borrows from Yunjun, 2019.
