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
4. Solve a minimum-cost-flow problem that pairs positive and negative residues through low-cost paths.
5. Integrate the corrected gradients through the valid mask.
6. If the valid mask has disconnected regions, apply a bridge post-pass to set their relative $2\pi$ offsets from the unwrapped phase at the region boundaries (a spanning tree rooted at the largest region).
7. Grow SNAPHU-faithful connected-component labels from the final unwrapped phase by the ambiguity-wiggle reliability test.

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

## Bridge post-pass

If a valid mask splits the scene into disconnected regions, the relative $2\pi$ offset between those regions is not observed directly from the wrapped phase. Whirlwind unwraps each region, then builds a minimum spanning tree over the nearest boundary-pixel pairs (rooted at the largest region) and reads the relative level from the unwrapped phase in a small box at each bridge endpoint, rounding it to an integer number of cycles.

Our implementation is a port of isce3's NISAR GUNW bridging algorithm, which borrows from Yunjun, 2019.
