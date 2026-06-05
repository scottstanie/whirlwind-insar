# Algorithm notes

Whirlwind is a 2D minimum-cost-flow phase unwrapper for InSAR interferograms. The public Python entry point is:

```python
unw, conncomp = whirlwind.unwrap(igram, corr, nlooks, mask=mask)
```

The input interferogram is complex. Whirlwind unwraps `angle(igram)` and uses `corr`, `nlooks`, and the optional valid-pixel mask to decide where integer 2pi cycle corrections are most likely.

## Pipeline

1. Compute wrapped phase from the complex interferogram.
2. Compute residues from 2x2 wrapped-gradient loops.
3. Build coherence-based edge costs using the Carballo/Lee statistical model.
4. Solve a minimum-cost-flow problem that pairs positive and negative residues through low-cost paths.
5. Integrate the corrected gradients through the valid mask.
6. Grow SNAPHU-style connected-component labels from the same cost model.
7. If the valid mask has disconnected regions, apply a bridge post-pass to set relative 2pi offsets when the coarse-scale evidence is clear.

## Why minimum-cost flow

Wrapped gradients can be locally inconsistent because of noise. Those inconsistencies appear as residues. A valid unwrapped phase needs those residues to be neutralized by integer cycle corrections. Minimum-cost flow chooses a set of correction paths that balances the residues while preferring low-coherence or likely wrap-line edges.

This is the same broad algorithm family as SNAPHU: residues, statistical costs, network flow, then integration. Whirlwind's implementation is Rust-backed and tuned for the 2D coherence-cost path exposed by `whirlwind.unwrap`.

## Inputs

| Input | Meaning |
|---|---|
| `igram` | Complex wrapped interferogram. |
| `corr` | Coherence or correlation in `[0, 1]`. |
| `nlooks` | Effective number of looks for the coherence estimator. |
| `mask` | Optional boolean array where `True` means valid. |

Pass a mask for real scenes with water, shadow, layover, nodata, or other invalid pixels. Without a mask, filled invalid pixels can create artificial residues along boundaries.

## Outputs

| Output | Meaning |
|---|---|
| `unw` | Float32 unwrapped phase in radians. |
| `conncomp` | Uint32 connected-component labels. `0` means background or dropped pixels. |

The unwrapped phase is congruent with the wrapped input modulo 2pi. Connected components are useful when comparing against SNAPHU-style products or when downstream code needs component labels.

## Bridge post-pass

If a valid mask splits the scene into disconnected regions, the relative 2pi offset between those regions is not observed directly from the wrapped phase. Whirlwind unwraps each region and then uses a coarse connected view of the scene to choose the relative integer offset when the evidence is clear. This fixes narrow disconnected-region cases such as the A_025 river example in the NISAR comparison.

## Notes for developers

The core implementation lives in `crates/whirlwind-core`. The Python extension lives in `crates/whirlwind-py`, and the top-level Python wrapper is in `python/whirlwind`.

For the long-form derivation and implementation details, see the technical ATBD: [ATBD-whirlwind.md](../ATBD-whirlwind.md).
