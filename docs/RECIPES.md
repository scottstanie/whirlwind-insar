# Options and recipes

The defaults work for most interferograms:

```python
import whirlwind as ww

unw, conncomp = ww.unwrap(igram, corr, nlooks, mask=mask)
```

This page covers the inputs worth getting right and the optional steps for harder scenes. Every option below is a keyword argument to `unwrap`. In the CLI, most have a matching flag with dashes, e.g. `--goldstein-alpha 0.7`.

## Inputs

**`mask`**: boolean, `True` = valid. The default is `(igram != 0) & (corr > 0)`. Pass your own mask for water, layover, shadow, or nodata. Invalid pixels left unmasked create spurious residues, which slow the solve and can pull errors into good regions. Masked pixels come back as `NaN` in the phase and `0` in the labels.

**`nlooks`**: the effective number of independent looks behind `corr`. It sets how much the cost model trusts the coherence: more looks means a narrower, more confident cost. It must be at least 1. Values above 300 are clamped, with a warning. For a multilooked interferogram, the effective looks are usually lower than the window size (e.g. 3 × 6 = 18), because neighboring SAR samples are correlated.

## Noisy or decorrelated scenes

These three optional steps change only what the solver sees. The integer cycles it finds are applied back to your original wrapped phase, so the output always equals the input phase plus whole cycles.

**Interpolate low-coherence pixels.** Replaces the phase of each valid pixel with coherence below `interp_cutoff` by a distance-weighted average of nearby high-coherence pixels. This helps with scattered decorrelated speckle, e.g. persistent-scatterer or phase-linked interferograms:

```python
unw, conncomp = ww.unwrap(igram, corr, nlooks, mask=mask, interpolate=True)
```

**Goldstein filter.** An adaptive spectral filter that smooths the wrapped phase before the solve. `0.7` is a typical strength:

```python
unw, conncomp = ww.unwrap(igram, corr, nlooks, mask=mask, goldstein_alpha=0.7)
```

**Downsample the solve.** Unwraps a coherently averaged copy at the given factor to pick each block's 2π cycle, then maps those cycles back onto the full-resolution phase. `nlooks` stays the effective looks of your input `corr`. Detail smaller than a block can be lost, so this suits noisy scenes more than clean ones:

```python
unw, conncomp = ww.unwrap(igram, corr, nlooks, mask=mask, downsample=8)
```

Whether any of these helps depends on the scene, so compare against the default on your own data. `ww.interpolate` and `ww.goldstein` are also available as standalone functions.

## Regions separated by the mask

When the mask cuts the scene into separate pieces, the default `bridge=True` post-pass sets the 2π offset between them. For regular gaps such as NISAR subswath gaps, `connect_gaps=True` lets the solver level the pieces directly. See [Bridging disconnected regions](BRIDGING.md).

## Steep gradients

`phase_grad_window=(parallel, perpendicular)` sets the window used to estimate the local phase slope that enters the cost (SNAPHU's `KPARDPSI` / `KPERPDPSI`, default `(7, 7)`). A larger window gives a steadier slope estimate on dense fringes but blurs it across wrap lines.

By default, Whirlwind also makes the steepest edges free to cut (at most 3% of valid edges, and only those steeper than 1 radian), so that real discontinuities such as glacier shear margins don't get priced as if they were smooth. See [the algorithm notes](ALGORITHM.md#aliased-gradient-robustness-guard).

## Connected components

The label map has its own options. See [Connected components](CONNCOMP_TUNING.md).

## Threads

Whirlwind uses all logical CPUs by default. To run several unwraps side by side, limit each process with `WHIRLWIND_NUM_THREADS=4` or call `ww.set_num_threads(4)` before the first unwrap.
