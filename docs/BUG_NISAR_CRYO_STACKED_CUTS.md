# NISAR cryosphere stacked-cut artifact

**Status (2026-07-20): ROOT CAUSE ISOLATED - the linear Carballo cost surface,
not the solver, capacity, or preprocessing. The PHASS cost surface on
whirlwind's own unchanged capacity-1 parity solver scores 98.97% per-comp
(from 50.57%). See "PHASS cost surface" below.**

The hardest frame in the 1,382-frame NISAR GUNW campaign is:

```text
NISAR_L2_PR_GUNW_009_074_A_137_010_7700_SH_
20260102T085052_20260102T085123_20260114T085053_20260114T085123_
X05010_N_P_J_001.h5
```

At the benchmark settings (`nlooks=16`, water-only mask), Whirlwind has 50.57%
per-production-component ambiguity agreement with the production SNAPHU unwrap.
The valid mask has exactly one 4-connected integration region, so the bridge
post-pass is structurally a no-op. This is not one of the strong-ionosphere
river failures fixed by the smaller, size-monotone bridge.

## Symptom and likely mechanism

After global alignment, the ambiguity difference is dominated by two regions:

| Whirlwind − production ambiguity | valid pixels | fraction |
| -------------------------------: | -----------: | -------: |
| 0 cycles | 1,973,864 | 52.21% |
| −3 cycles | 1,769,969 | 46.82% |

The transition is not one three-cycle edge. It is three stacked one-cycle cut
lines: all 18,454 adjacent edges on which the ambiguity changes move by one
cycle (apart from 21 two-cycle edge outliers). This matches the known limitation
of the default capacity-1 linear network on steep smooth signals: multiple units
of correction cannot share an interior arc, so the flow lays parallel cuts.
Here those cuts split almost half the glacier by three cycles.

This is especially risky to "heal" as a blind image post-process. Cryosphere
scenes can contain real high phase gradients and discontinuities, so collapsing
a large integer step merely because it looks sharp could erase real signal.

## Solver invariant check

`WHIRLWIND_DEBUG=1` shows a complete, balanced solve:

```text
[pd_full] iter=0 excess=5467 deficit=5467
...
[ssp1] DONE: stranded_sources=0 remaining_excess=0
[pd_full] FINAL total_cost=7589085 remaining_excess=0
[pd_full] ADAPTIVE FINAL remaining_excess=0
```

This rules out the stranded-residue / incomplete-SSP bug that caused the old
Ridgecrest block tear. The artifact is the completed solution under the current
unit-capacity linear objective.

## Preprocessing and solver A/B

None of the easy preprocessing controls fixes the frame:

| Variant | agreement |
| ------- | --------: |
| baseline | 50.56% |
| interpolate, cutoff 0.1 | 50.20% |
| interpolate, cutoff 0.2 | 8.82% |
| interpolate, cutoff 0.3 | 4.53% |
| Goldstein α=0.7 | 50.48% |
| downsample 2× | 39.42% |
| downsample 4× | 1.96% |
| downsample 8× | 3.98% |
| phase-gradient window 15×15 | 50.67% |
| phase-gradient window 31×31 | 51.08% |
| experimental convex solver | 38.9% |
| experimental tiled solver | 47.6% |

An effective-look sweep from 4 through the model cap of 80 reaches only 53.54%
(at 80). The product metadata records 16 azimuth × 26 range looks, but changing
the effective-look assumption therefore does not explain the split. Whirlwind's
experimental PHASS-style reuse solver did not finish after 4 min 53 s and was
stopped, so that implementation is not currently a practical workaround.

## Actual isce3 PHASS result

Actual isce3 PHASS, run through `tophu.PhassUnwrap` with its defaults, completes
in 29.8 s and is a strong result on this frame. PHASS labels 80.62% of the input
mask as 24 connected components. On pixels carrying a nonzero PHASS label and a
production label, it has 99.19% ambiguity agreement with production after the
same production-component alignment used by the benchmark. Its largest
component alone covers 79.63% of the input mask and has 99.9998% agreement.
Most importantly, it does not contain Whirlwind's coherent three-cycle split.

PHASS is conservative here: it declines to label 19.38% of the input mask. Raw
unwrapped values returned at those `cc == 0` pixels are not valid output and
must be excluded; including every finite raw value gives the misleading 83.59%
score and a visibly patchy plot.

The result is also robust to the main PHASS acceptance control. With
`good_correlation=0.7` and `min_pixels_region=200` held fixed:

| correlation threshold | labeled mask | agreement on labeled pixels | components |
| --------------------: | -----------: | --------------------------: | ---------: |
| 0.10 | 94.26% | 99.845% | 9 |
| 0.15 | 90.17% | 99.825% | 10 |
| 0.20 (default) | 80.62% | 99.194% | 24 |
| 0.25 | 71.75% | 91.278% | 39 |
| 0.30 | 60.56% | 86.321% | 59 |

At threshold 0.10, the largest PHASS component covers 94.06% of the input mask
and has 99.9996% agreement where production is labeled. This nearly removes the
coverage tradeoff while retaining the artifact-free solution. The lower
threshold is therefore a useful robustness check, not evidence that production
must be treated as truth.

This is independent evidence that the Whirlwind block is an artifact even
though production SNAPHU is not ground truth, and it makes shared/multi-use cuts
a promising direction.

Source inspection confirms that the multi-use behavior is active, not merely a
description of PHASS: its ASSP search assigns zero reduced cost whenever an arc
already carries nonzero flow, and its nominal per-arc capacity checks are
commented out. That is the same first-unit-cost / later-units-free rule used by
Whirlwind's `reuse` network. PHASS is faster because its implementation seeds
all remaining supplies in one shortest-path forest and commits many compatible
paths per iteration; this frame drains in 40 iterations.

It is still not a clean ablation of multi-use flow. PHASS also uses a different
cost surface (squared coherence, high-coherence clamping, and zero cost on
wrapped gradients of at least one radian), then adds correlation-threshold cuts,
adjusts region seed ambiguities, and rejects small components. A Whirlwind
solver experiment needs to separate the batched ASSP mechanics, PHASS costs,
and conservative region output.

## Next solver work

The principled next experiment is a whole-frame multi-unit flow objective that
allows several corrections to share an arc while charging a validated marginal
cost for every unit. The actual PHASS result is the strongest positive control
for this direction. The existing Whirlwind reuse solver allows shared cuts but
uses a zero marginal cost after the first unit and is too slow here; the current
convex prototype is faster but its objective produces a worse solution. This
needs a solver/cost experiment, not a default interpolation or Goldstein change.

## Uncapacitated linear solve: NEGATIVE result (2026-07-20)

The multi-unit experiment above is now implemented:
`WHIRLWIND_UNWRAP_SOLVER=multi` runs `unwrap_linear_multi`, a true Costantini
uncapacitated linear MCF on the exact parity solver machinery (every arc
multi-unit, every unit charged the full arc cost; `network.rs multi_mode`).
On this frame it completes cleanly and finds a genuinely cheaper optimum -
total cost 7,373,481 vs 7,589,085 for capacity-1 (-2.8%) - and the agreement
is unchanged: 50.61% vs 50.57% per-comp. The three stacked one-cycle cuts
collapse into shared crossings, but the glacier is still split by -3 cycles.

**Conclusion: the split is the optimum of the linear Carballo cost surface
itself, not a capacity artifact.** Arc capacity is ruled out alongside
preprocessing. What still distinguishes PHASS is its cost surface
(`PhassUnwrapper.cc`): squared coherence scaled by 100, a clamp of every
cost above `int(good_corr^2*100)=49` up to 255, and - critically - **zero
cost wherever the wrapped phase gradient is >= 1.0 radian** (both pixels
valid). On a fast-flowing glacier the shear margins alias past 1 rad, so
PHASS charges nothing for cuts along them while whirlwind's Carballo cost
still charges for crossing them; conversely PHASS's clamp makes the smooth
interior uniformly expensive. `scripts/phass_cost_ablation.py` grafts the
PHASS cost surface (and single-ingredient variants) onto whirlwind's own
capacity-1 parity solver via `unwrap_linear_ext_costs`.

## PHASS cost surface on whirlwind's solver: FIXED (2026-07-20)

`scripts/phass_cost_ablation.py` on this frame, same solver
(`unwrap_linear_ext_costs`, capacity-1, PD(8)+SSP), same inputs and metric as
the benchmark, no bridge (single mask region):

| variant | ingredients | match | per-comp |
| ------- | ----------- | ----: | -------: |
| baseline (Carballo cost) | slope-centered statistical cost | 52.2% | 50.6% |
| `phass` | corr² + clamp + gradzero | 96.76% | **98.97%** |
| `phass-nogradzero` | corr² + clamp | 85.41% | 87.92% |
| `phass-noclamp` | corr² + gradzero | 96.76% | 98.97% |

Runtime 21-24 s, complete balanced solves. The remaining disagreement is thin
±1-cycle streaks along the shear margins (see
`compare/phass-cost-ww/phass/full.png` next to the downloaded frames).

Ingredient ranking on this frame:

1. **Plain squared-coherence cost** (`uchar(min(γ²)·100)`) does most of the
   work (50.6 → 87.9%). The Carballo cost's 7×7 slope-centering is the
   suspect: on fringe rates near Nyquist the biased local-gradient estimate
   makes cuts across the fast-flow interior artificially cheap, while PHASS
   charges pure coherence with no slope term.
2. **Zero cost where the wrapped gradient ≥ 1 rad** adds the rest
   (87.9 → 99.0%): the aliased shear margins become free to cut along, so
   corrections concentrate there instead of splitting the interior.
3. The high-coherence clamp (>49 → 255) is irrelevant here.

Capacity is genuinely not the issue: with the PHASS surface even the
capacity-1 network reaches 99%, and with the Carballo surface the
uncapacitated solver still splits the glacier.

### Open questions before shipping anything

- Regression: does the PHASS surface hurt the frames the Carballo cost wins?
  (Paired no-bridge spot-checks on other hard frames in
  `compare/phass-cost-ww/<product>/`; the 13-frame NISAR parity set is the
  real gate.)
- Integration path: a `cost_model="phass"` (or auto-detected steep-scene
  fallback) in the Rust cost builder, composed with the existing bridge +
  snaphu-conncomp pipeline.
- Whether Carballo's slope window (`phase_grad_window`) can be repaired
  instead (e.g. gradient-magnitude gating of the slope term) to keep one
  cost model.

## Reproduce

```bash
H5=/path/to/NISAR_L2_PR_GUNW_009_074_A_137_010_7700_SH_..._001.h5

PYTHONPATH=python WHIRLWIND_DEBUG=1 python aws-batch/compare_gunw.py "$H5" \
  --out-dir cryo-debug --force

# The benchmark wrapper now exposes focused A/B controls:
PYTHONPATH=python python aws-batch/compare_gunw.py "$H5" \
  --out-dir cryo-downsample-4 --downsample 4 --force
PYTHONPATH=python python aws-batch/compare_gunw.py "$H5" \
  --out-dir cryo-interp --interpolate --interp-cutoff 0.1 --force
```
