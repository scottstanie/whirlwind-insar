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

## The fix: an aliased-gradient validity guard on Carballo (2026-07-20)

The PHASS result above is not the fix to ship - swapping the whole cost surface
also swaps in squared coherence and the clamp, and it regresses other frames.
The useful question is *which part of the Carballo cost is wrong here*, and the
answer is its **domain of validity**, not its shape.

The Carballo arc cost is a log-likelihood ratio `-log(p1/p0)`: the evidence that
this edge carries a 2π cycle jump, given the locally expected slope. That
conditioning only means something while the wrapped observation still
discriminates between hypotheses. Once the true fringe rate passes Nyquist - a
glacier shear margin - one wrapped difference is consistent with many true
slopes, the likelihood ratio collapses toward 1, and the honest cost is 0.
Whirlwind instead reports the model's confident answer, which makes cutting
*along* the real discontinuity expensive, so the solver lays a cheaper cut
straight through the smooth interior. That is the -3 cycle block.

`cost::SlopeGuard` encodes exactly that: where the RAW per-edge wrapped `|Δφ|`
reaches a threshold, the cost is 0. It keys off the raw difference, not the
smoothed slope, because a box average over a shear margin is diluted by its
gentle neighbours - precisely where the model stops discriminating. It is gated
on `gamma > 0` so mask-boundary edges (masked pixels enter as `0+0j`) never
trigger. Off by default; `WHIRLWIND_SLOPE_GUARD_RAD` / `_MODE` enable it.

Cryo frame, same default solver and pipeline in every arm - only the guard
changes:

| arm | match | per-comp |
| --- | ----: | -------: |
| baseline (guard off) | 52.21% | 50.57% |
| `zerocost` 0.8 rad | 98.76% | 98.96% |
| `zerocost` 1.0 rad (PHASS's threshold) | 98.96% | 99.14% |
| `zerocost` 1.4 rad | 99.41% | 99.57% |
| **`zerocost` 2.0 rad** | **99.58%** | **99.73%** |
| `zeroslope` 1.0 rad | 52.17% | 50.55% |
| *(full PHASS cost surface, for reference)* | *96.76%* | *98.97%* |

Three things to read off this:

1. **The guard beats adopting the PHASS surface wholesale** (99.73% vs 98.97%,
   and 99.58% vs 96.76% on raw match). The statistically-grounded cost is kept;
   only its validity domain is declared.
2. **Higher thresholds are better, safer, and faster** (2.0 rad: best score,
   fewest edges touched, 26 s vs 30 s baseline). Freeing moderate gradients at
   0.8 rad discards recoverable signal.
3. **`zeroslope` does nothing** (50.55% vs 50.57% baseline). This kills the
   competing hypothesis that the *slope estimator* is at fault. Falling back to
   the zero-slope cost keeps the coherence weighting, and a coherent flat-slope
   cut is expensive - so aliased edges stay hard to cut and the block survives.
   The defect is the model answering confidently out of domain, not answering
   with a bad slope.

A separate measurement rules out the estimator independently: whirlwind
box-averages raw wrapped angles (an arithmetic mean of a circular quantity,
`cost/mod.rs`), which is wrong in principle near the ±π branch cut. On this
frame it understates steep slopes by only 11%, and just 2.5% of edges exceed
1 rad, so it cannot explain a 48-point gap. Worth fixing on its own merits
(a complex-domain circular mean is wrap-safe and coherence-weighted), but it
is not this bug.

Visual check (`slope-guard/cryo_009_074_A_137/zerocost-2.0/carballo/full.png`):
the -3 cycle block is gone, the ambiguity difference is blank apart from a thin
streak along the shear feature itself, and whirlwind returns a single connected
component covering the frame. No rip artifact.

### Validation on the 15 hard campaign frames

Same harness, `--arms baseline zerocost-1.0 zerocost-2.0`, per-component
agreement. `prod` = fraction of the frame the production unwrap actually
labels; `alias` = fraction of valid edges above 1 rad.

| frame | prod | alias | baseline | 1.0 rad | 2.0 rad |
| ----- | ---: | ----: | -------: | ------: | ------: |
| 074_A_137 (cryo) | 0.95 | 2.5% | 50.57% | **99.14%** | **99.73%** |
| 077_A_036 | 0.97 | 2.2% | 54.83% | **99.26%** | 54.16% |
| 035_D_123 | 0.96 | 12.7% | 59.59% | 62.98% | 60.74% |
| 143_D_060 | **0.05** | 49.8% | 62.26% | **25.73%** | 69.05% |
| 015_D_054 | 0.84 | 5.2% | 99.51% | 99.97% | 99.71% |
| 159_D_056 | 0.93 | 2.6% | 99.46% | 99.95% | 99.62% |
| 148_A_019 | 0.62 | 21.8% | 99.96% | 99.98% | 99.98% |
| 106_A_036 | 0.91 | 6.3% | 99.83% | 99.35% | 99.44% |
| 127_D_069 | 0.97 | 4.7% | 99.96% | 99.72% | 99.72% |
| 048_D_075 | 0.61 | 16.3% | 99.97% | 99.96% | 99.98% |
| 033_A_019 / 045_D_052 / 049_A_035 / 055_D_073 / 055_D_071 | ≥0.97 | ≤1.4% | ≥99.92% | unchanged (±0.01) | unchanged |

**A second broken frame is explained by the same mechanism.** `077_A_036` was
recorded above as "a real within-region solve issue" that only downsampling
helped (54.83% → 85.33% at 4x). The guard at 1 rad takes it to **99.26%** with
no downsampling. Two of the campaign's worst frames are one bug.

**But the threshold does not generalize.** 1 rad fixes `077_A_036`; 2 rad does
nothing for it (54.16%). 2 rad is best on the cryo frame; 1 rad wrecks
`143_D_060` (62% → 26%). No single radian value is right for all three.

### What actually discriminates: the aliased FRACTION, not coherence

The obvious refinement - fire only on coherent edges, so a steep gradient means
discontinuity rather than noise - is **wrong**, and the data says so plainly.
Aliased edges are low-coherence in *every* frame, including the ones the guard
fixes:

| frame | mean edge coherence | mean coherence of aliased edges | aliased AND γ>0.4 |
| ----- | ------------------: | ------------------------------: | ----------------: |
| 074_A_137 (fixed) | 0.381 | 0.241 | 0.6% |
| 077_A_036 (fixed) | 0.416 | 0.209 | 0.2% |
| 143_D_060 (broken) | 0.198 | 0.181 | 0.2% |

A coherence gate at 0.4 would fire on 0.6% of the cryo frame instead of 2.5% -
throwing away most of the fix - while barely changing `143_D_060`. Coherence
does not separate the good cases from the bad one.

The quantity that does is **how much of the cost field the guard erases**.
Freeing 2-3% of edges (`074`, `077`) fixes those frames; freeing 50%
(`143_D_060` at 1 rad) destroys the solve. That damage is real, not a metric
artifact of the frame's 5% production coverage: the unwrapped field goes
visibly blotchy and the ambiguity error widens to ±6 cycles
(`slope-guard-frames/zerocost-1.0/...143_D_060.../carballo/full.png`).

This points at an **aliased-edge budget** rather than a fixed radian threshold:
pick the threshold per frame as a high quantile of the raw `|Δφ|` distribution,
so the guard frees at most a few percent of edges, floored around 1 rad. On
`074`/`077` (2-3% aliased) that selects ~1 rad and keeps both fixes; on
`143_D_060` (50% aliased) it pushes the threshold toward π, which disables the
guard exactly where it does harm. One scene-independent knob instead of a
per-scene radian value.

### The budget rule, measured

`WHIRLWIND_SLOPE_GUARD_BUDGET` with a 1 rad floor, on the six frames that moved
under any arm:

| frame | baseline | budget 0.05 | budget 0.03 | best FIXED threshold |
| ----- | -------: | ----------: | ----------: | -------------------: |
| 074_A_137 (cryo) | 50.57% | 99.14% | **99.38%** | 99.73% (2.0) |
| 077_A_036 | 54.83% | 99.26% | **99.26%** | 99.26% (1.0) |
| 035_D_123 | 59.59% | 94.21% | **94.43%** | 62.98% (1.0) |
| 143_D_060 | 62.26% | 69.09% | **62.32%** | 25.73% (1.0) ← wrecked |
| 106_A_036 | 99.83% | 99.64% | 99.65% | 99.44% (2.0) |
| 127_D_069 | 99.96% | 99.72% | 99.72% | 99.72% (both) |

At budget 0.03: **worst regression −0.25 pp, mean +21.28 pp.** One
scene-independent knob fixes all three broken frames, leaves the decorrelated
frame alone instead of destroying it, and regresses the healthy frames less
than either fixed threshold. `035_D_123` is fixed *only* by the budget rule -
no fixed threshold got it above 63% - because what it needed was a threshold
selective enough to free 3% rather than the 12.7% that 1 rad frees there.

The results are stable across 0.03 and 0.05, so this is not knife-edge tuning.
The radian floor is load-bearing in the other direction: on frames with few
aliased edges it stops the budget from overspending, which is why `074`/`077`
at budget 0.05 reproduce the fixed-1 rad numbers exactly.

Recommended setting: **`WHIRLWIND_SLOPE_GUARD_BUDGET=0.03`,
`WHIRLWIND_SLOPE_GUARD_RAD=1.0`** (floor).

### 13-frame parity gate: PASSED

The set that established ww-orig equivalence, same arms:

| result | value |
| ------ | ----: |
| frames improved | 5 |
| frames unchanged | 8 |
| frames **regressed** | **0** |
| worst change | −0.01 pp |
| mean change | +0.04 pp |
| mean per-comp | 98.91% → 98.95% |

Every frame at 100% stays at 100%. The historically weakest frame, `D_075`,
improves 88.07% → 88.21%; `A_020` improves +0.19 pp; `D_077` +0.11 pp. Nothing
moves down by more than a rounding step.

So the guard is not a tradeoff on this set - it is free. Combined with the hard
frames (worst regression −0.25 pp, mean +21.28 pp), `budget=0.03` with a 1 rad
floor is defensible as a **default**, not just an opt-in mode. Remaining
judgement call before flipping it: whether to re-run the full 1,382-frame
campaign first, which is the only way to see the tail.

### Risk: the guard behaves differently in noise than on a steep margin

Freeing an edge is only justified when a large gradient means "real
discontinuity". Under pure decorrelation `Δφ` is uniform, so `P(|Δφ| ≥ 1) =
(π−1)/π ≈ 68%` - a noisy frame would have most of its edges freed. Measured
aliased-edge fraction across the hard set confirms the spread: the cryo frame
is only 2.5% at 1 rad (a genuinely steep *coherent* margin, the intended
target), while `143_D_060` (mean coherence 0.243) is **49.8%**, and
`148_A_019` 21.8%, `048_D_075` 16.3%.

If those frames regress, the principled refinement is a coherence gate - fire
the guard only where the edge is coherent enough that a steep gradient implies a
discontinuity rather than noise. In decorrelated areas the Carballo cost is
already low, so the guard has little to add there anyway.

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

### Slope guard

The guard is off unless `WHIRLWIND_SLOPE_GUARD_RAD` is set, so it composes with
any existing entry point:

```bash
WHIRLWIND_SLOPE_GUARD_RAD=2.0 PYTHONPATH=python \
  python aws-batch/compare_gunw.py "$H5" --out-dir cryo-guard --force
```

The A/B tooling reuses `compare_gunw.py`'s cached `full_arrays.npz`, so the
inputs and the agreement metric are byte-for-byte the benchmark's. Each arm is
a separate process because `cost::slope_guard()` caches in a `OnceLock`.

```bash
# threshold sweep on one frame (baseline, 0.8/1.0/1.4/2.0 rad, zeroslope)
scripts/run_slope_guard_sweep.sh <compare-dir>/<product>/full_arrays.npz <out-dir>

# many frames x arms -> sweep.md + sweep.json with a regression summary
PYTHONPATH=python python scripts/slope_guard_frame_sweep.py \
  --compare-dir <compare-dir> --out-dir <out-dir> \
  --arms baseline zerocost-1.0 zerocost-2.0
```

The 13-frame parity set has no cached arrays yet; generate them once with
`compare_gunw.py`, then sweep the cheap way:

```bash
PYTHONPATH=python python aws-batch/compare_gunw.py \
  /Volumes/.../nisar_gunw/*.h5 --out-dir <parity-compare-dir> --nlooks 16
PYTHONPATH=python python scripts/slope_guard_frame_sweep.py \
  --compare-dir <parity-compare-dir> --out-dir <parity-guard-dir> \
  --arms baseline zerocost-2.0
```
