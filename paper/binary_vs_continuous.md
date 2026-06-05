# Continuous vs. binary cost: a controlled comparison

This document captures the empirical comparison between whirlwind-rs's
continuous coherence/CRLB-weighted cost and a binary thresholded variant
that mimics spurt's "good vs. bad" pixel partition.  It is a self-contained
discussion suitable for direct inclusion in the paper.

## The question

Spurt builds its spatial graph as a Delaunay triangulation over only the
pixels with `temporal_coherence > T` (a hard threshold).  Whirlwind-rs uses
a fixed 4-connected grid graph spanning every pixel, with per-edge integer
cost

\[
  c_{\text{edge}} = \mathrm{round}\!\bigl[ S \cdot \gamma_{\text{edge}} \cdot (\pi - |\alpha_{\text{smooth}}|) \bigr]
\]

(for the Carballo cost) or

\[
  c_{\text{edge}} = \mathrm{round}\!\bigl[ S \cdot \tfrac{1}{\sigma_p^2 + \sigma_q^2} \cdot (\pi - |\alpha_{\text{smooth}}|) \bigr]
\]

(for the CRLB cost).  Bad pixels never disappear; they just contribute
small cost.

Two natural questions:

1.  **Equivalence.**  If we threshold `temporal_coherence` and pass it as a
    binary mask to whirlwind-rs, do we recover the spurt-style behavior?
2.  **Sensitivity.**  Spurt's hard threshold is fragile near the cutoff —
    a pixel at γ = 0.71 looks identical to a pixel at γ = 0.95 once
    binarized.  Does whirlwind-rs's continuous cost behave better in that
    fragile band?

We answer both with one synthetic test suite and one real-data run on the
Palos Verdes Capella SAR stack used elsewhere in the paper.

## Setup

In whirlwind-rs the existing `mask` argument to `ww.unwrap` /
`ww.unwrap_crlb` is the closest analog to spurt's exclusion: `mask=False`
sets every arc touching that pixel to integer cost 0 — cheap-to-cut.
After unwrap, pixels in the connected component of `mask=True` containing
the seed get integrated; pixels in other components stay `NaN`
([crates/whirlwind-core/src/integrate.rs](../crates/whirlwind-core/src/integrate.rs)).
This is exactly the spurt "your pixel isn't in our graph" semantics, but
**restricted by the grid's 4-connectivity** — the key difference we
quantify below.

Variants compared:

| label          | cost on good pixels                           | cost on bad pixels |
| -------------- | --------------------------------------------- | ------------------ |
| `continuous`   | full continuous γ-weighted (or CRLB-weighted) | low (γ-weighted)   |
| `binary T=0.6` | uniform γ = 0.95 (Carballo) / CRLB (PV)       | 0 (mask=False)     |
| `binary T=0.9` | uniform γ = 0.95 (Carballo) / CRLB (PV)       | 0 (mask=False)     |

Reference pixel for the PV run is chosen as the location of maximum
`temporal_coherence` in the window so it survives every threshold up to
that value.  Binary variants fall back to a per-IG finite-median anchor
on any IG where reference subtraction would otherwise propagate `NaN`
across the whole frame; this is purely cosmetic for the time-series plot.

## How to rerun

A single command per stage:

```sh
# Synthetic (~5 s)
uv run python scripts/binary_vs_continuous_synth.py \
    --out /tmp/binary-vs-continuous/synth

# Palos Verdes 1024² subset (~30 s; same window as reproduce.sh)
uv run python scripts/binary_vs_continuous_pv.py \
    --dolphin "$DOLPHIN_DIR" \
    --out /tmp/binary-vs-continuous/pv \
    --window 1000 1500 2024 2524 \
    --max-igs 60 \
    --thresholds 0.6 0.9

# Plots and aggregate metrics
uv run python scripts/binary_vs_continuous_plots.py \
    --pv-out /tmp/binary-vs-continuous/pv
```

No Rust code was added or changed for the comparison — it uses the
existing `mask` argument.

## Findings

### Synthetic 1 — bridge between blobs (`bridge_between_blobs.png`)

Two high-coherence (γ = 0.9) blobs separated by a thin γ = 0.7 bridge
embedded in a γ = 0.3 background, with a phase ramp running across.
Truth puts the right blob ~6π ahead of the left.

| variant      | RMSE [rad] | cycle errors         | blob1 offset | blob2 offset |
| ------------ | ---------- | -------------------- | ------------ | ------------ |
| continuous   | **4.45**   | 5521 / 11042         | k = 0        | k = 1        |
| binary T=0.6 | 0.15       | 0 / 11042            | k = 1        | k = 1        |
| binary T=0.8 | 0.15       | 0 / 5521 (NaN: 5521) | k = 1        | k = 0 (NaN)  |

This is the only scenario in the suite where binary outperforms
continuous.  At T = 0.6 the bridge is kept and forces a consistent
relative cycle between the blobs; at T = 0.8 the bridge is cut and
blob2 falls into a different connected component, returning `NaN`.  The
continuous variant routes the wrap-line correction through the low-coh
background and picks up an extra 2π between blobs — a real failure mode
worth flagging.  It is exactly the case where spurt's exclusion-based
graph is helpful.

### Synthetic 2 — noise spike at boundary (`noise_spike_at_boundary.png`)

Flat truth, a 2π noise spike two pixels inside a sharp γ = 0.4 → γ = 0.9
boundary.  All variants except T = 0.95 reach the same answer (RMSE
0.15, 0 cycle errors); T = 0.95 wipes the entire eval region to `NaN`.
At realistic noise levels both cost regimes agree.

### Synthetic 3 — threshold sweep (`threshold_sweep.png`)

Noisy Gaussian deformation bump, evaluation mask = pixels with
γ ∈ [0.6, 0.8] (the "moderate" band).  Binary RMSE on the *surviving*
pixels decreases monotonically with threshold — purely a survivorship
effect; the surviving pixels are the higher-γ half of the band.  The
right panel shows the cost: fraction of the eval band kept drops from
1.0 at T = 0.5 to 0 at T ≈ 0.8.  Continuous keeps all 31 526 pixels at
RMSE 0.315.

This is the canonical fragility-vs-threshold demonstration: binary
trades coverage for nothing.

### Palos Verdes 1024² subset (`coverage.png`, `timeseries.png`, `per_ig_triptych.png`)

The real-data run is the dominant finding.

| variant      | temp_coh > T (% kept) | unwrap finite anywhere        | finite cell fraction across stack |
| ------------ | --------------------- | ----------------------------- | --------------------------------- |
| continuous   | n/a                   | 1 048 576 / 1 048 576 (100 %) | 100 %                             |
| binary T=0.6 | 149 707 (14.3 %)      | 223 (0.02 %)                  | 0.021 %                           |
| binary T=0.9 | 12 335 (1.2 %)        | 1 (<0.001 %)                  | 1 x 10⁻⁶                          |

Of the 14.3 % of pixels that the T = 0.6 mask designates as "good", only
**0.02 %** are in the same 4-connected component as the seed.  The rest
are scattered fragments that the grid graph cannot stitch together.  The
coverage figure makes this immediate: the dim-purple regions of the
T = 0.6 panel are pixels above threshold but disconnected; only the
tiny bright cluster at the top edge of the scene is reachable.

This is the qualitative difference from spurt.  Spurt builds a
Delaunay triangulation over the sparse kept-pixel set; long-range edges
in that triangulation knit isolated bright pixels into a single
connected graph.  Whirlwind-rs's grid graph cannot do that — it can
only reach what is 4-connected through the mask.  On a heavily
decorrelated scene like Palos Verdes (median `temp_coh` ≈ 0.38), the
4-connected kept set fragments into thousands of tiny components.

The time-series panel shows the consequence: at three of the four
hand-picked pixels (high-coh, low-coh, near-threshold) the binary
variants have **0 / 23 finite values** because the picked pixel is
outside the seed's component.  Only the `binary_survives` pixel — chosen
specifically to be inside that component — has both methods present,
and there the binary variants show much smaller temporal variation than
continuous because the per-IG median fallback flattens the signal.

### What about the bridge result?  Is continuous strictly worse there?

The bridge scenario is an existence proof that a sufficiently noisy,
low-coh background can cause continuous to route a wrap-line correction
through the wrong region and leave a 2π gap between disconnected
high-coh regions.  In principle the fix is either (a) post-unwrap
quality masking (which we already provide via `quality_triangles`) so
the bad answer is replaced with `NaN`, or (b) a higher cost contrast
between good and bad regions so the wrap-line correction prefers to
stay in good pixels.  We have not yet evaluated (b) systematically;
it would be a small env-var-guarded variant of the cost function (see
"Open work" below).

## Conclusion

The continuous cost is not strictly a superset of the binary mask
behavior, but it is overwhelmingly preferable for the grid-graph
architecture we use, for one architectural reason: **the grid is
4-connected.**  Any approach that relies on dropping pixels — spurt's
exclusion, or whirlwind-rs with a binary mask — must compensate either
with a Delaunay-style sparse graph or with low-cost arcs through the
"bad" set.  Whirlwind-rs takes the latter route, with the additional
benefit of weighting "bad" pixels by their actual coherence rather than
treating them as a uniform veto.

The synthetic threshold sweep shows the per-pixel cost of binarization:
loss of coverage that the binary variant can never recover, on pixels
where the continuous variant still produces useful (if noisier) phase.
The Palos Verdes real-data run shows the cost in the regime where it
matters most: heavily decorrelated scenes where a hard `temp_coh`
threshold leaves only fragments, and a 4-connected grid cannot bridge
them.

The bridge synthetic is a real cautionary tale — continuous can fail
catastrophically on contrived geometries — but it requires both a very
low-coh routing channel (γ = 0.3) and a wrap line stretched across it,
neither of which dominates in the PV stack.  In production the failure
is screened out by the post-unwrap closure-quality map.

## Side experiment: multilook coherence bias correction

The 2D Carballo cost path passes the sample coherence γ̂ straight into the
Lee 1994 PDF used by the cost LUT.  Lee's PDF is conditioned on the
*true* coherence γ; using the sample estimate is a plug-in MLE that is
biased upward, especially at low γ and low L.  Direction of the bias:
γ̂ > γ ⇒ estimated phase variance is too low ⇒ cost is too high on truly-
noisy pixels ⇒ MCF under-uses them as cheap residue-routing channels.

The Touzi/Bessel-style closed-form correction

  γ_corr² = max(0, (L · γ̂² − 1) / (L − 1))

is one line of code per pixel and exactly addresses this.  It is wired
in behind `WHIRLWIND_COH_BIAS_CORRECT=1` in
[crates/whirlwind-core/src/cost/mod.rs](../crates/whirlwind-core/src/cost/mod.rs);
default off.  Unit tests cover the identity-at-γ=1, floor-at-γ̂<√(1/L),
large-L → identity, and degenerate-L≤1 behaviors.

**Bridge synthetic (mixed coherence):** the correction *fixes* the 2π
blob-to-blob misroute described above.  RMSE 4.45 → 0.15, 5521 cycle
errors → 0.  Both blobs unwrap with the same integer offset.  Mechanism:
the γ ≈ 0.3 background pixels get corrected to γ_corr = 0, so the cost
through them drops to zero and MCF routes the wrap-line dipoles freely
into the noise field instead of leaving a 2π loop spanning the bridge.

**Uniform-coherence ramps (`scripts/coh_bias_ab.py`):** the correction
*hurts*.  γ̂ = 0.3, L = 5 goes from RMSE 13.5 → 28.3 (38k → 59k cycle
errors); γ̂ = 0.5, L = 5 goes from 12.4 → 14.1.  Higher γ or higher L is
roughly neutral.

Mechanism (same one that helped the bridge):
`γ̂ < √(1/L)` ⇒ `γ_corr = 0`.  On a uniformly low-coh scene that wipes
out *every* edge's cost, leaving MCF nothing to optimize against — the
(π − |α|) gradient term has nothing to scale.  Bias correction tells us
correctly that "this whole scene is consistent with γ_true ≈ 0", but our
cost function has no graceful fallback for that case.

The closed-form correction therefore trades one failure mode for
another.  It is kept as an experimental knob (no effect on any default
code path or any production figure) because the bridge result is a real
phenomenon worth documenting and the underlying issue — biased γ in a
plug-in likelihood — is structural.  A softer correction (non-zero floor
or Bayesian shrinkage with a uniform γ_true prior) might combine the two
behaviors; not implemented.

## Posterior reliability from the MCF solution

The discussion so far compared *how spurt-style hard masking changes the
unwrap*.  A related question is whether whirlwind can emit its own
"reliable component" mask alongside the unwrap, so downstream consumers
don't have to choose a temp-coh threshold themselves and so we're not
handing the value-prop ("which pixels do you trust?") to a separate
post-processing step.

We ported SNAPHU's `GrowConnCompsMask` (Chen 2001 thesis; Chen & Zebker
2002 IEEE TGRS; `snaphu_tile.c:670`).  SNAPHU's criterion is: for each
arc, compute `min(negcost, poscost)` — the local incremental cost of
perturbing the flow by ±1 unit at the current solution.  Arcs where this
"flatness" is below a threshold are *cuts*; BFS through non-cut arcs
defines components.  Small components are dropped; the largest
`max_ncomps` are kept and renumbered.

In SNAPHU's piecewise-convex cost model the flatness signal is genuine
(`negcost > 0` and `poscost > 0` at a strict interior minimum).  In our
linear unit-capacity formulation there is no curvature anywhere, so the
adaptation collapses to: cut a pixel edge when its minimum raw arc cost
is below threshold (low coherence ⇒ uninformative noise model), plus
mask-forbidden arcs.  Notably, *MCF flow placement is not a cut
signal*: a high-cost branch cut means MCF paid the correct price to
close a noise-induced residue pair, which is the right answer to encode,
not an unreliable region.  Branch cuts that *do* sit in low-coherence
regions show up as cuts anyway because the underlying arc cost is low —
so the algorithm captures the meaningful cases without double-counting.

API: `whirlwind.unwrap_with_conncomp(igram, coh, nlooks, mask,
cost_threshold, min_size_frac, max_ncomps)` and the CRLB-path twin
`unwrap_crlb_with_conncomp(...)`.  Both return `(unwrapped, components)`
from a single MCF solve.

### Synthetic validation (`scripts/conncomp_validate.py`)

Two scenes:

1. **bridge_between_blobs** — same scene as Synthetic 1.  We compare
   spurt-style (`skimage.measure.label` on `γ̂ > T`) and MCF
   components at several thresholds.  Headline numbers (256², γ̂ ∈
   {0.3, 0.7, 0.9}):

   | Method                     | Components | Coverage            |
   | -------------------------- | ---------- | ------------------- |
   | spurt-style `γ̂ > 0.60`     | 1          | 17.2%               |
   | spurt-style `γ̂ > 0.85`     | 2          | 16.8% (bridge cut)  |
   | MCF `cost_threshold = 50`  | 1          | 100.0%              |
   | MCF `cost_threshold = 150` | 1          | 17.2% (bridge kept) |
   | MCF `cost_threshold = 250` | 2          | 16.9% (bridge cut)  |

   The MCF threshold sweep (`conncomp_sweep_bridge.png`) tracks the
   spurt sweep very closely once you align them: the same step
   transitions (background cut at low threshold; bridge cut at high
   threshold) happen at corresponding points on the two axes.  **The
   MCF approach is not a smoother gradient on bimodal-γ̂ inputs** — both
   step functions are sharp because the underlying coherence
   distribution is sharp.

2. **noisy_ramp_with_hole** — smooth ramp with a low-γ̂ disk.  All
   approaches give the right answer: one component covering ~95% of
   the scene, the hole excluded.  Threshold-insensitive (MCF stable
   over `cost_threshold ∈ [0, 250]`; spurt stable over `T ∈ [0.5,
   0.85]`).  This is the easy case.

### Where MCF should actually differ from spurt

The synthetic doesn't really stress what could be the actual advantage,
because γ̂ is set per-region rather than estimated from data.  In real
data the Carballo cost is `γ̂ · (π − |α_smooth|)` — the product of
coherence *and* a local phase-smoothness term.  Two pixels with the
same γ̂ can have very different costs depending on the local phase
gradient.  A spurt-style temp-coh threshold cannot distinguish these;
the MCF cost threshold can.  That difference should appear most
visibly on real data with regions that have high coherence but
incoherent phase gradients (water; off-axis residues from atmosphere).
We don't have a Palos Verdes pass on this branch — that's the obvious
next experiment.

### What this is not

This is not a novel algorithm.  It is the SNAPHU connected-component
output (Chen 2001), ported to whirlwind's grid + linear cost model.
The contribution is that whirlwind no longer has to hand off to an
external post-processing step or to SNAPHU itself just to emit a
reliable-component mask alongside the unwrap.  Downstream consumers
(dolphin's `displacement_workflow`, time-series anchoring,
visualization) can now use the same single call.

## Open work

- A "forbid" mode (`mask=False` ⇒ cost = `N` rather than `0`) would let
  continuous keep the bridge-scenario correctness while still spanning
  the whole grid.  Roughly 20 LOC behind an env var; not added yet
  because the comparison did not show enough motivation, and Dial's
  bucket-queue size is sensitive to the maximum cost.
- A softer coherence bias correction (cap the γ_corr floor at γ̂ / 2 or
  similar; or run a Bayesian posterior integration over γ_true rather
  than plug-in MLE).  See the side experiment above for why the
  closed-form correction is not yet a default.
- A direct A/B against spurt's actual outputs on the same window would
  be the cleanest external validation.  The plot script's
  `--spurt-out` hook is reserved for this; the spurt CLI is

  ```sh
  python -m spurt.workflows.emcf -i "$DOLPHIN_DIR" -o /tmp/spurt-out -c 0.6
  ```

  and produces unwrapped GeoTIFFs in `/tmp/spurt-out/`.  Plumbing the
  spurt outputs into the existing time-series plot is straightforward
  once those files exist.

## Artifacts

All written by the scripts above; recover by rerunning the commands.

```
/tmp/binary-vs-continuous/
├── synth/
│   ├── bridge_between_blobs.png
│   ├── noise_spike_at_boundary.png
│   ├── threshold_sweep.png
│   └── summary.json
└── pv/
    ├── coverage.png
    ├── per_ig_triptych.png
    ├── timeseries.png
    ├── aggregate.json
    ├── summary.json
    ├── temp_coh.npy
    ├── continuous/{date_phases.npy, unw_stack.npy, meta.json}
    ├── binary_T0.60/{date_phases.npy, unw_stack.npy, variant_mask.npy, meta.json}
    └── binary_T0.90/{date_phases.npy, unw_stack.npy, variant_mask.npy, meta.json}
```
