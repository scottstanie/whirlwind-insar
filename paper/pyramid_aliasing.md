# Multilook-first aliasing, and a pyramidal alternative

Companion to `crates/whirlwind-core/src/pyramid.rs` and
`scripts/dense_fringe_pyramid.py`. Written to answer a concrete worry about the
`multilook`-first path: *what happens to a steep, dense-fringe signal (a volcano
eruption bowl, an earthquake near-field) when we down-look by a lot, unwrap the
coarse grid, and use that as a starting point?*

## TL;DR

* **The worry is real and the failure is dramatic.** Single-shot
  multilook-first (`ww.unwrap(..., multilook=L)`) ALIASES any signal whose
  full-res fringe rate `g` (rad/pixel) satisfies `L·g > π`, and the
  block-replicated coarse `K` can never recover it. With `L = 8` a *mild* cone
  at `g = 0.2π` collapses from **98 %** K-correct (full-res) to **6 %**. This is
  the "averaged past aliasing, then locked onto the wrong K" case exactly.
* **The most important finding is a base-solver bug, not a multilook one.** The
  linear Carballo coherence cost ([`unwrap`]) mis-routes the CORNERS of a smooth
  steep signal (the steepest part of a bowl) because the concentric wrap-line
  rings must drain at the image boundary and the capacity-1 frame arcs can't
  carry the stacked flow (same pathology as the ignored `diagonal_ramp_512`
  regression). On a *perfectly clean* `0.7π` bowl `unwrap` scores only **88 %**,
  all errors in the corners, while `unwrap_reuse` and `unwrap_convex` score
  **100 %**. (Noise dithers the ring alignment, so the pathology is worst on
  clean/synthetic data.) For many "scary steep signal" cases the real fix is a
  better default *cost*, not a pyramid - see the Assessment section.
* **A pyramidal coarse-to-fine unwrap is a sound optional tool** for the
  noisy-and-steep regime. Refining by powers of two and unwrapping only the
  *residual* against the upsampled coarser solution (the previous level's `K` as
  a prior) recovers full resolution without the single big multilook jump.
  `unwrap_pyramid` defaults its per-level solver to `"reuse"` (corner-safe).
* **The default is conservative: `base_factor=1`** - a single full-resolution
  reuse solve that never aliases. `base_factor=N>1` opts into a fixed cascade
  for noise suppression; `base_factor=0` opts into the **experimental** automatic
  Itoh-violation-rate probe (synthetic-tuned, with a near-Nyquist constant-ramp
  blind spot - not recommended on real data yet).
* **It is not magic.** Nothing recovers a signal genuinely aliased at full
  resolution (`g > π`, a hard Nyquist wall), the experimental auto-probe has a
  constant-ramp blind spot (below), and the multi-level cascade has not been
  shown to beat a simpler 2-level scheme - see the Assessment.
* **Large frames are covered.** `tile_size>0` tiles the finest levels to bound
  peak memory; the coarsest (absolute) level reuses the anchored tiled path, the
  residual (relative) levels use a lightweight prediction-relative tiler.

## Why multilook-first aliases

Coherently down-looking the complex igram by `L` averages `LxL` blocks. The
coarse grid's pixel spacing is `L` original pixels, so a full-res phase gradient
`g` rad/pixel becomes `L·g` rad per *coarse* pixel. Phase unwrapping rests on
Itoh's assumption that adjacent samples differ by less than half a cycle
(`|Δφ| < π`); above that the wrapped gradient `Δψ = wrap(Δφ)` folds to the wrong
branch and the solver confidently integrates too few cycles.

So multilook-first is safe only while `L·g < π`, i.e. `g < π/L`. For `L = 8`
that is `g < π/8 ≈ 0.39 rad/pixel` - eight times *stricter* than the true
full-resolution Nyquist limit `g < π`. Worse, block-replicating the coarse
solution bakes the wrong `K` in permanently: no later stage can add the cycles
the coarse grid threw away. This is invisible on the broad, gently-sloped scenes
multilook-first was tuned for and catastrophic on a dense-fringe deformation
sharing the same frame.

## The pyramidal scheme

Classic multigrid / multi-resolution unwrapping. Build a schedule
`base, base/2, …, 1` and refine coarse→fine:

1. **Coarsest level** (factor `base`): an ordinary whole-image unwrap of the
   `basex` down-looked igram. This level alone must be unaliased (`base·g < π`).
2. **Each finer level** `f`: bilinearly upsample the previous (coarser) level's
   *unwrapped* phase to this grid → `pred`. Rotate this level's complex igram by
   `exp(−i·pred)`, so its phase becomes `wrap(angle − pred)` - the **residual**
   wrapped phase - while its magnitude (hence coherence) is untouched. Because
   `pred` already carries the large-scale gradient, the residual gradient is
   small (well under π), so a plain unwrap solves it without aliasing. The
   level's phase is `pred + unwrap(residual)`.

`pred` is exactly the requested "previous solved K as a prior": per pixel,
`round((pred − angle)/2π)` is the integer cycle the coarse solve believes this
pixel sits in, and the residual unwrap only corrects deviations from it.
Refining all the way to `f = 1` always returns a *full-resolution* surface -
never the blocky block-replicated field of single-shot multilook.

Implementation: `whirlwind_core::pyramid::unwrap_pyramid` /
`ww.unwrap_pyramid(igram, corr, nlooks, mask=None, base_factor=0,
solver="reuse", tile_size=0)`.

### Base solver: why not the linear coherence cost (the "corners" bug)

A smooth radial bowl has *no* phase noise yet still produces residues from
discretization, and its wrap-lines are concentric rings that can only terminate
at the image boundary. The linear coherence cost routes each ring's endpoints to
the frame, but the frame arcs have unit capacity, so the stacked flow at the
corners (where the bowl is steepest and the rings densest) overflows onto
interior arcs and flips whole corner blocks by ±2π. On a perfectly clean `0.7π`
bowl this is **100 % of the wrong pixels in the corners** (`r > 120` of `180`),
dragging `unwrap` to 88 %. `unwrap_reuse` (multi-unit arcs) and `unwrap_convex`
(quadratic cost) both place no spurious interior flow and score 100 %. The
effect is loud on clean synthetic signals because the pathology needs
exactly-aligned rings; a noisy real bowl dithers them. The pyramid defaults to
`"reuse"` so the synthetic and real cases behave the same. See
`panels_corner_solver.png`.

### Automatic `base_factor` (Itoh-violation-rate probe)

The probe measures the **Itoh-violation rate** of each `fx` down-looked igram:
the fraction of adjacent coarse-pixel wrapped phase differences whose magnitude
exceeds `0.6π`. This directly tests the aliasing (Nyquist) condition. An
*absolute* threshold does not work, because phase NOISE alone pushes the rate
high (≈0.25–0.35 at 4 looks / γ≈0.2) without any aliasing. The discriminator is
the *direction of change* with `f`:

* **Noise** makes the rate *fall or hold* as `f` grows - coherent averaging
  suppresses it (noisy `0.2π` cone: `f=1 → 0.33`, `f=2 → 0.26`).
* **Aliasing** makes it *jump* once `f·g > π` (that same cone: `f=4 → 0.42`; a
  clean `0.6π` bowl: `f=1 → 0.0`, `f=2 → 0.28`).

So `auto_base_factor` walks `1, 2, 4, …` and keeps doubling while the next level
either sits below a benign noise `FLOOR = 0.05` *or* the rate *meaningfully
decreases* (drops by ≥ `DECR = 0.02` - coherent averaging still suppressing
noise on an unaliased grid). It stops the first time the rate holds flat or
rises (the aliasing fold) and returns the factor before it. Both conditions are
needed: an absolute threshold alone never downsamples noisy data (noise keeps
the rate high); a decrease rule alone never downsamples clean gentle data (the
rate is already ≈0). This fixed two failure modes found while iterating - an
initial residue-density probe over-downsampled *clean* steep bowls (too few
residues to trip any floor even when aliased), and an absolute-threshold Itoh
probe over-downsampled *very noisy* mild signals (the aliasing jump was buried
under the noise floor).

**Limitation - the constant-ramp blind spot.** The probe (like any local
gradient/curl measure) detects aliasing through the wrapped jumps it creates,
which appear where the gradient *varies* (every real localized signal: bowls,
point sources, faults). A near-constant-rate ramp aliases *coherently* -
adjacent aliased pixels' wrapped gradient folds back small - so the probe cannot
cleanly see the aliasing onset. For mild-to-moderate constant rates it still
keeps base = 1 (the f=1 rate is low and the f=2 jump is visible), and the reuse
solver handles the unaliased full-res signal - `pyrA` is 100 % on the clean cone
up to `g = 0.7π`. But at the very steepest rates (`g ≳ 0.8π`) the f=1 rate is
*already* high from the steep signal itself, the trend is ambiguous, and the
probe over-downsamples and aliases (cone `g = 0.9π`: `pyrA` = 1 %). This is a
fundamental ambiguity (an aliased ramp is indistinguishable from a gentle ramp
without external information), not a tuning bug - pass an explicit
`base_factor=1` for a scene you know is a near-Nyquist constant-rate ramp.

### Tiling the finest levels (memory)

The MCF graph is `mxn` nodes regardless of residue count, so the finest level is
the memory bottleneck on a large frame. With `tile_size>0`, the **coarsest**
level (absolute phase, no prediction) reuses `unwrap_tiled`, whose global coarse
anchor is exactly right for absolute phase. The **residual** levels are
*relative* to a global prediction, so the anchored tiled path is actively
harmful there (it region-votes a near-flat field into garbage). Instead they use
a lightweight tiler: solve each overlapping tile independently (trivial - the
residual is small-gradient), gauge each to a common cycle by removing its
rounded-2π median, and feather-composite. Because every tile's residual is
referenced to the *same* prediction, no inter-tile 2π reconciliation is needed.
Tiled K matches untiled to within seam noise.

## Synthetic dense-fringe results

`scripts/dense_fringe_pyramid.py` (deterministic, seed 0, 384² grid) builds two
truths - a constant-rate **cone** (`φ = g·r`) and a steep **bowl** / paraboloid
(`φ = a·r²`, edge rate `g_edge`) - simulates Goodman-noise igrams, and reports
K-correct fraction (per-pixel integer-cycle agreement with truth) for `full`,
`ml4`, `ml8`, `pyr2`, `pyr4`, and `pyrA` (auto `base_factor`). The pyramid uses
its default reuse base solver.

**Fringe-rate sweep, cone, γ = 0.95 (8 looks).** Single-shot multilook collapses
at its thresholds (`ml4` at `g > π/4`, `ml8` at `g > π/8`); fixed-base pyramids
alias once `base·g > π`. `pyrA` is 100 % up to `g = 0.7π`: the Itoh probe sees
the f=2 violation jump on the (unaliased) clean cone and *keeps* base = 1, where
the corner-safe reuse solver already nails the full-resolution cone. At the
extreme `g = 0.9π` `pyrA` collapses to 1 % - this is the constant-ramp blind
spot (a cone is a near-constant radial ramp): so near the Nyquist wall the f=1
violation rate is itself already high from the steep signal, the probe
mis-reads the trend and over-downsamples. `full` (87→83 %) is the only method
that degrades gracefully at the very steepest rates; set `base_factor=1`
explicitly for a scene you know is a near-Nyquist ramp.

| g (xπ) | full |  ml4 |  ml8 | pyr2 | pyr4 | pyrA |
| -----: | ---: | ---: | ---: | ---: | ---: | ---: |
|    0.1 |   98 |   98 |   97 |  100 |  100 |  100 |
|    0.2 |   94 |   91 |    6 |  100 |  100 |  100 |
|    0.3 |   91 |    3 |    5 |  100 |    7 |  100 |
|    0.5 |   89 |    2 |    2 |   98 |    4 |  100 |
|    0.7 |   87 |    3 |    2 |    1 |   14 |  100 |
|    0.9 |   83 |    1 |    1 |    1 |    2 |    1 |

**Fringe-rate sweep, bowl, γ = 0.95 (8 looks).** The varying-gradient case (a
volcano bowl). `pyrA` is 100 % across the whole range - the probe downsamples
where the corner annulus is still unaliased and backs off where it isn't -
beating the fixed bases `pyr2`/`pyr4`, which degrade once their coarsest level
aliases at the steep edge.

| g_edge (xπ) | full |  ml4 |  ml8 | pyr2 | pyr4 | pyrA |
| ----------: | ---: | ---: | ---: | ---: | ---: | ---: |
|         0.2 |   95 |   94 |   83 |  100 |  100 |  100 |
|         0.3 |   89 |   88 |    6 |  100 |  100 |  100 |
|         0.4 |   89 |   72 |    5 |  100 |   96 |  100 |
|         0.5 |   88 |   50 |    4 |  100 |   66 |  100 |
|         0.6 |   88 |    3 |    5 |  100 |   15 |  100 |
|         0.8 |   87 |    2 |    3 |   80 |    8 |  100 |
|         0.9 |   87 |    2 |    2 |   64 |   11 |  100 |

**Noise sweep, mild rate g = 0.2π, 4 looks, falling coherence.** The regime that
justifies multilooking: as γ falls the full-res solve drowns in noise residues,
but the coarse grids stay unaliased. `pyrA` matches the best fixed base at every
γ - the Itoh probe reads the noise *falling* with `f` and downsamples as far as
it safely can.

|    γ | full |  ml4 |  ml8 | pyr2 | pyr4 | pyrA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 |   90 |   92 |    6 |  100 |  100 |  100 |
| 0.35 |   85 |   91 |    6 |   98 |   98 |   98 |
| 0.30 |   76 |   90 |    6 |   97 |   97 |   97 |
| 0.25 |   69 |   90 |    6 |   95 |   95 |   95 |
| 0.20 |   39 |   82 |    6 |   93 |   93 |   93 |

Across all three sweeps `pyrA` is the best or tied-best method except at the
near-Nyquist constant-ramp blind spot (cone `g = 0.9π`, above). `ml8` is stuck
near 6 % whenever `8·g > π`, regardless of how clean the data is. Figures
(`curves_cone.png`, `curves_bowl.png`, `curves_noise.png`,
`panels_steep_bowl.png`, `panels_corner_solver.png`) are written to `--out`.

## Recommendation

* Keep the single-shot `multilook` path for what it is good at (cheap noise
  suppression on gently-sloped scenes), but **do not apply a large `multilook`
  blindly** to scenes that may contain dense fringes.
* `unwrap_pyramid`'s **default is `base_factor=1, solver="reuse"`** - a single
  corner-safe full-resolution solve that never aliases. This is deliberately
  conservative: it is the right default for an unknown scene.
* Opt into a fixed cascade (`base_factor=2`–`4`) only when you *know* the scene
  is noisy enough to need it AND shallow enough that `base_factor·g < π`.
* `base_factor=0` (the automatic Itoh probe) is **experimental**: synthetic-tuned
  and with a near-Nyquist constant-ramp blind spot. Don't rely on it on real data
  yet.
* Set `tile_size>0` on large frames to bound peak memory.

## Assessment - is the pyramid the right path?

Honest verdict after building it: **the diagnosis is the durable result; the
pyramid is a reasonable safety tool but not obviously the primary fix.**

**What is solid.** (1) The aliasing characterization - multilook-first is safe
only while `L·g < π`, eight times stricter than the true Nyquist limit, and the
wrong `K` is unrecoverable. That justifies *not* applying a big blind multilook
regardless of what replaces it. (2) The residual-against-prediction mechanism is
standard multigrid and is sound as an *optional* tool for the noisy-and-steep
regime.

**What is fragile / oversold.** (1) The automatic base probe is a
synthetic-tuned heuristic with a fundamental constant-ramp blind spot - hence
it is no longer the default. (2) Everything here is synthetic (cones, bowls,
Goodman noise); the residual tiler's "residual stays within a cycle" assumption
is untested when the coarse prediction is poor.

**2-level vs N-level - the cascade earns its keep, but only in heavy noise.**
A natural simplification is a *2-level* scheme: coarse solve at `base`, then one
residual pass straight to full resolution (skip the intermediate octaves). Both
share the coarse solve and the residual-against-prediction trick. Measured
(`scripts/pyramid_2_vs_n.py`, gentle `g=0.08π` cone, reuse solver, 3 seeds):

| base |    γ | looks | full | 2-level |  N-level |
| ---: | ---: | ----: | ---: | ------: | -------: |
|    8 | 0.60 |     4 | 99.9 |    99.9 |     99.9 |
|    8 | 0.15 |     4 | 54.9 |    85.1 | **89.8** |
|    8 | 0.12 |     4 | 35.6 |    55.4 | **86.8** |
|   16 | 0.18 |     4 | 89.6 |    60.0 | **76.6** |
|   16 | 0.12 |     4 | 35.6 |    24.1 | **46.9** |

They **tie on clean and mild-noise data** - there the single jump is fine, so for
most scenes the extra levels are unnecessary machinery. But in the
**extreme-noise** regime the cascade wins decisively (up to +31 points), and by
more the larger the `base`. Mechanism: 2-level's lone full-resolution residual
pass sees the *full* per-pixel noise (its effective looks are not multiplied), so
under heavy noise it drowns just like plain full-res; N-level's intermediate
residuals run on down-looked grids with effective looks scaled by `f²`, staying
unwrappable and handing a clean prediction to the next octave. A single
`base→full` jump skips that progressive denoising. So the cascade is not
redundant - it is precisely the machinery that makes the pyramid useful in the
one regime (very low coherence) that justified it. A reasonable product choice:
default to a small `base` (cheap, 2-level-equivalent) and only spend the full
cascade when a large `base` is warranted.

**Can `base` be auto-selected, or is it another SNAPHU-style expert knob?**
This is the question that decides whether the pyramid is *usable* or just
*capable*. The encouraging answer: unlike SNAPHU's many interacting parameters,
the pyramid has essentially **one** knob with a *physical, measurable* ceiling
(`base·g < π`, the coarsest-level Nyquist limit) and an **asymmetric** failure
mode - too-small `base` only forgoes some denoising (a few K-points), too-large
`base` aliases (catastrophic). So "pick the largest non-aliasing base" is a
single well-posed estimation problem, not a search over a knob soup.

`scripts/pyramid_auto_base.py` tests it over a (steepness, coherence) grid,
comparing the data-driven Itoh-probe choice against an unknowable per-cell
*oracle* and against every fixed default, scored by **regret**.

*Regret* is borrowed from decision theory: for one scene, it is how much
K-correct (% of pixels on the right integer cycle) you give up by using a
strategy's `base` instead of the best `base` you could possibly have chosen for
that scene. Formally, for scene `s` and strategy `π` choosing base `bπ(s)` from
`{1,2,4,8,16}`,

> `regret(π, s) = max_b K(s, b) − K(s, bπ(s))`,

where `K(s, b)` is the K-correct of the pyramid at base `b`. The `max_b` term is
the **oracle** - the best achievable at any base, computed here by brute force
because we know the synthetic truth (it is *not* available at run time; it only
defines the ceiling we measure against). So regret = 0 means "chose the best
possible base"; regret = 30 means "left 30 K-points on the table versus the best
base for that scene." We report it two ways across the grid: **mean** (typical
cost of the strategy) and **worst-case** (its biggest single failure - the
number that matters for a default you can't babysit). Lower is better; a strategy
with low mean *and* low worst-case is one you can ship unattended.

| strategy                      | mean regret | worst-case regret |
| ----------------------------- | ----------: | ----------------: |
| **probe (auto)**              |     **0.8** |           **8.1** |
| fixed `base=1` (conservative) |         5.9 |              38.9 |
| fixed `base=2`                |         3.1 |              36.0 |
| fixed `base=4`                |        21.0 |              92.5 |
| fixed `base=8`                |        25.4 |              89.8 |

The probe lands **within ~1 K-point of the oracle on average** and never loses
more than 8, while *no* fixed default is safe across the grid (every one has a
30–90-point worst case - exactly the "works on almost every scene but only with
the right setting" trap). This is the structural reason auto-selection looks
tractable here where it isn't for SNAPHU: the goal is a single physical quantity
(the aliasing onset), it is directly observable (the violation rate jumps when a
down-look folds the fringes), and guessing low is cheap while guessing high is
ruinous - so a conservative estimator is near-optimal.

Two honest caveats keep this from being "solved": (1) the probe still has the
near-Nyquist constant-rate-ramp blind spot (`g≳0.8π`, documented above), out of
this grid's range; (2) all synthetic. But the regret structure is strong enough
that promoting the probe from experimental to default is well-motivated *once it
survives a real scene* - and even a wrong probe guess is bounded by the
asymmetry, unlike a mis-set SNAPHU parameter.

**The deeper fix this surfaced.** The single most important finding is nearly
orthogonal to the pyramid: the **linear coherence cost scores only 88 % on a
perfectly clean, noise-free steep bowl** (all errors in the corners) while
`reuse`/`convex` score 100 %. That is a base-solver / cost-model issue, not a
multilook or pyramid one. For a large class of "scary steep signal" cases the
right fix is therefore **a better default cost**, not a pyramid - full-res
reuse/convex already nail clean steep bowls. This is independent evidence
pointing the same way as the in-flight `convex_cost_design.md` NISAR work.

**Why the better cost is better, and when.** The linear cost penalises `|flow|`
only - "don't route flow here unless you must" - with *no preferred direction*.
The corner/boundary failure follows directly: when concentric wrap-line rings
all drain to the image boundary, the unit-capacity frame arcs stack up, the
solver has no signal for the true per-arc cycle count, and the overflow spills
onto interior arcs. `convex` uses `(k − offset)² / σ²` where `offset` encodes the
local smoothed phase gradient, so it actively *pulls* each arc's flow toward the
physically-correct integer and gets the topology right. `reuse` fixes the same
failure differently: arcs become multi-unit at zero marginal cost after the
first push, so the stacked drainage flows through instead of spilling. So the
better cost helps exactly where there is **large-scale structure to get right**
(steep gradients, boundary-draining wrap-lines, dense fringes) and stops helping
under **pure high noise** (many independent residue dipoles that pair locally) -
there the preferred-offset signal is itself noisy and there is no regional
topology to recover.

**Cost of the better cost - usually negative (it is faster).** Measured
single-threaded (so the cost model is what is compared, not parallelism) on
Goodman-noise cones, time relative to linear with K-correct in parens:

| scene                 | linear       | reuse       | convex          |
| --------------------- | ------------ | ----------- | --------------- |
| 256², γ=0.7           | 34 ms (99)   | 1.36x (100) | **0.90x** (100) |
| 512², γ=0.7           | 245 ms (94)  | 1.03x (100) | **0.52x** (100) |
| 1024², γ=0.7          | 2546 ms (89) | 0.70x (100) | **0.23x** (100) |
| 512², γ=0.6, ~260 res | 272 ms (94)  | 0.91x (100) | **0.48x** (100) |
| 512², γ=0.4, ~7k res  | 367 ms (90)  | 1.12x (100) | 1.77x (100)     |
| 512², γ=0.3, ~23k res | 543 ms (90)  | 1.68x (99)  | 2.68x (98)      |

The accuracy fix and the speed-up are the *same* phenomenon: with a correct
preferred offset (convex) or free reuse, the solver reaches the optimum in far
fewer augmentations instead of thrashing on the boundary-stacking pathology, and
the margin *grows* with image size. The slowdown only appears in the
low-coherence, tens-of-thousands-of-residues regime (convex up to ~2.7x, reuse
~1.7x), where every arc carries multi-unit flow - and that is the same regime
where one would be multilooking/pyramiding anyway, so the relevant comparison
there is against a coarse solve, not full-res.

**Caveats on this evidence.** Single-threaded synthetic cones with i.i.d.
Goodman noise; real scenes have spatially-correlated noise and different
absolute residue counts. And `convex`'s `offset` definition is known to be
finicky - `convex_cost_design.md` and the `WHIRLWIND_DEVIATION_COST`
negative-result note record an earlier wrong offset choice that hurt real NISAR
data. The synthetic win may partly reflect that a smoothed-gradient offset is
ideal on smooth synthetic cones. So "make convex the default" still needs a
real-scene A/B before being trusted; `reuse` is the lower-risk first step (same
cost shape question does not arise - it changes only arc capacity, not the cost).

**Suggested path forward (next increments, not this PR).**
1. Evaluate making `reuse` (or `convex`) the default solver in the top-level
   `unwrap`, pending real-data validation. That removes the corner failures
   everywhere with zero pyramid machinery.
2. Then the pyramid/multilook becomes a *narrow* tool for the genuinely
   irreducible regime - noisy AND steep enough that the noise-suppressing
   multilook you would need itself aliases. Worth measuring how large that
   regime actually is on real data (in a volcano near-field the steep pixels are
   often the *coherent* ones, so the noisy∩steep overlap may be small).
3. Validate on one real dense-fringe scene (a known eruption or earthquake pair)
   before trusting any of this; that is the missing piece that would settle
   whether the pyramid is needed or a better base solver makes it redundant.
   `scripts/cost_model_real_ab.py` runs the linear/reuse/convex A/B on a
   downloaded GUNW `.h5` (the sandbox can't fetch one: Earthdata/ASF are
   network-blocked and there are no credentials).
4. If the auto-probe is pursued, combine the Itoh-violation rate with a spectral
   (FFT-peak) fringe-rate estimate to cover the constant-ramp blind spot, and
   calibrate `FLOOR`/`DECR` on real scenes.

## Regression coverage

* `crates/whirlwind-core/src/pyramid.rs` unit tests: bilinear upsample, factor
  schedule, `reuse_solver_fixes_clean_bowl_corners` (the corner bug - reuse >
  linear + 5 pp), `pyramid_recovers_clean_dense_cone`,
  `pyramid_coarsest_must_be_unaliased` (the `base·g < π` wall),
  `itoh_probe_separates_noise_from_aliasing`, `auto_base_recovers_clean_bowl`,
  and `tiled_finest_level_matches_untiled`.
* `python/tests/test_unwrap.py::TestPyramid`:
  `test_recovers_steep_bowl_that_multilook_destroys` (pyramid > 75 % where `ml8`
  < 30 %), `test_beats_fullres_in_heavy_noise`,
  `test_reuse_solver_fixes_clean_bowl_corners`, `test_auto_base_factor`,
  `test_tiled_finest_level_matches_untiled`, `test_unknown_solver_raises`, and
  `test_base_factor_one_linear_matches_plain_unwrap`.
