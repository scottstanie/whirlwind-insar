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
* **A pyramidal coarse-to-fine unwrap is a genuinely better default.** Refining
  by powers of two and unwrapping only the *residual* against the upsampled
  coarser solution (the previous level's `K` as a prior) recovers full
  resolution without the single big jump. With the automatic base and reuse
  solver below, `pyrA` is the best or tied-best method across almost the entire
  dense-fringe / noise sweep — it never drowns in noise (unlike full-res) and
  almost never aliases (unlike `ml*`). The one exception is a near-Nyquist
  *constant-rate ramp* (`g ≳ 0.8π`), the auto-probe's documented blind spot,
  where you should pass `base_factor=1` explicitly.
* **The "corners" worry was a real bug — in the base solver, not the pyramid.**
  The linear Carballo coherence cost ([`unwrap`]) mis-routes the CORNERS of a
  smooth steep signal (the steepest part of a bowl) because the concentric
  wrap-line rings must drain at the image boundary and the capacity-1 frame arcs
  can't carry the stacked flow (same pathology as the ignored `diagonal_ramp_512`
  regression). On a *perfectly clean* `0.7π` bowl `unwrap` scores only **88 %**,
  all errors in the corners, while `unwrap_reuse` and `unwrap_convex` score
  **100 %**. (Noise dithers the ring alignment, so the pathology is worst on
  clean/synthetic data.) The pyramid therefore defaults its per-level solver to
  `"reuse"`.
* **`base_factor` is chosen automatically.** `base_factor=0` (the default) runs
  an Itoh-violation-rate probe and picks the largest down-look that has not yet
  started aliasing the steepest fringe present.
* **It is still not magic.** Nothing recovers a signal genuinely aliased at full
  resolution (`g > π`, a hard Nyquist wall), and the auto-probe has a
  constant-ramp blind spot (below).
* **Large frames are covered.** `tile_size>0` tiles the finest levels to bound
  peak memory; the coarsest (absolute) level reuses the anchored tiled path, the
  residual (relative) levels use a lightweight prediction-relative tiler.

## Why multilook-first aliases

Coherently down-looking the complex igram by `L` averages `L×L` blocks. The
coarse grid's pixel spacing is `L` original pixels, so a full-res phase gradient
`g` rad/pixel becomes `L·g` rad per *coarse* pixel. Phase unwrapping rests on
Itoh's assumption that adjacent samples differ by less than half a cycle
(`|Δφ| < π`); above that the wrapped gradient `Δψ = wrap(Δφ)` folds to the wrong
branch and the solver confidently integrates too few cycles.

So multilook-first is safe only while `L·g < π`, i.e. `g < π/L`. For `L = 8`
that is `g < π/8 ≈ 0.39 rad/pixel` — eight times *stricter* than the true
full-resolution Nyquist limit `g < π`. Worse, block-replicating the coarse
solution bakes the wrong `K` in permanently: no later stage can add the cycles
the coarse grid threw away. This is invisible on the broad, gently-sloped scenes
multilook-first was tuned for and catastrophic on a dense-fringe deformation
sharing the same frame.

## The pyramidal scheme

Classic multigrid / multi-resolution unwrapping. Build a schedule
`base, base/2, …, 1` and refine coarse→fine:

1. **Coarsest level** (factor `base`): an ordinary whole-image unwrap of the
   `base×` down-looked igram. This level alone must be unaliased (`base·g < π`).
2. **Each finer level** `f`: bilinearly upsample the previous (coarser) level's
   *unwrapped* phase to this grid → `pred`. Rotate this level's complex igram by
   `exp(−i·pred)`, so its phase becomes `wrap(angle − pred)` — the **residual**
   wrapped phase — while its magnitude (hence coherence) is untouched. Because
   `pred` already carries the large-scale gradient, the residual gradient is
   small (well under π), so a plain unwrap solves it without aliasing. The
   level's phase is `pred + unwrap(residual)`.

`pred` is exactly the requested "previous solved K as a prior": per pixel,
`round((pred − angle)/2π)` is the integer cycle the coarse solve believes this
pixel sits in, and the residual unwrap only corrects deviations from it.
Refining all the way to `f = 1` always returns a *full-resolution* surface —
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

The probe measures the **Itoh-violation rate** of each `f×` down-looked igram:
the fraction of adjacent coarse-pixel wrapped phase differences whose magnitude
exceeds `0.6π`. This directly tests the aliasing (Nyquist) condition. An
*absolute* threshold does not work, because phase NOISE alone pushes the rate
high (≈0.25–0.35 at 4 looks / γ≈0.2) without any aliasing. The discriminator is
the *direction of change* with `f`:

* **Noise** makes the rate *fall or hold* as `f` grows — coherent averaging
  suppresses it (noisy `0.2π` cone: `f=1 → 0.33`, `f=2 → 0.26`).
* **Aliasing** makes it *jump* once `f·g > π` (that same cone: `f=4 → 0.42`; a
  clean `0.6π` bowl: `f=1 → 0.0`, `f=2 → 0.28`).

So `auto_base_factor` walks `1, 2, 4, …` and keeps doubling while the next level
either sits below a benign noise `FLOOR = 0.05` *or* the rate *meaningfully
decreases* (drops by ≥ `DECR = 0.02` — coherent averaging still suppressing
noise on an unaliased grid). It stops the first time the rate holds flat or
rises (the aliasing fold) and returns the factor before it. Both conditions are
needed: an absolute threshold alone never downsamples noisy data (noise keeps
the rate high); a decrease rule alone never downsamples clean gentle data (the
rate is already ≈0). This fixed two failure modes found while iterating — an
initial residue-density probe over-downsampled *clean* steep bowls (too few
residues to trip any floor even when aliased), and an absolute-threshold Itoh
probe over-downsampled *very noisy* mild signals (the aliasing jump was buried
under the noise floor).

**Limitation — the constant-ramp blind spot.** The probe (like any local
gradient/curl measure) detects aliasing through the wrapped jumps it creates,
which appear where the gradient *varies* (every real localized signal: bowls,
point sources, faults). A near-constant-rate ramp aliases *coherently* —
adjacent aliased pixels' wrapped gradient folds back small — so the probe cannot
cleanly see the aliasing onset. For mild-to-moderate constant rates it still
keeps base = 1 (the f=1 rate is low and the f=2 jump is visible), and the reuse
solver handles the unaliased full-res signal — `pyrA` is 100 % on the clean cone
up to `g = 0.7π`. But at the very steepest rates (`g ≳ 0.8π`) the f=1 rate is
*already* high from the steep signal itself, the trend is ambiguous, and the
probe over-downsamples and aliases (cone `g = 0.9π`: `pyrA` = 1 %). This is a
fundamental ambiguity (an aliased ramp is indistinguishable from a gentle ramp
without external information), not a tuning bug — pass an explicit
`base_factor=1` for a scene you know is a near-Nyquist constant-rate ramp.

### Tiling the finest levels (memory)

The MCF graph is `m×n` nodes regardless of residue count, so the finest level is
the memory bottleneck on a large frame. With `tile_size>0`, the **coarsest**
level (absolute phase, no prediction) reuses `unwrap_tiled`, whose global coarse
anchor is exactly right for absolute phase. The **residual** levels are
*relative* to a global prediction, so the anchored tiled path is actively
harmful there (it region-votes a near-flat field into garbage). Instead they use
a lightweight tiler: solve each overlapping tile independently (trivial — the
residual is small-gradient), gauge each to a common cycle by removing its
rounded-2π median, and feather-composite. Because every tile's residual is
referenced to the *same* prediction, no inter-tile 2π reconciliation is needed.
Tiled K matches untiled to within seam noise.

## Synthetic dense-fringe results

`scripts/dense_fringe_pyramid.py` (deterministic, seed 0, 384² grid) builds two
truths — a constant-rate **cone** (`φ = g·r`) and a steep **bowl** / paraboloid
(`φ = a·r²`, edge rate `g_edge`) — simulates Goodman-noise igrams, and reports
K-correct fraction (per-pixel integer-cycle agreement with truth) for `full`,
`ml4`, `ml8`, `pyr2`, `pyr4`, and `pyrA` (auto `base_factor`). The pyramid uses
its default reuse base solver.

**Fringe-rate sweep, cone, γ = 0.95 (8 looks).** Single-shot multilook collapses
at its thresholds (`ml4` at `g > π/4`, `ml8` at `g > π/8`); fixed-base pyramids
alias once `base·g > π`. `pyrA` is 100 % up to `g = 0.7π`: the Itoh probe sees
the f=2 violation jump on the (unaliased) clean cone and *keeps* base = 1, where
the corner-safe reuse solver already nails the full-resolution cone. At the
extreme `g = 0.9π` `pyrA` collapses to 1 % — this is the constant-ramp blind
spot (a cone is a near-constant radial ramp): so near the Nyquist wall the f=1
violation rate is itself already high from the steep signal, the probe
mis-reads the trend and over-downsamples. `full` (87→83 %) is the only method
that degrades gracefully at the very steepest rates; set `base_factor=1`
explicitly for a scene you know is a near-Nyquist ramp.

| g (×π) | full | ml4 | ml8 | pyr2 | pyr4 | pyrA |
|-------:|-----:|----:|----:|-----:|-----:|-----:|
| 0.1 | 98 | 98 | 97 | 100 | 100 | 100 |
| 0.2 | 94 | 91 | 6 | 100 | 100 | 100 |
| 0.3 | 91 | 3 | 5 | 100 | 7 | 100 |
| 0.5 | 89 | 2 | 2 | 98 | 4 | 100 |
| 0.7 | 87 | 3 | 2 | 1 | 14 | 100 |
| 0.9 | 83 | 1 | 1 | 1 | 2 | 1 |

**Fringe-rate sweep, bowl, γ = 0.95 (8 looks).** The varying-gradient case (a
volcano bowl). `pyrA` is 100 % across the whole range — the probe downsamples
where the corner annulus is still unaliased and backs off where it isn't —
beating the fixed bases `pyr2`/`pyr4`, which degrade once their coarsest level
aliases at the steep edge.

| g_edge (×π) | full | ml4 | ml8 | pyr2 | pyr4 | pyrA |
|------------:|-----:|----:|----:|-----:|-----:|-----:|
| 0.2 | 95 | 94 | 83 | 100 | 100 | 100 |
| 0.3 | 89 | 88 | 6 | 100 | 100 | 100 |
| 0.4 | 89 | 72 | 5 | 100 | 96 | 100 |
| 0.5 | 88 | 50 | 4 | 100 | 66 | 100 |
| 0.6 | 88 | 3 | 5 | 100 | 15 | 100 |
| 0.8 | 87 | 2 | 3 | 80 | 8 | 100 |
| 0.9 | 87 | 2 | 2 | 64 | 11 | 100 |

**Noise sweep, mild rate g = 0.2π, 4 looks, falling coherence.** The regime that
justifies multilooking: as γ falls the full-res solve drowns in noise residues,
but the coarse grids stay unaliased. `pyrA` matches the best fixed base at every
γ — the Itoh probe reads the noise *falling* with `f` and downsamples as far as
it safely can.

| γ | full | ml4 | ml8 | pyr2 | pyr4 | pyrA |
|-----:|-----:|----:|----:|-----:|-----:|-----:|
| 0.50 | 90 | 92 | 6 | 100 | 100 | 100 |
| 0.35 | 85 | 91 | 6 | 98 | 98 | 98 |
| 0.30 | 76 | 90 | 6 | 97 | 97 | 97 |
| 0.25 | 69 | 90 | 6 | 95 | 95 | 95 |
| 0.20 | 39 | 82 | 6 | 93 | 93 | 93 |

Across all three sweeps `pyrA` is the best or tied-best method except at the
near-Nyquist constant-ramp blind spot (cone `g = 0.9π`, above). `ml8` is stuck
near 6 % whenever `8·g > π`, regardless of how clean the data is. Figures
(`curves_cone.png`, `curves_bowl.png`, `curves_noise.png`,
`panels_steep_bowl.png`, `panels_corner_solver.png`) are written to `--out`.

## Recommendation

* Keep the single-shot `multilook` path for what it is good at (cheap noise
  suppression on gently-sloped scenes), but **do not apply a large `multilook`
  blindly** to scenes that may contain dense fringes.
* Prefer `unwrap_pyramid` (default `base_factor=0` auto, `solver="reuse"`) as the
  safer noise-suppressing path: across the synthetic sweep `pyrA` is the best or
  tied-best method almost everywhere. Override with `base_factor=1` for a scene
  you suspect is a near-Nyquist constant-rate ramp (`g ≳ 0.8π`), the probe's
  documented blind spot.
* Set `tile_size>0` on large frames to bound peak memory.
* Future work: combine the Itoh probe with a spectral (FFT-peak) fringe-rate
  estimate to cover the constant-ramp blind spot, and calibrate `FLOOR`/`DECR` /
  the reuse-vs-convex base-solver choice on real (not just synthetic) scenes.

## Regression coverage

* `crates/whirlwind-core/src/pyramid.rs` unit tests: bilinear upsample, factor
  schedule, `reuse_solver_fixes_clean_bowl_corners` (the corner bug — reuse >
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
