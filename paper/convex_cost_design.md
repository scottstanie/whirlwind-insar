# SNAPHU-style convex cost: prototype design

Companion to `paper/phass_experiments.md`. The reuse follow-up
(2026-05-28 part 2) lands whirlwind at PV 99.75 % / NISAR 92.70 %
K-match vs SNAPHU at α=0; the path-dependence probe (three Dijkstra
backends produce 0.005 %-identical answers) rules out solver-order
tie-breaking as the cause of the residual NISAR error. That leaves
the cost model itself as the missing piece.

This doc plans the **convex-cost prototype** — the substantive next
lane. Goal: get NISAR α=0 to within a few pp of dolphin PHASS
(97.93 %) without changing the solver architecture beyond what
convex MCF strictly requires.

## What SNAPHU's smooth cost actually is

For an arc between adjacent residue nodes with integer flow `k`:

```
c_e(k) = (k · nshortcycle − offset_e)² / σ²_e
```

* `nshortcycle` is a fixed integer (SNAPHU uses 100, scaling phase
  cycles to integer units). For us: keep as 1 since flow is already
  in integer cycles.
* `offset_e` is the per-arc *preferred* integer flow. **CORRECTION
  (2026-05-28, verified against SNAPHU source):** SNAPHU computes
  `offset = nshortcycle · (dpsi − avgdpsi)` — the DEVIATION of the raw
  wrapped gradient from its local box-mean — NOT `avgdpsi` alone
  (`snaphu_cost.c:1115-1116`; `dpsi` in cycles from `snaphu_util.c:149`).
  This deviation spikes toward ±1 cycle at an isolated wrap line (raw ≈ ±π
  while the box-mean ≈ 0) yet is ≈0 in smooth regions — the wrap-line
  routing signal the smoothed gradient lacks. The absolute (ramp-scale)
  flow comes from SNAPHU's coarse `unwrappedest` shift
  (`snaphu_cost.c:1127-1132`); whirlwind's analog is the anchor/cascade, so
  the deviation cost belongs in the per-tile solve, not whole-image. The
  original "offset = avgdpsi" here was the implementation's bug (Suspect 5).
* `σ²_e` is the per-arc noise variance. SNAPHU uses Lee 1994 PDF
  conditioned on coherence; we can do the same via the existing
  `cost::lee_pdf` LUT.

So the cost is a parabola in `k`, minimum at `k = offset_e`, opening
at rate `1/σ²_e`.

## Why it should help the residual NISAR island

The current Carballo cost is linear in `|k|` with `k=0` preferred —
*all* arcs prefer no flow. The 958k-pixel `-3 cycle` blob has a
self-consistent solution paying `3 × (linear cost) × 958k` more
than SNAPHU's solution. Under linear cost the routing finds a
*local* optimum; the global "3-cycle-off-here, 0 elsewhere"
solution survives because each individual arc pays a modest premium.

Under quadratic cost, the same 3-cycle deviation pays `9 × (cost) ×
958k` — an order of magnitude more, and crucially, *splitting* the
3-cycle blob into three separate 1-cycle blobs costs only
`3 × 1² × 958k = 3×`. Quadratic curvature makes large coherent
errors structurally expensive in a way linear cost cannot.

The `offset_e` term is the second half of the same idea: it tells
each arc what its flow *should* be based on the smoothed gradient
geometry, so the cost minimum follows the actual wrap-line topology
rather than always preferring `k = 0`.

## Implementation plan (phased)

### Phase 1: the cost path

Add `cost::compute_snaphu_smooth_costs(igram, corr, nlooks, mask)`
returning `(offsets: Vec<i32>, weights: Vec<i32>)` of length
`num_forward`. Implementation mirrors the existing
`compute_carballo_costs`:

* Same 7×7 mask-aware smoothed phase gradient via
  `smooth_phase_gradients_with_mask`.
* Per-arc offset: `round(α_smooth / (2π))` rounded to integer.
  For typical IG arcs `|α_smooth| < π`, so offset ∈ {-1, 0, 1}.
* Per-arc weight: `1 / σ²_e` where `σ²_e` comes from the Lee 1994
  variance at `γ = (γ_a + γ_b) / 2` and the given `nlooks`. The
  existing `cost::lut::get_or_build` already returns the PDF; we
  need a moment integral over it for the variance. Simpler
  approximation: `σ² ≈ (1 − γ²) / (2L γ²)` (Just/Bamler 1994 small-
  angle approximation). Calibrate against full numerical variance.

Output the two `Vec<i32>` arrays scaled by `COST_SCALE = 100` so
they stay integer and comparable to Carballo magnitudes.

### Phase 2: Network state

Add to `Network`:
* `pub offsets: Vec<i32>` — length `num_forward`. Zero when not in
  convex mode.
* `pub weights: Vec<i32>` — length `num_forward`.
* `pub convex_mode: bool`.

`Network::new_convex_with_mask(...)` constructs in convex mode with
the offsets/weights filled in. Existing `new_with_mask` stays
linear-mode (convex_mode = false).

### Phase 3: marginal cost in Dial

Add `Network::marginal_cost(arc) -> i32`:
```
fwd = canonical-direction arc index
sign = +1 if arc is forward, -1 if reverse
f_after = flow_count[fwd] + sign  // flow after pushing one unit
f_before = flow_count[fwd]
return weights[fwd] * ((f_after - offsets[fwd])² - (f_before - offsets[fwd])²)
       = weights[fwd] * (2 * sign * (f_before - offsets[fwd]) + 1)
```

Negative when pushing flow *toward* offset; positive otherwise.

In dial.rs relax sites: if `net.convex_mode`, use marginal_cost
instead of arc_cost. (Similar branch pattern to the existing
`is_used` check.)

### Phase 4: Bellman-Ford pre-pass for initial potentials

Convex marginal costs can be negative. Standard SSP requires
non-negative reduced costs. Initial potentials must absorb the
negativity.

SPFA (Bellman-Ford with queue) on the residual graph from a virtual
super-source connected to every excess node at cost 0. Sets
`net.potential[v] = −dist[v]` so initial reduced costs are ≥ 0.

After each primal-dual iteration: the standard `π[v] −= dist[v]`
update keeps reduced costs ≥ 0 *provided* the marginal cost on each
arc didn't change in a way the update can't absorb. Each
augmentation changes one arc's flow by ±1, which shifts that arc's
marginal cost by `2·weight`. The potential update can absorb this
as long as the augmenting path's reduced cost was already 0 on that
arc (tight, which it is after Dijkstra). Net effect: convex MCF
runs as standard SSP after the Bellman-Ford warm-up.

### Phase 5: integrate

`integrate::integrate_with_mask` already reads signed flow via
`net.arc_flow(g, arc)`, which returns the right value in any mode.
No change needed.

### Phase 6: top-level + Python

Add `pub fn unwrap_convex(...)` mirroring `unwrap_reuse`. Python
binding `whirlwind.unwrap_convex(...)`. Companion experiment script
`scripts/phass_experiments/run_convex.py`.

## Decision gates

* **End of Phase 3**: synthetic test passes (the
  `diagonal_ramp_512_reuse` analog under convex). If the synthetic
  fails, the algorithm is wrong before we try real data.
* **End of Phase 6**: PV K-match ≥ 99 %, NISAR K-match ≥ 95 %. If
  PV regresses or NISAR doesn't improve, the convex cost is
  computed wrong (offset polarity, σ² calibration) and we debug
  before pushing further.

## First-run results (2026-05-28 evening)

Phase 5+6 implementation completed. `unwrap_convex` in Rust and Python;
`scripts/phass_experiments/run_convex.py` for reproduction.

The end-to-end `diagonal_ramp_512_convex` test passes (max error 0.0
rad) — algorithm is sound on synthetic.

But on real scenes, the picture is split:

| scene | mode | wall | K=match | `|dK|`=1 | `|dK|`≥2 |
|---|---|---:|---:|---:|---:|
| PV    | baseline (unit-cap)  | 0.7 s | 90.67 % | 1.09 % |  8.25 % |
| PV    | reuse                | 3.7 s | 99.75 % | 0.25 % |  0.00 % |
| PV    | **convex**           | 14.6 s | **99.68 %** | 0.32 % | **0.00 %** |
| NISAR | baseline (unit-cap)  |  75 s | 80.01 % | 1.71 % | 18.28 % |
| NISAR | reuse                |  93 s | 92.70 % | 0.24 % |  7.06 % |
| NISAR | **convex**           | 402 s | **68.55 %** | 5.97 % | **25.48 %** |

PV: convex matches reuse essentially perfectly. The diagonal_ramp test
passes. The cost machinery and solver are correct.

NISAR: convex regresses past the unit-cap baseline. The +3 cycle blob
that reuse partly fixed grows back, and a new error mode appears. So
the convex cost as I've implemented it is *not* what SNAPHU produces
on the same data — somewhere between cost formulation and solver
interaction, the prototype solves a different problem.

Plausible suspects (in order of cheapness to test):
* ~~**σ² calibration.**~~ Tested — see below; ruled out.
* **Offset polarity.** The Carballo per-direction split (DOWN gets
  `+α`, UP gets `−α`) translates to opposite signed offsets on the
  two arcs of one pixel edge. The math checks out on paper but
  empirically: try the opposite convention and see if NISAR moves.
* **Whole-image vs tiled.** SNAPHU runs 9×9 tiles + a separate
  stitching pass. Whole-image convex MCF can lock into a different
  global optimum on a large scene.
* **Negative reduced costs without Bellman-Ford.** The analysis in
  Phase 4 said the standard SSP potential update keeps reduced
  costs ≥ 0 even in convex mode, but the proof assumes properties
  that may not hold across iterations. The release-build `dial.rs`
  `debug_assert!(rc >= 0)` is stripped; negative reduced costs would
  silently corrupt the Dijkstra result.

### Suspect 2 (offset polarity): ruled out, but exposes the real bug

Added `WHIRLWIND_CONVEX_OFFSET_FLIP=1` env-gated toggle that negates
every per-arc offset. Re-ran NISAR. **68.80 % → 68.78 %**. The blob
sizes and bounding boxes are identical to the third decimal. So
offset polarity isn't the cause either.

But running this diagnostic forced us to look at the *distribution*
of NISAR offsets, which revealed the actual problem:

```
NISAR offsets (= round(α_smooth · 100 / 2π), bounded ±50):
  |offset|=0:    38 %
  |offset|=1-5:  54 %
  |offset|>10:    0.4 %
  |offset|>25:    0.00 %
  max |offset|:  22
```

Maximum offset on NISAR is **22**, far below the saturation point at
±50 where wrap-line guidance kicks in (the cost-symmetric point
between k=0 and k=±1). Most arcs have |offset|≤5, which is too small
to matter — at |offset|=5 the cost ratio between k=0 and k=±1 is
still 361×.

Root cause: the **7×7 box smoothing** that the offset reads from
washes wrap lines out. A wrap line is a sharp ±π discontinuity over
~1 pixel; box-averaged across 7 pixels (mixing both sides of the
wrap) the smoothed gradient collapses to ~0. We measured max
|α_smooth| ≈ 1.35 rad on NISAR — well below the theoretical π that a
true wrap-line arc should produce.

So convex cost on NISAR is *effectively* `w · k² · 10000` everywhere
(pure quadratic, no offset structure). That strongly resists multi-
cycle deviations, but with no offset signal to use as routing
guidance, it just makes flow more expensive everywhere — worse than
linear. **Suspects 2 alone is not the cause; the *preprocessing* that
feeds the offset computation is.**

Plot: `plots/nisar_convex_panel.png` shows the convex Δ K panel
dominated by a 2.5M-pixel +4 cycle blob in the upper-right, even
worse than baseline's 1.6M-pixel blob in the same region.

### Suspect 5 (new): offset smoothing kernel

What we'd actually want for the offset: a wrap-line *indicator*, not
a smoothed gradient. Candidates to try:

1. **Smaller smoothing kernel** (3×3 instead of 7×7). Less averaging
   across wrap-line discontinuities; the offset would track the local
   wrapped gradient more closely. Cheapest test.
2. **Raw (unsmoothed) gradient.** Maximally preserves wrap-line
   discontinuities but is per-arc noisy. Equivalent to PHASS's
   `phase_diff_th` cut input. Useful as a sanity check; we expect
   the noisy offsets to make things worse, but a few percent better
   K-match would confirm the diagnosis.
3. **Median-of-wrapped-gradients** within a window. Robust to the
   discontinuity (the median across a wrap line picks one side or
   the other, not the average).
4. **Replace box-smoothing with a wrap-aware filter** that detects
   the discontinuity and adjusts. This is PHASS's amplitude-Canny
   path, but driven from phase instead of intensity.

(1) is the obvious cheap first try.

### Suspect 1 (σ²): ruled out

Replaced `just_bamler_variance` with `lut::get_or_build_variance`,
which numerically integrates the full Lee 1994 PDF
`σ² = ∫_{-π}^{π} α² · p(α|γ,L) dα` over a 1024-point γ grid and
returns interpolated values per arc. 1024-sample mid-point rule per
γ; 256k PDF evaluations one-time per `nlooks`. Same `compute_snaphu_smooth_costs`
otherwise.

| scene | mode | wall | K=match |
|---|---|---:|---:|
| `diagonal_ramp_512_convex` | passes (0.97 s, was 91 s) | — | max err 0.0 rad |
| PV    | convex, Lee var |  8.8 s | 99.59 % |
| NISAR | convex, Just/Bamler | 402 s | 68.55 % |
| NISAR | convex, Lee var     | 409 s | **68.80 %** |

NISAR moved 0.25 pp — well inside numerical noise. σ² calibration is
*not* the cause of the regression. Suspect 1 ruled out.

(Side effect: synthetic test now runs 90× faster because the Lee
variance gives more reasonable weights than the Just/Bamler small-
angle approximation does at low γ. The numerical correctness was
unchanged but the bucket-queue / heap workload depended on weight
magnitudes.)

Three suspects remain (offset polarity, whole-image vs tiled, and
the no-Bellman-Ford assumption from Phase 4). Holding the convex
prototype here per the original decision to pause + diagnose.

## Resolution (2026-05-28 evening): offset fixed + solver made sound — convex is NOT a win

All three remaining suspects addressed:
* **Offset (Suspect 5):** switched to SNAPHU's true `nshortcycle·(dpsi −
  avgdpsi)` deviation (was `avgdpsi` alone). Offsets now reach ±~100 and
  carry wrap-line signal. Unit test `deviation_offset_zero_on_ramp_nonzero_at_feature`.
* **Solver soundness (no-Bellman-Ford assumption):** replaced by
  `Network::preload_convex_min` — pre-load each arc to `k* = round(offset/100)`,
  adjust excess; at `k*` all residual marginals are ≥0 so zero potentials are
  valid and Dijkstra/heap stays sound (textbook ordered-parallel-arc convex MCF,
  no negative cycles, no Bellman-Ford). Soundness test
  `preload_makes_all_marginals_nonnegative`. The old `unwrap_convex` solve was
  silently corrupt in release (negative undo-arc marginals after a push, with
  `debug_assert!(rc>=0)` stripped) — that is now fixed.

**Empirical verdict — the convex cost is correct + sound but NOT the win:**
| scene / mode | linear | convex (fixed) |
|---|---|---|
| Atlanta 5× whole-image | 11.4% | 11.0% (no help) |
| Atlanta 5× tiled256+anchor | 47.3% | 51.7% (+4.4%) |
| NISAR tiled512+anchor (mainland) | 99.81% | 99.54% (REGRESSION), 91s vs 4s |

Convex helps Atlanta modestly (well short of snaphu's 97.9% / multilook's 97.7%)
and REGRESSES NISAR while being ~20× slower. It does NOT fix the col-4032
spurious sliver (present under both costs → a residue-pairing tie-break, not a
cost-shape issue). **Conclusion: keep convex as a correct, sound, opt-in lane
(`unwrap_convex`, `WHIRLWIND_TILE_CONVEX=1`), NOT the default. The dominant lever
for noisy scenes is coherent multilooking (noise/residue suppression), not the
per-arc cost shape.** This also retires the "linear cost can't route noise"
framing: the cost is ~4% of Atlanta's gap.

## Out of scope (for this prototype)

* **Combining convex + reuse.** Convex cost should subsume reuse
  semantics — multi-unit flow is naturally allowed by the
  marginal-cost formulation. Reuse mode stays as the cheaper
  no-Goldstein default; convex is the higher-quality lane.
* **Klein cycle-cancellation.** Hope to avoid by virtue of the
  Phase 4 reasoning above. If we hit negative reduced costs in
  practice we'll revisit.
* **Parallel-arc decomposition.** A cleaner formulation that
  expresses convex MCF as N-parallel-arc linear MCF. More
  arcs = more memory, but more standard. Defer unless the direct
  marginal-cost approach breaks down.
