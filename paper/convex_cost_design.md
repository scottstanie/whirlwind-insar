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
* `offset_e` is the per-arc *preferred* integer flow. SNAPHU
  computes `offset = round(avgdpsi · nshortcycle / 2π)` where
  `avgdpsi` is the box-smoothed wrapped phase gradient. The arc
  actively wants its flow to equal `offset`.
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
* **σ² calibration.** I used the Just/Bamler small-angle
  approximation `(1 − γ²) / (2L γ²)`. SNAPHU uses the full Lee 1994
  variance. For low coh / few looks the two diverge noticeably.
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

Hold the convex prototype here for now — the PV result and the
synthetic test confirm the implementation works in principle, but the
NISAR regression means convex-as-implemented is not the answer.
Diagnosis on which of the four suspects above is the cause is the
next step before deciding to push convex further.

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
