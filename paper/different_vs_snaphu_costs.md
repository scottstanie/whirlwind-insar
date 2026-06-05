# Why whirlwind α=0 doesn't match SNAPHU

A self-contained internal note. The TL;DR for the paper: **whirlwind's
per-arc cost is linear-in-flow with unit capacity; SNAPHU's smooth-mode
per-arc cost is convex (quadratic) in flow with an arc-specific preferred
non-zero offset.** That difference is structural, not a magnitude-tuning
issue, and it explains why a "magnitude-matched linear" port of SNAPHU's
smooth cost into whirlwind makes things *worse*.

## The two costs side by side

For an arc connecting adjacent residue nodes, with per-arc coherence
`γ_edge` (= min of endpoint coherences), arc flow `k ∈ ℤ`, and locally
windowed wrapped gradient `α_smooth`:

|                               | whirlwind (Carballo)                               | SNAPHU smooth                                |
| ----------------------------- | -------------------------------------------------- | -------------------------------------------- |
| Cost shape                    | `γ_edge · max(0, π − sign(k)·α_smooth)` per unit ` | k                                            | `                                                        | `(k·nshortcycle − offset_smooth)² / σ²_edge`            |
| In flow `k`                   | **Linear** per unit                                | **Quadratic**, with non-zero minimiser       |
| Arc capacity                  | **1** (unit)                                       | Many cycles (network sized to allow it)      |
| Coherence weighting           | γ, raw                                             | `(1−ρ)^rhopow` (rhopow ≈ 8.4 at L=100)       |
| Phase-gradient input          | `α_smooth` (7x7 box of wrapped gradient)           | `offset = round(avgdpsi · nshortcycle / 2π)` |
| Role of the smoothed gradient | **Cost modifier**: arcs in regions with `          | α_smooth                                     | ≈ π` get low cost - "this region looks like a wrap line" | **Cost offset**: the arc actively wants `flow ≈ offset` |
| Solver                        | SSP / primal-dual Dijkstra, integer cost           | MCM-flavoured convex MCF                     |

## Why a "use SNAPHU's exact numbers" port fails in our solver

Two qualitative properties of the SNAPHU cost are *unrepresentable* by a
single integer cost-per-unit-flow:

1. **Curvature.** A quadratic cost penalises 2 cycles 4x more than 1.
   A linear cost only penalises it 2x. So with linear cost, MCF
   doesn't strongly resist routing multi-cycle paths through cheap
   noise channels.
2. **Per-arc preferred non-zero offset.** SNAPHU's quadratic has its
   minimum at `flow = offset`. The arc actively *wants* a particular
   non-zero flow set by the local windowed gradient. A linear cost can
   only ever say "flow=0 is cheapest, with a per-unit deviation
   penalty" - it cannot pull `k` toward an offset value.

Worse: linearising the quadratic on the *per-unit-flow* axis (slope at
`k=0`, no offset) gives an arc cost proportional to `|offset| / σ²`.
With the proper `(1−ρ)^8.4` weighting and high coherence, `1/σ²` is
*very large* - so a high-coherence smooth arc, on which we want flow=0,
ends up with a *low* per-unit cost in the linearisation (because the
parabola is shallow near its minimum at `k=offset`). MCF then gleefully
routes flow through coherent smooth regions, which is the opposite of
the desired behaviour. **Match the numbers, lose the shape, get a
worse answer.** This is what happened in the first
`scripts/ww_cost_experiment.py` attempt.

## The "use deviation as cost input" patch - tested, also worse

Hypothesis: since whirlwind's `α_smooth` is a cost *modifier* but
SNAPHU's `avgdpsi` is a cost *offset*, maybe whirlwind should feed
the *per-arc deviation* `wrap(dpsi_arc − dpsi_smoothed_7x7)` into its
`(π − |α|)` formula. Single-arc noise outliers inside a coherent ramp
would then become locally cheap routing channels - emulating SNAPHU's
"this arc's preferred flow is non-zero" without changing the solver.

Result on the NISAR 9x9 reference scene at α=0 (no Goldstein):

| α=0 mode                    | wall | cc>0 coverage | K-agreement vs SNAPHU on cc=1 mainland |
| --------------------------- | ---: | ------------: | -------------------------------------: |
| baseline (`α_smooth` input) | 84 s |        3.47 % |                            **92.52 %** |
| deviation input             | 87 s |        1.71 % |                          **86.50 %** ↓ |

Why it loses: the substitution does succeed in making noise arcs cheap
to route through - but in a smooth coherent ramp with random per-arc
noise, *those noise arcs have no geometric link to true wrap-line
topology*. MCF routes 2π discontinuities through them and creates
K-flips in the wrong places.

Conclusion: **the 7x7 smoothing in the current cost is load-bearing,
not incidental**. It's the mechanism that confines cheap-routing to
regions that *consistently* look like wrap lines (multiple aligned
arcs). Replacing it with raw-minus-smoothed destroys that regional
preference.

(The toggle is preserved as `WHIRLWIND_DEVIATION_COST=1` for the
documented negative result - see `crates/whirlwind-core/src/cost/mod.rs`.)

## What about PHASS?

PHASS (`isce3/cxx/isce3/unwrap/phass`, where Geoff got the primal-dual
idea) uses a *different* trick: still linear cost, but with **hard
zero-cost cuts** in two places.

1. **Phase-gradient cuts**: any arc with `|wrap(Δphase_raw)| ≥ 1.0 rad`
   gets cost = 0. This is the equivalent of SNAPHU's "this arc has a
   non-zero preferred flow" - except instead of a quadratic that pulls
   `k` toward `offset`, PHASS just declares the arc free so MCF will
   use it preferentially.
2. **Amplitude edge cuts** (when amplitude data is provided): a Canny
   detector on backscatter finds geometric edges (coastline,
   layover); arcs straddling them get cost = 0. Big win on
   SWOT lakes; not directly relevant to our use cases.

The underlying cost is `min(γ_p, γ_q)² · 100`, saturated at
`good_corr² · 100`. Coherence-squared (no `(π − |α|)` term at all),
saturated in coherent regions so MCF doesn't agonise over the high-γ
plateau. Arc capacity is 4 (vs. our 1), but the algorithm is unit-flow
augmentation throughout, so capacity > 1 just caps pathological paths.

PHASS's hard cuts are the part that's interesting for closing the
no-filter gap in whirlwind: they're *single-arc* cheap routing
channels, not regional. They achieve what the deviation-cost
substitution attempted, but PHASS makes them work because the
coherence cost is **saturated** in the surrounding coherent area
(every other arc is at the 255 ceiling), so a single arc dropping to 0
is a uniquely cheap channel, not one of many.

## So is Goldstein really necessary?

**Practically, with the current cost: yes.** α = 0.7 gives 99.9 %
K-match with SNAPHU on the NISAR scene at 27x the speed. That's the
working answer in PR #19.

**Structurally: no.** Both SNAPHU and PHASS unwrap that scene without
any input filtering. The cost shape and routing channels do all the
work. To close the no-filter gap in whirlwind without copying SNAPHU's
solver, the practical paths are (in order of how much code they need):

- **Hard cuts at large `|wrap(dpsi_raw)|`** (PHASS-style). Smallest
  intervention - just modify the cost computation. Tested in
  [[phass-experiments]]; doesn't help in our solver.
- **Saturated coherence cost** (also PHASS-style). Replace the
  `(π − |α|)` modulation with a flat 255 ceiling above some γ. Tested
  in [[phass-experiments]]; doesn't help in our solver. (But the
  *actual* PHASS algorithm via `dolphin --unwrap-method PHASS` hits
  97.9 % K-match with SNAPHU at α=0 - so the idea works, our port of
  it doesn't.)
- **Iterative-recost SSP**: re-evaluate per-arc marginal cost
  `c(f+1) − c(f)` before each Dijkstra in `ssp.rs`. Lets the solver
  see a convex cost without changing topology. Closest to SNAPHU's
  MCM. Larger change; not tried here.
- **Goldberg parallel-arc convex reduction**: replace each arc with K
  parallel arcs of increasing marginal cost. Lets the existing solver
  express any monotone convex cost at Kx the arcs. Larger still.

## Reproduction

```bash
# Build editable install
cd /Users/staniewi/repos/whirlwind-insar
maturin develop --release

# NISAR α=0 baseline vs deviation cost
cd /tmp/ww-nisar
python test_deviation_cost.py baseline   # default Carballo cost
WHIRLWIND_DEVIATION_COST=1 \
  python test_deviation_cost.py deviation
python test_deviation_cost.py compare    # cross-tabulate vs SNAPHU 9x9
```

SNAPHU 9x9 reference TIFFs live next to the NISAR inputs at
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar/`. The whirlwind
α=0.5 and α=0.7 references are alongside (saved by earlier work).
