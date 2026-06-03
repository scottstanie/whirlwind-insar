# Why a whole-image MCF solve "runs away" (and snaphu single-tile doesn't)

Working note (2026-06), issue #65. Answers the question: *if whirlwind uses a
snaphu-style convex cost and a sound solver, why does a standalone whole-image
solve drift to a wrong large-scale winding ("run away") while snaphu's
single-tile solve does not?*

## The phenomenon (D_077, a steep-ramp NISAR GUNW frame, water-masked, nlooks=16)

Per-component match vs production, ww whole-image, by **center-crop size**:

| crop | ww convex | ww reuse (linear) | snaphu single-tile |
|---:|---:|---:|---:|
| 256 | 99.99% | 99.99% | 99.99% |
| 512 | 77% | 80% | 99.98% |
| 1024 | 32% | 52% | 99.98% |
| 2048 | 24% | 22% | 99.98% |
| full (~4200) | 2% | 1% | 99.30% |

Three facts fall out:

1. **It is scale-dependent.** At a 256-px crop ww = snaphu = 99.99%. The error
   grows monotonically with the solved domain.
2. **Convex is not the cure.** ww's linear (reuse) cost is as good or *better*
   than the convex cost at every size. Swapping ww's coherence→weight curve for
   snaphu's exact empirical `sigsqrho` moved the 512 crop by only +3 pts.
3. **The ww solver is sound.** Negative-cycle canceling (`cycle_cancel`) finds
   **zero** negative residual cycles in the converged flow at 512/1024 (the
   independent Bellman-Ford certificate in `tests/convex_solver_probe.rs`
   agrees). By the negative-cycle optimality theorem, ww's flow **is** the
   global minimum-cost flow of its cost. So the runaway is not a solver bug — it
   is the cost's own optimum.

## The mechanism

Phase unwrapping by MCF assigns an integer cycle count `k_e` to every pixel-edge
gradient. The cost is **separable per edge**: `Σ_e c_e(k_e)`, with
`c_e` increasing in `|k_e|` away from a per-edge preferred value (linear for
Carballo/PHASS, quadratic for snaphu-smooth). The unwrapped surface is the path
integral of `dpsi_e + k_e` — so the *large-scale winding* is the **accumulation**
of the per-edge `k_e` over a path.

The key asymmetry: **a per-edge cost can penalize the magnitude of each `k_e`,
but it cannot see the accumulated winding.** A "run-away" is not one edge with a
huge `k`; it is a vast, smoothly-varying field of *small* per-edge corrections
(mostly 0, with occasional ±1 along slip lines) that **integrate** to a wrong
large-scale ramp. Each edge's `k_e` is small, so:

- **Linear cost** charges `γ·|k_e|` — a coherent block of small corrections
  costs the same per unit as scattered ones; nothing makes the *coordinated*
  winding expensive. This is the classic linear run-away.
- **Convex cost** charges `w·(k_e·ns − offset_e)²` — its curvature only bites
  when a *single* edge carries large `k`. For a runaway built from `k_e ∈
  {0, ±1}`, the quadratic term never engages: `(±1·ns)²` per slip edge is the
  same whether those slips form the true winding or a wrong one. So convex
  curvature does **not**, on its own, forbid the accumulated runaway.

What *would* forbid it is an **absolute reference** that ties the winding to a
known large-scale field — exactly snaphu's optional `unwrappedest` offset shift
(`snaphu_cost.c:1127-1132`): `offset_e += (ns/2π)·(est[h] − est[t])`. That makes
each edge *prefer* the coarse field's expected integer flow, so the cost's
optimum tracks the reference instead of drifting. Whirlwind's analogue is the
coarse anchor / cascade — but `unwrap_convex` (the standalone whole-image path)
does not apply it.

## So why does snaphu single-tile reach 99% without an estimate file?

snaphu-py writes **no** `ESTFILE` (verified: `_unwrap.py:372-404`), so its smooth
cost is the same local-deviation parabola whirlwind uses — and the analysis above
says *that cost's global optimum can run away too*. Two non-exclusive reasons
snaphu still lands at 99% single-tile:

- **(b) Heuristic, anchored solver — the operative effect, and the cleanest analogy.**
  snaphu does not compute the global min-cost flow. It (i) initializes a feasible
  flow with a *linear* MCF/MST (`init=mcf`, CS2), then (ii) runs `TreeSolve` for a
  bounded number of flow increments (`nflow = 1..maxflow=4`). It stays *near* the
  coherence-following init and never explores far enough to discover the
  lower-cost runaway. This is precisely an **A\*-with-an-admissible-heuristic**
  vs **exhaustive-Dijkstra** distinction: snaphu's good init + bounded search is
  the heuristic that keeps it on the true winding; whirlwind's solver is "too
  good" — it finds the true global optimum, which is the runaway. (NISAR
  production additionally tiles + `SINGLETILEREOPTIMIZE`, both regularizers.)

- **(a) Possible residual cost difference — open.** Under a numpy replica of
  snaphu's *exact* smooth cost, the full-frame runaway scores ~12× more expensive
  than production, hinting snaphu's cost ranks the runaway as bad. But ww's
  offset is algebraically the same deviation `ns·(dpsi − avgdpsi)` and the weight
  A/B barely moved the result — so if a faithful-cost term matters it is *not*
  the weight (candidates left: the ρ-gated `0.5·avgdpsi` low-coherence branch,
  `nshortcycle` 100 vs 200, masked-edge handling). Settling (a) vs (b) cleanly
  needs ww's *exact* objective evaluated on matched surfaces (the dual flow↔phase
  bookkeeping in `integrate.rs`); not yet done. Operationally it does not change
  the fix.

## Why tiling works — and what it costs

Tiling is **spatial regularization**: inside a 256–512 px tile the cost's optimum
*is* the truth (top of the table), because there isn't enough domain for a
coordinated runaway to be cheaper. Stitching tiles then re-imposes a consistent
winding. This is why whirlwind's default (tiled + global coarse anchor + cascade)
avoids the run-away — but the stitching is where the **blocky / streaky / stripy
seam artifacts** (#61–#64) come from: they are the seams of the regularizer.

## The fix that follows

An **absolute anchor** kills the runaway directly. Verified end-to-end (pure
orchestration of `ww.unwrap`, `scripts/flatten_refine_d077.py`): flatten the
wrapped phase by an anchor `A`, unwrap the **residual whole-image** (no tiling →
no seams), add `A` back.

- Oracle anchor (`A` = production) → **100% match, seamless, instant** (the
  residual has near-zero flow). Proof the mechanism is sound: with a correct
  absolute reference, ww's whole-image solve does not run away.
- Tiled-solve anchor → reproduces the tiled quality (the hybrid is **anchor-
  limited**). So the lever is **anchor quality**, and the proven lever for that
  is **multilook** (coherent averaging suppresses noise → fewer residues → less
  to run away; see `atlanta_failure`), not the cost shape.

This is the user's own "fast tiled mostly-right → whole-image solve" hybrid: the
anchor is the admissible heuristic; the whole-image residual solve removes the
tiling seams.

## Metric caveat

The benchmark scores against production, and **NISAR production unwrapping is
snaphu** (`cost=smooth, init=mcf`), with the input being `exp(i·wrap(production))`.
So "snaphu vs production" is a near **self-match** (≈99% by construction), while
whirlwind is scored against snaphu's particular winding. On **synthetic
known-truth** ramps (γ 0.25–0.9, up to 40 cycles, ≤1024 px) snaphu ≡ ww-convex ≡
ww-reuse to within cycle-accuracy 0.99+. So part of the headline gap is the
metric, not unwrap correctness — quantifying ww's *intrinsic* correctness (vs
wrapped-data self-consistency, not vs snaphu) is a separate, worthwhile check.
