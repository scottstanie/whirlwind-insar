# Unwrapper comparison: whirlwind vs PHASS vs snaphu (internal)

Working notes (2026-06) to answer: *what is actually different* between whirlwind,
PHASS, and snaphu, so we can land a whirlwind that **mostly matches single-tile
snaphu on NISAR with no artifacts**. Based on a static survey of the source trees
(`~/repos/{isce3,libwhirlwind,whirlwind,whirlwind-insar,tophu}`) + the NISAR GUNW
benchmark (`paper/nisar_gunw_bench.md`).

## Lineage (one cost family, three implementations)

`~/repos/whirlwind` (original **Python**) → `~/repos/libwhirlwind` (**C++** header-only,
"work in progress, not for general use") → `~/repos/whirlwind-insar` (**Rust**, current).
All three use the **same Carballo linear coherence cost** and a primal-dual MCF solver;
the Rust version is a 3–72x faster rewrite (pixel-identical mod 2π to libwhirlwind) that
*added* tiling, the coarse-anchor/cascade, CRLB cost, solve-free conncomps, and the
flow-reuse + (prototype) convex modes. There is no `whirlwind-cpp` - that's `libwhirlwind`.

**Correction to the working recollection** ("original whirlwind was close to PHASS but
with a cost closer to snaphu"): half right at best. The *solver* is **not** PHASS's
quality-guided region-grow - it's classical MCF. The *cost* is **not** snaphu's convex
cost - it's the Carballo **linear** cost (coherence-weighted, phase-gradient-aware). The
only "snaphu-ish" thing is using coherence as a statistical weight; the cost *shape* has
always been linear, never convex. So whirlwind is its own point: **a phase-aware
linear-cost MCF**, distinct from both PHASS (pure-coherence cost) and snaphu (convex cost).

## The three, side by side

|               | **whirlwind** (Rust)                                                                                                                      | **PHASS** (isce3)                                                                                           | **snaphu**                                                         |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Cost          | Carballo **linear**: `γ·max(0, π∓α)` per unit, direction-aware; γ=min-endpoint coherence, α=7x7-smoothed phase gradient (**phase-aware**) | **pure coherence**: `min(coh_i,coh_j)`, no phase info                                                       | **convex/statistical (MAP)**: `w·(k − offset)²`, quadratic in flow |
| Marginal cost | **constant** (linear)                                                                                                                     | constant                                                                                                    | **increasing** (curvature)                                         |
| Solver        | primal-dual MCF + Dial bucket-queue; **flow-reuse** (uncapacitated, reused arcs free)                                                     | primal-dual MCF (ASSP) + **flow-reuse**, then region-grow                                                   | nonlinear network-flow (curvature)                                 |
| Whole-image   | **runs away** on noisy/steep (linear optimum ≠ truth) → needs tiling                                                                      | **runs away** too (same solver family)                                                                      | **single-tile is the quality ceiling**                             |
| Tiling        | per-tile MCF + global coarse anchor + multi-scale cascade + feathered composite + gated multi-shift                                       | whole-image (tiled at caller level, e.g. tophu)                                                             | tile + secondary tile-MCF merge                                    |
| Conncomps     | **solve-free** from cost grid; `min_size_px=100`; **~53 / frame**, recall ~94%                                                            | region-grow flood-fill; `good_correlation=0.7`, `min_region=200`; **hundreds–thousands**, gives up <0.7 coh | BFS post-solve; `min_region_size=300`                              |

## The crux: cost *shape*, not magnitude

This is the one structural difference that explains everything:

- **snaphu's cost is convex** (increasing marginal cost). Pushing one cycle is cheap;
  pushing a *block* of coherent cycles the wrong way gets progressively expensive. That
  curvature makes the **true unwrap the global optimum**, so a single whole-image solve is
  well-posed and is the quality ceiling. Tiling in snaphu is purely memory/speed and only
  *approximates* the single-tile answer.
- **whirlwind's (and PHASS's) cost is linear** (constant marginal cost). A coherent block
  of cycles routed the wrong way costs the same per unit as small corrections, so the
  whole-image optimum can sit far from the truth on noisy/steep data - the **"run-away"**.
  whirlwind's tiling + coarse-anchor + cascade is a **spatial regularizer** that bounds
  where the wrong optimum can act; it's compensating for the cost, not adding genuine
  value over a correct whole-image solve. The tile-block / checkerboard / vertical-streak
  artifacts (issues #61/#62) are the **seams of that compensation**.

whirlwind's Carballo cost is *better than PHASS's* (phase-aware vs pure-coherence → fewer,
larger conncomps and higher recall), but it is **not** convex, so it inherits the
run-away. `paper/different_vs_snaphu_costs.md` already argued this is structural, not a
tuning issue.

## Where whirlwind stands vs the goal

- **"Like PHASS, but gives up on fewer pixels and makes dozens not thousands of
  conncomps"** - *whirlwind already achieves this.* Phase-aware cost + solve-free conncomp
  + absolute 100-px floor give ~53 components and ~94% recall on NISAR, vs PHASS's
  hundreds–thousands and its hard `good_correlation=0.7` give-up. ✅
- **"Like snaphu, but faster / match single-tile snaphu with no artifacts"** - *not there.*
  This needs snaphu's **cost curvature** so the whole-image solve is well-posed (then the
  artifacts vanish because there's nothing to reconcile). whirlwind has a convex-cost
  prototype, but it's unsound/untuned whole-image (negative marginal costs from the offset
  break Dial's non-negative-reduced-cost assumption without a Bellman-Ford/SPFA pre-pass).

## Empirical backbone (NISAR GUNW, water-masked, nlooks=16, per-component match)

Decisive head-to-head on the two steep-ramp D-frames where ww fails:

| frame | ww reuse whole-image | ww **convex** prototype whole-image | ww tiled (best 512/256) | **snaphu single-tile** |
| ----- | -------------------: | ----------------------------------: | ----------------------: | ---------------------: |
| D_077 |           1% (119 s) |                       **2%** (86 s) |              72% (58 s) |      **99.3% (736 s)** |
| D_078 |           7% (111 s) |                      **24%** (63 s) |              72% (43 s) |      **99.9% (749 s)** |

Three results, all clean:

1. **The convex *concept* is proven by snaphu.** snaphu's sound convex/statistical cost
   gets **99.3–99.9% single-tile with no artifacts** on the exact steep ramps where ww
   tops out at 72%. Single-tile convex IS the quality ceiling - confirming the thesis.
2. **whirlwind's convex *prototype* is broken whole-image** (2–24%, *worse* than the
   tiled linear cost) - it does not realize the convex benefit (unsound: the parabola
   offset creates negative marginal costs that Dial's non-negative-reduced-cost Dijkstra
   can't handle without a Bellman-Ford/SPFA pre-pass). So the lever is a *sound* convex
   cost, not the current prototype.
3. **Speed is not ww's problem.** snaphu single-tile = **~12 min/frame**; ww tiled = 2–20 s,
   ww whole-image = 119–290 s. ww is **6–60x faster than snaphu**. "Like snaphu but faster"
   reduces to "match snaphu quality at ww speed."

- **Tile size/overlap can't fully fix the steep ramps.** Best D_077: 512/64=48%,
  512/256=**72%**, 1024/512=54%, 2048/1024=54%, single=1%. Raising the default overlap
  (64→256, far below snaphu's 400 floor) is a strict +24-pt win but caps at ~72% - the
  linear cost, not the granularity, is the limit. (`ww_gunw_overlap/`)

## Harness

**tophu 0.2.0** is installed (`~/miniforge3/envs/mapping-312`, source `~/repos/tophu`) and
exposes `SnaphuUnwrap` / `ICUUnwrap` / `PhassUnwrap` behind a uniform `UnwrapCallback` +
`multiscale_unwrap(...)`. This is the right harness to compare snaphu/icu/phass against
whirlwind on the GUNW frames with one interface (run in the mapping-312 env, or
`pip install tophu` into the project venv). Our `scripts/snaphu_nisar_compare.py` already
does snaphu directly; tophu adds ICU + PHASS for free.

## Path to the goal

1. **Implement a SOUND convex/statistical cost - this is the lever, now empirically
   confirmed.** snaphu's convex single-tile hits 99.3–99.9% clean on the frames where ww
   tops out at 72%; ww's convex *prototype* gets 2–24% because it's unsound (negative
   marginal costs from the parabola offset break the Dial/Dijkstra non-negativity
   assumption - `preload_convex_min` is insufficient). The real work: a convex per-arc
   cost (snaphu-style MAP/smooth) with a solver that tolerates negative reduced costs
   (Bellman-Ford/SPFA potential init + Klein cycle-cancelling, or a cost-scaling MCF).
   With ww's fast machinery this should give **snaphu quality at a fraction of snaphu's
   ~12-min single-tile runtime**, and the tile artifacts vanish (whole-image becomes
   well-posed, so no tiling/multi-shift reconciliation to leave seams).
2. **Cheap immediate win (ship now):** raise the default tile overlap 64 → ~256 - a strict
   +24-pt improvement on steep ramps (48→72%), runtime cost is fine. Doesn't fix the
   ceiling but removes the worst of the checkerboard while the cost work lands.
3. **Validate against tophu** (snaphu/icu/phass uniform harness, `scripts/tophu_compare.py`)
   on the full GUNW set so "matches single-tile snaphu" is measured directly, and to see
   where PHASS sits (the "give up less / fewer conncomps" axis).

## Bottom line

whirlwind is already a fast, phase-aware-linear-cost MCF that **beats PHASS** on
conncomp count + recall, and is **6–60x faster than snaphu**. The single remaining gap to
"matches single-tile snaphu, no artifacts" is the **cost shape**: a sound convex cost. The
tile-block/checkerboard/streak artifacts are not tiling bugs to patch one-by-one - they're
symptoms of the linear cost needing tiling as a crutch. Fix the cost and they disappear.
