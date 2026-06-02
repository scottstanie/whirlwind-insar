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
the Rust version is a 3–72× faster rewrite (pixel-identical mod 2π to libwhirlwind) that
*added* tiling, the coarse-anchor/cascade, CRLB cost, solve-free conncomps, and the
flow-reuse + (prototype) convex modes. There is no `whirlwind-cpp` — that's `libwhirlwind`.

**Correction to the working recollection** ("original whirlwind was close to PHASS but
with a cost closer to snaphu"): half right at best. The *solver* is **not** PHASS's
quality-guided region-grow — it's classical MCF. The *cost* is **not** snaphu's convex
cost — it's the Carballo **linear** cost (coherence-weighted, phase-gradient-aware). The
only "snaphu-ish" thing is using coherence as a statistical weight; the cost *shape* has
always been linear, never convex. So whirlwind is its own point: **a phase-aware
linear-cost MCF**, distinct from both PHASS (pure-coherence cost) and snaphu (convex cost).

## The three, side by side

| | **whirlwind** (Rust) | **PHASS** (isce3) | **snaphu** |
|---|---|---|---|
| Cost | Carballo **linear**: `γ·max(0, π∓α)` per unit, direction-aware; γ=min-endpoint coherence, α=7×7-smoothed phase gradient (**phase-aware**) | **pure coherence**: `min(coh_i,coh_j)`, no phase info | **convex/statistical (MAP)**: `w·(k − offset)²`, quadratic in flow |
| Marginal cost | **constant** (linear) | constant | **increasing** (curvature) |
| Solver | primal-dual MCF + Dial bucket-queue; **flow-reuse** (uncapacitated, reused arcs free) | primal-dual MCF (ASSP) + **flow-reuse**, then region-grow | nonlinear network-flow (curvature) |
| Whole-image | **runs away** on noisy/steep (linear optimum ≠ truth) → needs tiling | **runs away** too (same solver family) | **single-tile is the quality ceiling** |
| Tiling | per-tile MCF + global coarse anchor + multi-scale cascade + feathered composite + gated multi-shift | whole-image (tiled at caller level, e.g. tophu) | tile + secondary tile-MCF merge |
| Conncomps | **solve-free** from cost grid; `min_size_px=100`; **~53 / frame**, recall ~94% | region-grow flood-fill; `good_correlation=0.7`, `min_region=200`; **hundreds–thousands**, gives up <0.7 coh | BFS post-solve; `min_region_size=300` |

## The crux: cost *shape*, not magnitude

This is the one structural difference that explains everything:

- **snaphu's cost is convex** (increasing marginal cost). Pushing one cycle is cheap;
  pushing a *block* of coherent cycles the wrong way gets progressively expensive. That
  curvature makes the **true unwrap the global optimum**, so a single whole-image solve is
  well-posed and is the quality ceiling. Tiling in snaphu is purely memory/speed and only
  *approximates* the single-tile answer.
- **whirlwind's (and PHASS's) cost is linear** (constant marginal cost). A coherent block
  of cycles routed the wrong way costs the same per unit as small corrections, so the
  whole-image optimum can sit far from the truth on noisy/steep data — the **"run-away"**.
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
  conncomps"** — *whirlwind already achieves this.* Phase-aware cost + solve-free conncomp
  + absolute 100-px floor give ~53 components and ~94% recall on NISAR, vs PHASS's
  hundreds–thousands and its hard `good_correlation=0.7` give-up. ✅
- **"Like snaphu, but faster / match single-tile snaphu with no artifacts"** — *not there.*
  This needs snaphu's **cost curvature** so the whole-image solve is well-posed (then the
  artifacts vanish because there's nothing to reconcile). whirlwind has a convex-cost
  prototype, but it's unsound/untuned whole-image (negative marginal costs from the offset
  break Dial's non-negative-reduced-cost assumption without a Bellman-Ford/SPFA pre-pass).

## Empirical backbone (NISAR GUNW, water-masked, nlooks=16)

- **No tile size/overlap fully fixes the steep-ramp D-frames.** Best D_077 per-component
  match: 512/64=48%, 512/256=**72%**, 1024/512=54%, 2048/256=54%, single-tile=1%. Bigger
  overlap helps (the 64-px default is far below snaphu's 400-px floor) but is non-monotonic
  and caps well short of clean — consistent with "the linear cost, not the granularity, is
  the limit." (`ww_gunw_overlap/`)
- **snaphu single-tile runtime baseline:** *pending* (jobs `b301s2xsg`/`bj7mnxb8e`; clearly
  minutes — confirms ww's tiled 2–20 s is a large speed win and even whole-image 119–290 s
  is competitive; speed is not the problem, artifacts are).
- **Convex whole-image on D_077/D_078:** *pending* (the decisive cost-shape test).

## Harness

**tophu 0.2.0** is installed (`~/miniforge3/envs/mapping-312`, source `~/repos/tophu`) and
exposes `SnaphuUnwrap` / `ICUUnwrap` / `PhassUnwrap` behind a uniform `UnwrapCallback` +
`multiscale_unwrap(...)`. This is the right harness to compare snaphu/icu/phass against
whirlwind on the GUNW frames with one interface (run in the mapping-312 env, or
`pip install tophu` into the project venv). Our `scripts/snaphu_nisar_compare.py` already
does snaphu directly; tophu adds ICU + PHASS for free.

## Path to the goal

1. **The structural lever is the convex/statistical cost.** Make the convex mode *sound
   and tuned whole-image* (Bellman-Ford/SPFA for the negative-reduced-cost case, or the
   `preload_convex_min` already added) and re-evaluate it **whole-image** (not tiled, where
   tiling already hides the linear cost's problem — the reason convex earlier "didn't
   win"). If convex-whole-image is clean on D_077/D_078, that's the fix *and* the
   explanation, and tiling reverts to memory-only.
2. **Meanwhile**, raise the default tile overlap (64 → ~tile/2) — it's a strict improvement
   on the steep ramps (48→72% on D_077) at a runtime cost we've now established is fine.
3. **Validate against tophu** (snaphu/icu/phass) on the full GUNW set so "matches
   single-tile snaphu" is measured directly, not inferred.
