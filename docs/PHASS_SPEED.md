# Whirlwind performance: why it's ~15–40× faster than SNAPHU (and ~2–4× slower than PHASS)

Two questions come up on the benchmark: **why is whirlwind so much faster than
SNAPHU** (the headline ~15–40× on single-tile NISAR frames), and **why is isce3
PHASS still ~2–4× faster than whirlwind** (e.g. D_077 ≈ 20 s vs ≈ 37 s). This note
answers both. (Aside: a tiled or reoptimize warm-start can cut runtime but **not**
full-frame memory — the final single-tile solve still allocates the whole-image
network.)

## Why ~15–40× faster than SNAPHU?

On these NISAR frames whirlwind unwraps in ~14–41 s vs single-tile SNAPHU's
~500–900 s. The honest framing first: **this is the same algorithm class.** SNAPHU
and whirlwind both reduce 2D unwrapping to a minimum-cost flow on the residue
network — whirlwind is *not* trading quality for speed here (that is the PHASS story
below); it matches SNAPHU's per-component quality. So the gap is *how the MCF is
built and solved*, not a different or weaker problem. Four factors, roughly by
impact:

1. **A fixed linear cost vs nonlinear statistical costs.** SNAPHU's costs are
   *statistical* (MAP-derived) and **nonlinear** functions of the per-arc flow — its
   2001 paper is titled "…statistical models for cost functions in **nonlinear
   optimization**" (Chen & Zebker, *JOSA A*) — so its network-flow solve carries
   convex, multi-unit arc costs and is correspondingly heavier. whirlwind uses a
   **fixed linear** Carballo (Lee-1994) per-arc cost and a capacity-1 MCF, so it
   solves a single, lighter network. Same residue-pairing problem, a simpler (and
   so cheaper) cost model — at no measured quality cost on these frames. *(This is
   the factor I can least precisely quantify without profiling SNAPHU's internals;
   the implementation and configuration factors below are the surer bets.)*

2. **An efficient *serial* solver — not parallelism.** It is tempting to credit
   rayon, but measured (`scripts/rayon_bench.py`): 1 thread vs 12 is only
   **~1.2–1.3×** (D_077 43.6 → 37.4 s), because the PD/SSP solver is largely serial
   and rayon only parallelizes the O(mn) cost/residue/conncomp build. The telling
   number: **whirlwind on a single thread is still ~13× faster than single-tile
   SNAPHU.** So the gap is the *solver and cost model*, not core count — a lean
   per-arc linear cost and a tuned Dial-bucket Dijkstra MCF (with the recent
   per-source-rescan fix, ~1.4–2.4×). SNAPHU v2 is mature, portable C; parallelism
   is not where the difference comes from.

3. **Single-tile is SNAPHU's *slow* configuration.** SNAPHU's operational strength
   is **tiling** — bounded per-tile graphs plus a single-tile reoptimize pass — and
   that path is fast and battle-tested. A single-tile solve over a full ~18-Mpx
   NISAR frame is the largest, slowest setup for it: we are benchmarking at SNAPHU's
   worst case. Which is exactly the result worth stating — **whirlwind reaches
   SNAPHU-quality on the whole frame in one pass, with no tiling**, so the tile
   bookkeeping, the seam artifacts, and the reoptimize step drop out of the pipeline.

4. *(minor)* whirlwind grows connected-component labels directly from the cost grid
   **without** an MCF solve, so conncomps do not add a second global pass.

**Bottom line, fair to SNAPHU:** it is still the quality reference, and its tiled
path is fast. whirlwind's edge is not a cleverer algorithm — it is that a single
fixed-cost linear MCF, solved efficiently in parallel, hits SNAPHU's single-tile
quality far faster, so you do not *need* to tile a NISAR frame to unwrap it quickly.

## Why is PHASS faster than whirlwind? An algorithm-class difference — but PHASS is *not* "no flow"

A common over-simplification is "PHASS just region-grows, it doesn't solve a
flow." That is wrong. PHASS **builds residues and a cost graph and runs a
min-cost-flow `solve(node_patch)`**, then flood-fills. Its speed comes from three
things, none of which is "skip the flow":

1. **Hard-cut structure.** PHASS injects hard `cost = 0 / 255` barriers from a
   Canny edge detector, ≥ 1 rad phase-gradient cuts, and good-coherence clamps.
   These *drastically* slash the connectivity of the flow problem, so the
   `solve()` it runs is over a much smaller / fragmented graph.
2. **Flow reuse** across solves.
3. **Heuristic post-processing** instead of exact convergence: a flood-fill seed,
   a per-region **histogram-mode** wrap vote, and **discarding any region
   < 200 px**.

whirlwind's `unwrap_linear`, by contrast, solves the **exact global min-cost flow
on the full (uncut) graph**, draining *every* residue to integer balance. That
exactness is the quality lead: on the NISAR GUNW set whirlwind scores 99–100 %
per-component vs PHASS's 48–99 % (94.7 % on D_077), because PHASS's hard cuts,
mode vote and small-region discards each shed accuracy for speed.

So the trade is **heuristic-cut-down-flow + flood-fill + reuse (PHASS)** vs
**exact global MCF (whirlwind)** — not "no search vs search."

## Where whirlwind actually spent its time

Profiled on D_077 (4176 × 4257, `WHIRLWIND_DEBUG=1`):

| stage | time | notes |
|---|---|---|
| **SSP fallback** | ~45 s | 927 single-source Dijkstras after PD hands off |
| PD (8 full Dijkstra) | ~13 s | early iters drain thousands of units, later ones < 100 |
| cost / residue / integrate / conncomp | ~8 s | all O(*mn*), rayon-parallel |

More PD iterations are strictly slower for identical quality (the residual past
iter 8 is "stranded" for multi-source PD), so 8 is the sweet spot and SSP draining
is the real cost.

## The fix that came out of it (~1.4–2.4×, shipped)

Drilling into the SSP fallback with `WHIRLWIND_DEBUG` showed the dominant cost was
**not** the Dijkstra traversals but `max_reduced_cost_par` — an O(E) scan over
~38 M arcs that ran **once per source** purely to size the Dial buckets:
**34 s of D_077's 61 s (~52 % of runtime, ~76 % of the SSP phase).**

A naive "scan once" hoist is unsafe: the capped potential update *grows*
potentials, so the max reduced cost rises and a stale, too-small Dial `k` would
alias the circular buckets and return a wrong answer. The fix maintains `max_rc`
**across** sources (one tight scan up front) and rescans **only on overflow** — if
any relaxation observes `rc ≥ k`, it discards that source's partial Dijkstra,
recomputes `max_rc` tight, and retries, so an under-sized `k` can never commit.

**Result:** D_077 61 → 37 s, D_075 2.44×, A_035 1.65×, several others 1.4–1.5×;
the optimal cost is byte-identical, per-component match is unchanged on all 13
NISAR frames, and the 79 core tests stay green. Residue-light frames are
unchanged (they barely touch the SSP fallback).

Profile it yourself: `scripts/prof_pdssp.py` (a PD-iters sweep) and
`WHIRLWIND_DEBUG=1`, which prints `max_reduced_cost_scan=…ms`.

## Remaining levers (not yet done)

- **Reliability region-growing / reoptimize warm-start** — borrow PHASS's idea as
  an *init*: a fast coarse/tiled pass produces a near-solution, the single-tile
  MCF then *reoptimizes* from it, so far fewer residues remain for PD/SSP. Speed
  only (the final solve is still full-image memory); estimated 2–3× more.
- **Parallelize the SSP source loop** — disjoint residual components are
  independent.
