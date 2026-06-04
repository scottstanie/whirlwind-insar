# PHASS vs whirlwind: why PHASS's single tile is faster, and what we did about it

isce3's **PHASS** single-tile unwrap runs ~2–4× faster than whirlwind's
single-tile linear MCF (e.g. D_077 ≈ 20 s vs ≈ 60 s, before the fix below). This
note explains the difference and the speed win it led to.

## It is an algorithm-class difference — but PHASS is *not* "no flow"

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
