# Whirlwind, SNAPHU, and PHASS performance notes

This note explains the main runtime difference in the 13-frame NISAR GUNW
benchmark:

- Whirlwind: 10.5-27.3 s per frame.
- SNAPHU, single tile: 465-1242 s per frame, or about 25-115x slower.
- SNAPHU, 3x3 tiled plus reoptimize: 97-201 s per frame.
- PHASS: 5.5-22.6 s per frame. Whirlwind is usually slower than PHASS, with a
  median runtime ratio of about 1.5x and a maximum of about 2.8x on this sweep.

The quality comparison in the public docs uses per-connected-component 2pi
ambiguity agreement with the production NISAR GUNW unwrap. Whirlwind agrees with
production SNAPHU on at least 98.8 percent of pixels on 12 of 13 frames; the
remaining frame, D_075, scores 88.2 percent. The sweep's own SNAPHU runs
(`cost=smooth`, `init=mcf`, both single-tile and 3x3 plus reoptimize) also score
88.2 percent on D_075, so that frame appears to reflect a production-reference
configuration mismatch rather than a Whirlwind-only failure. PHASS ranges from
48.4 to 99.6 percent on the same frames.

A tiled or warm-started path can reduce runtime, but it does not remove the peak
memory cost of a final whole-frame solve if the final solve still builds the
whole-image network.

## Why Whirlwind is faster than single-tile SNAPHU

Whirlwind and SNAPHU are in the same broad algorithm family: compute residues,
assign statistical edge costs, solve a network-flow problem, then integrate the
corrected gradients. The speed difference in this benchmark is mostly in the
flow representation and shortest-path implementation.

1. Fixed linear costs. SNAPHU uses nonlinear, flow-dependent statistical costs.
   Whirlwind's default NISAR path uses a fixed linear Carballo/Lee per-arc cost
   and capacity-1 flow. That keeps the same residue-pairing structure, but makes
   the network problem lighter.

2. Shortest-path implementation. Whirlwind uses a tuned Dial-bucket Dijkstra
   path for the fixed integer-cost problem. The speedup is not mainly from core
   count: `scripts/rayon_bench.py` measured only about 1.2-1.3x from 1 thread to
   12 threads on D_077, because the PD/SSP solver is mostly serial. The
   parallel work is mainly the O(mn) cost, residue, and connected-component
   setup.

3. SNAPHU single-tile is the slow configuration. SNAPHU's operational path is
   tiled unwrapping plus reoptimization. A single-tile solve over an entire
   NISAR frame is the largest and slowest way to run it. That comparison is
   still useful because Whirlwind reaches near-production-SNAPHU agreement on
   most frames in one whole-frame pass, without the tiled solve and reoptimize
   workflow.

4. Connected components are cheap in Whirlwind. Whirlwind grows component labels
   directly from the cost grid rather than running a second global flow solve.

The short version: Whirlwind is not using a fundamentally weaker problem than
single-tile SNAPHU for this comparison. It is using a simpler fixed-cost MCF and
a faster implementation of the corresponding shortest-path work.

## Why PHASS is usually faster than Whirlwind

PHASS is faster because it reduces and fragments the problem before making
region-level decisions. In the isce3/tophu PHASS path used by the sweep, PHASS
does build residues and a cost graph, runs a min-cost-flow solve on node patches,
and then flood-fills regions.

The main speed factors are:

1. Hard cuts and barriers. PHASS adds hard cost barriers from edge and
   phase-gradient tests, plus coherence-based constraints. That reduces graph
   connectivity and makes each solve smaller.

2. Flow reuse. Previously used flow paths are cheap to reuse, so repeated solves
   do not pay the same cost as an independent global solve.

3. Region-level post-processing. After the flow step, PHASS uses flood fill, a
   per-region histogram-mode wrap vote, and drops very small regions. These
   choices are fast, but they are also where some disagreement with production
   SNAPHU can enter on difficult NISAR frames.

Whirlwind's `unwrap_linear` path instead solves the fixed-cost MCF over the full
uncut graph and drains every residue to integer balance. That costs more
shortest-path work, especially on residue-heavy scenes, but avoids PHASS's hard
cut, region-vote, and small-region-discard tradeoffs.

So the useful distinction is:

- PHASS: fragmented/reused flow plus flood-fill and region decisions.
- Whirlwind: whole-frame fixed-cost MCF with exact residue balance for that
  cost model.

## Current bottleneck

On residue-heavy NISAR frames, Whirlwind spends most of its time in the
primal-dual and SSP shortest-path work. The O(mn) steps--cost construction,
residue computation, integration, and connected-component labeling--are
parallel and are not the main limit.

Useful profiling entry points:

- `scripts/prof_pdssp.py`: sweeps the number of primal-dual iterations.
- `WHIRLWIND_DEBUG=1`: prints detailed PD/SSP timings, including reduced-cost
  scan time.

## Possible speed work

- Reliability or tiled warm-start: use a fast coarse or tiled pass to start the
  whole-frame MCF closer to the final solution. This targets runtime only; the
  final whole-frame solve still has whole-frame memory use.
- Parallel SSP over independent residual components.
