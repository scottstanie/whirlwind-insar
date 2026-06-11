# Whirlwind, SNAPHU, and PHASS performance notes

This note explains the main runtime differences in the 13-frame NISAR GUNW
benchmark:

- Whirlwind: 10.5-27.3 s per frame.
- SNAPHU, single tile: 465-1242 s per frame, 25-115x the Whirlwind runtime.
- SNAPHU, 3x3 tiled plus reoptimize: 97-201 s per frame.
- PHASS: 5.5-22.6 s per frame. PHASS beats Whirlwind on 12 of 13 frames, with a median runtime ratio of about 1.5x and a maximum of about 2.8x.

The quality comparison in the public docs uses per-connected-component 2pi ambiguity agreement with the production NISAR GUNW unwrap. Whirlwind agrees with production SNAPHU on at least 98.8 percent of pixels on 12 of 13 frames.  The remaining frame, D_075, scores 88.2 percent, and the sweep's own SNAPHU runs (`cost=smooth`, `init=mcf`, both single-tile and 3x3 plus reoptimize) score the same 88.2 percent there, which points to a configuration mismatch against the production reference rather than a Whirlwind failure. PHASS ranges from 48.4 to 99.6 percent on the same frames.

## Why Whirlwind is faster than single-tile SNAPHU

Whirlwind and SNAPHU are in the same broad algorithm family: compute residues, assign statistical edge costs, solve a network-flow problem, then integrate the corrected gradients. The speed difference in this benchmark comes from the flow representation and the shortest-path implementation.

1. Fixed linear costs. SNAPHU evaluates nonlinear, flow-dependent statistical costs, and re-evaluates them as flow changes. Whirlwind's default NISAR path uses a fixed linear Carballo/Lee per-arc cost with unit-capacity arcs (only the zero-cost boundary gutter ring is multi-unit). The residue-pairing structure is the same; the fixed integer costs make the network problem lighter.

2. Shortest-path implementation. Whirlwind uses a tuned Dial bucket-queue Dijkstra (`shortest_path/dial.rs`) for the fixed integer-cost problem. Core count contributes little: `scripts/rayon_bench.py` measured about 1.2-1.3x going from 1 to 12 threads on D_077, because the PD/SSP solver is mostly serial. The parallel gain is concentrated in the O(mn) cost, residue, and connected-component setup.

3. Single tile is the NISAR production configuration. The isce3 GUNW defaults run SNAPHU with `ntiles: [1, 1]` (`cost=smooth`, `init=mcf`), so the single-tile column is the configuration NISAR actually ships, and the comparison is like for like: one whole-frame statistical-cost solve against one whole-frame fixed-cost solve. Other pipelines commonly tile SNAPHU for speed; the 3x3 run shows what that buys here (97-201 s), and in this sweep it also raised peak RSS on 12 of 13 frames because the tile workers run in parallel.

In short, Whirlwind solves the same kind of whole-frame network problem as single-tile SNAPHU, with a simpler fixed-cost model and a faster implementation of the shortest-path work that dominates the runtime.

## Why PHASS is usually faster than Whirlwind

The isce3 PHASS used by the sweep also computes residues and per-arc costs and runs a flow solve over the whole frame, then derives the unwrapped surface from regions. Its speed comes from approximations on both sides of that solve:

1. Cheap discharge corridors. A Canny edge detector and phase-gradient thresholds zero the arc costs along likely discontinuities, and the remaining costs are coherence-based bytes (0-255). Residues drain along the zero-cost corridors nearly for free, so augmenting paths stay short.

2. Free reuse of flowed arcs. Inside its successive-shortest-paths solve, any arc that already carries flow has zero cost for later paths (`ASSP.cc`), so later residue pairings funnel along existing flow lines. Each iteration gets cheaper as flow accumulates, at the price of exactness: the solution is no longer a minimum-cost flow for the stated costs.

3. Region-level repair. After the flow step, PHASS flood-fills regions bounded by the flow lines, picks each region's 2pi ambiguity by a histogram-mode vote, and drops regions smaller than a pixel-count floor. These steps are fast, and they are also where PHASS diverges from production SNAPHU on hard frames (48.4 percent on D_075, 67.0 on A_025, 75.4 on A_030).

Whirlwind's `unwrap_linear` path solves the fixed-cost MCF over the full uncut graph and drains every residue to integer balance. That costs more shortest-path work, especially on residue-heavy scenes, and it avoids the cut, vote, and small-region-discard heuristics. Condensed:

- PHASS: approximate flow (free reuse, zero-cost cut corridors) plus flood fill and per-region wrap votes.
- Whirlwind: exact whole-frame fixed-cost MCF, every residue drained.

## Current bottleneck

On residue-heavy NISAR frames, Whirlwind spends most of its time in the primal-dual and SSP shortest-path work. The O(mn) steps (cost construction, residue computation, integration, and connected-component labeling) are parallel and contribute little to the total.

Useful profiling entry points:

- `scripts/prof_pdssp.py`: sweeps the number of primal-dual iterations.
- `WHIRLWIND_DEBUG=1`: prints detailed PD/SSP timings, including reduced-cost scan time.

## Possible speed work

- Reliability-ordered or tiled warm start: use a fast coarse pass to start the whole-frame MCF near its final solution. This targets runtime only; peak memory still includes the whole-image network for the final solve.
- Parallel SSP over independent residual components.
