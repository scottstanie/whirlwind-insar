# Whirlwind, SNAPHU, and PHASS performance notes

This note explains the runtime pattern in the [13-frame NISAR GUNW benchmark](NISAR_SUMMARY.md): Whirlwind is much faster than single-tile SNAPHU and usually slower than PHASS. The measured runtime, memory, and quality numbers live in the NISAR comparison page and the raw [`nisar_4way_results.csv`](nisar_4way_results.csv).

The quality metric in that comparison is per-connected-component 2pi ambiguity
agreement with the production NISAR GUNW unwrap.

## Why Whirlwind is faster than single-tile SNAPHU

Whirlwind and SNAPHU are in the same broad algorithm family: compute residues, assign statistical edge costs, solve a network-flow problem, then integrate the corrected gradients. The speed difference in this benchmark comes from the flow representation and the shortest-path implementation.

1. Fixed linear costs. SNAPHU evaluates nonlinear, flow-dependent statistical costs, and re-evaluates them as flow changes. Whirlwind's default NISAR path uses a fixed linear Carballo/Lee per-arc cost with unit-capacity arcs (only the zero-cost boundary gutter ring is multi-unit). The fixed integer costs make the network problem open to use faster MCF algorithms.

2. Shortest-path implementation. Whirlwind uses a [Dial bucket-queue Dijkstra](https://www.geeksforgeeks.org/dsa/dials-algorithm-optimized-dijkstra-for-small-range-weights/) (`shortest_path/dial.rs`) for the fixed integer-cost problem. This optimized implementation is possible because we use integer costs, and have a small range of possible edge weights (as opposed to a general floating point solver).

3. The solver that SNAPHU uses can handle non-linear, which is necessary to solver MCF problems in the way that SNAPHU poses the topography-unwrapping problem (mode="topo"). Whirlwind focuses on unwrapping deformation fields; SNAPHU's cost model is simpler for this problem as well, but it uses the same underling MCF solver for both the non-convex "topo" mode and the "defo"/"smooth" modes.

In short, Whirlwind solves the same kind of network problem as SNAPHU, but has a simpler fixed-cost model and a faster implementation of the shortest-path work that dominates the runtime.

## Why PHASS is usually faster than Whirlwind

The isce3 PHASS used by the sweep also computes residues and per-arc costs and runs a flow solve over the whole frame, then derives the unwrapped surface from regions. Its speed comes from approximations on both sides of that solve:

1. Cheap discharge corridors. A Canny edge detector and phase-gradient thresholds zero the arc costs along likely discontinuities, and the remaining costs are coherence-based bytes (0-255). Residues drain along the zero-cost corridors nearly for free, so augmenting paths stay short.

2. Free reuse of flowed arcs. Inside its successive-shortest-paths solve, any arc that already carries flow has zero cost for later paths (`ASSP.cc`), so later residue pairings funnel along existing flow lines. Each iteration gets cheaper as flow accumulates, at the price of exactness: the solution is no longer a minimum-cost flow for the stated costs.

3. Region-level repair. After the flow step, PHASS flood-fills regions bounded by the flow lines, picks each region's 2pi ambiguity by a histogram-mode vote, and drops regions smaller than a pixel-count floor. These steps are fast, and they are also where PHASS diverges from production SNAPHU on hard frames.

Whirlwind's `unwrap_linear` path solves the fixed-cost MCF over the full uncut graph and drains every residue to integer balance. That costs more shortest-path work, especially on residue-heavy scenes, and it avoids the cut, vote, and small-region-discard heuristics. Condensed:

- PHASS: approximate flow (free reuse, zero-cost cut corridors) plus flood fill and per-region wrap votes.
- Whirlwind: exact whole-frame fixed-cost MCF, every residue drained.

## Profiling Whirlwind

To profile Whirlwind, use the following entry points:

- `scripts/prof_pdssp.py`: sweeps the number of primal-dual iterations.
- `WHIRLWIND_DEBUG=1`: prints detailed PD/SSP timings, including reduced-cost scan time.

Whirlwind spends most of its time in the primal-dual and SSP shortest-path work. The O(mn) steps (cost construction, residue computation, integration, and connected-component labeling) are parallel and contribute little to the total.
