# Tiled unwrap + a corrected diagnosis of the no-Goldstein gap

Last updated: 2026-05-28 (evening, session 2).

Companion to `handoff.md`, `convex_cost_design.md`, `phass_experiments.md`.
This doc records (1) the **tiled coherence unwrap** that is now the
fast/low-memory no-Goldstein path, (2) several **corrections** to earlier
conclusions after reading the SNAPHU 2.0.7, ISCE3 PHASS, and original C++
`libwhirlwind` sources, and (3) what's still open.

## tl;dr

| NISAR vs SNAPHU 9×9 | wall | K-match | \|dK\|≥2 | memory |
|---|---:|---:|---:|---|
| `unwrap` whole-image | 78 s | 80.0 % | 18.3 % | whole-scene |
| **`unwrap` tiled ts=512** | **3.5 s** | **96.6 %** | **2.7 %** | per-tile |
| tiled ts=1024 | 5.9 s | 92.0 % | 5.8 % | per-tile |
| `unwrap_reuse` | 93 s | 92.7 % | 7.1 % | whole-scene |
| Goldstein α=0.7 | 38 s | 99.9 % | — | whole-scene |

**Tiling at ts≈512 beats whole-image (80→96.6 %), runs ~22× faster, and
bounds memory to tile scale** — the "fast unwrapper that isn't a memory
exploder" goal. It is exposed on the existing Python `unwrap` via
`tile_size` / `tile_overlap` (mirrors `unwrap_crlb`).

## Why tiling *beats* whole-image (not just lower memory)

A whole-image MCF finds the true cost-optimum; tiling can only approximate
it. Yet tiled is *more correct*. The resolution: **the linear coherence
cost is a poor proxy for correctness** — it lets residues pair over long
distances and stack multi-cycle ramps cheaply, so its global optimum
*contains* the runaway (the whole-image 80 % is the genuine cost-optimum,
not a solver failure — confirmed earlier by ruling out path-order). Tiling
**constrains the search to locally-paired solutions**, acting as a spatial
regularizer that compensates for the cost's defect. Evidence:
* Smaller tiles → *more* correct (512 ≫ 2048 ≈ whole-image). More
  constraint → better is the signature of regularization, not optimization.
* A clean window unwrapped **standalone** had K-std ≈ 1; the *same pixels*
  extracted from the **whole-scene** solve had K-std ≈ 6–10. The global
  solve injects the error.
* The reference itself is SNAPHU **9×9 — tiled**. SNAPHU's quality comes
  substantially from tiling + its secondary network, so matching it favors
  a tiled solver.

The principled way to make "single-tile is the ceiling" true again is to
fix the *cost* (convex curvature → no runaway to exploit); then tiling is
purely memory/speed and `single_tile_reoptimize` (a final whole-image
refine warm-started from the tiled result) works. With the current linear
cost a final global pass would regress straight back to 80 %.

## The stitch is the crux

Per-tile MCF picks its own integer ambiguity; tiles are reconciled by an
integer 2π offset per tile, found from the overlap. The reconciler evolved:

1. **median-of-continuous-diff + greedy BFS** — REGRESSED NISAR to 47 %
   (`|dK|=1`=23 %). A bad pairwise stitch propagates to every downstream
   tile, and the continuous median rounds the wrong way when a wrap line
   splits the overlap.
2. **weighted mode of rounded offsets + max-confidence spanning tree
   (Prim)** — 47 → 92–96 %. Mode is robust to a split overlap; growing the
   tree highest-confidence-first keeps low-confidence seams as leaves so a
   wrong ±1 can't cascade.
3. **consensus voting** (current) — after the Prim seed, every tile adopts
   the coherence-weighted mode of the offset implied by *all* its
   neighbours, iterated. Resolves seams a tree locks in.

Residual failure (still visible, "would-not-ship"): in **low-coherence
regions a whole tile can lock onto a wrong offset** (a locally-consistent
wrong island that local voting can't flip) — plus ordinary per-pixel
salt-and-pepper in decorrelated areas (partly inherent; SNAPHU has it too).
The fix is a **global secondary reconciliation network** (SNAPHU's
`AssembleTiles`): connected unwrapped *regions* across tiles, one integer
offset each, solved by a *global* MCF whose arc costs sum over each seam —
where flipping a wrong island is globally cheaper, so it can't survive.
That is the next task.

## Corrections to earlier conclusions (source-verified)

Read against `~/repos/snaphu-v2.0.7/src`, `~/repos/isce3/cxx/isce3/unwrap/phass`,
`~/repos/libwhirlwind`, `~/repos/whirlwind`.

1. **The convex `offset` is the wrong quantity.** SNAPHU smooth/defo cost is
   `(flow·nshortcycle + offset)²/sigsq` with
   `offset = nshortcycle·(dpsi_raw − avgdpsi)` (snaphu_cost.c:1113-1120) —
   the *deviation* of the raw wrapped gradient from its 7×7 boxcar mean, so
   the preferred unwrapped gradient is the smooth neighborhood gradient
   (`flow* = avgdpsi − dpsi`). whirlwind's `compute_snaphu_smooth_costs`
   uses `offset = round(α_smooth·100/2π)` (the smoothed gradient itself),
   capped at ±50 so it can never even express a ±1 preference. This — not
   "σ² calibration" or "offset polarity" — is why convex collapsed to a
   pure quadratic with no wrap-line signal.
2. **The convex solver is unsound, so its 68.8 % is uninformative.** No
   Bellman-Ford pre-pass (the code's own comments say it's mandatory);
   marginal cost goes negative after the first push; convex is force-routed
   to the binary heap which runs plain Dijkstra with no negative-edge
   handling (`debug_assert!(rc>=0)` stripped in release). "Convex regresses"
   was a corrupt solve, not evidence against convex cost. Fix: deviation
   offset + warm-start each arc's flow to its parabola minimum `k*` (makes
   initial marginals ≥0 → existing Dijkstra valid, no extra memory). Run it
   *inside tiles* so it stays cheap/bounded.
3. **Unit-capacity is faithful, not a port regression.** The original
   `whirlwind/_unwrap.py:38` hardcodes `capacity=1`; the per-arc cost was
   always a single linear scalar. But `libwhirlwind` ships `UncapacitatedMixin`
   (multi-unit) and `RectangularGridGraph<P>` (parallel arcs = the textbook
   convex reduction) — both unused by the unwrap path. `unwrap_reuse` is
   essentially turning on uncapacitated flow. Parallel-arc convex is the
   "clean" convex route but is a **memory exploder** (K× arcs); the in-place
   warm-start route above is preferred.
4. **`reuse` amplifies runaway on hard scenes.** It won NISAR (92.7 %) but
   on noisy data the free-highway extends spurious wrap-lines for free
   (Atlanta K-std 10.5 vs baseline 8.5). It is *not* the robust default.

## Atlanta (S-1, OPERA frame): a real failure, not a bad reference

Earlier this was mis-diagnosed twice. Final, correct status: **the
`opera.displacement` reference is valid** — `snaphu-py` (SNAPHU 2.0.7,
`ntiles=(2,2)`, 5× subsampled) unwraps `opera.int.tif` cleanly in ~1 min.
So whirlwind producing K-std 8–10 garbage there is **our problem**. Prep is
*not* the cause: `opera.int.phs` ≡ `angle(opera.int.tif)` exactly, the
provided coherence is not overstating (the IG's own phase coherence is
higher), and we already mask NaN→0 so `nan_to_num` is a no-op. The earlier
dolphin-PHASS run that looked catastrophic (K-std 328) was itself borked on
some input convention — disregard it. Atlanta is a genuinely noisy
(median coh 0.62) scene where the **linear cost's runaway** dominates;
tiling helps (8.5→3.2 K-std) but the real fixes are the secondary network +
convex cost. **Judge no-Goldstein quality on NISAR/PV; treat Atlanta as the
hard target the secondary-network + convex work must crack.**

Open idea (from S.S.): **multiscale** — unwrap a coarsened (e.g. 5×) version
to fix the large-scale ambiguities, then constrain the full-res solve to it.
Aliases fast/small features but stabilizes the big picture; worth testing
once the secondary network lands.

## Where the code lives

* `crates/whirlwind-core/src/tile.rs`: `unwrap_tiled`, `unwrap_one_tile_coh`,
  `stitching_offset_coh` (mode + confidence), Prim seed + consensus voting.
* `crates/whirlwind-py/src/lib.rs`: `unwrap(..., tile_size, tile_overlap)`.
* `scripts/phass_experiments/`: `run_atlanta.py` (modes incl. `tiled`,
  `goldstein`), `analyze_atlanta.py` (K-match vs OPERA, global + per-cc),
  `plot_atlanta.py`, `plot_nisar_tiled.py`.

## Decision rule for the next round

Goal = a shippable no-Goldstein unwrap: build the **secondary
reconciliation network** (kills the low-coherence whole-tile offset
islands), then **convex cost done right inside tiles** (deviation offset +
warm-start; removes the runaway at the source and unlocks
`single_tile_reoptimize`). Goldstein α=0.7 remains the working production
default (99.9 %, 38 s) meanwhile.
