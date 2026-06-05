# Tiled unwrap + a corrected diagnosis of the no-Goldstein gap

Last updated: 2026-05-28 (evening, session 2).

Companion to `handoff.md`, `convex_cost_design.md`, `phass_experiments.md`.
This doc records (1) the **tiled coherence unwrap** that is now the
fast/low-memory no-Goldstein path, (2) several **corrections** to earlier
conclusions after reading the SNAPHU 2.0.7, ISCE3 PHASS, and original C++
`libwhirlwind` sources, and (3) what's still open.

## tl;dr

| NISAR vs SNAPHU 9x9       |      wall |    K-match |  \|dK\|≥2 | memory      |
| ------------------------- | --------: | ---------: | --------: | ----------- |
| `unwrap` whole-image      |      78 s |     80.0 % |    18.3 % | whole-scene |
| **`unwrap` tiled ts=512** | **3.5 s** | **96.6 %** | **2.7 %** | per-tile    |
| tiled ts=1024             |     5.9 s |     92.0 % |     5.8 % | per-tile    |
| `unwrap_reuse`            |      93 s |     92.7 % |     7.1 % | whole-scene |
| Goldstein α=0.7           |      38 s |     99.9 % |         - | whole-scene |

**Tiling at ts≈512 beats whole-image (80→96.6 %), runs ~22x faster, and
bounds memory to tile scale** - the "fast unwrapper that isn't a memory
exploder" goal. It is exposed on the existing Python `unwrap` via
`tile_size` / `tile_overlap` (mirrors `unwrap_crlb`).

> **Update (session 3, committed e24e0ed / 8aa7a1d):** the no-Goldstein path
> is now the **shippable default** and beats the table above. Tiled + a
> **global coarse anchor** + a **multi-scale cascade** + a **feathered seam
> composite** reach **99.79 % K-match (0 % multi-cycle) in 3.9 s** on this
> NISAR frame - visually identical to SNAPHU 9x9, no Goldstein. For noisy
> S-1-class scenes a `multilook=8` down-look first gets Atlanta to **97.7 %**
> (SNAPHU = 97.9 %). Goldstein α=0.7 is now a **legacy/alternative**, not the
> recommended default; convex cost is a longer-term lever, **not** the fix
> (its solver is unsound - see Corrections #2 below). Full method + figures:
> [`report_anchor_cascade.md`](report_anchor_cascade.md). The sections below
> (why tiling wins, the stitch, the secondary net, coarse-refine, the
> corrections, Atlanta) remain accurate and are the foundation that path is
> built on.

## Why tiling *beats* whole-image (not just lower memory)

A whole-image MCF finds the true cost-optimum; tiling can only approximate
it. Yet tiled is *more correct*. The resolution: **the linear coherence
cost is a poor proxy for correctness** - it lets residues pair over long
distances and stack multi-cycle ramps cheaply, so its global optimum
*contains* the runaway (the whole-image 80 % is the genuine cost-optimum,
not a solver failure - confirmed earlier by ruling out path-order). Tiling
**constrains the search to locally-paired solutions**, acting as a spatial
regularizer that compensates for the cost's defect. Evidence:
* Smaller tiles → *more* correct (512 ≫ 2048 ≈ whole-image). More
  constraint → better is the signature of regularization, not optimization.
* A clean window unwrapped **standalone** had K-std ≈ 1; the *same pixels*
  extracted from the **whole-scene** solve had K-std ≈ 6–10. The global
  solve injects the error.
* The reference itself is SNAPHU **9x9 - tiled**. SNAPHU's quality comes
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

1. **median-of-continuous-diff + greedy BFS** - REGRESSED NISAR to 47 %
   (`|dK|=1`=23 %). A bad pairwise stitch propagates to every downstream
   tile, and the continuous median rounds the wrong way when a wrap line
   splits the overlap.
2. **weighted mode of rounded offsets + max-confidence spanning tree
   (Prim)** - 47 → 92–96 %. Mode is robust to a split overlap; growing the
   tree highest-confidence-first keeps low-confidence seams as leaves so a
   wrong ±1 can't cascade.
3. **consensus voting** (current) - after the Prim seed, every tile adopts
   the coherence-weighted mode of the offset implied by *all* its
   neighbours, iterated. Resolves seams a tree locks in.

Residual failure (still visible, "would-not-ship"): in **low-coherence
regions a whole tile can lock onto a wrong offset** (a locally-consistent
wrong island that local voting can't flip) - plus ordinary per-pixel
salt-and-pepper in decorrelated areas (partly inherent; SNAPHU has it too).
The fix is a **global secondary reconciliation network** (SNAPHU's
`AssembleTiles`): connected unwrapped *regions* across tiles, one integer
offset each, solved by a *global* MCF whose arc costs sum over each seam -
where flipping a wrong island is globally cheaper, so it can't survive.
That is the next task.

## Corrections to earlier conclusions (source-verified)

Read against `~/repos/snaphu-v2.0.7/src`, `~/repos/isce3/cxx/isce3/unwrap/phass`,
`~/repos/libwhirlwind`, `~/repos/whirlwind`.

1. **The convex `offset` is the wrong quantity.** SNAPHU smooth/defo cost is
   `(flow·nshortcycle + offset)²/sigsq` with
   `offset = nshortcycle·(dpsi_raw − avgdpsi)` (snaphu_cost.c:1113-1120) -
   the *deviation* of the raw wrapped gradient from its 7x7 boxcar mean, so
   the preferred unwrapped gradient is the smooth neighborhood gradient
   (`flow* = avgdpsi − dpsi`). whirlwind's `compute_snaphu_smooth_costs`
   uses `offset = round(α_smooth·100/2π)` (the smoothed gradient itself),
   capped at ±50 so it can never even express a ±1 preference. This - not
   "σ² calibration" or "offset polarity" - is why convex collapsed to a
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
   convex reduction) - both unused by the unwrap path. `unwrap_reuse` is
   essentially turning on uncapacitated flow. Parallel-arc convex is the
   "clean" convex route but is a **memory exploder** (Kx arcs); the in-place
   warm-start route above is preferred.
4. **`reuse` amplifies runaway on hard scenes.** It won NISAR (92.7 %) but
   on noisy data the free-highway extends spurious wrap-lines for free
   (Atlanta K-std 10.5 vs baseline 8.5). It is *not* the robust default.

## Atlanta (S-1, OPERA frame): a real failure, not a bad reference

Earlier this was mis-diagnosed twice. Final, correct status: **the
`opera.displacement` reference is valid** - `snaphu-py` (SNAPHU 2.0.7,
`ntiles=(2,2)`, 5x subsampled) unwraps `opera.int.tif` cleanly in ~1 min.
So whirlwind producing K-std 8–10 garbage there is **our problem**. Prep is
*not* the cause: `opera.int.phs` ≡ `angle(opera.int.tif)` exactly, the
provided coherence is not overstating (the IG's own phase coherence is
higher), and we already mask NaN→0 so `nan_to_num` is a no-op. The earlier
dolphin-PHASS run that looked catastrophic (K-std 328) was itself borked on
some input convention - disregard it. Atlanta is a genuinely noisy
(median coh 0.62) scene where the **linear cost's runaway** dominates;
tiling helps (8.5→3.2 K-std) but the real fixes are the secondary network +
convex cost. **Judge no-Goldstein quality on NISAR/PV; treat Atlanta as the
hard target the secondary-network + convex work must crack.**

Open idea (from S.S.): **multiscale** - unwrap a coarsened (e.g. 5x) version
to fix the large-scale ambiguities, then constrain the full-res solve to it.
Aliases fast/small features but stabilizes the big picture; worth testing
once the secondary network lands.

## Secondary reconciliation: min-cost-flow (session 2b)

The greedy/consensus stitch left visible **whole-tile offset errors** in
low-coherence patches (a tile locked to a wrong 2π offset that per-tile
voting can't flip - it can't *break a satisfied seam*). The fix is SNAPHU's
secondary-network idea at tile scale: a **global min-cost tension** over the
tile grid, `minimize Σ w·|measured − (o_a − o_b)|` over integer per-tile
offsets, solved as a residue **min-cost-flow on the planar dual** of the
tile grid (`reconcile_offsets_mcf` in `tile.rs`, with a small self-contained
SPFA-based MCF). Because it minimizes the *summed* seam cost globally, it
flips a wrong island whenever that lowers total cost - breaking a satisfied
seam when warranted. Guarded by `reconcile_mcf_breaks_low_confidence_wrong_seam`.

NISAR vs SNAPHU 9x9 (ts=512, no Goldstein):

| reconciler            |      wall |    K-match |  \|dK\|≥2 |
| --------------------- | --------: | ---------: | --------: |
| consensus voting      |     3.5 s |     96.6 % |     2.7 % |
| **MCF secondary net** | **3.4 s** | **97.5 %** | **1.7 %** |

97.5 % now exceeds `unwrap_reuse` (92.7 %) and approaches dolphin-PHASS
(97.9 %) at ~18x its speed and bounded memory. The residual (worst-crop
~83 %) is **no longer the reconciliation** - the MCF optimally reconciles
the measured seams, but those per-seam measurements are themselves wrong in
genuinely low-coherence overlaps. Closing that needs the **convex per-tile
cost** (deviation offset; reduces per-tile error so seam measurements are
trustworthy) and possibly larger overlap / better seam-confidence - the next
lever. **96–97.5 % is a fast/low-memory foundation, not a shippable
unwrap quality yet** (per S.S.).

Longer-term product framing (S.S.): if the primal-dual + tiling path can get
*near* SNAPHU/PHASS quality but markedly faster, ship it as a "2–3x speedup"
option. Tiling already shows ~18–22x on NISAR; the gating factor is quality
parity, which is the convex-cost work.

## Coarse region-refinement → 99 % (session 2c)

The MCF secondary net left **2 rectangular high-coherence artifacts** (a +1 and
a +3 constant-offset block): sub-tile regions the per-tile MCF unwrapped a few
cycles off, bounded by a 2π discontinuity *ring*. They survive seam
reconciliation because the seam *measurement* across them is consistent (the
whole sub-region is uniformly shifted), so the reconciler bakes the shift in.

What distinguishes the artifact from the correct unwrap is the **global
smoothness**: the wrong block has a 2π ring through *high coherence*, which a
correct unwrap never cuts. So a coherence-aware post-pass removes them
(`coarse_refine` in `tile.rs`). Per-pixel jump detection fragments under phase
noise (22 M components on NISAR), so we **coarsen 8x** (block-mean - noise
averages out, the 100s-of-px artifacts survive), group coarse pixels into
regions by no-jump connectivity, and shift each region by the integer that
zeroes its **coherence-weighted** boundary jumps. High-coh rings are expensive
→ flipped away; legitimate low-coh cuts are cheap → kept.

NISAR vs SNAPHU 9x9 (ts=512, no Goldstein), full pipeline:

| stage                                |     K-match |   \|dK\|≥2 |
| ------------------------------------ | ----------: | ---------: |
| tiled + consensus stitch             |      96.6 % |      2.7 % |
| + MCF secondary net                  |      97.5 % |      1.7 % |
| + coarse region-refine               |      99.2 % |     0.21 % |
| + global coarse anchor               |     99.63 % |     0.00 % |
| **+ multi-scale cascade (f=16,8,4)** | **99.89 %** | **0.00 %** |
| + feathered seam composite           |     99.79 % |     0.00 % |

**A strong intermediate at 99.2 %; the final winning path** adds a **global
coarse anchor** (snaps each region's integer cycle level to a seam-free
multilooked whole-image solve → reaches the no-seam wrong islands the relative
vote misses; 99.2→99.63, kills the multi-cycle streak), a **multi-scale
cascade** (`coarse_refine` at f=16,8,4 → 99.63→99.89), and a **feathered seam
composite** (blends overlaps to erase tile-seam lines - trades 0.10 % mainland
K-match to cut seam tears 3–5x; 99.89→99.79). All visually match SNAPHU 9x9.
Full method + figures: [`report_anchor_cascade.md`](report_anchor_cascade.md).

## Where the code lives

* `crates/whirlwind-core/src/tile.rs`: `unwrap_tiled`, `unwrap_one_tile_coh`,
  `stitching_offset_coh` (mode + confidence), Prim seed + consensus voting.
* `crates/whirlwind-py/src/lib.rs`: `unwrap(..., tile_size, tile_overlap)`.
* `scripts/phass_experiments/`: `run_atlanta.py` (modes incl. `tiled`,
  `goldstein`), `analyze_atlanta.py` (K-match vs OPERA, global + per-cc),
  `plot_atlanta.py`, `plot_nisar_tiled.py`.

## Decision rule (updated session 3)

**The shippable no-Goldstein unwrap exists now: tiled + global coarse anchor +
multi-scale cascade + feathered composite = 99.79 % K-match, 0 % multi-cycle,
3.9 s on NISAR** (default path, committed e24e0ed / 8aa7a1d). For noisy /
moderate-coherence scenes pass `multilook=L` (Atlanta S-1 → 97.7 %). The
secondary reconciliation network was built (`reconcile_offsets_mcf`) and the
coarse anchor + cascade superseded the planned per-region secondary MCF on the
critical path. Goldstein α=0.7 is now a **legacy/alternative**, not the
recommended default.

**Convex cost is a longer-term lever, NOT the current fix.** Its solver is
unsound (no Bellman-Ford; negative marginal costs routed through plain
Dijkstra - see Corrections #2). The general win it *could* bring is letting the
*fine* per-tile solve survive noisy phase without pre-multilooking (so noisy
scenes wouldn't need `multilook=`). Remaining levers: expose/auto-tune
`multilook`; convex/statistical per-tile cost; finer seam healing.

## Stage 3 (warm-started full-image reoptimize): why it was harder than 50 LOC

(Folded in from the former `docs/TILING_DESIGN.md` so the dead-ends aren't
re-explored. Stage 3 = SNAPHU's `-S`: warm-start the primal-dual over the full
graph from the tiled flow - single-piece memory, short wall time because the
seed is near-optimal.) The 50-LOC estimate missed two interacting constraints:

1. **Unit-capacity model.** Each forward grid arc has capacity 1; reverse
   residual arcs have cost `-c`. Once a forward arc is saturated by a warm
   start, its residual reverse becomes available with negative cost.
2. **Dial Dijkstra requires non-negative reduced costs.** With the default zero
   potentials, a saturated warm-start arc immediately violates this
   (`c - π[tail] + π[head] = -c < 0` on the residual reverse) and
   `primal_dual::run` asserts on the first iteration.

Two workarounds were prototyped (both since closed):

- **(G) Don't saturate the bitvec.** Apply only the divergence of the warm-start
  flow to `excess`; PD then routes the residual imbalance on the standard
  residual graph (no negative reduced costs). *Feasible* (`div(init + corr) =
  residue`, round-trips exactly through PD) *but not equivalent to a true
  warm-start*: when two stitching errors' divergence pairs match up *across*
  stitches, PD routes flow through a corridor and every edge along it gets its
  cycle count shifted by 1, leaving a 2π error in some region. (Exposed on a 64²
  ramp with 4x4 tiles: the median stitch left 6 source/sink pairs; PD balanced
  all but one, which routed non-locally → unwrap 2π off at one pixel vs
  non-tiled.)
- **(C) Saturate the bitvec *and* recompute potentials by SPFA.** Set `π = -d`
  (shortest-path distance in the residual graph). SPFA converges given no
  negative cycles - but the warm-start flow from a median-stitched unwrap
  *contained* negative residual cycles (SPFA detected one immediately): the seed
  flow can be cost-reduced by cancelling a cycle. The standard fix is **Klein's
  cycle-cancelling algorithm**, itself non-trivial and competitive in cost with
  just calling `unwrap_crlb` from scratch on the full image.

**Recommendation.** A working "polish" without Klein cycle-cancellation is to
**not warm-start at all**: tile (cheap memory, fast feedback), and for users who
want guaranteed equivalence with non-tiled, call `unwrap_crlb` directly on the
full image (no memory benefit, trivial correctness). A genuine warm-start polish
that recovers `unwrap_crlb`'s answer faster than `unwrap_crlb` itself appears to
require either Stage 2 (region-grow on conncomps + secondary MCF on a coarsened
graph) or Klein + Bellman-Ford.
