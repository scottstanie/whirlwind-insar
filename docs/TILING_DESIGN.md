# Tiling for whirlwind-rs — design + artifact analysis

> **Status:** Tiling is not implemented. The single-piece solver is already
> parallelized internally (cost build, residue compute, potential update,
> max-reduced-cost scan; see `docs/PERFORMANCE.md`), which covers most of
> the speedup tiling would give on small-to-medium scenes. The naive
> overlap-median stitch (stage 1 below) has known failure modes on rugged
> scenes (Chen & Zebker 2002, Fig 7), so any future tiling effort should
> probably skip straight to stage 2 (per-region MCF) — at which point
> stage 1 below is mainly useful as background on why stage 2 is needed.

## Why we need it

`unwrap()` is single-pass single-piece. The current scaling (per `docs/PERFORMANCE.md`) is **~115 bytes/pixel** working set and **O(R)** per Dijkstra where R is the residue count. For a full Sentinel-1 IW frame (~100 Mpx) that's:

- **~11.5 GiB RAM** — at the edge of typical single-machine memory.
- **~10 minutes** end-to-end wall time on residue-dense scenes.

Both are addressed by partitioning the image into independently-unwrapped tiles, processing them in parallel via `rayon`, then stitching.

## Why naive tiling is dangerous

Most engineers' first instinct is "just unwrap each tile separately and add a 2π offset per tile to align them at the seams." This works exactly when:

- Every tile is unwrapped to a *globally consistent* shape internally (i.e., the tile's interior has at most one connected component of valid pixels).
- The seams have enough valid, well-correlated pixels to estimate the integer-cycle offset reliably.

It fails — sometimes catastrophically — when a phase discontinuity (e.g., a decorrelated mountain layover front, a coastline, a shadow region) creates an **internally-isolated region** inside a tile. That region's unwrapping is off by ±2πk relative to the rest of its own tile, and **no single per-tile offset** can fix it. The error then propagates outward into neighboring tiles via the median-offset stitch and contaminates the global solution. Chen & Zebker (2002) document this clearly in Figs. 7 and 8 of their paper, with rugged Alaska topography as the canonical pathological case.

A correctness analysis I'll keep in mind throughout this doc:

> If every tile is internally correct, simple stitching produces a globally-correct solution. If even one tile is internally wrong, simple stitching cannot recover.

## How SNAPHU solves it

`Phase Unwrapping for Large SAR Interferograms: Statistical Segmentation and Generalized Network Models` (Chen & Zebker 2002) is the reference. Their full algorithm:

1. **Primary unwrap** each tile independently (their unwrapper = SNAPHU's nonlinear MCF; ours = whirlwind primal-dual SSP).
2. **Region-grow inside each tile** using the *minimum incremental cost* per arc:
   `c_s = min(g(χ₀+1), g(χ₀−1))`  (eq. 4 of their paper).
   Add neighbouring pixels to a region while reachable arcs have `c_s` below threshold. Enforce a minimum region size; merge tiny regions into their lowest-cost neighbour.
3. **Build a secondary network**:
   - Secondary nodes = region corners (where ≥ 3 regions meet) and tile corners.
   - Secondary arcs = boundary segments between adjacent regions; one arc per boundary.
   - Secondary arc cost = sum of incremental costs of the primary arcs the secondary arc traces (eq. 5). This **preserves the MAP-estimation framework** — the secondary problem is mathematically a coarsened version of the primary problem, not an ad-hoc heuristic.
4. **Initialize secondary flows** by aligning tiles top-to-bottom + left-to-right (trivial path-integration analog).
5. **Solve the secondary MCF** with the same nonlinear-MCF solver as the primary.
6. **Integrate** to recover the unwrapped phase.
7. **Optional** (`-S` flag): re-run the unwrapper on the *full* image using the tiled solution as the initial flow. This is essentially "polish using full-resolution single-piece, but warm-started." Memory-wise it's no win, but wall-time-wise it's fast because the initial flow is near-optimal.

The crucial part is **step 2** — without per-region offsets, an internally-broken tile cannot be fixed at the secondary stage.

## Our staged plan

Three stages, each producing a working unwrapper. Stage 1 is what we'll actually implement first; 2 and 3 are deferred but designed in such that we don't paint ourselves into a corner.

### Stage 1: tile + overlapping median stitch (with conflict diagnostics)

The simple thing. Useful in practice on most scenes, and gives us a baseline to measure the value of Stage 2 against.

```text
shape (m, n), tile_size T (default 1024), overlap O (default 64).
n_tiles_r = ceil((m - O) / (T - O))
n_tiles_c = ceil((n - O) / (T - O))

for each (tr, tc) in parallel via rayon:
    crop tile rect [tr*(T-O) : tr*(T-O)+T,  tc*(T-O) : tc*(T-O)+T]
    unw[tr, tc] = unwrap_single(...)         # our existing pipeline

# Stitch row 0 left-to-right.
offsets[0, 0] = 0
for tc in 1..n_tiles_c:
    overlap = pixels shared between tile (0, tc-1) and (0, tc)
    diffs = (right_unw - left_unw)[overlap]    # in radians
    cycles = round(diffs / 2π)
    offset = 2π * mode(cycles)                  # most-common integer offset
    confidence = fraction of cycles that equal mode(cycles)
    offsets[0, tc] = offsets[0, tc-1] - offset

# Same for each column, starting from row 0 of that column.
# Apply offsets and stitch.
```

**Three diagnostics we expose:**

1. `confidence[tr, tc]` — fraction of overlap pixels agreeing with the modal cycle. If < 0.7, the boundary is ambiguous (heavy noise or an internal discontinuity inside the tile). Surface this in the result struct so the user knows when to distrust a region.
2. `n_residues_per_tile` — high counts hint at the tile being internally broken.
3. `conflict_check` — for any 4-tile corner, the four pairwise offsets should sum to 0 (mod 2π). If not, log the corner.

**What stage 1 gets right:**
- Flat-to-rolling terrain with high coherence (most Sentinel-1 over agriculture / urban).
- Linear scaling: peak RAM ≈ `K · (T+O)² · 115 bytes` where K is rayon thread count. For T=1024, K=8: ≈ 1 GiB peak vs ~11.5 GiB single-piece on a Sentinel-1 IW frame.
- Wall time: ≈ `(num_tiles / K) · per_tile_time`. On the same frame, ≈ 30 s instead of 600 s.

**What stage 1 gets wrong (honest version):**
- Rugged terrain (Alaska, Andes, Himalayas) where layover/shadow creates isolated regions inside tiles.
- Large water bodies or urban shadow zones straddling boundaries — the median offset is estimated over no-coherence pixels and is garbage.
- Phase fringes that are denser than the overlap width.

For these failure modes, the diagnostic output will *show* the failure (low confidence, residue hotspots) but the algorithm won't fix it. That's stage 2's job.

### Stage 2: Carballo region-grow + secondary MCF (deferred)

This is the actual port of Chen-Zebker 2002 §III–IV. Pseudocode is in their paper and the SNAPHU source (`snaphu_tile.c`). Key implementation points for our codebase:

- Region growing uses our cost arrays directly: `c_s[arc] = min(cost_fwd[arc], cost_fwd[transpose(arc)])`. We already compute these in `cost::compute_carballo_costs`.
- The secondary network is non-grid (arbitrary topology). Our `RectangularGridGraph` won't work for it; we'd need a generic `CsrGraph`. New code.
- The secondary MCF uses the **same primal-dual SSP loop** with the same Dial Dijkstra. Reuse, don't reimplement.
- Secondary arc costs need to follow eq. 5: sum of primary incremental costs along the traced boundary. We track this during region-growth.

The amount of new code is moderate (~600 LOC). Stage 2 is what the user runs when stage-1 diagnostics show low confidence. The two stages can share most of the codebase.

### Stage 3: single-tile reoptimize (further deferred)

SNAPHU's `-S` mode. Once we have a tiled solution as the initial flow, run the primal-dual once over the *full* graph starting from that flow. Memory is back to single-piece, but wall time is short because the initial flow is near-optimal.

Implementation requirement: `primal_dual::run` needs to accept a *starting flow* (currently it always starts from zero flow). One field added to `Network` (initial saturation bitmap from the tiled result) plus a few lines to skip the "all forward unsaturated" initialization. Maybe 50 LOC.

## Detailed plan for stage 1

```text
// crates/whirlwind-core/src/tile.rs

pub struct TileConfig {
    pub tile_size: usize,        // e.g. 1024
    pub overlap: usize,          // e.g. 64
    pub nthreads: Option<usize>, // None = use rayon's global pool
    pub min_confidence: f32,     // e.g. 0.7 — flag low-confidence seams
}

pub struct TileResult {
    pub unwrapped: Array2<f32>,
    pub confidence: Array2<f32>,        // (n_tile_rows, n_tile_cols-1) + (n_tile_rows-1, n_tile_cols)
    pub n_residues_per_tile: Array2<u32>,
    pub flagged_corners: Vec<(usize, usize)>,
}

pub fn unwrap_tiled(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    cfg: TileConfig,
) -> Result<TileResult, UnwrapError>;
```

Implementation order, smallest-step-first:

1. `tile_rects(m, n, T, O) -> Vec<TileRect>` — pure function, easy to test.
2. Single-threaded unwrap-each-tile loop. No stitching yet. Verify each tile unwraps independently.
3. **Horizontal stitch** along row 0: median offset of column-overlap region. Test with the 1024² diagonal ramp split into a 2×2 tile grid — must reproduce the single-piece result to atol=1e-2.
4. Generalize to all rows / all columns. Verify on 1024² → 2×2, 4×4, 8×8 splits.
5. Rayon parallelism on the per-tile unwrap (the stitch is sequential).
6. Diagnostics: confidence + residue count + corner-cycle check.
7. Bench: add a `tiled_1024_split_4x4` row in `examples/bench_scale.rs` and compare wall time + peak RSS to single-piece.
8. Real-data run: re-run `scripts/run_real_data.py` with tiled vs single-piece on Palos Verdes; PNG-diff the two unwraps; confirm any differences are within 2π (mod 2π) of each other.

## Verification gates

The plan is wrong if any of these fail; they're cheap to check:

| Gate | Check | If it fails |
|---|---|---|
| Clean diagonal ramp tiled vs single-piece | `max(|unw_tiled - unw_single - 2πk|) < 1e-2` for some integer k | Stitching is broken; debug median calculation |
| Noisy gaussian bump 1024² with γ=0.85, nlooks=10 | Same agreement as above within 1e-2 | Stage-1 limits reached even on synthetic data — should not happen for γ=0.85 |
| Palos Verdes (real, γ=0.9 median, no layover) | Visual diff vs single-piece is uniform 2π | Likely a stitch sign error |
| Rosamond (real, γ=0.21 median, sparse coverage) | Diagnostics flag low-confidence boundaries | This is the expected stage-1 failure mode — should *not* be silent |

The third gate is the important one. If stage 1 quietly produces broken output on rugged scenes, we shouldn't ship it; we should ship it *with the diagnostic that flags the failure*. That's the difference between "naive tiling that lies" and "tiling with honest scope."

## Comparison vs SNAPHU's tiling

| Aspect | SNAPHU | whirlwind-rs stage 1 | whirlwind-rs stage 2 (planned) |
|---|---|---|---|
| Primary tile unwrap | nonlinear MCF | our primal-dual SSP | same |
| Reassembly granularity | per arbitrarily-shaped region | per whole tile | per region (port of SNAPHU) |
| Reassembly cost model | statistical (eq. 5) | mode of integer overlap diffs | statistical (eq. 5) |
| Handles isolated-region tiles | yes | no — flagged but not fixed | yes |
| Optional full-image polish | `-S` flag | not planned | stage 3 |
| Code complexity | high (~2K LOC `snaphu_tile.c`) | low (~300 LOC) | moderate (+600 LOC over stage 1) |
| Expected speedup on Sentinel-1 frame | ~5× wall, fits memory | ~10× wall, fits memory | ~7× wall, fits memory |

## Risks and unknowns

- **Stage 1 will produce visible artifacts on Alaska-style scenes.** We accept this, surface it via diagnostics, and let users opt in to stage 2 when needed. Stage 1 is intended for the 80 % of routine InSAR work that isn't pathological.
- **Mode-of-cycle stitch is not robust to bimodal overlap distributions.** If the overlap region happens to be split by a wrap-line that's within the overlap window, modes can flip-flop. Mitigation: pick the median over a sufficiently large overlap (≥ 64 px), and the diagnostic catches the case when no clear mode exists.
- **Memory at very-large scale.** Even tiled, a 100-Mpx Sentinel-1 frame holds the *output array* in memory. That's another 400 MiB. Fine on a workstation, painful on small VMs. Not addressed here.
- **Connected-component handling.** SNAPHU writes a connected-components label image and treats large invalid regions as their own components. We don't have a conncomp output yet; stage 1's `confidence` array is its substitute, but a real conncomp output is a separate piece of work that's orthogonal to tiling.

## Recommendation

Implement **stage 1** with diagnostics. Re-run `scripts/bench.py` on real data with both tiled and single-piece configurations; if the diagnostics flag a high-residue / low-confidence boundary on a scene that matters for the paper, implement stage 2 for that scene specifically. Stage 3 is academically nice but not on the critical path — `primal_dual::run` accepting an initial flow is a small change we can do later if we ever need it.
