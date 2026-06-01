//! Tiled 2D unwrap: split a large image into overlapping tiles, unwrap each
//! tile independently, then stitch by reconciling per-tile integer ambiguity
//! offsets in the overlap regions.
//!
//! Why this matches non-tiled output in coherent regions: per-tile MCF picks
//! its own integer ambiguity for the wrap-line endpoints, but the overlap
//! between two adjacent tiles is the same patch of phase data — if both
//! tiles unwrapped it well, the per-pixel difference is exactly an integer
//! multiple of 2π (per-IG global offset between the two tiles). Taking the
//! CRLB-weighted median of that difference and rounding to a 2π multiple
//! gives the correct stitching offset.
//!
//! Failure mode: in overlap regions that are heavily decorrelated, both
//! tiles' per-IG unwrap is unreliable, the per-pixel difference is noisy,
//! and the median may snap to the wrong 2π multiple. Acceptable per spec —
//! those pixels aren't trustworthy in either tiled or non-tiled output.

use crate::cost;
use crate::grid::RectangularGridGraph;
use crate::integrate;
use crate::network::Network;
use crate::primal_dual;
use crate::residue;
use crate::UnwrapError;
use ndarray::{s, Array2, ArrayView2};
use num_complex::Complex32;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::f32::consts::TAU;

/// One tile's bounds in the parent image, in row/col indices.
#[derive(Debug, Clone, Copy)]
pub struct Tile {
    pub r0: usize,
    pub r1: usize, // exclusive
    pub c0: usize,
    pub c1: usize, // exclusive
}

impl Tile {
    pub fn rows(&self) -> usize {
        self.r1 - self.r0
    }
    pub fn cols(&self) -> usize {
        self.c1 - self.c0
    }
}

/// Decompose an `m × n` image into a regular grid of tiles of size up to
/// `tile_size × tile_size` with `overlap` overlap between adjacent tiles.
/// Tiles on the right / bottom edge are smaller if `m`/`n` don't divide
/// evenly; every tile has at least `min(tile_size, m)` rows and similarly
/// for columns (no degenerate tiny tiles).
pub fn decompose(m: usize, n: usize, tile_size: usize, overlap: usize) -> Vec<Tile> {
    assert!(tile_size >= 4, "tile_size must be ≥ 4");
    assert!(overlap < tile_size, "overlap must be < tile_size");
    let step = tile_size - overlap;

    // Generate row starts: 0, step, 2*step, ... last one set so r1 = m.
    let row_starts = axis_starts(m, tile_size, step);
    let col_starts = axis_starts(n, tile_size, step);

    let mut tiles = Vec::with_capacity(row_starts.len() * col_starts.len());
    for &r0 in &row_starts {
        let r1 = (r0 + tile_size).min(m);
        for &c0 in &col_starts {
            let c1 = (c0 + tile_size).min(n);
            tiles.push(Tile { r0, r1, c0, c1 });
        }
    }
    tiles
}

fn axis_starts(total: usize, tile_size: usize, step: usize) -> Vec<usize> {
    if total <= tile_size {
        return vec![0];
    }
    let mut starts = vec![0_usize];
    loop {
        let last = *starts.last().unwrap();
        let next = last + step;
        if next + tile_size >= total {
            // Set the last start so the tile lands exactly on the image edge.
            starts.push(total - tile_size);
            return starts;
        }
        starts.push(next);
    }
}

/// Layout of a tile grid: tiles indexed by (row, col); neighbour relations.
struct TileGrid {
    tiles: Vec<Tile>,
    /// (rows, cols) of the tile grid (not the pixel image).
    grid_rows: usize,
    grid_cols: usize,
}

impl TileGrid {
    fn from_decomposition(m: usize, n: usize, tile_size: usize, overlap: usize) -> Self {
        let step = tile_size - overlap;
        let n_rows = axis_starts(m, tile_size, step).len();
        let n_cols = axis_starts(n, tile_size, step).len();
        let tiles = decompose(m, n, tile_size, overlap);
        assert_eq!(tiles.len(), n_rows * n_cols);
        TileGrid {
            tiles,
            grid_rows: n_rows,
            grid_cols: n_cols,
        }
    }
    fn index_of(&self, gr: usize, gc: usize) -> usize {
        gr * self.grid_cols + gc
    }
    fn rc_of(&self, idx: usize) -> (usize, usize) {
        (idx / self.grid_cols, idx % self.grid_cols)
    }
    /// Adjacent tile indices in (right, down) order, or None if at the edge.
    fn neighbours(&self, idx: usize) -> [Option<usize>; 4] {
        let (gr, gc) = self.rc_of(idx);
        let right = if gc + 1 < self.grid_cols {
            Some(self.index_of(gr, gc + 1))
        } else {
            None
        };
        let down = if gr + 1 < self.grid_rows {
            Some(self.index_of(gr + 1, gc))
        } else {
            None
        };
        let left = if gc > 0 {
            Some(self.index_of(gr, gc - 1))
        } else {
            None
        };
        let up = if gr > 0 {
            Some(self.index_of(gr - 1, gc))
        } else {
            None
        };
        [right, down, left, up]
    }
}

/// Tiled CRLB-weighted 2D unwrap. See module docs.
pub fn unwrap_crlb_tiled(
    igram: ArrayView2<Complex32>,
    variance: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    tile_size: usize,
    overlap: usize,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != variance.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), variance.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    // If the image fits in one tile, fall back to the single-tile path.
    if tile_size >= m && tile_size >= n {
        return crate::unwrap_crlb(igram, variance, mask);
    }
    assert!(overlap >= 2, "overlap must be ≥ 2 for median-based stitching");

    let grid = TileGrid::from_decomposition(m, n, tile_size, overlap);
    let n_tiles = grid.tiles.len();

    // 1) Unwrap each tile in parallel.
    let tile_unws: Vec<Result<Array2<f32>, UnwrapError>> = grid
        .tiles
        .par_iter()
        .map(|t| unwrap_one_tile(igram, variance, mask, t))
        .collect();
    let tile_unws: Vec<Array2<f32>> = tile_unws
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;

    // 2) BFS over tiles, accumulating per-tile integer-2π offsets relative
    //    to the seed tile (index 0).
    let mut offsets_2pi: Vec<i64> = vec![0; n_tiles];
    let mut visited = vec![false; n_tiles];
    visited[0] = true;
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(0);
    while let Some(idx) = q.pop_front() {
        for nb in grid.neighbours(idx).into_iter().flatten() {
            if visited[nb] {
                continue;
            }
            let k = stitching_offset(
                &grid.tiles[idx],
                &tile_unws[idx],
                &grid.tiles[nb],
                &tile_unws[nb],
                variance,
            );
            // unw[nb] + 2π·offset_nb ≡ unw[idx] + 2π·offset_idx in overlap
            //   ⇒ offset_nb = offset_idx − k
            offsets_2pi[nb] = offsets_2pi[idx] - k;
            visited[nb] = true;
            q.push_back(nb);
        }
    }

    // 3) Composite: write the *core* of each tile (its full extent, minus
    //    overlap with already-written neighbours) into the output. Each
    //    pixel ends up coming from exactly one tile — the first one to
    //    claim it in BFS order. For an inner pixel that's covered by 4
    //    overlapping tiles, we deterministically pick the top-leftmost.
    let mut out = Array2::<f32>::from_elem((m, n), f32::NAN);
    for (idx, (tile, unw)) in grid.tiles.iter().zip(tile_unws.iter()).enumerate() {
        let off = offsets_2pi[idx] as f32 * TAU;
        for ti in 0..tile.rows() {
            let gi = tile.r0 + ti;
            for tj in 0..tile.cols() {
                let gj = tile.c0 + tj;
                if out[(gi, gj)].is_nan() {
                    out[(gi, gj)] = unw[(ti, tj)] + off;
                }
            }
        }
    }

    Ok(out)
}

/// Tiled coherence-cost 2D unwrap (Carballo cost), the coherence twin of
/// [`unwrap_crlb_tiled`]. Split into overlapping tiles, unwrap each tile
/// independently in parallel, then BFS-stitch by a coherence-weighted
/// overlap-median 2π reconciliation.
///
/// This is the memory- and robustness-motivated path: per-tile MCF keeps
/// flow local (a misrouted residue can't accumulate cycle errors across the
/// whole frame the way a single whole-image solve does), and peak memory is
/// bounded by the tile size rather than the full scene.
pub fn unwrap_tiled(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    tile_size: usize,
    overlap: usize,
    multilook: usize,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != corr.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), corr.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    // MULTILOOK-FIRST (for noisy / moderate-coherence scenes, e.g. Sentinel-1).
    // whirlwind's linear cost mis-routes through noisy phase; coherently
    // down-looking ×`multilook` suppresses that noise, after which the SAME
    // tiled+anchor+cascade pipeline reaches SNAPHU quality. We tile the coarse
    // (a whole-image coarse solve still has residual runaway), then upsample.
    if multilook > 1 {
        let (cig, ccorr, cmask) = multilook_complex(igram, corr, mask, multilook);
        let (cm, cn) = cig.dim();
        // Small coarse tiles bound the runaway (validated: whole-image coarse
        // ≈83% vs tiled-coarse ≈97.7% on Atlanta). 1 ⇒ no further multilook.
        let cts = 128.min(cm).min(cn);
        let cov = (cts / 4).max(2);
        let coarse = unwrap_tiled(
            cig.view(), ccorr.view(), nlooks * (multilook * multilook) as f32,
            Some(cmask.view()), cts, cov, 1,
        )?;
        return Ok(upsample_blockrep(&coarse, multilook, m, n));
    }
    if tile_size >= m && tile_size >= n {
        return crate::unwrap(igram, corr, nlooks, mask);
    }
    assert!(overlap >= 2, "overlap must be ≥ 2 for median-based stitching");

    let grid = TileGrid::from_decomposition(m, n, tile_size, overlap);
    let _n_tiles = grid.tiles.len();

    // 1) Unwrap each tile in parallel.
    let tile_unws: Vec<Result<Array2<f32>, UnwrapError>> = grid
        .tiles
        .par_iter()
        .map(|t| unwrap_one_tile_coh(igram, corr, nlooks, mask, t))
        .collect();
    let tile_unws: Vec<Array2<f32>> =
        tile_unws.into_iter().collect::<Result<Vec<_>, _>>()?;

    // 2) Reconcile per-tile integer-2π offsets by a GLOBAL min-cost-flow
    //    secondary network (SNAPHU's `AssembleTiles` idea at tile scale).
    //    Measure each adjacent-pair seam gradient, then solve a min-cost
    //    tension on the tile grid so a coherent low-confidence wrong island
    //    is flipped when that lowers the total weighted seam cost — which
    //    per-tile/region heuristics cannot do (they can't break a satisfied
    //    seam). `gh`/`gv` are the measured offset gradients; `wh`/`wv` their
    //    coherence-weighted confidences.
    let (rows, cols) = (grid.grid_rows, grid.grid_cols);
    let mut gh = vec![0_i64; rows * cols.saturating_sub(1)];
    let mut wh = vec![0_i64; rows * cols.saturating_sub(1)];
    let mut gv = vec![0_i64; rows.saturating_sub(1) * cols];
    let mut wv = vec![0_i64; rows.saturating_sub(1) * cols];
    for gr in 0..rows {
        for gc in 0..cols {
            let idx = grid.index_of(gr, gc);
            if gc + 1 < cols {
                let nb = grid.index_of(gr, gc + 1);
                let (k, w) = stitching_offset_coh(
                    &grid.tiles[idx], &tile_unws[idx], &grid.tiles[nb], &tile_unws[nb], corr,
                );
                // stitch returns k with off_nb = off_idx − k ⇒ gh = off_nb − off_idx = −k
                gh[gr * (cols - 1) + gc] = -k;
                wh[gr * (cols - 1) + gc] = w;
            }
            if gr + 1 < rows {
                let nb = grid.index_of(gr + 1, gc);
                let (k, w) = stitching_offset_coh(
                    &grid.tiles[idx], &tile_unws[idx], &grid.tiles[nb], &tile_unws[nb], corr,
                );
                gv[gr * cols + gc] = -k;
                wv[gr * cols + gc] = w;
            }
        }
    }
    let offsets_2pi = reconcile_offsets_mcf(rows, cols, &gh, &wh, &gv, &wv);

    // 3) FEATHERED composite. A hard "first-tile-wins" switch leaves a faint
    //    1-px seam line wherever two tiles' (offset-aligned) values differ by a
    //    fraction of a cycle. Instead, blend overlapping tiles with a
    //    triangular taper (peaks at tile centre, ≥1 at edges) so that small
    //    disagreement is spread smoothly across the overlap. A genuine 2π tear
    //    must NOT be averaged into a half-cycle, so we gate: pass 1 finds each
    //    pixel's dominant (max-weight) tile value as a reference; pass 2
    //    weighted-averages only tiles whose value agrees within π of it.
    let taper = |p: usize, len: usize| -> f32 { (p + 1).min(len - p) as f32 };
    let mut refv = Array2::<f32>::from_elem((m, n), f32::NAN);
    let mut refw = Array2::<f32>::zeros((m, n));
    for (idx, (tile, unw)) in grid.tiles.iter().zip(tile_unws.iter()).enumerate() {
        let off = offsets_2pi[idx] as f32 * TAU;
        let (tr, tc) = (tile.rows(), tile.cols());
        for ti in 0..tr {
            let gi = tile.r0 + ti;
            let wr = taper(ti, tr);
            for tj in 0..tc {
                let v = unw[(ti, tj)];
                if !v.is_finite() {
                    continue;
                }
                let w = wr * taper(tj, tc);
                let gj = tile.c0 + tj;
                if w > refw[(gi, gj)] {
                    refw[(gi, gj)] = w;
                    refv[(gi, gj)] = v + off;
                }
            }
        }
    }
    let mut acc = Array2::<f32>::zeros((m, n));
    let mut wsum = Array2::<f32>::zeros((m, n));
    for (idx, (tile, unw)) in grid.tiles.iter().zip(tile_unws.iter()).enumerate() {
        let off = offsets_2pi[idx] as f32 * TAU;
        let (tr, tc) = (tile.rows(), tile.cols());
        for ti in 0..tr {
            let gi = tile.r0 + ti;
            let wr = taper(ti, tr);
            for tj in 0..tc {
                let v = unw[(ti, tj)];
                if !v.is_finite() {
                    continue;
                }
                let gj = tile.c0 + tj;
                let val = v + off;
                if (val - refv[(gi, gj)]).abs() < TAU * 0.5 {
                    let w = wr * taper(tj, tc);
                    acc[(gi, gj)] += w * val;
                    wsum[(gi, gj)] += w;
                }
            }
        }
    }
    let mut out = Array2::<f32>::from_elem((m, n), f32::NAN);
    for gi in 0..m {
        for gj in 0..n {
            if wsum[(gi, gj)] > 0.0 {
                out[(gi, gj)] = acc[(gi, gj)] / wsum[(gi, gj)];
            }
        }
    }

    // 4) Coarse region-refinement: remove residual whole-region 2π-offset
    //    artifacts (rectangular blocks bounded by high-coherence 2π rings)
    //    that the per-tile MCF / seam reconciliation leaves behind. A
    //    globally-consistent coarse anchor (one seam-free whole-image solve on
    //    a heavily-multilooked copy) pins each region's integer cycle level —
    //    this reaches low-coherence "wrong islands" that share no
    //    high-confidence seam with the mainland, which the relative
    //    largest-region vote cannot. `.ok()` ⇒ a degenerate coarse solve falls
    //    back to the anchorless vote rather than failing the whole unwrap.
    // `WHIRLWIND_NO_ANCHOR=1` disables the global anchor (falls back to the
    // relative largest-region vote) — for before/after comparison only.
    // Default: build the global coarse anchor and refine coarse→fine
    // (f=16,8,4). The coarse→fine cascade re-anchors at each scale, so a wrong
    // block fragmented at one scale is caught whole at a coarser one, and
    // region boundaries resolve below the single-pass 8-px granularity.
    // `WHIRLWIND_NO_ANCHOR=1` reverts to the old single-f=8, anchorless vote
    // (for before/after comparison).
    if std::env::var("WHIRLWIND_NO_ANCHOR").is_ok() {
        coarse_refine(&mut out, corr, mask, 8, None);
    } else {
        let anchor = compute_coarse_anchor(igram, corr, nlooks, mask, 8);
        let av = anchor.as_ref().map(|a| a.view());
        for &f in &[16usize, 8, 4] {
            coarse_refine(&mut out, corr, mask, f, av);
        }
    }

    // 5) Heal residual thin MCF "sliver" artifacts: a ≤4-px vertical/horizontal
    //    run the per-tile residue-pairing left a constant integer #cycles off
    //    from a surround that AGREES on both sides (a tie-break in noise, not a
    //    cost-shape error — present under linear and convex costs alike, so no
    //    cost removes it). Bounded + coherence-gated; cannot fire across a real
    //    fringe (its two sides sit at different levels). `WHIRLWIND_NO_HEAL=1`
    //    disables it.
    if std::env::var("WHIRLWIND_NO_HEAL").is_err() {
        heal_thin_slivers(&mut out, corr, mask, 0.2, 4, 6);
    }
    Ok(out)
}

/// Coherence above which a branch cut counts as "through coherent terrain".
const COH_CUT_THR: f32 = 0.7;
/// Coherent-cut rate (coherence-weighted cuts per valid pixel) above which the
/// gated multi-shift re-solve fires. Empirically the fragmented GUNW A_016 sits
/// at ≈6.7e-3 while every clean / noisy-but-fine scene (NISAR, Atlanta, the four
/// clean GUNW frames) is ≤5.6e-4 — a >3× margin on each side.
const COH_CUT_FLOOR: f64 = 1.5e-3;

#[inline]
fn wrap_to_pi(d: f32) -> f32 {
    d - TAU * (d / TAU).round()
}

/// Rate of branch cuts that pass through HIGH-coherence pixels, per valid pixel.
///
/// A correct unwrap never tears coherent terrain, so a significant rate is the
/// signature of a tile-seam artifact or a wrong global winding. For each 4-neighbour
/// arc with min endpoint coherence > `coh_thr`, the integer flow is
/// `round((Δunw − wrap(Δφ)) / 2π)`; we sum `|flow|·coherence` over those arcs and
/// divide by the valid-pixel count. `φ = arg(igram)`.
fn coherent_cut_rate(
    igram: ArrayView2<Complex32>,
    unw: &Array2<f32>,
    corr: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    coh_thr: f32,
) -> f64 {
    let (m, n) = unw.dim();
    let valid = |i: usize, j: usize| {
        mask.map(|mk| mk[(i, j)]).unwrap_or(true) && unw[(i, j)].is_finite()
    };
    let mut sum = 0.0_f64;
    let mut nvalid = 0_usize;
    for i in 0..m {
        for j in 0..n {
            if !valid(i, j) {
                continue;
            }
            nvalid += 1;
            if j + 1 < n && valid(i, j + 1) {
                let c = corr[(i, j)].min(corr[(i, j + 1)]);
                if c > coh_thr {
                    let dphi = wrap_to_pi(igram[(i, j + 1)].arg() - igram[(i, j)].arg());
                    let flow = ((unw[(i, j + 1)] - unw[(i, j)] - dphi) / TAU).round().abs();
                    if flow > 0.0 {
                        sum += (flow * c) as f64;
                    }
                }
            }
            if i + 1 < m && valid(i + 1, j) {
                let c = corr[(i, j)].min(corr[(i + 1, j)]);
                if c > coh_thr {
                    let dphi = wrap_to_pi(igram[(i + 1, j)].arg() - igram[(i, j)].arg());
                    let flow = ((unw[(i + 1, j)] - unw[(i, j)] - dphi) / TAU).round().abs();
                    if flow > 0.0 {
                        sum += (flow * c) as f64;
                    }
                }
            }
        }
    }
    if nvalid == 0 {
        0.0
    } else {
        sum / nvalid as f64
    }
}

/// Gated multi-shift tiled unwrap — the default for large frames.
///
/// Runs the standard tile grid. A correct unwrap never tears coherent terrain, so
/// if the result has a high `coherent_cut_rate` (> [`COH_CUT_FLOOR`]) — the
/// signature of a tile-SEAM artifact or a wrong global WINDING on a fragmented
/// scene (e.g. NISAR GUNW A_016, a decorrelation-split frame) — it re-runs on tile
/// grids shifted by fractions of the tile step (a seam in one grid is interior in
/// another) and returns the result with the FEWEST coherent cuts. The shift is
/// realised by zero-padding the top-left by `s` (those pixels are masked out), so
/// no change to the tile decomposition is needed.
///
/// No-op (1× cost) on clean scenes (rate ≈ 0); ~4× on the rare fragmented frame
/// that needs it (speed is not the constraint there). Validated: A_016 55% → 97%
/// with the high-coherence seam-strip artifact removed, and the four clean GUNW
/// frames + NISAR + Atlanta unchanged (gate does not fire).
pub fn unwrap_tiled_robust(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    tile_size: usize,
    overlap: usize,
    multilook: usize,
) -> Result<Array2<f32>, UnwrapError> {
    let base = unwrap_tiled(igram, corr, nlooks, mask, tile_size, overlap, multilook)?;
    let (m, n) = igram.dim();
    // Only the standard tiled path has shiftable seams (multilook coarsens first;
    // a single-tile solve has no seams).
    if multilook > 1 || (tile_size >= m && tile_size >= n) {
        return Ok(base);
    }
    let rate0 = coherent_cut_rate(igram, &base, corr, mask, COH_CUT_THR);
    if rate0 <= COH_CUT_FLOOR {
        return Ok(base);
    }
    let step = tile_size - overlap;
    let mut best = base;
    let mut best_rate = rate0;
    for &s in &[step / 2, step / 4, (3 * step) / 4] {
        if s == 0 {
            continue;
        }
        let mut pig = Array2::<Complex32>::zeros((m + s, n + s));
        pig.slice_mut(s![s.., s..]).assign(&igram);
        let mut pco = Array2::<f32>::zeros((m + s, n + s));
        pco.slice_mut(s![s.., s..]).assign(&corr);
        let pmask = mask.map(|mk| {
            let mut p = Array2::<bool>::from_elem((m + s, n + s), false);
            p.slice_mut(s![s.., s..]).assign(&mk);
            p
        });
        let cand_padded = unwrap_tiled(
            pig.view(),
            pco.view(),
            nlooks,
            pmask.as_ref().map(|a| a.view()),
            tile_size,
            overlap,
            1,
        )?;
        let cand = cand_padded.slice(s![s.., s..]).to_owned();
        let r = coherent_cut_rate(igram, &cand, corr, mask, COH_CUT_THR);
        if r < best_rate {
            best_rate = r;
            best = cand;
        }
    }
    Ok(best)
}

/// A tiny successive-shortest-path min-cost-flow. Uses SPFA (Bellman-Ford
/// queue) for the shortest-path step so it tolerates the negative residual-arc
/// costs without maintaining potentials — fine because the tile graph is
/// small. Arcs are stored in forward/reverse pairs: arc `e` and `e ^ 1`.
struct Mcf {
    head: Vec<i32>,
    to: Vec<usize>,
    next: Vec<i32>,
    cap: Vec<i64>,
    cost: Vec<i64>,
}

impl Mcf {
    fn new(n: usize) -> Self {
        Mcf { head: vec![-1; n], to: Vec::new(), next: Vec::new(), cap: Vec::new(), cost: Vec::new() }
    }

    fn add_arc(&mut self, u: usize, v: usize, cap: i64, cost: i64) -> usize {
        let id = self.to.len();
        self.to.push(v);
        self.cap.push(cap);
        self.cost.push(cost);
        self.next.push(self.head[u]);
        self.head[u] = id as i32;
        self.to.push(u);
        self.cap.push(0);
        self.cost.push(-cost);
        self.next.push(self.head[v]);
        self.head[v] = (id + 1) as i32;
        id
    }

    /// Undirected seam u↔v at cost `w` (a unit of correction in either
    /// direction costs `w`). Returns `(fe, be)`; net flow u→v is
    /// `used(fe) − used(be)`.
    fn add_seam(&mut self, u: usize, v: usize, w: i64) -> (usize, usize) {
        const INF: i64 = 1 << 50;
        (self.add_arc(u, v, INF, w), self.add_arc(v, u, INF, w))
    }

    /// Flow pushed on forward arc `e` (== current cap of its reverse partner).
    fn used(&self, e: usize) -> i64 {
        self.cap[e ^ 1]
    }

    /// Balance node supplies (`> 0` source, `< 0` sink, Σ = 0) at min cost.
    fn solve(&mut self, supply: &[i64]) {
        let n = self.head.len();
        let mut sup = supply.to_vec();
        loop {
            let Some(src) = (0..n).find(|&i| sup[i] > 0) else { break };
            let mut dist = vec![i64::MAX; n];
            let mut pe = vec![-1_i32; n];
            let mut inq = vec![false; n];
            dist[src] = 0;
            let mut q = std::collections::VecDeque::new();
            q.push_back(src);
            inq[src] = true;
            while let Some(u) = q.pop_front() {
                inq[u] = false;
                let mut e = self.head[u];
                while e != -1 {
                    let ei = e as usize;
                    let v = self.to[ei];
                    if self.cap[ei] > 0
                        && dist[u] != i64::MAX
                        && dist[u] + self.cost[ei] < dist[v]
                    {
                        dist[v] = dist[u] + self.cost[ei];
                        pe[v] = ei as i32;
                        if !inq[v] {
                            q.push_back(v);
                            inq[v] = true;
                        }
                    }
                    e = self.next[ei];
                }
            }
            let mut sink = usize::MAX;
            let mut bd = i64::MAX;
            for i in 0..n {
                if sup[i] < 0 && dist[i] < bd {
                    bd = dist[i];
                    sink = i;
                }
            }
            if sink == usize::MAX {
                break; // unbalanced — shouldn't happen (Σ supply == 0, connected)
            }
            let mut f = sup[src].min(-sup[sink]);
            let mut v = sink;
            while v != src {
                let ei = pe[v] as usize;
                f = f.min(self.cap[ei]);
                v = self.to[ei ^ 1];
            }
            let mut v = sink;
            while v != src {
                let ei = pe[v] as usize;
                self.cap[ei] -= f;
                self.cap[ei ^ 1] += f;
                v = self.to[ei ^ 1];
            }
            sup[src] -= f;
            sup[sink] += f;
        }
    }
}

/// Globally reconcile per-tile integer-2π offsets by min-cost tension:
/// minimize `Σ w · |measured − (o_a − o_b)|` over integer offsets `o`,
/// solved as a residue min-cost-flow on the planar dual of the tile grid
/// (SNAPHU's `AssembleTiles` secondary network at tile scale). Unlike the
/// per-tile/region heuristics, this **can break a satisfied seam** when that
/// lowers the total weighted seam cost — the property needed to flip a
/// coherent wrong island in a low-coherence patch.
///
/// * `gh[gr*(cols-1)+gc]` = measured `o[gr][gc+1] − o[gr][gc]`, confidence `wh`.
/// * `gv[gr*cols+gc]`     = measured `o[gr+1][gc] − o[gr][gc]`, confidence `wv`.
///
/// Returns one integer offset per tile (row-major; global offset arbitrary).
fn reconcile_offsets_mcf(
    rows: usize,
    cols: usize,
    gh: &[i64],
    wh: &[i64],
    gv: &[i64],
    wv: &[i64],
) -> Vec<i64> {
    let mut o = vec![0_i64; rows * cols];
    if rows == 0 || cols == 0 {
        return o;
    }
    if rows == 1 {
        for gc in 1..cols {
            o[gc] = o[gc - 1] + gh[gc - 1];
        }
        return o;
    }
    if cols == 1 {
        for gr in 1..rows {
            o[gr] = o[gr - 1] + gv[gr - 1];
        }
        return o;
    }

    // Residue (curl) per interior face: the clockwise 2×2-tile loop with
    // top-left tile (fr, fc). Zero iff the four seam measurements close.
    let nf = (rows - 1) * (cols - 1);
    let outer = nf;
    let face = |fr: usize, fc: usize| fr * (cols - 1) + fc;
    let mut supply = vec![0_i64; nf + 1];
    for fr in 0..rows - 1 {
        for fc in 0..cols - 1 {
            supply[face(fr, fc)] = gh[fr * (cols - 1) + fc] + gv[fr * cols + (fc + 1)]
                - gh[(fr + 1) * (cols - 1) + fc]
                - gv[fr * cols + fc];
        }
    }
    supply[outer] = -supply[..nf].iter().sum::<i64>();

    // Dual graph: a node per interior face plus one outer node; one seam-edge
    // per tile seam, between the two faces it borders (boundary → outer).
    let mut mcf = Mcf::new(nf + 1);
    let mut he = vec![(0usize, 0usize); rows * (cols - 1)];
    let mut ve = vec![(0usize, 0usize); (rows - 1) * cols];
    for gr in 0..rows {
        for gc in 0..cols - 1 {
            let below = if gr < rows - 1 { face(gr, gc) } else { outer };
            let above = if gr >= 1 { face(gr - 1, gc) } else { outer };
            he[gr * (cols - 1) + gc] = mcf.add_seam(above, below, wh[gr * (cols - 1) + gc].max(1));
        }
    }
    for gr in 0..rows - 1 {
        for gc in 0..cols {
            let right = if gc < cols - 1 { face(gr, gc) } else { outer };
            let left = if gc >= 1 { face(gr, gc - 1) } else { outer };
            ve[gr * cols + gc] = mcf.add_seam(left, right, wv[gr * cols + gc].max(1));
        }
    }
    mcf.solve(&supply);

    // Apply the integer correction (net dual flow) to each seam so the field
    // is curl-free, then integrate: row 0 along `gh`, then columns down `gv`.
    let mut gh2 = gh.to_vec();
    let mut gv2 = gv.to_vec();
    // `gh` and `gv` enter the face curl with mirror-opposite signs, so their
    // corrections take opposite signs of the net dual flow (verified by the
    // `reconcile_mcf_breaks_low_confidence_wrong_seam` test).
    for (i, &(fe, be)) in he.iter().enumerate() {
        gh2[i] += mcf.used(fe) - mcf.used(be);
    }
    for (i, &(fe, be)) in ve.iter().enumerate() {
        gv2[i] -= mcf.used(fe) - mcf.used(be);
    }
    for gc in 1..cols {
        o[gc] = o[gc - 1] + gh2[gc - 1];
    }
    for gr in 1..rows {
        for gc in 0..cols {
            o[gr * cols + gc] = o[(gr - 1) * cols + gc] + gv2[(gr - 1) * cols + gc];
        }
    }
    o
}

#[inline]
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Coarse-scale region reconciliation: a noise-robust post-pass that removes
/// the residual whole-region 2π-offset artifacts the tile reconciliation
/// leaves (a sub-tile region the per-tile MCF unwrapped a few cycles off,
/// showing as a rectangular block bounded by a 2π discontinuity *ring*).
///
/// Per-pixel jump detection fragments under phase noise, so we coarsen `unw`
/// by `f`× (block mean over valid pixels — noise averages out, large-scale
/// offsets survive), group coarse pixels into regions by no-jump connectivity,
/// then shift each region by the integer that zeroes its **coherence-weighted**
/// boundary jumps (high-coherence rings are expensive → flipped away;
/// legitimate low-coherence cuts are cheap → kept). The per-region integer
/// offset (× 2π) is added back to the full-resolution `unw` in place.
/// Build a globally-consistent coarse "anchor" unwrap to pin per-region cycle
/// levels to. Multilook the COMPLEX igram by `lk`× (coherent down-look — never
/// average wrapped phase, which is meaningless across 2π), unwrap the tiny
/// coarse image in ONE whole-image solve (no tiles ⇒ no seams ⇒ one
/// self-consistent surface), and block-replicate it back to full resolution.
///
/// Down-looking by `lk` multiplies the effective looks by `lk²`, so coarse
/// coherence is far higher than the full-res input — the coarse solve is
/// reliable and free of the long-distance runaway that corrupts a full-res
/// whole-image solve. The anchor is consumed only to choose each region's
/// INTEGER 2π level (via a coherence-weighted mode over the whole region), so a
/// sub-cycle smoothing error in the anchor does not propagate, and a local
/// anchor error is outvoted by the rest of its region.
/// Coherent ×`lk` down-look of the complex igram: unit-phasor block mean (the
/// physically-correct coherent average — never average wrapped phase across
/// 2π), block-mean coherence, and validity = a majority of the block valid.
/// Suppresses noise and re-estimates phase; effective looks scale by `lk²`.
pub(crate) fn multilook_complex(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    lk: usize,
) -> (Array2<Complex32>, Array2<f32>, Array2<bool>) {
    let (m, n) = igram.dim();
    let (cm, cn) = (m / lk, n / lk);
    let mut cig = Array2::<Complex32>::zeros((cm, cn));
    let mut ccorr = Array2::<f32>::zeros((cm, cn));
    let mut cmask = Array2::<bool>::from_elem((cm, cn), false);
    for ci in 0..cm {
        for cj in 0..cn {
            let (mut zs, mut cs, mut cnt) = (Complex32::new(0.0, 0.0), 0.0_f32, 0_usize);
            for di in 0..lk {
                for dj in 0..lk {
                    let (i, j) = (ci * lk + di, cj * lk + dj);
                    let ok = mask.map(|mk| mk[(i, j)]).unwrap_or(true) && igram[(i, j)].norm() > 0.0;
                    if ok {
                        zs += igram[(i, j)];
                        cs += corr[(i, j)];
                        cnt += 1;
                    }
                }
            }
            if cnt * 2 >= lk * lk {
                let mag = zs.norm();
                cig[(ci, cj)] = if mag > 0.0 { zs / mag } else { zs };
                ccorr[(ci, cj)] = cs / cnt as f32;
                cmask[(ci, cj)] = true;
            }
        }
    }
    (cig, ccorr, cmask)
}

/// Block-replicate a coarse field to `(m, n)` (nearest-neighbour). The trailing
/// `< lk` strip (when `m`/`n` aren't divisible by `lk`) has no coarse cell and
/// stays NaN.
fn upsample_blockrep(coarse: &Array2<f32>, lk: usize, m: usize, n: usize) -> Array2<f32> {
    let (cm, cn) = coarse.dim();
    let mut out = Array2::<f32>::from_elem((m, n), f32::NAN);
    for i in 0..m {
        for j in 0..n {
            let (ci, cj) = (i / lk, j / lk);
            if ci < cm && cj < cn && coarse[(ci, cj)].is_finite() {
                out[(i, j)] = coarse[(ci, cj)];
            }
        }
    }
    out
}

fn compute_coarse_anchor(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    lk: usize,
) -> Option<Array2<f32>> {
    let (m, n) = igram.dim();
    if m / lk < 2 || n / lk < 2 {
        return None;
    }
    let (cig, ccorr, cmask) = multilook_complex(igram, corr, mask, lk);
    // One whole-image solve on the tiny coarse image; effective looks ×lk².
    let cunw = crate::unwrap(cig.view(), ccorr.view(), nlooks * (lk * lk) as f32, Some(cmask.view())).ok()?;
    // Block-replicate to full res (we only consume round((anchor−unw)/2π)).
    Some(upsample_blockrep(&cunw, lk, m, n))
}

fn coarse_refine(
    unw: &mut Array2<f32>,
    coh: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    f: usize,
    anchor: Option<ArrayView2<f32>>,
) {
    use std::collections::HashMap;
    let (m, n) = unw.dim();
    let (mh, mw) = (m / f, n / f);
    if mh < 2 || mw < 2 {
        return;
    }
    let tau = TAU as f64;
    let valid =
        |unw: &Array2<f32>, i: usize, j: usize| {
            mask.map(|mk| mk[(i, j)]).unwrap_or(true) && unw[(i, j)].is_finite()
        };

    // 1) Coarsen: block-mean unw / coh over valid pixels; require ≥ f valid.
    let mut cunw = vec![0_f64; mh * mw];
    let mut ccoh = vec![0_f64; mh * mw];
    let mut cvalid = vec![false; mh * mw];
    for ci in 0..mh {
        for cj in 0..mw {
            let (mut s, mut sc, mut cnt) = (0_f64, 0_f64, 0_usize);
            for di in 0..f {
                for dj in 0..f {
                    let (i, j) = (ci * f + di, cj * f + dj);
                    if valid(unw, i, j) {
                        s += unw[(i, j)] as f64;
                        sc += coh[(i, j)] as f64;
                        cnt += 1;
                    }
                }
            }
            if cnt >= f {
                let idx = ci * mw + cj;
                cunw[idx] = s / cnt as f64;
                ccoh[idx] = sc / cnt as f64;
                cvalid[idx] = true;
            }
        }
    }

    // 2) Regions = connected components over no-jump coarse edges.
    let cyc = |a: usize, b: usize| -> i64 { ((cunw[b] - cunw[a]) / tau).round() as i64 };
    let mut parent: Vec<usize> = (0..mh * mw).collect();
    for ci in 0..mh {
        for cj in 0..mw {
            let idx = ci * mw + cj;
            if !cvalid[idx] {
                continue;
            }
            if cj + 1 < mw && cvalid[idx + 1] && cyc(idx, idx + 1) == 0 {
                let (ra, rb) = (uf_find(&mut parent, idx), uf_find(&mut parent, idx + 1));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
            if ci + 1 < mh && cvalid[idx + mw] && cyc(idx, idx + mw) == 0 {
                let (ra, rb) = (uf_find(&mut parent, idx), uf_find(&mut parent, idx + mw));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }

    // 3) Per-region integer 2π offset.
    let mut off: HashMap<usize, i64> = HashMap::new();
    if let Some(anchor) = anchor {
        // ANCHOR MODE. Snap each region's offset to a globally-consistent
        // coarse whole-image unwrap. The anchor carries no tile seams, so it
        // resolves the integer ambiguity even for low-coherence regions the
        // largest-region vote (below) cannot reach (their boundary edges
        // carry near-zero coherence weight). We take the coherence-weighted
        // MODE of round((anchor − unw)/2π) over each whole region, so a local
        // sub-region anchor error is outvoted — only the region's dominant
        // (correct) integer survives.
        let mut canchor = vec![0_f64; mh * mw];
        let mut cavalid = vec![false; mh * mw];
        for ci in 0..mh {
            for cj in 0..mw {
                let (mut s, mut cnt) = (0_f64, 0_usize);
                for di in 0..f {
                    for dj in 0..f {
                        let (i, j) = (ci * f + di, cj * f + dj);
                        if anchor[(i, j)].is_finite() {
                            s += anchor[(i, j)] as f64;
                            cnt += 1;
                        }
                    }
                }
                if cnt >= f {
                    canchor[ci * mw + cj] = s / cnt as f64;
                    cavalid[ci * mw + cj] = true;
                }
            }
        }
        let mut votes: HashMap<usize, HashMap<i64, f64>> = HashMap::new();
        for idx in 0..mh * mw {
            if !cvalid[idx] || !cavalid[idx] {
                continue;
            }
            let k = ((canchor[idx] - cunw[idx]) / tau).round() as i64;
            let r = uf_find(&mut parent, idx);
            *votes
                .entry(r)
                .or_default()
                .entry(k)
                .or_insert(0.0) += ccoh[idx].max(1e-6);
        }
        for (r, vmap) in &votes {
            let best = vmap
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|kv| *kv.0)
                .unwrap_or(0);
            off.insert(*r, best);
        }
    } else {
        // NO-ANCHOR FALLBACK. Inter-region edges (jump + coherence² weight):
        // edge A→B with jump j wants `off_A − off_B = j` to zero the boundary
        // jump. Iterative coherence-weighted-mode vote anchored to the largest
        // region. Distinct regions share only jump edges (no satisfied seams),
        // so the vote has no degenerate fixed point.
        let mut sizes: HashMap<usize, usize> = HashMap::new();
        for idx in 0..mh * mw {
            if cvalid[idx] {
                *sizes.entry(uf_find(&mut parent, idx)).or_insert(0) += 1;
            }
        }
        let Some((&anchor_region, _)) = sizes.iter().max_by_key(|kv| *kv.1) else { return };

        let mut adj: HashMap<usize, Vec<(usize, i64, f64)>> = HashMap::new();
        for ci in 0..mh {
            for cj in 0..mw {
                let idx = ci * mw + cj;
                if !cvalid[idx] {
                    continue;
                }
                let ra = uf_find(&mut parent, idx);
                let mut consider = |idx2: usize, adj: &mut HashMap<usize, Vec<(usize, i64, f64)>>| {
                    let j = cyc(idx, idx2);
                    if j != 0 {
                        let rb = uf_find(&mut parent, idx2);
                        let w = ccoh[idx].min(ccoh[idx2]).powi(2);
                        adj.entry(ra).or_default().push((rb, j, w));
                        adj.entry(rb).or_default().push((ra, -j, w));
                    }
                };
                if cj + 1 < mw && cvalid[idx + 1] {
                    consider(idx + 1, &mut adj);
                }
                if ci + 1 < mh && cvalid[idx + mw] {
                    consider(idx + mw, &mut adj);
                }
            }
        }

        let regions: Vec<usize> = adj.keys().copied().filter(|&r| r != anchor_region).collect();
        for _ in 0..200 {
            let mut changed = false;
            for &r in &regions {
                let mut votes: HashMap<i64, f64> = HashMap::new();
                for &(nb, j, w) in &adj[&r] {
                    *votes.entry(off.get(&nb).copied().unwrap_or(0) + j).or_insert(0.0) += w;
                }
                let best = votes
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|kv| *kv.0)
                    .unwrap_or(0);
                if best != off.get(&r).copied().unwrap_or(0) {
                    off.insert(r, best);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    // 5) Apply per-region integer offset back to full-resolution unw.
    for ci in 0..mh {
        for cj in 0..mw {
            let idx = ci * mw + cj;
            if !cvalid[idx] {
                continue;
            }
            let r = uf_find(&mut parent, idx);
            let d = off.get(&r).copied().unwrap_or(0);
            if d == 0 {
                continue;
            }
            let add = tau as f32 * d as f32;
            for di in 0..f {
                for dj in 0..f {
                    let (i, j) = (ci * f + di, cj * f + dj);
                    if valid(unw, i, j) {
                        unw[(i, j)] += add;
                    }
                }
            }
        }
    }
}

/// Heal thin "sliver" artifacts the MCF residue-pairing can leave: a thin
/// (≤ `max_w` px wide) run of pixels unwrapped a constant nonzero integer
/// number of cycles off from a coherent surround that AGREES on both sides.
///
/// These are tie-break artifacts — a spurious branch cut the unit-capacity MCF
/// laid down in moderate-coherence noise (e.g. NISAR col 4032, a 2-px −1 sliver
/// over ~420 rows on coh≈0.65 where SNAPHU is flat). They are NOT a cost-shape
/// problem (present under both the linear and convex costs), so no per-arc cost
/// removes them; a bounded integer-consistency cleanup is the right tool.
///
/// A run is snapped iff the coherent pixel just past EACH end sits the SAME
/// nonzero integer `c` cycles above the run (`cL == cR == c ≠ 0`) — i.e. the
/// run is a thin island `c` cycles below an otherwise-continuous surround. That
/// cannot hold across a real fringe (its two sides sit at different levels), so
/// genuine signal is untouched; and a real ≤`max_w`-px feature a full cycle off
/// from an agreeing surround is an artifact, not deformation. Coherence-gated
/// (`coh > min_coh`); runs in both orientations (the row pass catches vertical
/// lines, the column pass horizontal); iterated so adjacent slivers settle,
/// with fixes computed against the pre-iteration field then applied together.
///
/// `max_w = 1` reduces to the original immediate-neighbour 1-px heal.
fn heal_thin_slivers(
    unw: &mut Array2<f32>,
    corr: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    min_coh: f32,
    max_w: usize,
    iters: usize,
) {
    let (m, n) = unw.dim();
    let ok = |unw: &Array2<f32>, i: usize, j: usize| {
        mask.map(|mk| mk[(i, j)]).unwrap_or(true) && unw[(i, j)].is_finite() && corr[(i, j)] > min_coh
    };
    // Pixels within this of the run base are treated as the same integer level.
    let same_level = std::f32::consts::PI;
    for _ in 0..iters {
        let mut fixes: Vec<(usize, usize, f32)> = Vec::new();
        // Row pass: vertical-line slivers (a run along columns within one row).
        for i in 0..m {
            let mut j = 1;
            while j + 1 < n {
                if !ok(unw, i, j) || !ok(unw, i, j - 1) {
                    j += 1;
                    continue;
                }
                let cl = ((unw[(i, j - 1)] - unw[(i, j)]) / TAU).round() as i64;
                if cl == 0 {
                    j += 1;
                    continue; // left neighbour is same level — not a left edge
                }
                let base = unw[(i, j)];
                let mut e = j; // extend the same-level run rightward, bounded by max_w
                while e + 1 < n
                    && e + 1 - j < max_w
                    && ok(unw, i, e + 1)
                    && (unw[(i, e + 1)] - base).abs() < same_level
                {
                    e += 1;
                }
                if e + 1 < n && ok(unw, i, e + 1) {
                    let cr = ((unw[(i, e + 1)] - unw[(i, e)]) / TAU).round() as i64;
                    if cr == cl && cr != 0 {
                        let d = cl as f32 * TAU;
                        for jj in j..=e {
                            fixes.push((i, jj, d));
                        }
                        j = e + 1;
                        continue;
                    }
                }
                j += 1;
            }
        }
        // Column pass: horizontal-line slivers (a run along rows within one col).
        for jc in 0..n {
            let mut i = 1;
            while i + 1 < m {
                if !ok(unw, i, jc) || !ok(unw, i - 1, jc) {
                    i += 1;
                    continue;
                }
                let cu = ((unw[(i - 1, jc)] - unw[(i, jc)]) / TAU).round() as i64;
                if cu == 0 {
                    i += 1;
                    continue;
                }
                let base = unw[(i, jc)];
                let mut e = i;
                while e + 1 < m
                    && e + 1 - i < max_w
                    && ok(unw, e + 1, jc)
                    && (unw[(e + 1, jc)] - base).abs() < same_level
                {
                    e += 1;
                }
                if e + 1 < m && ok(unw, e + 1, jc) {
                    let cd = ((unw[(e + 1, jc)] - unw[(e, jc)]) / TAU).round() as i64;
                    if cd == cu && cd != 0 {
                        let d = cu as f32 * TAU;
                        for ii in i..=e {
                            fixes.push((ii, jc, d));
                        }
                        i = e + 1;
                        continue;
                    }
                }
                i += 1;
            }
        }
        if fixes.is_empty() {
            break;
        }
        for (i, j, d) in fixes {
            unw[(i, j)] += d;
        }
    }
}

/// Per-tile base solver. The linear Carballo cost has a corner / capacity-1
/// boundary-stacking bug on smooth STEEP signals (paper/pyramid_aliasing.md:
/// ~88% on a clean 0.7π bowl vs 100% for reuse/convex). `reuse` (PHASS
/// flow-reuse) and `convex` (SNAPHU quadratic) are corner-safe.
#[derive(Clone, Copy, PartialEq)]
enum TileSolver {
    Linear,
    Reuse,
    Convex,
}

/// `WHIRLWIND_TILE_SOLVER=linear|reuse|convex` (default set in code below).
/// Legacy `WHIRLWIND_TILE_CONVEX=1` still selects convex.
fn tile_solver() -> TileSolver {
    use std::sync::OnceLock;
    static F: OnceLock<TileSolver> = OnceLock::new();
    *F.get_or_init(|| match std::env::var("WHIRLWIND_TILE_SOLVER").as_deref() {
        Ok("reuse") => TileSolver::Reuse,
        Ok("convex") => TileSolver::Convex,
        Ok("linear") => TileSolver::Linear,
        _ if std::env::var("WHIRLWIND_TILE_CONVEX").is_ok() => TileSolver::Convex,
        // DEFAULT: reuse (PHASS flow-reuse). Validated 2026-05-31 across a clean
        // steep ramp (linear FAILs 12.6 rad / reuse PASSes 0.0 — the corner bug),
        // NISAR (linear 99.84% → reuse 99.96%), and a 5-frame NISAR-GUNW sweep
        // (reuse within ~0.4% of linear, better on 3/5). Corner-safe and a net
        // win; ~slower than linear but speed is not the constraint vs SNAPHU.
        _ => TileSolver::Reuse,
    })
}

fn unwrap_one_tile_coh(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    tile: &Tile,
) -> Result<Array2<f32>, UnwrapError> {
    let ig = igram.slice(s![tile.r0..tile.r1, tile.c0..tile.c1]);
    let co = corr.slice(s![tile.r0..tile.r1, tile.c0..tile.c1]);
    let mk = mask
        .as_ref()
        .map(|m| m.slice(s![tile.r0..tile.r1, tile.c0..tile.c1]));
    let (tm, tn) = ig.dim();
    let wrapped_phase = ig.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mk);
    let graph = RectangularGridGraph::new(tm + 1, tn + 1);
    // Per-tile base solver (WHIRLWIND_TILE_SOLVER). reuse/convex are corner-safe
    // where the linear Carballo cost stacks flow at steep-signal boundaries.
    let mut net = match tile_solver() {
        TileSolver::Convex => {
            let (offsets, weights) = cost::compute_snaphu_smooth_costs(ig, co, nlooks, mk);
            let mut net = Network::new_convex_with_mask(&graph, residues.view(), &offsets, &weights, mk);
            net.preload_convex_min(&graph);
            net
        }
        TileSolver::Reuse => {
            let costs = cost::compute_carballo_costs(ig, co, nlooks, mk);
            Network::new_reuse_with_mask(&graph, residues.view(), &costs, mk)
        }
        TileSolver::Linear => {
            let costs = cost::compute_carballo_costs(ig, co, nlooks, mk);
            Network::new_with_mask(&graph, residues.view(), &costs, mk)
        }
    };
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mk.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mk)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    Ok(unw)
}

/// Coherence-weighted integer-2π offset between two overlapping tiles, plus a
/// confidence score. Returns `(k, confidence)` where adding `k·2π` to `unw_b`
/// aligns it to `unw_a`, and `confidence` is the coherence weight that voted
/// for the winning integer.
///
/// Uses the weighted **mode of per-pixel rounded offsets** rather than the
/// median of continuous diffs: when a wrap line crosses the overlap the two
/// tiles disagree on one side, and the continuous median can land between two
/// integers and round the wrong way — the mode robustly picks the integer the
/// majority (by coherence weight) of overlap pixels agree on. The confidence
/// (winning-bin weight) lets the caller stitch high-agreement seams first.
fn stitching_offset_coh(
    tile_a: &Tile,
    unw_a: &Array2<f32>,
    tile_b: &Tile,
    unw_b: &Array2<f32>,
    corr: ArrayView2<f32>,
) -> (i64, i64) {
    let r0 = tile_a.r0.max(tile_b.r0);
    let r1 = tile_a.r1.min(tile_b.r1);
    let c0 = tile_a.c0.max(tile_b.c0);
    let c1 = tile_a.c1.min(tile_b.c1);
    if r0 >= r1 || c0 >= c1 {
        return (0, 0);
    }
    // Weighted histogram of rounded integer offsets.
    let mut bins: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
    for gi in r0..r1 {
        for gj in c0..c1 {
            let a = unw_a[(gi - tile_a.r0, gj - tile_a.c0)];
            let b = unw_b[(gi - tile_b.r0, gj - tile_b.c0)];
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            let k = ((b - a) / TAU).round() as i64;
            let c = corr[(gi, gj)];
            let w = if c.is_finite() && c > 0.0 {
                let cc = c.min(1.0) as f64;
                cc * cc
            } else {
                1e-3
            };
            *bins.entry(k).or_insert(0.0) += w;
        }
    }
    // Winning integer = highest total coherence weight.
    let mut best_k = 0_i64;
    let mut best_w = -1.0_f64;
    for (&k, &w) in &bins {
        if w > best_w {
            best_w = w;
            best_k = k;
        }
    }
    if best_w < 0.0 {
        return (0, 0);
    }
    (best_k, best_w.round() as i64)
}

fn unwrap_one_tile(
    igram: ArrayView2<Complex32>,
    variance: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    tile: &Tile,
) -> Result<Array2<f32>, UnwrapError> {
    let ig = igram.slice(s![tile.r0..tile.r1, tile.c0..tile.c1]);
    let va = variance.slice(s![tile.r0..tile.r1, tile.c0..tile.c1]);
    let mk = mask.as_ref().map(|m| m.slice(s![tile.r0..tile.r1, tile.c0..tile.c1]));

    let (tm, tn) = ig.dim();
    let wrapped_phase = ig.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mk);
    let costs = cost::compute_crlb_costs(ig, va, mk);
    let graph = RectangularGridGraph::new(tm + 1, tn + 1);
    let mut net = Network::new_with_mask(&graph, residues.view(), &costs, mk);
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mk.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mk)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    Ok(unw)
}

/// CRLB-weighted median of `(unw_b - unw_a) / 2π`, rounded to the nearest
/// integer. Returns the integer K such that adding `K · 2π` to `unw_b`'s
/// values aligns it with `unw_a` in the overlap region. Returns 0 if there
/// is no overlap (shouldn't happen for adjacent tiles) or no valid pixels.
fn stitching_offset(
    tile_a: &Tile,
    unw_a: &Array2<f32>,
    tile_b: &Tile,
    unw_b: &Array2<f32>,
    variance: ArrayView2<f32>,
) -> i64 {
    let r0 = tile_a.r0.max(tile_b.r0);
    let r1 = tile_a.r1.min(tile_b.r1);
    let c0 = tile_a.c0.max(tile_b.c0);
    let c1 = tile_a.c1.min(tile_b.c1);
    if r0 >= r1 || c0 >= c1 {
        return 0;
    }
    // Collect (value, weight) for overlap pixels with finite difference.
    let mut samples: Vec<(f32, f32)> =
        Vec::with_capacity((r1 - r0) * (c1 - c0));
    for gi in r0..r1 {
        for gj in c0..c1 {
            let a = unw_a[(gi - tile_a.r0, gj - tile_a.c0)];
            let b = unw_b[(gi - tile_b.r0, gj - tile_b.c0)];
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            let diff_2pi = (b - a) / TAU;
            // Weight ∝ 1 / variance (CRLB). Skip nodata.
            let v = variance[(gi, gj)];
            let w = if v.is_finite() && v > 0.0 { 1.0 / v } else { 1e-3 };
            samples.push((diff_2pi, w));
        }
    }
    if samples.is_empty() {
        return 0;
    }
    // Weighted median.
    samples.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let total_w: f32 = samples.iter().map(|&(_, w)| w).sum();
    let mut cum = 0.0_f32;
    let mut median = samples[0].0;
    for &(v, w) in &samples {
        cum += w;
        if cum >= 0.5 * total_w {
            median = v;
            break;
        }
    }
    median.round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unwrap_crlb;

    #[test]
    fn decompose_single_tile_when_image_fits() {
        let tiles = decompose(100, 100, 128, 16);
        assert_eq!(tiles.len(), 1);
        assert_eq!((tiles[0].r0, tiles[0].r1, tiles[0].c0, tiles[0].c1), (0, 100, 0, 100));
    }

    #[test]
    fn decompose_regular_overlap() {
        // 200x200 image, 128 tile, 32 overlap → step=96. Starts: 0, 96, then
        // 96+128=224 >= 200, so push 200-128=72? Hmm.
        // Actually: from 96, next start = 192, 192+128=320 ≥ 200 → push 200-128=72.
        // But 72 < 96, weird. Let me re-derive.
        // axis_starts(200, 128, 96): start=[0]. last=0, next=96, 96+128=224 ≥ 200,
        // so push 200-128=72, return. → [0, 72]. Yes that's the behaviour.
        // (72 < 96 is fine — last tile's start moves to overlap more, not less.)
        let starts = axis_starts(200, 128, 96);
        assert_eq!(starts, vec![0, 72]);
    }

    #[test]
    fn decompose_evenly_divides() {
        // 256 = 2 × 128, with 0 overlap → 2 starts.
        let starts = axis_starts(256, 128, 128);
        // axis_starts(256, 128, 128): start=[0]. last=0, next=128, 128+128=256 not >= 256? 256>=256 → push 256-128=128.
        assert_eq!(starts, vec![0, 128]);
    }

    #[test]
    fn stitching_offset_recovers_integer_step() {
        // Two horizontally-adjacent 8×16 tiles with 8-pixel column overlap.
        // tile_a spans cols 0..16, tile_b spans cols 8..24 → overlap cols 8..16.
        // tile_b's pixels are at +3·2π relative to tile_a; stitching should
        // give K = +3 (so subtracting 3·2π from tile_b aligns it with tile_a).
        use ndarray::Array2;
        let m = 8;
        let tile_a = Tile { r0: 0, r1: m, c0: 0, c1: 16 };
        let tile_b = Tile { r0: 0, r1: m, c0: 8, c1: 24 };
        let unw_a = Array2::<f32>::zeros((m, 16));
        let unw_b = Array2::<f32>::from_elem((m, 16), 3.0 * TAU);
        let var = Array2::<f32>::from_elem((m, 24), 0.1);
        let k = stitching_offset(&tile_a, &unw_a, &tile_b, &unw_b, var.view());
        assert_eq!(k, 3, "stitching should recover the planted +3·2π step");
    }

    // Build seam-gradient arrays from a per-tile truth field on an R×C grid.
    fn seams_from_truth(rows: usize, cols: usize, truth: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let mut gh = vec![0_i64; rows * (cols - 1)];
        let mut gv = vec![0_i64; (rows - 1) * cols];
        for gr in 0..rows {
            for gc in 0..cols {
                if gc + 1 < cols {
                    gh[gr * (cols - 1) + gc] = truth[gr * cols + gc + 1] - truth[gr * cols + gc];
                }
                if gr + 1 < rows {
                    gv[gr * cols + gc] = truth[(gr + 1) * cols + gc] - truth[gr * cols + gc];
                }
            }
        }
        (gh, gv)
    }

    #[test]
    fn reconcile_mcf_recovers_consistent_offsets() {
        // Consistent seams (zero curl) → MCF pushes no flow → exact recovery.
        let (rows, cols) = (4usize, 5usize);
        let truth: Vec<i64> = (0..rows * cols).map(|t| (2 * (t / cols) + 3 * (t % cols)) as i64).collect();
        let (gh, gv) = seams_from_truth(rows, cols, &truth);
        let wh = vec![100_i64; gh.len()];
        let wv = vec![100_i64; gv.len()];
        let off = reconcile_offsets_mcf(rows, cols, &gh, &wh, &gv, &wv);
        for t in 0..rows * cols {
            assert_eq!(off[t] - off[0], truth[t] - truth[0], "tile {t}");
        }
    }

    #[test]
    fn reconcile_mcf_breaks_low_confidence_wrong_seam() {
        // One vertical and one (row-0) horizontal seam are corrupted with LOW
        // confidence; every other seam is high confidence. The min-cost flow
        // must correct the two cheap seams (not reroute through the expensive
        // ones), recovering the planted ramp — the property a region-flip
        // heuristic cannot guarantee.
        let (rows, cols) = (4usize, 5usize);
        let truth: Vec<i64> = (0..rows * cols).map(|t| (2 * (t / cols) + 3 * (t % cols)) as i64).collect();
        let (mut gh, mut gv) = seams_from_truth(rows, cols, &truth);
        let mut wh = vec![100_i64; gh.len()];
        let mut wv = vec![100_i64; gv.len()];
        // Corrupt gv at (gr=1, gc=2) by +7, low confidence.
        gv[1 * cols + 2] += 7;
        wv[1 * cols + 2] = 1;
        // Corrupt gh at (gr=0, gc=1) by −4, low confidence (row 0 is on the
        // integration path, so a wrong sign would show up directly).
        gh[0 * (cols - 1) + 1] -= 4;
        wh[0 * (cols - 1) + 1] = 1;
        let off = reconcile_offsets_mcf(rows, cols, &gh, &wh, &gv, &wv);
        for t in 0..rows * cols {
            assert_eq!(
                off[t] - off[0],
                truth[t] - truth[0],
                "tile {t}: MCF failed to correct the low-confidence wrong seams"
            );
        }
    }

    #[test]
    fn coarse_refine_flips_block_offset() {
        use ndarray::Array2;
        // Smooth ramp (gradient ≪ π) with a planted +2-cycle rectangular block
        // offset (block edges aligned to the 8× coarsen grid). coarse_refine
        // must flip the block back so the field is smooth again.
        let (m, n) = (64usize, 64usize);
        let truth = Array2::from_shape_fn((m, n), |(i, j)| 0.1 * i as f32 + 0.07 * j as f32);
        let mut unw = truth.clone();
        for i in 16..48 {
            for j in 16..48 {
                unw[(i, j)] += 2.0 * TAU;
            }
        }
        let coh = Array2::<f32>::from_elem((m, n), 0.9);
        coarse_refine(&mut unw, coh.view(), None, 8, None);
        // Field should equal truth up to one global integer-cycle constant.
        let kglob = ((unw[(0, 0)] - truth[(0, 0)]) / TAU).round();
        let mut maxres = 0.0_f32;
        for i in 0..m {
            for j in 0..n {
                let r = (unw[(i, j)] - truth[(i, j)] - TAU * kglob).abs();
                maxres = maxres.max(r);
            }
        }
        assert!(maxres < 1e-3, "coarse_refine left a block offset: max residual {maxres} rad");
    }

    #[test]
    fn coarse_refine_anchor_fixes_isolated_low_coh_island() {
        use ndarray::Array2;
        // The failure the global anchor exists to fix: a wrong-offset block in
        // a LOW-coherence patch that is itself surrounded by an INVALID moat,
        // so it shares no coarse no-jump edge with the high-coherence mainland.
        // The relative largest-region vote (None) cannot reach it (no edges);
        // the anchor snaps it absolutely.
        let (m, n) = (64usize, 64usize);
        let truth = Array2::from_shape_fn((m, n), |(i, j)| 0.05 * i as f32 + 0.04 * j as f32);
        let mut unw = truth.clone();
        // Island [16,48)×[16,48) offset by +2 cycles; ring [8,16)∪[48,56) invalid.
        let mut mask = Array2::<bool>::from_elem((m, n), true);
        for i in 0..m {
            for j in 0..n {
                let in_ring = (8..56).contains(&i) && (8..56).contains(&j)
                    && !((16..48).contains(&i) && (16..48).contains(&j));
                if in_ring {
                    mask[(i, j)] = false;
                }
                if (16..48).contains(&i) && (16..48).contains(&j) {
                    unw[(i, j)] += 2.0 * TAU;
                }
            }
        }
        // Low coherence on the island, high on the mainland.
        let coh = Array2::from_shape_fn((m, n), |(i, j)| {
            if (16..48).contains(&i) && (16..48).contains(&j) { 0.3 } else { 0.9 }
        });
        // A correct anchor (= truth) — the global coarse solve's role.
        let anchor = truth.clone();
        coarse_refine(&mut unw, coh.view(), Some(mask.view()), 8, Some(anchor.view()));
        let kglob = ((unw[(20, 20)] - truth[(20, 20)]) / TAU).round();
        let mut maxres = 0.0_f32;
        for i in 16..48 {
            for j in 16..48 {
                let r = (unw[(i, j)] - truth[(i, j)] - TAU * kglob).abs();
                maxres = maxres.max(r);
            }
        }
        assert!(maxres < 1e-3, "anchor failed to fix isolated island: max residual {maxres} rad");
    }

    #[test]
    fn heal_thin_slivers_removes_1px_2px_3px_and_spares_fringe() {
        use ndarray::Array2;
        // Smooth ramp + three slivers of width 1, 2, 3 each offset by an integer
        // cycle in a coherent area: the bounded continuity-cleanup must snap all
        // three back (the col-4032 case is the width-2 one). The smooth ramp
        // (gradient ≪ π) must be untouched.
        let (m, n) = (60usize, 60usize);
        let truth = Array2::from_shape_fn((m, n), |(i, j)| 0.05 * i as f32 + 0.04 * j as f32);
        let mut unw = truth.clone();
        for i in 5..55 {
            unw[(i, 10)] += TAU; // width-1, +1
            unw[(i, 25)] -= TAU; // width-2, -1
            unw[(i, 26)] -= TAU;
            unw[(i, 40)] += TAU; // width-3, +1
            unw[(i, 41)] += TAU;
            unw[(i, 42)] += TAU;
        }
        let coh = Array2::<f32>::from_elem((m, n), 0.8);
        heal_thin_slivers(&mut unw, coh.view(), None, 0.2, 4, 6);
        let mut maxres = 0.0_f32;
        for i in 0..m {
            for j in 0..n {
                maxres = maxres.max((unw[(i, j)] - truth[(i, j)]).abs());
            }
        }
        assert!(maxres < 1e-3, "slivers not healed: max residual {maxres}");
    }

    #[test]
    fn heal_thin_slivers_spares_real_fringe_step() {
        use ndarray::Array2;
        // A genuine 2π step (left half one cycle below the right half) is a REAL
        // discontinuity, not a thin sliver: the two sides do NOT agree on a
        // common surround, so the cleanup must leave it alone. (A wide block, not
        // a ≤4-px run.)
        let (m, n) = (40usize, 40usize);
        let mut unw = Array2::from_shape_fn((m, n), |(i, j)| 0.03 * i as f32 + 0.03 * j as f32);
        for i in 0..m {
            for j in 20..n {
                unw[(i, j)] += TAU; // right half a full cycle up
            }
        }
        let before = unw.clone();
        let coh = Array2::<f32>::from_elem((m, n), 0.8);
        heal_thin_slivers(&mut unw, coh.view(), None, 0.2, 4, 6);
        let mut maxdelta = 0.0_f32;
        for i in 0..m {
            for j in 0..n {
                maxdelta = maxdelta.max((unw[(i, j)] - before[(i, j)]).abs());
            }
        }
        assert!(maxdelta < 1e-6, "cleanup wrongly touched a real 2π step: max delta {maxdelta}");
    }

    #[test]
    fn tiled_coherence_matches_single_tile_on_smooth_input() {
        use crate::unwrap;
        use ndarray::Array2;
        // Smooth ramp with no wraps: tiled coherence unwrap must agree with
        // the whole-image coherence unwrap (up to a global integer cycle).
        let m = 80;
        let n = 80;
        let truth: Array2<f32> =
            Array2::from_shape_fn((m, n), |(i, j)| 0.04 * i as f32 + 0.03 * j as f32);
        let igram = truth.mapv(|p| Complex32::new(p.cos(), p.sin()));
        let corr = Array2::<f32>::from_elem((m, n), 0.9);

        let whole = unwrap(igram.view(), corr.view(), 10.0, None).unwrap();
        let tiled = unwrap_tiled(igram.view(), corr.view(), 10.0, None, 32, 8, 1).unwrap();

        let align = |u: &Array2<f32>| -> Array2<f32> {
            let off = u.iter().zip(truth.iter()).map(|(&u, &t)| u - t).sum::<f32>()
                / (u.len() as f32);
            let k = (off / TAU).round();
            u.mapv(|v| v - TAU * k)
        };
        let wa = align(&whole);
        let ta = align(&tiled);
        let max_err = wa
            .iter()
            .zip(ta.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-3,
            "tiled and whole-image coherence unwrap should agree on smooth input, max diff {max_err}"
        );
    }

    #[test]
    fn tiled_unwrap_matches_single_tile_on_smooth_input() {
        use ndarray::Array2;
        // Smooth phase ramp that has no wraps — non-tiled unwrap is trivial,
        // tiled unwrap should also produce a smooth field.
        let m = 64;
        let n = 64;
        let truth: Array2<f32> = Array2::from_shape_fn((m, n), |(i, j)| {
            0.05 * i as f32 + 0.03 * j as f32
        });
        let igram = truth.mapv(|p| Complex32::new(p.cos(), p.sin()));
        let var = Array2::<f32>::from_elem((m, n), 0.1);

        let non_tiled = unwrap_crlb(igram.view(), var.view(), None).unwrap();
        let tiled = unwrap_crlb_tiled(igram.view(), var.view(), None, 24, 8).unwrap();

        // Both should be smooth. Compare to truth after aligning the
        // global integer-cycle offset.
        let align = |u: &Array2<f32>| -> Array2<f32> {
            let off = u.iter().zip(truth.iter()).map(|(&u, &t)| u - t).sum::<f32>()
                / (u.len() as f32);
            let k = (off / TAU).round();
            u.mapv(|v| v - TAU * k)
        };
        let non_tiled_a = align(&non_tiled);
        let tiled_a = align(&tiled);

        let max_err = non_tiled_a
            .iter()
            .zip(tiled_a.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-3,
            "tiled and non-tiled should agree on smooth input, max diff {max_err}"
        );
    }

    #[test]
    fn coherent_cut_rate_zero_on_clean_high_when_tearing_coherent_terrain() {
        use ndarray::Array2;
        // A clean smooth ramp unwrapped correctly has NO coherent cuts.
        let (m, n) = (64, 64);
        let truth: Array2<f32> =
            Array2::from_shape_fn((m, n), |(i, j)| 0.05 * i as f32 + 0.03 * j as f32);
        let igram = truth.mapv(|p| Complex32::new(p.cos(), p.sin()));
        let corr = Array2::<f32>::from_elem((m, n), 0.95);
        let clean = coherent_cut_rate(igram.view(), &truth, corr.view(), None, COH_CUT_THR);
        assert!(clean < 1e-9, "clean ramp must have ~0 coherent-cut rate, got {clean}");

        // Inject a spurious +1 cycle "island" across coherent terrain (a branch-cut
        // loop) — the coherent-cut rate must jump well above the gate floor.
        let mut torn = truth.clone();
        for i in 20..40 {
            for j in 20..40 {
                torn[(i, j)] += TAU;
            }
        }
        let torn_rate = coherent_cut_rate(igram.view(), &torn, corr.view(), None, COH_CUT_THR);
        assert!(
            torn_rate > COH_CUT_FLOOR,
            "tearing coherent terrain must exceed the gate floor, got {torn_rate}"
        );
    }

    #[test]
    fn unwrap_tiled_robust_is_noop_on_clean_scene() {
        use ndarray::Array2;
        // On a clean tiled scene the gate must NOT fire: robust == plain tiled.
        let (m, n) = (96, 96);
        let truth: Array2<f32> =
            Array2::from_shape_fn((m, n), |(i, j)| 0.04 * i as f32 + 0.03 * j as f32);
        let igram = truth.mapv(|p| Complex32::new(p.cos(), p.sin()));
        let corr = Array2::<f32>::from_elem((m, n), 0.9);
        let plain = unwrap_tiled(igram.view(), corr.view(), 10.0, None, 32, 8, 1).unwrap();
        let robust = unwrap_tiled_robust(igram.view(), corr.view(), 10.0, None, 32, 8, 1).unwrap();
        let max_diff = plain
            .iter()
            .zip(robust.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-6, "robust must equal plain tiled on a clean scene (no gate), diff {max_diff}");
    }
}
