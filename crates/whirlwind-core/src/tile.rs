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

    #[test]
    fn flow_from_unwrap_roundtrips_through_integrate() {
        use crate::cost;
        use crate::integrate as integ;
        use crate::network::Network;
        use crate::residue;
        use ndarray::Array2;

        // Unwrap a moderate-noise IG, extract its flow via flow_from_unwrap,
        // and re-integrate via integrate_with_initial_flow on a freshly-built
        // warm-started network whose PD-flow is all-zero. The result must
        // reproduce the original unwrap, proving that the (init_flow ↔
        // unwrap) round-trip is exact and that the warm-start excess
        // adjustment balances residue charges in aggregate.
        let m = 48;
        let n = 56;
        let truth: Array2<f32> = Array2::from_shape_fn((m, n), |(i, j)| {
            0.4 * i as f32 + 0.25 * j as f32
                + 4.0 * ((i as f32 / 12.0).sin() * (j as f32 / 10.0).cos())
        });
        let igram = truth.mapv(|p| Complex32::new(p.cos(), p.sin()));
        let var = Array2::<f32>::from_elem((m, n), 0.05);

        let unw0 = crate::unwrap_crlb(igram.view(), var.view(), None).unwrap();

        let wrapped = igram.mapv(|z| z.arg());
        let graph = RectangularGridGraph::new(m + 1, n + 1);
        let (flow, n_clamped) =
            integ::flow_from_unwrap(wrapped.view(), unw0.view(), &graph);
        assert_eq!(
            n_clamped, 0,
            "smooth unwrap should not need any |k|>1 clamping"
        );

        let residues = residue::compute_with_mask(wrapped.view(), None);
        let costs = cost::compute_crlb_costs(igram.view(), var.view(), None);
        let net = Network::new_with_initial_flow(
            &graph,
            residues.view(),
            &costs,
            None,
            &flow,
        );
        // Total excess should be zero (init_flow + residues balance globally).
        assert!(
            net.is_balanced(),
            "warm-started net should be globally balanced (Σ excess == 0)"
        );

        // Re-integrate via the combined-flow integration — net is fresh
        // (PD never ran), so the result is determined entirely by the
        // init flow we extracted, and must round-trip back to the unwrap.
        let reintegrated =
            integ::integrate_with_initial_flow(wrapped.view(), &graph, &net, &flow);
        let max_err = unw0
            .iter()
            .zip(reintegrated.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-3,
            "flow round-trip should reproduce unwrap (max err {max_err})"
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
}
