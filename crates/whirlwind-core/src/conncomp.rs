//! SNAPHU-style connected component growing from a solved MCF network.
//!
//! After [`crate::primal_dual::run`] (or the SSP fallback) returns, every
//! pixel-edge in the original phase image is backed by two forward residue
//! arcs (one per Carballo direction). We label each pixel-edge as a *cut*
//! when at least one of its underlying arcs is unreliable:
//!
//! 1. The arc is forbidden (both directions saturated — masked-out pixel).
//! 2. The minimum raw forward cost across the two directions is below
//!    `cost_threshold` — the noise model is uninformative here, so any
//!    routing decision across this edge wasn't strongly supported by data.
//!
//! Then BFS the remaining pixels through non-cut edges to label connected
//! components. Small components are dropped; the top `max_ncomps` by size are
//! kept and renumbered 1..=max_ncomps.
//!
//! This is the gridded adaptation of `GrowConnCompsMask` in SNAPHU's
//! `snaphu_tile.c`. SNAPHU works with convex piecewise costs and tests
//! `min(negcost, poscost)` — the local cost-function flatness around the
//! chosen flow. In our linear unit-capacity setting that collapses to the
//! raw arc cost (no curvature anywhere), and MCF flow placement is
//! deliberately *not* a cut signal: a high-cost branch cut means MCF paid
//! the right price to close a noise-induced residue pair, which is the
//! correct answer to encode, not an unreliable region. Low-cost edges where
//! MCF *does* place flow show up as cuts anyway by the raw-cost test.

use crate::grid::RectangularGridGraph;
use crate::network::Network;
use ndarray::{Array2, ArrayView2};
use std::collections::VecDeque;

/// Parameters for [`grow_components`]. Defaults mirror SNAPHU's `defparams`
/// scaled to whirlwind's `COST_SCALE = 100` (Carballo cost ranges 0..~314).
#[derive(Debug, Clone)]
pub struct ConnCompParams {
    /// Cut a pixel edge when min raw forward cost across the two underlying
    /// arcs is ≤ this. Higher → fewer/larger components.
    pub cost_threshold: i32,
    /// Drop components covering less than this fraction of valid pixels.
    pub min_size_frac: f32,
    /// Keep at most this many components (largest by size). 0 → keep all.
    pub max_ncomps: u32,
}

impl Default for ConnCompParams {
    fn default() -> Self {
        Self {
            cost_threshold: 50,
            min_size_frac: 0.01,
            max_ncomps: 64,
        }
    }
}

#[inline]
fn edge_is_cut(net: &Network, fwd1: usize, fwd2: usize, thresh: i32) -> bool {
    let nf = net.num_forward();
    let sat = |a: usize| net.is_arc_saturated(a);
    let forbidden1 = sat(fwd1) && sat(fwd1 + nf);
    let forbidden2 = sat(fwd2) && sat(fwd2 + nf);
    if forbidden1 || forbidden2 {
        return true;
    }
    net.cost_fwd[fwd1].min(net.cost_fwd[fwd2]) <= thresh
}

/// Grow components on the pixel grid using a solved MCF network. Returns a
/// `(m_phase, n_phase)` `u32` label array; 0 = unassigned (cut off or below
/// `min_size_frac`).
pub fn grow_components(
    g: &RectangularGridGraph,
    net: &Network,
    pixel_mask: Option<ArrayView2<bool>>,
    params: &ConnCompParams,
) -> Array2<u32> {
    let m_phase = g.m - 1;
    let n_phase = g.n - 1;
    assert!(m_phase >= 1 && n_phase >= 1);
    if let Some(mm) = pixel_mask {
        assert_eq!(mm.dim(), (m_phase, n_phase), "pixel mask must be (m-1, n-1)");
    }

    let valid = |i: usize, j: usize| pixel_mask.map(|m| m[(i, j)]).unwrap_or(true);
    let n_valid: usize = (0..m_phase)
        .flat_map(|i| (0..n_phase).map(move |j| (i, j)))
        .filter(|&(i, j)| valid(i, j))
        .count();
    let min_size =
        ((params.min_size_frac as f64 * n_valid as f64).ceil() as usize).max(1);

    let mut labels = Array2::<u32>::zeros((m_phase, n_phase));
    let mut next_label: u32 = 0;
    let mut sizes: Vec<usize> = vec![0]; // sizes[0] is a placeholder

    for si in 0..m_phase {
        for sj in 0..n_phase {
            if labels[(si, sj)] != 0 || !valid(si, sj) {
                continue;
            }
            next_label += 1;
            let label = next_label;
            let mut q: VecDeque<(usize, usize)> = VecDeque::new();
            q.push_back((si, sj));
            labels[(si, sj)] = label;
            let mut size = 0_usize;

            while let Some((i, j)) = q.pop_front() {
                size += 1;

                // Right: pixel edge (i, j)-(i, j+1) ↔ down(i,j+1)+up(i+1,j+1)
                if j + 1 < n_phase && labels[(i, j + 1)] == 0 && valid(i, j + 1) {
                    let fwd1 = g.down_arc(i, j + 1).unwrap();
                    let fwd2 = g.up_arc(i + 1, j + 1).unwrap();
                    if !edge_is_cut(net, fwd1, fwd2, params.cost_threshold) {
                        labels[(i, j + 1)] = label;
                        q.push_back((i, j + 1));
                    }
                }
                // Left: pixel edge (i, j-1)-(i, j) ↔ down(i,j)+up(i+1,j)
                if j >= 1 && labels[(i, j - 1)] == 0 && valid(i, j - 1) {
                    let fwd1 = g.down_arc(i, j).unwrap();
                    let fwd2 = g.up_arc(i + 1, j).unwrap();
                    if !edge_is_cut(net, fwd1, fwd2, params.cost_threshold) {
                        labels[(i, j - 1)] = label;
                        q.push_back((i, j - 1));
                    }
                }
                // Down: pixel edge (i, j)-(i+1, j) ↔ right(i+1,j)+left(i+1,j+1)
                if i + 1 < m_phase && labels[(i + 1, j)] == 0 && valid(i + 1, j) {
                    let fwd1 = g.right_arc(i + 1, j).unwrap();
                    let fwd2 = g.left_arc(i + 1, j + 1).unwrap();
                    if !edge_is_cut(net, fwd1, fwd2, params.cost_threshold) {
                        labels[(i + 1, j)] = label;
                        q.push_back((i + 1, j));
                    }
                }
                // Up: pixel edge (i-1, j)-(i, j) ↔ right(i,j)+left(i,j+1)
                if i >= 1 && labels[(i - 1, j)] == 0 && valid(i - 1, j) {
                    let fwd1 = g.right_arc(i, j).unwrap();
                    let fwd2 = g.left_arc(i, j + 1).unwrap();
                    if !edge_is_cut(net, fwd1, fwd2, params.cost_threshold) {
                        labels[(i - 1, j)] = label;
                        q.push_back((i - 1, j));
                    }
                }
            }
            sizes.push(size);
        }
    }

    // Drop too-small components; cap at max_ncomps (largest by size).
    let mut indices: Vec<u32> = (1..=next_label)
        .filter(|&l| sizes[l as usize] >= min_size)
        .collect();
    indices.sort_by(|&a, &b| sizes[b as usize].cmp(&sizes[a as usize]));
    if params.max_ncomps > 0 {
        indices.truncate(params.max_ncomps as usize);
    }
    let mut renumber = vec![0_u32; (next_label + 1) as usize];
    for (new_idx, &old) in indices.iter().enumerate() {
        renumber[old as usize] = (new_idx + 1) as u32;
    }
    labels.mapv_inplace(|l| renumber[l as usize]);
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost;
    use crate::primal_dual;
    use crate::residue;
    use ndarray::Array2;
    use num_complex::Complex32;

    /// Smooth ramp + one strong residue boundary. Components should respect
    /// the masked stripe even though the wrapped phase is otherwise smooth.
    #[test]
    fn split_by_mask_stripe() {
        let m = 32;
        let n = 32;
        let mut truth = Array2::<f32>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                truth[(i, j)] = 0.3 * (i as f32 + j as f32);
            }
        }
        let igram: Array2<Complex32> =
            truth.mapv(|p| Complex32::from_polar(1.0, p.sin().atan2(p.cos())));
        let corr = Array2::<f32>::from_elem((m, n), 0.9);
        let mut mask = Array2::<bool>::from_elem((m, n), true);
        // Stripe of invalid pixels straight down the middle.
        for i in 0..m {
            mask[(i, n / 2)] = false;
        }
        let mask_view = mask.view();
        let costs = cost::compute_carballo_costs(igram.view(), corr.view(), 5.0, Some(mask_view));
        let graph = RectangularGridGraph::new(m + 1, n + 1);
        let residues = residue::compute_with_mask(
            igram.mapv(|z| z.arg()).view(),
            Some(mask_view),
        );
        let mut net =
            Network::new_with_mask(&graph, residues.view(), &costs, Some(mask_view));
        primal_dual::run(&graph, &mut net, 50);
        let labels =
            grow_components(&graph, &net, Some(mask_view), &ConnCompParams::default());
        // Left and right halves should be in different components.
        let left = labels[(m / 2, n / 4)];
        let right = labels[(m / 2, 3 * n / 4)];
        assert!(left > 0, "left half should be labeled");
        assert!(right > 0, "right half should be labeled");
        assert_ne!(left, right, "mask stripe should separate components");
    }

    #[test]
    fn small_components_are_dropped() {
        let m = 64;
        let n = 64;
        let truth = Array2::<f32>::zeros((m, n));
        let igram: Array2<Complex32> = truth.mapv(|p| Complex32::from_polar(1.0, p));
        let corr = Array2::<f32>::from_elem((m, n), 0.9);
        let mut mask = Array2::<bool>::from_elem((m, n), true);
        // Isolated 2x2 island.
        for i in 0..m {
            for j in 0..n {
                if !(i < 2 && j < 2) && (i < 4 || j < 4) {
                    mask[(i, j)] = false;
                }
            }
        }
        let mask_view = mask.view();
        let costs = cost::compute_carballo_costs(igram.view(), corr.view(), 5.0, Some(mask_view));
        let graph = RectangularGridGraph::new(m + 1, n + 1);
        let residues = residue::compute_with_mask(
            igram.mapv(|z| z.arg()).view(),
            Some(mask_view),
        );
        let mut net =
            Network::new_with_mask(&graph, residues.view(), &costs, Some(mask_view));
        primal_dual::run(&graph, &mut net, 50);
        let params = ConnCompParams { min_size_frac: 0.01, ..Default::default() };
        let labels = grow_components(&graph, &net, Some(mask_view), &params);
        // The 2x2 island is 4 pixels; 1% of valid pixels in (64*64 - masked)
        // is well above 4, so the island gets zeroed.
        assert_eq!(labels[(0, 0)], 0);
        assert_eq!(labels[(1, 1)], 0);
        // Main region should still be labeled.
        assert!(labels[(32, 32)] > 0);
    }
}
