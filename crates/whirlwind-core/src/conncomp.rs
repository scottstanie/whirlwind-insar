//! SNAPHU-style connected component growing from a solved MCF network.
//!
//! After [`crate::primal_dual::run`] (or the SSP fallback) returns, every
//! pixel-edge in the original phase image is backed by two forward residue
//! arcs (one per Carballo direction). We label each pixel-edge as a *cut*
//! when at least one of its underlying arcs is unreliable:
//!
//! 1. The arc is forbidden (both directions saturated - masked-out pixel).
//! 2. The minimum raw forward cost across the two directions is below
//!    `cost_threshold` - the noise model is uninformative here, so any
//!    routing decision across this edge wasn't strongly supported by data.
//!
//! Then BFS the remaining pixels through non-cut edges to label connected
//! components. Small components are dropped; the top `max_ncomps` by size are
//! kept and renumbered 1..=max_ncomps.
//!
//! This is the gridded adaptation of `GrowConnCompsMask` in SNAPHU's
//! `snaphu_tile.c`. SNAPHU works with convex piecewise costs and tests
//! `min(negcost, poscost)` - the local cost-function flatness around the
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

/// Parameters for [`grow_components`]. The `cost_threshold` is compared against
/// the default Carballo coherence cost grid (`CARBALLO_COST_SCALE = 6`); the
/// default of 50 corresponds to a per-edge one-cycle probability of ~2.4e-4.
#[derive(Debug, Clone)]
pub struct ConnCompParams {
    /// Cut a pixel edge when min raw forward cost across the two underlying
    /// arcs is ≤ this. Higher → fewer/larger components.
    pub cost_threshold: i32,
    /// Drop components smaller than this many pixels. This ABSOLUTE floor is the
    /// binding control: at 80 m it is 0.8 km/side, at 30 m 0.3 km - scene-size-
    /// and pixel-spacing-invariant (matches SNAPHU's `minregionsize`). A small
    /// coherent island stays a usable, self-consistent component the caller can
    /// re-reference into; only sub-floor speckle is dropped.
    pub min_size_px: usize,
    /// Vestigial fractional floor, kept only as an anti-pathology cap on huge
    /// frames (it can only RAISE `min_size_px`, never lower it). At the default
    /// 1e-4 it stays negligible (<100 px below ~1M valid). Do NOT raise toward
    /// 0.01 - on a NISAR frame 1% is a ~25 km minimum feature, which orphans
    /// every island a user might want to reference into. The effective floor is
    /// `max(min_size_px, ceil(min_size_frac * n_valid))`, so this fraction only
    /// bites when it exceeds the absolute `min_size_px` control.
    pub min_size_frac: f32,
    /// Keep at most this many components (largest by size). 0 → keep all. The
    /// `min_size_px` floor is the real speckle control; this is only a guard
    /// against a pathological scene emitting tens of thousands of labels, set
    /// generously so it never clips a genuine feature the floor admits.
    pub max_ncomps: u32,
}

impl Default for ConnCompParams {
    fn default() -> Self {
        Self {
            cost_threshold: 50,
            min_size_px: 100,
            min_size_frac: 0.0001,
            max_ncomps: 1024,
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
    (net.cost_fwd[fwd1].min(net.cost_fwd[fwd2]) as i32) <= thresh
}

/// Grow components on the pixel grid using a solved MCF network. Returns a
/// `(m_phase, n_phase)` `u32` label array; 0 = unassigned (cut off or smaller
/// than `max(min_size_px, ceil(min_size_frac * n_valid))`), renumbered by size.
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
        assert_eq!(
            mm.dim(),
            (m_phase, n_phase),
            "pixel mask must be (m-1, n-1)"
        );
    }

    let valid = |i: usize, j: usize| pixel_mask.map(|m| m[(i, j)]).unwrap_or(true);
    let n_valid: usize = (0..m_phase)
        .flat_map(|i| (0..n_phase).map(move |j| (i, j)))
        .filter(|&(i, j)| valid(i, j))
        .count();
    // Absolute floor governs; the fraction only ever RAISES it (a generous cap
    // on huge frames), so a coherent island down to `min_size_px` is kept.
    let frac_floor = (params.min_size_frac as f64 * n_valid as f64).ceil() as usize;
    let min_size = params.min_size_px.max(frac_floor).max(1);

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
        let residues = residue::compute_with_mask(igram.mapv(|z| z.arg()).view(), Some(mask_view));
        let mut net = Network::new_with_mask(&graph, residues.view(), &costs, Some(mask_view));
        primal_dual::run(&graph, &mut net, 50);
        let labels = grow_components(&graph, &net, Some(mask_view), &ConnCompParams::default());
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
        // Two isolated islands separated from the main body by a masked moat:
        // a tiny 2x2 (4 px, below the 100-px floor) and a 12x12 (144 px, above).
        // A one-pixel masked ring around each isolates it.
        for i in 0..m {
            for j in 0..n {
                let small = i < 2 && j < 2;
                let big = (4..16).contains(&i) && (4..16).contains(&j);
                let moat = (i < 3 && j < 3) || ((3..17).contains(&i) && (3..17).contains(&j));
                if moat && !small && !big {
                    mask[(i, j)] = false;
                }
            }
        }
        let mask_view = mask.view();
        let costs = cost::compute_carballo_costs(igram.view(), corr.view(), 5.0, Some(mask_view));
        let graph = RectangularGridGraph::new(m + 1, n + 1);
        let residues = residue::compute_with_mask(igram.mapv(|z| z.arg()).view(), Some(mask_view));
        let mut net = Network::new_with_mask(&graph, residues.view(), &costs, Some(mask_view));
        primal_dual::run(&graph, &mut net, 50);
        // Default policy: absolute 100-px floor governs (frac is a negligible cap).
        let labels = grow_components(&graph, &net, Some(mask_view), &ConnCompParams::default());
        // The 2x2 island (4 px < 100) is dropped...
        assert_eq!(labels[(0, 0)], 0);
        // ...the 12x12 island (144 px >= 100) SURVIVES as its own component
        // (the whole point: a small coherent island is NOT dropped for being
        // disconnected - the caller can re-reference into it)...
        assert!(labels[(9, 9)] > 0);
        // ...and the main body is labeled.
        assert!(labels[(40, 40)] > 0);

        // The old 1% fraction would have orphaned the 12x12 island: assert the
        // absolute floor is what keeps it (raising min_size_px past 144 drops it).
        let strict = ConnCompParams {
            min_size_px: 300,
            ..Default::default()
        };
        let labels_strict = grow_components(&graph, &net, Some(mask_view), &strict);
        assert_eq!(labels_strict[(9, 9)], 0);
    }
}
