//! Flow-corrected phase integration.
//!
//! Start at (0, 0); integrate down column 0, then across each row. For each
//! pixel-edge, the integer cycle correction is the *net flow* between the
//! two residue nodes bordering that edge:
//!     net_flow = arc_flow(reverse_direction) - arc_flow(forward_direction)
//!
//! For masked inputs, `integrate_with_mask` seeds at the first valid pixel
//! and BFS-walks the connected valid region; masked pixels are left as NaN.

use crate::grid::RectangularGridGraph;
use crate::network::Network;
use ndarray::{Array2, ArrayView2};
use std::collections::VecDeque;
use std::f32::consts::TAU;

/// Wrapped difference of two phase values, ∈ [-π, π).
#[inline]
fn wrapped_diff(a: f32, b: f32) -> f32 {
    let d = a - b;
    d - TAU * (d / TAU).round()
}

pub fn integrate(
    wrapped_phase: ArrayView2<f32>,
    g: &RectangularGridGraph,
    net: &Network,
) -> Array2<f32> {
    let (m, n) = wrapped_phase.dim();
    assert_eq!(g.m, m + 1);
    assert_eq!(g.n, n + 1);

    let mut unw = Array2::<f32>::zeros((m, n));
    unw[(0, 0)] = wrapped_phase[(0, 0)];

    // Down column 0.
    let mut phi = wrapped_phase[(0, 0)] as f64;
    for i in 1..m {
        let dpsi = wrapped_diff(wrapped_phase[(i, 0)], wrapped_phase[(i - 1, 0)]);
        // Pixel edge between (i-1, 0) and (i, 0) is *vertical* — its residue
        // arcs are RIGHT (forward direction = j increasing) between residues
        // (i, 0) and (i, 1).
        let fwd = g.right_arc(i, 0).unwrap();
        let rev = g.left_arc(i, 1).unwrap();
        let net_flow = net.arc_flow(g, rev) - net.arc_flow(g, fwd);
        let dphi = dpsi + TAU * (net_flow as f32);
        phi += dphi as f64;
        unw[(i, 0)] = phi as f32;
    }

    // Across each row.
    for i in 0..m {
        let mut phi = unw[(i, 0)] as f64;
        for j in 1..n {
            let dpsi = wrapped_diff(wrapped_phase[(i, j)], wrapped_phase[(i, j - 1)]);
            // Pixel edge between (i, j-1) and (i, j) is *horizontal* — residue
            // arcs are DOWN (forward = i increasing) between residues
            // (i, j) and (i+1, j).
            let fwd = g.down_arc(i, j).unwrap();
            let rev = g.up_arc(i + 1, j).unwrap();
            let net_flow = net.arc_flow(g, fwd) - net.arc_flow(g, rev);
            let dphi = dpsi + TAU * (net_flow as f32);
            phi += dphi as f64;
            unw[(i, j)] = phi as f32;
        }
    }

    unw
}

/// Like [`integrate`] but skips masked-out pixels.
///
/// Seeds at the first valid pixel found in raster order, then BFS-walks the
/// 4-connected valid region. Masked pixels are left as NaN. If the valid
/// region has multiple disconnected components, only the one containing the
/// seed gets integrated; other components stay NaN.
pub fn integrate_with_mask(
    wrapped_phase: ArrayView2<f32>,
    g: &RectangularGridGraph,
    net: &Network,
    mask: Option<ArrayView2<bool>>,
) -> Array2<f32> {
    let (m, n) = wrapped_phase.dim();
    assert_eq!(g.m, m + 1);
    assert_eq!(g.n, n + 1);

    // No mask → fast path matches the original integrate.
    let Some(mask) = mask else {
        return integrate(wrapped_phase, g, net);
    };
    assert_eq!(mask.dim(), (m, n), "mask must be pixel-grid sized");

    let mut unw = Array2::<f32>::from_elem((m, n), f32::NAN);

    // Find the seed: first mask=true pixel in raster order.
    let mut seed: Option<(usize, usize)> = None;
    'outer: for i in 0..m {
        for j in 0..n {
            if mask[(i, j)] {
                seed = Some((i, j));
                break 'outer;
            }
        }
    }
    let Some((si, sj)) = seed else { return unw }; // all masked

    unw[(si, sj)] = wrapped_phase[(si, sj)];

    // BFS the valid 4-connected region.
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((si, sj));
    while let Some((i, j)) = queue.pop_front() {
        let phi_here = unw[(i, j)] as f64;
        let psi_here = wrapped_phase[(i, j)];

        // RIGHT neighbor (i, j+1): horizontal pixel-edge; residue arcs DOWN/UP.
        if j + 1 < n && mask[(i, j + 1)] && unw[(i, j + 1)].is_nan() {
            let dpsi = wrapped_diff(wrapped_phase[(i, j + 1)], psi_here);
            let fwd = g.down_arc(i, j + 1).unwrap();
            let rev = g.up_arc(i + 1, j + 1).unwrap();
            let net_flow = net.arc_flow(g, fwd) - net.arc_flow(g, rev);
            let dphi = dpsi + TAU * (net_flow as f32);
            unw[(i, j + 1)] = (phi_here + dphi as f64) as f32;
            queue.push_back((i, j + 1));
        }
        // LEFT neighbor (i, j-1): same edge, reversed sign.
        if j >= 1 && mask[(i, j - 1)] && unw[(i, j - 1)].is_nan() {
            let dpsi = wrapped_diff(wrapped_phase[(i, j - 1)], psi_here);
            let fwd = g.down_arc(i, j).unwrap();
            let rev = g.up_arc(i + 1, j).unwrap();
            // Going LEFT means we subtract the flow from RIGHT-direction integration.
            let net_flow = net.arc_flow(g, rev) - net.arc_flow(g, fwd);
            let dphi = dpsi + TAU * (net_flow as f32);
            unw[(i, j - 1)] = (phi_here + dphi as f64) as f32;
            queue.push_back((i, j - 1));
        }
        // DOWN neighbor (i+1, j): vertical pixel-edge; residue arcs RIGHT/LEFT.
        if i + 1 < m && mask[(i + 1, j)] && unw[(i + 1, j)].is_nan() {
            let dpsi = wrapped_diff(wrapped_phase[(i + 1, j)], psi_here);
            let fwd = g.right_arc(i + 1, j).unwrap();
            let rev = g.left_arc(i + 1, j + 1).unwrap();
            let net_flow = net.arc_flow(g, rev) - net.arc_flow(g, fwd);
            let dphi = dpsi + TAU * (net_flow as f32);
            unw[(i + 1, j)] = (phi_here + dphi as f64) as f32;
            queue.push_back((i + 1, j));
        }
        // UP neighbor (i-1, j).
        if i >= 1 && mask[(i - 1, j)] && unw[(i - 1, j)].is_nan() {
            let dpsi = wrapped_diff(wrapped_phase[(i - 1, j)], psi_here);
            let fwd = g.right_arc(i, j).unwrap();
            let rev = g.left_arc(i, j + 1).unwrap();
            let net_flow = net.arc_flow(g, fwd) - net.arc_flow(g, rev);
            let dphi = dpsi + TAU * (net_flow as f32);
            unw[(i - 1, j)] = (phi_here + dphi as f64) as f32;
            queue.push_back((i - 1, j));
        }
    }

    unw
}
