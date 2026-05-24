//! Flow-corrected phase integration.
//!
//! Start at (0, 0); integrate down column 0, then across each row. For each
//! pixel-edge, the integer cycle correction is the *net flow* between the
//! two residue nodes bordering that edge:
//!     net_flow = arc_flow(reverse_direction) - arc_flow(forward_direction)

use crate::grid::RectangularGridGraph;
use crate::network::Network;
use ndarray::{Array2, ArrayView2};
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
