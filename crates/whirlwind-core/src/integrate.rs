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

/// Like [`integrate`] but adds an externally-provided unit flow on top of
/// `net`'s flow per edge. Used to integrate the *combined* (warm-start +
/// PD-routed correction) flow when a network was built with
/// [`crate::network::Network::new_with_initial_flow`].
pub fn integrate_with_initial_flow(
    wrapped_phase: ArrayView2<f32>,
    g: &RectangularGridGraph,
    net: &Network,
    initial_flow: &[i8],
) -> Array2<f32> {
    let (m, n) = wrapped_phase.dim();
    assert_eq!(g.m, m + 1);
    assert_eq!(g.n, n + 1);
    assert_eq!(initial_flow.len(), g.num_forward);

    let init = |arc: usize| -> i32 { initial_flow[arc] as i32 };

    let mut unw = Array2::<f32>::zeros((m, n));
    unw[(0, 0)] = wrapped_phase[(0, 0)];

    let mut phi = wrapped_phase[(0, 0)] as f64;
    for i in 1..m {
        let dpsi = wrapped_diff(wrapped_phase[(i, 0)], wrapped_phase[(i - 1, 0)]);
        let fwd = g.right_arc(i, 0).unwrap();
        let rev = g.left_arc(i, 1).unwrap();
        let net_flow = (net.arc_flow(g, rev) + init(rev)) - (net.arc_flow(g, fwd) + init(fwd));
        let dphi = dpsi + TAU * (net_flow as f32);
        phi += dphi as f64;
        unw[(i, 0)] = phi as f32;
    }

    for i in 0..m {
        let mut phi = unw[(i, 0)] as f64;
        for j in 1..n {
            let dpsi = wrapped_diff(wrapped_phase[(i, j)], wrapped_phase[(i, j - 1)]);
            let fwd = g.down_arc(i, j).unwrap();
            let rev = g.up_arc(i + 1, j).unwrap();
            let net_flow = (net.arc_flow(g, fwd) + init(fwd)) - (net.arc_flow(g, rev) + init(rev));
            let dphi = dpsi + TAU * (net_flow as f32);
            phi += dphi as f64;
            unw[(i, j)] = phi as f32;
        }
    }

    unw
}

/// Like [`integrate_with_mask`] but adds an externally-provided unit flow
/// on top of `net`'s flow per edge. See [`integrate_with_initial_flow`].
pub fn integrate_with_mask_and_initial_flow(
    wrapped_phase: ArrayView2<f32>,
    g: &RectangularGridGraph,
    net: &Network,
    mask: Option<ArrayView2<bool>>,
    initial_flow: &[i8],
) -> Array2<f32> {
    let (m, n) = wrapped_phase.dim();
    assert_eq!(g.m, m + 1);
    assert_eq!(g.n, n + 1);
    assert_eq!(initial_flow.len(), g.num_forward);

    let Some(mask) = mask else {
        return integrate_with_initial_flow(wrapped_phase, g, net, initial_flow);
    };
    assert_eq!(mask.dim(), (m, n), "mask must be pixel-grid sized");

    let init = |arc: usize| -> i32 { initial_flow[arc] as i32 };
    let combined_flow = |arc: usize| -> i32 { net.arc_flow(g, arc) + init(arc) };

    let mut unw = Array2::<f32>::from_elem((m, n), f32::NAN);
    let mut seed: Option<(usize, usize)> = None;
    'outer: for i in 0..m {
        for j in 0..n {
            if mask[(i, j)] {
                seed = Some((i, j));
                break 'outer;
            }
        }
    }
    let Some((si, sj)) = seed else { return unw };
    unw[(si, sj)] = wrapped_phase[(si, sj)];

    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((si, sj));
    while let Some((i, j)) = queue.pop_front() {
        let phi_here = unw[(i, j)] as f64;
        let psi_here = wrapped_phase[(i, j)];

        if j + 1 < n && mask[(i, j + 1)] && unw[(i, j + 1)].is_nan() {
            let dpsi = wrapped_diff(wrapped_phase[(i, j + 1)], psi_here);
            let fwd = g.down_arc(i, j + 1).unwrap();
            let rev = g.up_arc(i + 1, j + 1).unwrap();
            let net_flow = combined_flow(fwd) - combined_flow(rev);
            let dphi = dpsi + TAU * (net_flow as f32);
            unw[(i, j + 1)] = (phi_here + dphi as f64) as f32;
            queue.push_back((i, j + 1));
        }
        if j >= 1 && mask[(i, j - 1)] && unw[(i, j - 1)].is_nan() {
            let dpsi = wrapped_diff(wrapped_phase[(i, j - 1)], psi_here);
            let fwd = g.down_arc(i, j).unwrap();
            let rev = g.up_arc(i + 1, j).unwrap();
            let net_flow = combined_flow(rev) - combined_flow(fwd);
            let dphi = dpsi + TAU * (net_flow as f32);
            unw[(i, j - 1)] = (phi_here + dphi as f64) as f32;
            queue.push_back((i, j - 1));
        }
        if i + 1 < m && mask[(i + 1, j)] && unw[(i + 1, j)].is_nan() {
            let dpsi = wrapped_diff(wrapped_phase[(i + 1, j)], psi_here);
            let fwd = g.right_arc(i + 1, j).unwrap();
            let rev = g.left_arc(i + 1, j + 1).unwrap();
            let net_flow = combined_flow(rev) - combined_flow(fwd);
            let dphi = dpsi + TAU * (net_flow as f32);
            unw[(i + 1, j)] = (phi_here + dphi as f64) as f32;
            queue.push_back((i + 1, j));
        }
        if i >= 1 && mask[(i - 1, j)] && unw[(i - 1, j)].is_nan() {
            let dpsi = wrapped_diff(wrapped_phase[(i - 1, j)], psi_here);
            let fwd = g.right_arc(i, j).unwrap();
            let rev = g.left_arc(i, j + 1).unwrap();
            let net_flow = combined_flow(fwd) - combined_flow(rev);
            let dphi = dpsi + TAU * (net_flow as f32);
            unw[(i - 1, j)] = (phi_here + dphi as f64) as f32;
            queue.push_back((i - 1, j));
        }
    }

    unw
}

/// Inverse of [`integrate`]: given an unwrapped phase, compute the
/// per-forward-grid-arc unit-flow assignment that `integrate` would need
/// to reproduce that unwrap from `wrapped_phase`.
///
/// Returned `flow` has one entry per forward grid arc; each value is 0 or
/// 1 (number of units of forward flow on that arc). Returns 0 wherever
/// either endpoint of the pixel edge is non-finite in `unw` (e.g. masked
/// regions left as NaN by `integrate_with_mask`).
///
/// `n_clamped` counts pixel edges where the unwrap demanded |k| > 1 (which
/// is unrepresentable in our unit-capacity network); those edges are clamped
/// to ±1 and the residual excess is left for the MCF solver to resolve.
pub fn flow_from_unwrap(
    wrapped_phase: ArrayView2<f32>,
    unw: ArrayView2<f32>,
    g: &RectangularGridGraph,
) -> (Vec<i8>, usize) {
    let (m, n) = wrapped_phase.dim();
    assert_eq!(unw.dim(), (m, n));
    assert_eq!(g.m, m + 1);
    assert_eq!(g.n, n + 1);

    let mut flow = vec![0i8; g.num_forward];
    let mut n_clamped = 0usize;

    // Vertical pixel edges (between (i-1, j) and (i, j)).
    // From `integrate`: dphi = dpsi + 2π · (arc_flow(left_arc(i, j+1))
    //                                       − arc_flow(right_arc(i, j))).
    for i in 1..m {
        for j in 0..n {
            let a = unw[(i, j)];
            let b = unw[(i - 1, j)];
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            let dphi = a - b;
            let dpsi = wrapped_diff(wrapped_phase[(i, j)], wrapped_phase[(i - 1, j)]);
            let k_raw = ((dphi - dpsi) / TAU).round() as i32;
            let k = k_raw.clamp(-1, 1);
            if k != k_raw {
                n_clamped += 1;
            }
            match k {
                1 => {
                    flow[g.left_arc(i, j + 1).unwrap()] = 1;
                }
                -1 => {
                    flow[g.right_arc(i, j).unwrap()] = 1;
                }
                _ => {}
            }
        }
    }

    // Horizontal pixel edges (between (i, j-1) and (i, j)).
    // From `integrate`: dphi = dpsi + 2π · (arc_flow(down_arc(i, j))
    //                                       − arc_flow(up_arc(i+1, j))).
    for i in 0..m {
        for j in 1..n {
            let a = unw[(i, j)];
            let b = unw[(i, j - 1)];
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            let dphi = a - b;
            let dpsi = wrapped_diff(wrapped_phase[(i, j)], wrapped_phase[(i, j - 1)]);
            let k_raw = ((dphi - dpsi) / TAU).round() as i32;
            let k = k_raw.clamp(-1, 1);
            if k != k_raw {
                n_clamped += 1;
            }
            match k {
                1 => {
                    flow[g.down_arc(i, j).unwrap()] = 1;
                }
                -1 => {
                    flow[g.up_arc(i + 1, j).unwrap()] = 1;
                }
                _ => {}
            }
        }
    }

    (flow, n_clamped)
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

    // No mask → fall back to the unmasked fast path.
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

