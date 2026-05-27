//! Flow-corrected phase integration.
//!
//! For each output pixel, the unwrapped phase is `wrapped[p] + 2π · K[p]`,
//! where `K[p]` is the running integer cycle count from the seed pixel along
//! the integration path. At each arc traversal we accumulate
//!
//! ```text
//! K[here] = K[prev] + wrap_n_cycle(ψ[here], ψ[prev]) + net_flow(arc)
//! ```
//!
//! with `wrap_n_cycle(a, b) ∈ {-1, 0, +1}` the integer offset such that
//! `wrap(a - b) = (a - b) + wrap_n_cycle(a, b) · 2π`, and `net_flow` the MCF
//! arc flow (the cycle correction MCF assigned to that edge).
//!
//! Doing this with integers — rather than accumulating the float quantity
//! `dpsi + 2π · net_flow` along the path — keeps the output exactly
//! congruent with the wrapped input modulo float rounding of the *single*
//! final multiplication `2π · K[p]`. The float-accumulator formulation
//! (whirlwind's old version, and SNAPHU's original `IntegratePhase`) has
//! error that grows with path length; the integer formulation has constant
//! error independent of image size.
//!
//! See Geoff Gunter's isce3 commit fe6cba72 for the same fix in SNAPHU.
//!
//! For masked inputs, `integrate_with_mask` seeds at the first valid pixel
//! and BFS-walks the connected valid region; masked pixels are left as NaN.

use crate::grid::RectangularGridGraph;
use crate::network::Network;
use ndarray::{Array2, ArrayView2};
use std::collections::VecDeque;
use std::f32::consts::TAU;

/// Integer N such that `wrap(a - b) = (a - b) + N · 2π`, i.e., the number
/// of 2π cycles that must be added to the raw difference to land it in
/// `(-π, π]`. Returns one of `{-1, 0, +1}` for `a, b ∈ [-π, π]`.
///
/// Uses the same `round` convention as the cost-pipeline's `wrap` so the
/// two stay companions: `wrap(a - b) = (a - b) - 2π · round((a - b) / 2π)`,
/// hence `N = -round((a - b) / 2π)`.
#[inline]
fn wrap_n_cycle(a: f32, b: f32) -> i32 {
    -((a - b) / TAU).round() as i32
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

    // Running integer cycle count `K[i, 0]` for the head-of-row pixel.
    // Each row's interior pixels inherit this and grow from it.
    let mut col0_cycles: i32 = 0;

    for i in 0..m {
        if i > 0 {
            // Vertical step (i-1, 0) → (i, 0). Pixel edge between
            // (i-1, 0) and (i, 0) is *vertical* — its residue arcs are
            // RIGHT (forward = j increasing) between residues (i, 0)
            // and (i, 1).
            let n_cyc = wrap_n_cycle(wrapped_phase[(i, 0)], wrapped_phase[(i - 1, 0)]);
            let fwd = g.right_arc(i, 0).unwrap();
            let rev = g.left_arc(i, 1).unwrap();
            let net_flow = net.arc_flow(g, rev) - net.arc_flow(g, fwd);
            col0_cycles += n_cyc + net_flow;
        }
        let mut cycles = col0_cycles;
        for j in 0..n {
            if j > 0 {
                // Horizontal step (i, j-1) → (i, j). Pixel edge is
                // *horizontal*; residue arcs are DOWN/UP between
                // (i, j) and (i+1, j).
                let n_cyc = wrap_n_cycle(wrapped_phase[(i, j)], wrapped_phase[(i, j - 1)]);
                let fwd = g.down_arc(i, j).unwrap();
                let rev = g.up_arc(i + 1, j).unwrap();
                let net_flow = net.arc_flow(g, fwd) - net.arc_flow(g, rev);
                cycles += n_cyc + net_flow;
            }
            unw[(i, j)] = wrapped_phase[(i, j)] + TAU * (cycles as f32);
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

    // Per-pixel integer cycle count `K[i, j]`. We track this alongside the
    // BFS instead of accumulating the unwrapped float; `unw[p]` is emitted
    // as `wrapped_phase[p] + 2π · K[p]` once per pixel.
    let mut cycles = Array2::<i32>::zeros((m, n));
    unw[(si, sj)] = wrapped_phase[(si, sj)];

    // BFS the valid 4-connected region.
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((si, sj));
    while let Some((i, j)) = queue.pop_front() {
        let psi_here = wrapped_phase[(i, j)];
        let k_here = cycles[(i, j)];

        // RIGHT neighbor (i, j+1): horizontal pixel-edge; residue arcs DOWN/UP.
        if j + 1 < n && mask[(i, j + 1)] && unw[(i, j + 1)].is_nan() {
            let n_cyc = wrap_n_cycle(wrapped_phase[(i, j + 1)], psi_here);
            let fwd = g.down_arc(i, j + 1).unwrap();
            let rev = g.up_arc(i + 1, j + 1).unwrap();
            let net_flow = net.arc_flow(g, fwd) - net.arc_flow(g, rev);
            let k = k_here + n_cyc + net_flow;
            cycles[(i, j + 1)] = k;
            unw[(i, j + 1)] = wrapped_phase[(i, j + 1)] + TAU * (k as f32);
            queue.push_back((i, j + 1));
        }
        // LEFT neighbor (i, j-1): same edge, reversed sign.
        if j >= 1 && mask[(i, j - 1)] && unw[(i, j - 1)].is_nan() {
            let n_cyc = wrap_n_cycle(wrapped_phase[(i, j - 1)], psi_here);
            let fwd = g.down_arc(i, j).unwrap();
            let rev = g.up_arc(i + 1, j).unwrap();
            // Going LEFT means we subtract the flow from RIGHT-direction integration.
            let net_flow = net.arc_flow(g, rev) - net.arc_flow(g, fwd);
            let k = k_here + n_cyc + net_flow;
            cycles[(i, j - 1)] = k;
            unw[(i, j - 1)] = wrapped_phase[(i, j - 1)] + TAU * (k as f32);
            queue.push_back((i, j - 1));
        }
        // DOWN neighbor (i+1, j): vertical pixel-edge; residue arcs RIGHT/LEFT.
        if i + 1 < m && mask[(i + 1, j)] && unw[(i + 1, j)].is_nan() {
            let n_cyc = wrap_n_cycle(wrapped_phase[(i + 1, j)], psi_here);
            let fwd = g.right_arc(i + 1, j).unwrap();
            let rev = g.left_arc(i + 1, j + 1).unwrap();
            let net_flow = net.arc_flow(g, rev) - net.arc_flow(g, fwd);
            let k = k_here + n_cyc + net_flow;
            cycles[(i + 1, j)] = k;
            unw[(i + 1, j)] = wrapped_phase[(i + 1, j)] + TAU * (k as f32);
            queue.push_back((i + 1, j));
        }
        // UP neighbor (i-1, j).
        if i >= 1 && mask[(i - 1, j)] && unw[(i - 1, j)].is_nan() {
            let n_cyc = wrap_n_cycle(wrapped_phase[(i - 1, j)], psi_here);
            let fwd = g.right_arc(i, j).unwrap();
            let rev = g.left_arc(i, j + 1).unwrap();
            let net_flow = net.arc_flow(g, fwd) - net.arc_flow(g, rev);
            let k = k_here + n_cyc + net_flow;
            cycles[(i - 1, j)] = k;
            unw[(i - 1, j)] = wrapped_phase[(i - 1, j)] + TAU * (k as f32);
            queue.push_back((i - 1, j));
        }
    }

    unw
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cost, network, residue, simulate};
    use ndarray::Array2;
    use num_complex::Complex32;
    use rand::SeedableRng;

    /// Unwrapped output must be congruent to the wrapped input modulo 2π
    /// up to f32 ULP at the unwrapped magnitude — by construction, since
    /// `unw[p] = wrapped[p] + 2π · K[p]` with `K` integer. This guards
    /// against any future regression to a float-accumulator integrator
    /// (SNAPHU's classic "millions of arcs sum drifts" bug; see Geoff
    /// Gunter's isce3 fix fe6cba72).
    #[test]
    fn unwrap_is_congruent_to_wrapped_input() {
        // Noisy 64² scene with a few hundred residues.
        let m = 64;
        let n = 64;
        let truth = simulate::diagonal_ramp((m, n));
        let gamma = Array2::<f32>::from_elem((m, n), 0.4);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (igram, cor) = simulate::simulate_ifg(&truth, &gamma, 4, &mut rng);
        let wrapped = igram.mapv(|z: Complex32| z.arg());

        let residues = residue::compute(wrapped.view());
        let costs = cost::compute_carballo_costs(igram.view(), cor.view(), 4.0, None);
        let g = RectangularGridGraph::new(m + 1, n + 1);
        let mut net = network::Network::new(&g, residues.view(), &costs);
        crate::primal_dual::run(&g, &mut net, 50);

        let unw = integrate(wrapped.view(), &g, &net);

        // wrap(unw - wrapped) should be ~0 at every pixel. The only float
        // error is the single multiplication `2π · K` and the subsequent
        // addition+wrap, so we expect ULP-scale residuals regardless of K.
        let mut max_residual: f32 = 0.0;
        for i in 0..m {
            for j in 0..n {
                let d = unw[(i, j)] - wrapped[(i, j)];
                let residual = (d - TAU * (d / TAU).round()).abs();
                if residual > max_residual {
                    max_residual = residual;
                }
            }
        }
        // 1e-4 rad is generous: f32 ULP at typical |unw| ~ 20 rad is ~3e-6,
        // and we never accumulate. The pre-fix float-accumulator integrator
        // would blow through this on much larger scenes; this 64² case is
        // mostly a sanity guard that the integer formulation is exact.
        assert!(
            max_residual < 1e-4,
            "non-congruent unwrap: max |wrap(unw - wrapped)| = {max_residual} rad"
        );
    }
}
