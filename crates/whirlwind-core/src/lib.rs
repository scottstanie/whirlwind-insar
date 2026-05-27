//! Whirlwind: Bayesian minimum-cost-flow phase unwrapper.
//!
//! Pipeline:
//! 1. Compute residues from wrapped phase.
//! 2. Compute Carballo-style Bayesian edge costs from the interferogram + coherence.
//! 3. Solve a min-cost flow problem on the residue grid (primal-dual SSP).
//! 4. Integrate the flow-corrected wrapped gradients to recover unwrapped phase.

pub mod closure;
pub mod conncomp;
pub mod cost;
pub mod goldstein;
pub mod grid;
pub mod integrate;
pub mod network;
pub mod primal_dual;
pub mod residual_graph;
pub mod residue;
pub mod shortest_path;
pub mod simulate;
pub mod ssp;
pub mod tile;

pub use conncomp::ConnCompParams;
pub use residual_graph::ResidualGraph;

use ndarray::{Array2, ArrayView2};
use num_complex::Complex32;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum UnwrapError {
    #[error("igram and coherence shape mismatch: {0:?} vs {1:?}")]
    ShapeMismatch((usize, usize), (usize, usize)),
    #[error("array too small: at least 2x2 required, got {0:?}")]
    TooSmall((usize, usize)),
}

/// Top-level phase unwrap (coherence-based cost — for raw boxcar IGs).
///
/// * `igram` — complex interferogram of shape `(m, n)`.
/// * `corr`  — sample coherence in `[0, 1]` of shape `(m, n)`.
/// * `nlooks` — effective number of looks (≥ 1).
/// * `mask` — optional valid-pixel mask (True = valid); same shape.
///
/// Returns an `(m, n)` unwrapped phase array (`f32`).
pub fn unwrap(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != corr.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), corr.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }

    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);

    let costs = cost::compute_carballo_costs(igram, corr, nlooks, mask);

    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new_with_mask(&graph, residues.view(), &costs, mask);

    // max_iter=50: per primal-dual iteration we run one multi-source Dijkstra
    // that batches every source's augmentation; the SSP fallback does one
    // Dijkstra *per source*. On very noisy data (hundreds of thousands of
    // residues) it's ~6× faster end-to-end to run more PD iters and skip
    // SSP entirely. See `examples/bench_scale.rs`.
    primal_dual::run(&graph, &mut net, 50);

    let unw = if mask.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mask)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    Ok(unw)
}

/// [`unwrap`] + SNAPHU-style connected components from the same MCF solve.
///
/// Equivalent to running `unwrap` and then growing components on the resulting
/// MCF network — but does the (expensive) solve only once. See
/// [`conncomp::grow_components`] for the cut/region-growing rules.
pub fn unwrap_with_components(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    params: ConnCompParams,
) -> Result<(Array2<f32>, Array2<u32>), UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != corr.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), corr.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let costs = cost::compute_carballo_costs(igram, corr, nlooks, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new_with_mask(&graph, residues.view(), &costs, mask);
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mask.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mask)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    let comps = conncomp::grow_components(&graph, &net, mask, &params);
    Ok((unw, comps))
}

/// Top-level phase unwrap (CRLB-weighted cost — for phase-linked IGs).
///
/// For interferograms formed from phase-linked SLCs (Dolphin, EVD, EMI),
/// the proper per-pixel noise weight is the CRLB-derived phase variance,
/// not the sliding-window sample coherence used by [`unwrap`].
///
/// * `igram` — complex interferogram, shape `(m, n)`.
/// * `variance` — per-pixel phase variance σ²_IG = σ²_a + σ²_b in rad²,
///   typically `crlb_<date_a>.tif + crlb_<date_b>.tif`. NoData = 0 is fine.
/// * `mask` — optional valid-pixel mask.
pub fn unwrap_crlb(
    igram: ArrayView2<Complex32>,
    variance: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != variance.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), variance.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let costs = cost::compute_crlb_costs(igram, variance, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new_with_mask(&graph, residues.view(), &costs, mask);
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mask.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mask)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    Ok(unw)
}

/// [`unwrap_crlb`] + SNAPHU-style connected components from the same solve.
pub fn unwrap_crlb_with_components(
    igram: ArrayView2<Complex32>,
    variance: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    params: ConnCompParams,
) -> Result<(Array2<f32>, Array2<u32>), UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != variance.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), variance.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let costs = cost::compute_crlb_costs(igram, variance, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new_with_mask(&graph, residues.view(), &costs, mask);
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mask.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mask)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    let comps = conncomp::grow_components(&graph, &net, mask, &params);
    Ok((unw, comps))
}

/// Top-level CRLB-weighted phase unwrap with a virtual ground node.
///
/// Adds a single ground node connected to every boundary residue with a
/// unit-capacity forward arc of cost `ground_cost`. Wrap-line endpoints
/// can then terminate at the image boundary independently of each other,
/// fixing the capacity-1 stacking limitation of [`unwrap_crlb`].
///
/// * `ground_cost = 0` — ground is free. Best for clean inputs whose
///   wrap-lines all exit at the boundary (e.g. smooth ramps with no
///   interior residues): MCF drains every boundary residue to ground
///   independently and places no spurious flow on interior arcs, leaving
///   the Itoh integration alone to recover the unwrap.
/// * `ground_cost > 0` — ground is preferred only when it's cheaper than
///   pairing with an opposite-sign interior residue along an internal
///   path. For data with dense interior residues (real noisy IGs), a
///   moderate positive cost keeps internal routing for the bulk of
///   residues while still draining boundary-only wrap-lines to ground.
pub fn unwrap_crlb_grounded(
    igram: ArrayView2<Complex32>,
    variance: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    ground_cost: i32,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != variance.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), variance.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let costs = cost::compute_crlb_costs(igram, variance, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new_with_mask_and_ground(
        &graph, residues.view(), &costs, mask, Some(ground_cost),
    );
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mask.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mask)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    Ok(unw)
}
