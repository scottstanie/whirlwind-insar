//! Whirlwind: Bayesian minimum-cost-flow phase unwrapper.
//!
//! Pipeline:
//! 1. Compute residues from wrapped phase.
//! 2. Compute Carballo-style Bayesian edge costs from the interferogram + coherence.
//! 3. Solve a min-cost flow problem on the residue grid (primal-dual SSP).
//! 4. Integrate the flow-corrected wrapped gradients to recover unwrapped phase.

pub mod cost;
pub mod grid;
pub mod integrate;
pub mod network;
pub mod primal_dual;
pub mod residue;
pub mod shortest_path;
pub mod simulate;
pub mod ssp;

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

/// Top-level phase unwrap.
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
