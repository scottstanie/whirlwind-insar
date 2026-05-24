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
    let residues = residue::compute(wrapped_phase.view());

    let costs = cost::compute_carballo_costs(igram, corr, nlooks, mask);

    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new(&graph, residues.view(), &costs);

    // max_iter=50 (vs the original Whirlwind's 8): per-iteration cost is one
    // multi-source Dijkstra that batches every source's augmentation, while
    // the SSP fallback does one Dijkstra *per source*. On very noisy data
    // (hundreds of thousands of residues), running more PD iters and skipping
    // SSP entirely is ~6× faster end to end. See `examples/bench_scale.rs`.
    primal_dual::run(&graph, &mut net, 50);

    let unw = integrate::integrate(wrapped_phase.view(), &graph, &net);
    Ok(unw)
}
