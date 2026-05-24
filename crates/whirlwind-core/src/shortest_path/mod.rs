//! Multi-source shortest paths on the residue residual graph.

pub mod dial;
pub mod heap;

use crate::grid::RectangularGridGraph;
use crate::network::Network;
use std::sync::OnceLock;

/// Backend selector for the multi-source Dijkstra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DijkstraBackend {
    /// Dial's bucket queue (default; fastest for bounded integer reduced costs).
    DialSerial,
    /// Parallel Dial's: process each bucket's nodes via rayon. Helps on
    /// residue-dense scenes where Dijkstra dominates.
    DialParallel,
    /// Binary-heap. Kept for comparison / debugging.
    Heap,
}

/// Read the backend from `WHIRLWIND_DIJKSTRA` (cached after first call).
/// Values: `heap`, `dial` / `serial` (default), `dial-par` / `parallel`.
///
/// Note on parallelism: the rayon-parallel Dial is a phase-1/phase-2 design
/// (parallel proposal collection + serial application). On the noisy-ramp
/// workloads we measured (M-series, 8 perf cores) it's *not* faster than
/// serial Dial because phase 2 dominates and is serial. It's kept as an
/// opt-in for further experimentation; for production use, leave the default.
pub fn backend() -> DijkstraBackend {
    static BE: OnceLock<DijkstraBackend> = OnceLock::new();
    *BE.get_or_init(|| match std::env::var("WHIRLWIND_DIJKSTRA").ok().as_deref() {
        Some("heap") => DijkstraBackend::Heap,
        Some("dial-par") | Some("parallel") => DijkstraBackend::DialParallel,
        // Default is serial Dial.
        _ => DijkstraBackend::DialSerial,
    })
}

/// Result of a multi-source Dijkstra over the residual graph from every
/// `excess_node` simultaneously, using reduced costs as arc lengths.
pub struct ShortestPaths {
    pub dist: Vec<i64>,
    pub pred_arc: Vec<i32>,     // arc id of the predecessor arc, -1 if none
    pub pred_node: Vec<i32>,    // tail of that arc
    pub source: Vec<i32>,       // which excess source reached this node, -1 if not reached
}

impl ShortestPaths {
    pub fn new(n_nodes: usize) -> Self {
        Self {
            dist: vec![i64::MAX; n_nodes],
            pred_arc: vec![-1; n_nodes],
            pred_node: vec![-1; n_nodes],
            source: vec![-1; n_nodes],
        }
    }

    pub fn was_reached(&self, node: usize) -> bool {
        self.source[node] >= 0
    }
}

/// Run multi-source Dijkstra over the residual graph using reduced costs.
/// Sources are nodes with positive excess; distance 0 at each.
///
/// Backend is selected once per process via `backend()` (env-var
/// `WHIRLWIND_DIJKSTRA`).
pub fn dijkstra_multi_source(
    g: &RectangularGridGraph,
    net: &Network,
) -> ShortestPaths {
    match backend() {
        DijkstraBackend::Heap => heap::run(g, net),
        DijkstraBackend::DialSerial => dial::run(g, net),
        DijkstraBackend::DialParallel => dial::run_parallel(g, net),
    }
}
