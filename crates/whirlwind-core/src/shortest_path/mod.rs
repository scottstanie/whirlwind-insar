//! Multi-source shortest paths on the residue residual graph.

pub mod dial;
pub mod heap;

use crate::grid::RectangularGridGraph;
use crate::network::Network;

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
/// By default uses Dial's bucket queue (faster for bounded integer reduced
/// costs, which is the primal-dual regime). Set `WHIRLWIND_DIJKSTRA=heap` to
/// force the binary-heap variant for comparison / debugging.
pub fn dijkstra_multi_source(
    g: &RectangularGridGraph,
    net: &Network,
) -> ShortestPaths {
    let use_heap = std::env::var("WHIRLWIND_DIJKSTRA").ok().as_deref() == Some("heap");
    if use_heap {
        heap::run(g, net)
    } else {
        dial::run(g, net)
    }
}
