//! Binary-heap multi-source Dijkstra on the residual graph.
//!
//! Dial's bucket queue is faster for bounded integer reduced costs; we'll add
//! it when profiling shows it matters. For now the heap version is correct
//! and ~O((V+E) log V).

use super::ShortestPaths;
use crate::grid::RectangularGridGraph;
use crate::network::Network;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub fn run(g: &RectangularGridGraph, net: &Network) -> ShortestPaths {
    let n_nodes = g.num_nodes();
    let mut sp = ShortestPaths::new(n_nodes);
    let mut visited = vec![false; n_nodes];
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();

    // Seed every excess node at distance 0.
    for s in net.excess_nodes() {
        sp.dist[s] = 0;
        sp.source[s] = s as i32;
        // sources have no predecessor — pred_arc stays -1, pred_node stays -1
        heap.push(Reverse((0, s)));
    }

    while let Some(Reverse((d, u))) = heap.pop() {
        if visited[u] {
            continue;
        }
        if d > sp.dist[u] {
            continue;
        }
        visited[u] = true;
        // sp.source[u] is already coherent with pred_node[u]: either u was a
        // seed (set above), or it was set at relaxation time below.

        let (ui, uj) = g.node_ij(u);
        let out = g.outgoing(ui, uj);
        for &(arc, v) in out.iter() {
            if net.is_arc_saturated(arc) {
                continue;
            }
            let rc = net.reduced_cost(g, arc);
            debug_assert!(rc >= 0, "negative reduced cost on residual arc {arc}: {rc}");
            let nd = d.saturating_add(rc);
            if nd < sp.dist[v] {
                sp.dist[v] = nd;
                sp.pred_arc[v] = arc as i32;
                sp.pred_node[v] = u as i32;
                // Coherent source attribution: v inherits u's source.
                sp.source[v] = sp.source[u];
                heap.push(Reverse((nd, v)));
            }
        }
    }
    sp
}
