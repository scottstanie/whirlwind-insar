//! Binary-heap multi-source Dijkstra on the residual graph.
//!
//! Dial's bucket queue is faster for bounded integer reduced costs; we'll add
//! it when profiling shows it matters. For now the heap version is correct
//! and ~O((V+E) log V).

use super::ShortestPaths;
use crate::network::Network;
use crate::residual_graph::ResidualGraph;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub fn run<G: ResidualGraph>(g: &G, net: &Network) -> ShortestPaths {
    let mut sp = ShortestPaths::new(net.num_nodes());
    run_into(g, net, &mut sp);
    sp
}

/// Reusable-buffer variant of [`run`].
pub fn run_into<G: ResidualGraph>(g: &G, net: &Network, sp: &mut ShortestPaths) {
    let n_nodes = net.num_nodes();
    sp.reset(n_nodes);
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();

    // Count sinks for early-exit.
    let mut sinks_left = 0_usize;
    let mut is_sink = vec![false; n_nodes];
    for (v, &e) in net.excess.iter().enumerate() {
        if e < 0 {
            is_sink[v] = true;
            sinks_left += 1;
        }
    }

    // Seed every excess node at distance 0.
    for s in net.excess_nodes() {
        sp.dist[s] = 0;
        // sources have no predecessor - pred_arc stays -1
        heap.push(Reverse((0, s)));
    }
    if sinks_left == 0 {
        return;
    }

    let mut buf: Vec<(usize, usize)> = Vec::with_capacity(8);
    while let Some(Reverse((d, u))) = heap.pop() {
        if sp.popped[u] {
            continue;
        }
        if d > sp.dist[u] {
            continue;
        }
        sp.popped[u] = true;
        if is_sink[u] {
            sinks_left -= 1;
            if sinks_left == 0 {
                return;
            }
        }
        let pot_u = net.potential[u];
        buf.clear();
        if u < g.num_nodes() {
            g.outgoing(u, &mut buf);
        }
        for &(arc, v) in net.extra_outgoing(u).iter() {
            buf.push((arc, v));
        }
        for &(arc, v) in buf.iter() {
            if net.is_arc_saturated(arc) {
                continue;
            }
            // Inline reduced_cost (tail=u, head=v are known from outgoing()).
            // Reuse: used arcs get reduced cost 0. Convex: use marginal cost.
            let rc = if net.is_used(arc) {
                0
            } else if net.convex_mode {
                net.marginal_cost(arc) - pot_u + net.potential[v]
            } else {
                net.arc_cost(g, arc) as i64 - pot_u + net.potential[v]
            };
            debug_assert!(rc >= 0, "negative reduced cost on residual arc {arc}: {rc}");
            let nd = d.saturating_add(rc);
            if nd < sp.dist[v] {
                sp.dist[v] = nd;
                sp.pred_arc[v] = arc as i64;
                heap.push(Reverse((nd, v)));
            }
        }
    }
}
