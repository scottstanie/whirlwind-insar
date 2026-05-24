//! Dial's bucket-queue Dijkstra.
//!
//! Reduced costs in primal-dual are non-negative integers bounded by some
//! `C_max`. A circular bucket queue of size `K = C_max + 1` lets us pop the
//! minimum in amortized O(1) and gives O(V + E + max_dist) total work per call
//! — competitive with the binary heap when C_max is small relative to V.

use super::ShortestPaths;
use crate::grid::RectangularGridGraph;
use crate::network::Network;

/// Multi-source Dijkstra over the residual graph using Dial's bucket queue.
pub fn run(g: &RectangularGridGraph, net: &Network) -> ShortestPaths {
    let n_nodes = g.num_nodes();
    let mut sp = ShortestPaths::new(n_nodes);

    // Find the maximum reduced cost over unsaturated arcs. This sets the
    // bucket count. The arc-scan is O(E), small relative to the Dijkstra
    // proper. Min 1 (covers the all-zero-cost edge case).
    let mut max_rc: i64 = 0;
    for a in 0..g.num_arcs() {
        if !net.is_arc_saturated(a) {
            let rc = net.reduced_cost(g, a);
            if rc > max_rc {
                max_rc = rc;
            }
        }
    }
    let k = (max_rc as usize).saturating_add(1).max(1);

    // Buckets store (node, queued_dist). queued_dist disambiguates stale
    // entries — when we pop a node whose sp.dist[u] no longer matches the
    // entry we pushed, we skip it.
    let mut buckets: Vec<Vec<(usize, i64)>> = vec![Vec::new(); k];
    let mut visited = vec![false; n_nodes];
    let mut pending: usize = 0;

    // Seed every excess node at distance 0.
    for s in net.excess_nodes() {
        sp.dist[s] = 0;
        sp.source[s] = s as i32;
        buckets[0].push((s, 0));
        pending += 1;
    }
    if pending == 0 {
        return sp;
    }

    let mut cur_bucket = 0_usize;
    let mut cur_dist: i64 = 0;
    let mut bucket_advances: usize = 0;

    while pending > 0 {
        // Advance to the next non-empty bucket. After K consecutive empty
        // advances we know no work remains.
        while buckets[cur_bucket].is_empty() {
            cur_bucket = (cur_bucket + 1) % k;
            cur_dist += 1;
            bucket_advances += 1;
            if bucket_advances > k {
                return sp;
            }
        }
        bucket_advances = 0;

        let (u, qd) = buckets[cur_bucket].pop().unwrap();
        pending -= 1;
        if visited[u] {
            continue;
        }
        // Stale: this entry was queued at qd but the node has since been
        // re-relaxed to a smaller dist.
        if sp.dist[u] != qd || qd != cur_dist {
            continue;
        }
        visited[u] = true;

        let (ui, uj) = g.node_ij(u);
        let out = g.outgoing(ui, uj);
        for &(arc, v) in out.iter() {
            if net.is_arc_saturated(arc) {
                continue;
            }
            let rc = net.reduced_cost(g, arc);
            debug_assert!(rc >= 0, "negative reduced cost on arc {arc}: {rc}");
            let nd = cur_dist + rc;
            if nd < sp.dist[v] {
                sp.dist[v] = nd;
                sp.pred_arc[v] = arc as i32;
                sp.pred_node[v] = u as i32;
                sp.source[v] = sp.source[u];
                let b = (nd as usize) % k;
                buckets[b].push((v, nd));
                pending += 1;
            }
        }
    }
    sp
}
