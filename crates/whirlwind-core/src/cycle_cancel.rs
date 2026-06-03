//! Negative-cycle canceling for the convex (SNAPHU-smooth) min-cost flow
//! (issue #65).
//!
//! The fast `primal_dual` batched augment produces a FEASIBLE flow but, at
//! NISAR scale, lands far above the convex optimum: it places ~all flow in a
//! few iterations at one stale Dijkstra snapshot's marginals, and nothing
//! re-optimizes the result (the SSP fallback only drains residual excess). The
//! converged flow then still admits negative-reduced-cost residual cycles —
//! and by the negative-cycle optimality theorem, a feasible flow is optimal
//! iff its residual graph has none.
//!
//! This module cancels them: repeatedly find a negative-marginal residual cycle
//! (SPFA from a virtual super-source) and push one unit around it. A cycle push
//! is excess-neutral (each node has one in- and one out-arc on the cycle), so
//! feasibility is preserved; the convex objective strictly decreases by the
//! (negative) sum of the cycle's marginals. Because the per-arc cost is convex
//! (marginal non-decreasing in flow), the same cycle's marginal sum rises after
//! a push, so the process converges to the optimum (no negative cycle).
//!
//! Used as a POLISH on the fast feasible flow — the cheaper the warm-start
//! (e.g. a tiled "mostly right" solve), the fewer cycles remain to cancel.

use crate::network::Network;
use crate::residual_graph::ResidualGraph;
use std::collections::VecDeque;

/// Find and cancel negative residual cycles until the convex flow is optimal
/// (no negative cycle) or `max_cycles` cancels are done. Returns the number of
/// cycles cancelled.
pub fn cancel_negative_cycles<G: ResidualGraph>(
    g: &G,
    net: &mut Network,
    max_cycles: usize,
) -> usize {
    let n = net.num_nodes();
    let total_arcs = 2 * net.num_forward();

    // Static CSR adjacency (arc ids grouped by tail). Topology is fixed across
    // the run; per-arc saturation and marginal cost are read LIVE during each
    // relaxation (they change only between cancels, when we push).
    let mut starts = vec![0u32; n + 1];
    for arc in 0..total_arcs {
        let (t, _h) = net.arc_endpoints(g, arc);
        starts[t + 1] += 1;
    }
    for i in 0..n {
        starts[i + 1] += starts[i];
    }
    let mut adj = vec![0u32; total_arcs];
    let mut cursor = starts.clone();
    for arc in 0..total_arcs {
        let (t, _h) = net.arc_endpoints(g, arc);
        adj[cursor[t] as usize] = arc as u32;
        cursor[t] += 1;
    }

    // Scratch reused across cancels.
    let mut dist = vec![0_i64; n];
    let mut pred_arc = vec![-1_i64; n];
    let mut in_q = vec![false; n];
    let mut cnt = vec![0_u32; n];
    let mut q: VecDeque<usize> = VecDeque::with_capacity(n);

    let mut cancelled = 0usize;
    while cancelled < max_cycles {
        // SPFA from a super-source: every node starts at dist 0 (so any
        // negative cycle anywhere in the residual graph is reachable).
        dist.iter_mut().for_each(|d| *d = 0);
        pred_arc.iter_mut().for_each(|p| *p = -1);
        in_q.iter_mut().for_each(|x| *x = true);
        cnt.iter_mut().for_each(|c| *c = 0);
        q.clear();
        q.extend(0..n);

        let mut hit: Option<usize> = None;
        while let Some(u) = q.pop_front() {
            in_q[u] = false;
            let du = dist[u];
            for &a in &adj[starts[u] as usize..starts[u + 1] as usize] {
                let arc = a as usize;
                if net.is_arc_saturated(arc) {
                    continue;
                }
                let c = net.marginal_cost(arc);
                let nd = du.saturating_add(c);
                let (_t, h) = net.arc_endpoints(g, arc);
                if nd < dist[h] {
                    dist[h] = nd;
                    pred_arc[h] = arc as i64;
                    if !in_q[h] {
                        in_q[h] = true;
                        cnt[h] += 1;
                        if cnt[h] as usize > n {
                            hit = Some(h);
                            break;
                        }
                        q.push_back(h);
                    }
                }
            }
            if hit.is_some() {
                break;
            }
        }

        let Some(start) = hit else {
            break; // no negative cycle: flow is the convex optimum
        };

        // Walk pred_arc back n steps to guarantee we are ON the cycle (the
        // `start` node may only be reachable FROM the cycle).
        let mut v = start;
        for _ in 0..n {
            let pa = pred_arc[v];
            if pa < 0 {
                break;
            }
            v = net.arc_endpoints(g, pa as usize).0; // tail
        }
        // Collect the directed cycle arcs by following preds until we return.
        let cyc_anchor = v;
        let mut arcs = Vec::new();
        loop {
            let pa = pred_arc[v];
            if pa < 0 {
                arcs.clear();
                break;
            }
            arcs.push(pa as usize);
            v = net.arc_endpoints(g, pa as usize).0; // tail
            if v == cyc_anchor {
                break;
            }
            if arcs.len() > n {
                arcs.clear();
                break;
            }
        }
        if arcs.is_empty() {
            break;
        }

        // Push one unit around the cycle. Excess-neutral; objective strictly
        // decreases by the cycle's (negative) marginal sum.
        for &arc in &arcs {
            net.push_unit(g, arc);
        }
        cancelled += 1;
    }
    cancelled
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::RectangularGridGraph;
    use crate::network::Network;
    use ndarray::Array2;

    fn total_cost(flow: &[i32], offsets: &[i32], weights: &[i32]) -> i64 {
        let ns = 100_i64;
        (0..flow.len())
            .map(|e| {
                let u = flow[e] as i64 * ns - offsets[e] as i64;
                weights[e] as i64 * u * u
            })
            .sum()
    }

    /// A center-cell loop wants k*=+1 on all 4 arcs; starting at flow=0 (NOT
    /// preloaded) is sub-optimal. Cancel must restore the +1 circulation and
    /// reach the same cost as preload-to-k*.
    #[test]
    fn cancel_reaches_optimum_on_known_loop() {
        let g = RectangularGridGraph::new(3, 3);
        let nf = g.num_forward;
        let mut offsets = vec![0_i32; nf];
        let mut weights = vec![1_i32; nf];
        let d = g.down_arc(1, 1).unwrap();
        let r = g.right_arc(2, 1).unwrap();
        let u = g.up_arc(2, 2).unwrap();
        let l = g.left_arc(1, 2).unwrap();
        for &a in &[d, r, u, l] {
            offsets[a] = 100; // parabola min at +1
            weights[a] = 5;
        }
        let residues = Array2::<i32>::zeros((3, 3));

        // Optimum (preload to k*) cost:
        let mut net_opt =
            Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);
        net_opt.preload_convex_min(&g);
        let opt_flow: Vec<i32> = (0..nf).map(|a| net_opt.arc_flow(&g, a)).collect();
        let opt_cost = total_cost(&opt_flow, &offsets, &weights);

        // Start at flow=0 (sub-optimal), cancel cycles, must match opt_cost.
        let mut net = Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);
        let k = cancel_negative_cycles(&g, &mut net, 100);
        let flow: Vec<i32> = (0..nf).map(|a| net.arc_flow(&g, a)).collect();
        let cost = total_cost(&flow, &offsets, &weights);
        assert!(k >= 1, "expected at least one cycle cancelled, got {k}");
        assert_eq!(cost, opt_cost, "cancel did not reach the convex optimum");
    }

    /// On the optimal (preloaded) flow, cancel must find nothing.
    #[test]
    fn cancel_is_noop_at_optimum() {
        let g = RectangularGridGraph::new(3, 3);
        let nf = g.num_forward;
        let mut offsets = vec![0_i32; nf];
        let mut weights = vec![1_i32; nf];
        for &a in &[
            g.down_arc(1, 1).unwrap(),
            g.right_arc(2, 1).unwrap(),
            g.up_arc(2, 2).unwrap(),
            g.left_arc(1, 2).unwrap(),
        ] {
            offsets[a] = 100;
            weights[a] = 5;
        }
        let residues = Array2::<i32>::zeros((3, 3));
        let mut net = Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);
        net.preload_convex_min(&g);
        let k = cancel_negative_cycles(&g, &mut net, 100);
        assert_eq!(k, 0, "found a negative cycle at the optimum");
    }
}
