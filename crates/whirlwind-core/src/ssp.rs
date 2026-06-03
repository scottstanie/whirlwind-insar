//! Successive Shortest Paths (SSP) fallback, run after the primal-dual loop to
//! drain any remaining excess. Two variants with very different graph-shape costs:
//!
//! * [`run`] — MULTI-source: seeds *every* excess node, runs a full multi-source
//!   Dijkstra, and augments ONE source→nearest-deficit path per iteration. Fast
//!   when few residues remain over a small / tiled graph; catastrophic on a
//!   whole-image graph (a near-graph-wide Dijkstra per single unit of flow).
//!   Used by the early-exit primal-dual path (`primal_dual::run`).
//! * [`run_single_source`] — SINGLE-source: one source at a time, early-exiting
//!   Dijkstra at the first popped deficit. ~10× faster on whole-image graphs
//!   (single-tile D_077 ≈1472 s → ≈158 s, same result). Used by the full-
//!   completion path (`primal_dual::run_full_dijkstra`), which leaves the valid
//!   (all-nodes-popped) potentials it requires. See ATBD §9.6.

use crate::network::Network;
use crate::residual_graph::ResidualGraph;
use crate::shortest_path::{ShortestPaths, dijkstra_multi_source_into};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub fn run<G: ResidualGraph>(g: &G, net: &mut Network) {
    let dbg = crate::primal_dual::debug_enabled();
    let mut safety = 0;
    let safety_limit = 4 * net.num_nodes();
    let mut sp = ShortestPaths::new(net.num_nodes());
    while net.excess_nodes().next().is_some() {
        if dbg && safety % 50 == 0 {
            let ex: i64 = net
                .excess
                .iter()
                .filter(|&&e| e > 0)
                .map(|&e| e as i64)
                .sum();
            let df: i64 = net
                .excess
                .iter()
                .filter(|&&e| e < 0)
                .map(|&e| -e as i64)
                .sum();
            eprintln!("[ssp] iter={safety} excess={ex} deficit={df}");
        }
        crate::primal_dual::record_ssp_iter();
        dijkstra_multi_source_into(g, net, &mut sp);
        let nearest_deficit = net
            .deficit_nodes()
            .filter(|&d| sp.was_reached(d))
            .min_by_key(|&d| sp.dist[d]);
        let Some(sink) = nearest_deficit else { return };

        // Walk pred_node back to find the actual source seed (where pred_arc<0).
        let mut arcs = Vec::new();
        let mut cur = sink;
        loop {
            let parc = sp.pred_arc[cur];
            if parc < 0 {
                break;
            }
            arcs.push(parc as usize);
            cur = sp.pred_node[cur] as usize;
            if arcs.len() > net.num_nodes() {
                arcs.clear();
                break;
            }
        }
        if arcs.is_empty() {
            return; // shouldn't happen, but be safe
        }
        let src = cur;
        for arc in arcs {
            net.push_unit(g, arc);
        }
        net.increase_excess(sink, 1);
        net.decrease_excess(src, 1);

        // Same potential update with d_max capping as primal_dual::run.
        // Key off `popped` so the cap stays valid under early-exit Dijkstra.
        let d_max = sp
            .dist
            .iter()
            .zip(sp.popped.iter())
            .filter_map(|(&d, &p)| if p { Some(d) } else { None })
            .max()
            .unwrap_or(0);
        for v in 0..net.num_nodes() {
            let dv = if sp.popped[v] { sp.dist[v] } else { d_max };
            net.potential[v] -= dv;
        }

        safety += 1;
        assert!(safety <= safety_limit, "SSP did not converge");
    }
}

/// Single-source SSP for the FULL-completion path (`primal_dual::run_full_dijkstra`).
///
/// One source at a time: a single-source Dijkstra that early-exits when the first
/// deficit (sink) is popped, augment one unit along that path, update potentials,
/// repeat over a fixed source list. ~10× faster than the multi-source [`run`] on
/// a whole-image graph because each search explores only the neighbourhood up to
/// the nearest sink instead of re-exploring the whole reachable graph per unit.
///
/// CORRECTNESS — requires non-negative reduced costs at entry, which the full-
/// completion PD path provides (all reachable nodes popped → exact potentials).
/// The per-source potential update preserves the invariant: popped nodes get
/// their exact distance (`π += d_sink − dist[v]`); non-popped nodes keep a zero
/// shift, which is exactly "cap at `d_sink`" in that frame — valid because any
/// unpopped node has `dist ≥ d_sink` by Dijkstra pop order. The `debug_assert`
/// proves it: it must NEVER fire on the full path. Do **not** use this after the
/// early-exit `run` (its `d_max`-capped potentials can be negative on frontier
/// arcs); use the multi-source [`run`] there.
pub fn run_single_source<G: ResidualGraph>(g: &G, net: &mut Network) {
    let dbg = crate::primal_dual::debug_enabled();
    let n_nodes = net.num_nodes();
    let mut dist = vec![i64::MAX; n_nodes];
    let mut pred_arc: Vec<i32> = vec![-1; n_nodes];
    let mut pred_node: Vec<i32> = vec![-1; n_nodes];
    let mut visited = vec![false; n_nodes]; // popped (finalized)
    let mut touched: Vec<usize> = Vec::new();
    let mut out_buf: Vec<(usize, usize)> = Vec::new();

    let sources: Vec<usize> = net.excess_nodes().collect();
    let total = sources.len();
    if dbg {
        eprintln!("[ssp1] single-source SSP: {total} sources");
    }

    for (idx, src) in sources.into_iter().enumerate() {
        if net.excess[src] <= 0 {
            continue; // already drained by a prior augmentation
        }
        if dbg && idx % 1000 == 0 {
            let ex: i64 = net.excess.iter().filter(|&&e| e > 0).map(|&e| e as i64).sum();
            eprintln!("[ssp1] source {idx}/{total}, excess_remaining={ex}");
        }
        crate::primal_dual::record_ssp_iter();

        // --- single-source Dijkstra, early-exit when first deficit popped ---
        dist[src] = 0;
        touched.push(src);
        let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
        heap.push(Reverse((0, src)));
        let mut sink_found: Option<(usize, i64)> = None;

        while let Some(Reverse((d, u))) = heap.pop() {
            if visited[u] || dist[u] != d {
                continue; // stale heap entry
            }
            visited[u] = true;
            if net.excess[u] < 0 {
                sink_found = Some((u, d));
                break;
            }
            let pot_u = net.potential[u];
            out_buf.clear();
            if u < g.num_nodes() {
                g.outgoing(u, &mut out_buf);
            }
            for &(arc, v) in out_buf.iter().chain(net.extra_outgoing(u).iter()) {
                if net.is_arc_saturated(arc) {
                    continue;
                }
                let rc = (net.arc_cost(g, arc) as i64) - pot_u + net.potential[v];
                // Must hold on the full-completion path (valid entry potentials +
                // the capped update below). NOT clamped: a fire means the
                // invariant broke and must be fixed, not papered over.
                debug_assert!(
                    rc >= 0,
                    "negative rc={rc} on arc {arc} in single-source SSP (src idx {idx})"
                );
                let nd = d + rc;
                if nd < dist[v] {
                    if dist[v] == i64::MAX {
                        touched.push(v);
                    }
                    dist[v] = nd;
                    pred_arc[v] = arc as i32;
                    pred_node[v] = u as i32;
                    heap.push(Reverse((nd, v)));
                }
            }
        }

        if let Some((sink, d_sink)) = sink_found {
            // Augment one unit src → … → sink.
            net.increase_excess(sink, 1);
            net.decrease_excess(src, 1);
            let mut cur = sink;
            loop {
                let pa = pred_arc[cur];
                if pa < 0 {
                    break;
                }
                net.push_unit(g, pa as usize);
                cur = pred_node[cur] as usize;
            }
            // Potential update (see CORRECTNESS above): popped nodes get exact;
            // non-popped keep a zero shift = capped at d_sink.
            for &v in &touched {
                if visited[v] {
                    net.potential[v] += d_sink - dist[v];
                }
            }
        }

        // Reset scratch for the next source (reuse allocations).
        for &v in &touched {
            dist[v] = i64::MAX;
            pred_arc[v] = -1;
            pred_node[v] = -1;
            visited[v] = false;
        }
        touched.clear();
    }
}
