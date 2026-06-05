//! Successive Shortest Paths (SSP) fallback, run after the primal-dual loop to
//! drain any remaining excess. Two variants with very different graph-shape costs:
//!
//! * [`run`] - MULTI-source: seeds *every* excess node, runs a full multi-source
//!   Dijkstra, and augments ONE source→nearest-deficit path per iteration. Fast
//!   when few residues remain over a small / tiled graph; catastrophic on a
//!   whole-image graph (a near-graph-wide Dijkstra per single unit of flow).
//!   Used by the early-exit primal-dual path (`primal_dual::run`).
//! * [`run_single_source`] - SINGLE-source: one source at a time, early-exiting
//!   at the first popped deficit, using **Dial's bucket queue** (not a binary
//!   heap). ~10x faster on whole-image graphs (single-tile D_077 ≈1472 s →
//!   ≈158 s) AND robust to the zero-cost masked "sea" on heavily-masked frames,
//!   which makes a binary heap balloon (millions of equal-distance entries) but
//!   which Dial processes in O(nodes) per bucket. Used by the full-completion
//!   path (`primal_dual::run_full_dijkstra`), which leaves the valid
//!   (all-nodes-popped) potentials it requires. See ATBD §9.6.

use crate::network::Network;
use crate::residual_graph::ResidualGraph;
use crate::shortest_path::dial::max_reduced_cost_par;
use crate::shortest_path::{ShortestPaths, dijkstra_multi_source_into};
use std::collections::VecDeque;

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

/// Single-source SSP for the FULL-completion path (`primal_dual::run_full_dijkstra`),
/// using **Dial's bucket queue** (not a binary heap).
///
/// One source at a time: a single-source Dial shortest-path that early-exits when
/// the first deficit (sink) is popped, augment one unit along that path, update
/// potentials, repeat over a fixed source list. On a small masked region this
/// early-exits in a tiny neighbourhood (D_077 ≈158 s vs ≈1472 s for the
/// multi-source [`run`]); on a heavily-masked frame the **zero-cost masked sea**
/// (masked arcs have cost 0, never forbidden) is traversed in O(nodes) via the
/// distance-0 bucket - a binary heap instead balloons to millions of equal-
/// distance entries and blows up memory.
///
/// Scratch (`dist`/`pred`/`popped`, the touched list, and the reusable bucket
/// vectors) is allocated once and reset per source only over `touched`. The Dial
/// bucket count `k = max_edge_reduced_cost + 1` is recomputed per source because
/// the potentials drift as we augment.
///
/// CORRECTNESS - requires non-negative reduced costs at entry, which the full-
/// completion PD path provides (all reachable nodes popped → exact potentials).
/// The per-source potential update preserves the invariant: popped nodes get
/// their exact distance (`π += d_sink − dist[v]`); non-popped nodes keep a zero
/// shift, which is exactly "cap at `d_sink`" in that frame - valid because any
/// unpopped node has `dist ≥ d_sink` by pop order. The `debug_assert` proves it:
/// it must NEVER fire on the full path. Do **not** use this after the early-exit
/// `run` (its `d_max`-capped potentials can be negative on frontier arcs); use
/// the multi-source [`run`] there.
pub fn run_single_source<G: ResidualGraph>(g: &G, net: &mut Network) {
    let dbg = crate::primal_dual::debug_enabled();
    let n_nodes = net.num_nodes();
    let mut dist = vec![i64::MAX; n_nodes];
    let mut pred_arc: Vec<i32> = vec![-1; n_nodes];
    let mut pred_node: Vec<i32> = vec![-1; n_nodes];
    let mut popped = vec![false; n_nodes];
    let mut touched: Vec<usize> = Vec::new();
    let mut out_buf: Vec<(usize, usize)> = Vec::with_capacity(8);
    // Dial buckets, reused across sources (grown to the largest k seen, cleared
    // between sources). Buckets store (node, queued_dist) so stale entries are
    // detected on pop. FIFO (`VecDeque`, pop FRONT) to match ww-orig's
    // `std::queue` Dial buckets: equal-distance ties across the cost-0 masked
    // sea must resolve BFS/fewest-hops first (short cuts), not LIFO/DFS (long
    // cuts) - see `shortest_path::dial::run_full_into` decl comment.
    let mut buckets: Vec<VecDeque<(usize, i64)>> = Vec::new();

    let sources: Vec<usize> = net.excess_nodes().collect();
    let total = sources.len();
    if dbg {
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
        eprintln!(
            "[ssp1] single-source SSP (Dial): {total} sources; excess={ex} deficit={df} balanced={}",
            ex == df
        );
    }
    // Count of sources that exhausted their Dijkstra WITHOUT reaching any
    // deficit. In a BALANCED, connected residual graph every remaining excess
    // node provably has an augmenting path to some deficit (residual reverse
    // arcs included), so a stranded source is a REACHABILITY/SSP BUG, not
    // expected control flow - surfaced here instead of being silently skipped.
    let mut stranded = 0_usize;
    let mut scan_ns: u128 = 0; // time in max_reduced_cost_par (now only on k-overflow)
    // Maintained ACROSS sources (was rescanned every source - ~half of D_077's
    // runtime). A valid upper bound on every arc's reduced cost; grows lazily (a
    // source that hits rc >= k recomputes it tight + retries). One tight scan here.
    let mut max_rc = {
        let _scan_t = std::time::Instant::now();
        let v = max_reduced_cost_par(g, net);
        scan_ns += _scan_t.elapsed().as_nanos();
        v
    };

    for (idx, src) in sources.into_iter().enumerate() {
        if net.excess[src] <= 0 {
            continue; // already drained by a prior augmentation
        }
        if dbg && idx % 1000 == 0 {
            let ex: i64 = net
                .excess
                .iter()
                .filter(|&&e| e > 0)
                .map(|&e| e as i64)
                .sum();
            eprintln!("[ssp1] source {idx}/{total}, excess_remaining={ex}");
        }
        crate::primal_dual::record_ssp_iter();

        // Dial circular-bucket count k = max edge reduced cost + 1. max_rc is
        // carried across sources (not rescanned each one). If a relaxation finds
        // rc >= k, potentials grew past the bound: discard this source's partial
        // Dijkstra, recompute max_rc tight (the ONLY O(E) rescan), and retry - so
        // an under-estimated k can never commit an aliased (wrong) path.
        let mut sink_found: Option<(usize, i64)> = None;
        // Captured at loop exit (the only exit is the success `break`) for the
        // STRANDED diagnostic, since k/cur_dist are now scoped to the retry loop.
        let dbg_cur_dist: i64;
        let dbg_k: usize;
        loop {
            let k = (max_rc as usize).saturating_add(1).max(1);
            if buckets.len() < k {
                buckets.resize_with(k, VecDeque::new);
            }

            dist[src] = 0;
            touched.push(src);
            buckets[0].push_back((src, 0));
            let mut pending = 1_usize;
            let mut cur_bucket = 0_usize;
            let mut cur_dist = 0_i64;
            let mut bucket_advances = 0_usize;
            let mut overflow = false;
            let mut max_rc_seen = 0_i64;

            while pending > 0 {
                while buckets[cur_bucket].is_empty() {
                    cur_bucket = (cur_bucket + 1) % k;
                    cur_dist += 1;
                    bucket_advances += 1;
                    if bucket_advances > k {
                        break; // nothing reachable remains (shouldn't happen w/ pending>0)
                    }
                }
                if buckets[cur_bucket].is_empty() {
                    break;
                }
                let (u, qd) = buckets[cur_bucket].pop_front().unwrap(); // FIFO (see decl)
                pending -= 1;
                if popped[u] {
                    continue;
                }
                // Stale: re-relaxed to a smaller dist since this entry was queued.
                if dist[u] != qd || qd != cur_dist {
                    continue;
                }
                popped[u] = true;
                if net.excess[u] < 0 {
                    sink_found = Some((u, cur_dist));
                    break;
                }
                let pot_u = net.potential[u];
                out_buf.clear();
                if u < g.num_nodes() {
                    g.outgoing(u, &mut out_buf);
                }
                // extra_outgoing is empty when there is no ground node (the
                // unwrap_linear case), so it does not allocate on the hot path.
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
                        "negative rc={rc} on arc {arc} in single-source Dial SSP (src idx {idx})"
                    );
                    if rc >= k as i64 {
                        overflow = true; // k too small (potentials grew) -> recompute + retry
                        break;
                    }
                    if rc > max_rc_seen {
                        max_rc_seen = rc;
                    }
                    let nd = cur_dist + rc;
                    if nd < dist[v] {
                        if dist[v] == i64::MAX {
                            touched.push(v);
                        }
                        dist[v] = nd;
                        pred_arc[v] = arc as i32;
                        pred_node[v] = u as i32;
                        buckets[(nd as usize) % k].push_back((v, nd));
                        pending += 1;
                    }
                }
                if overflow {
                    break;
                }
            }

            if overflow {
                // Discard the partial Dijkstra, recompute a tight max_rc, retry.
                for &v in &touched {
                    dist[v] = i64::MAX;
                    pred_arc[v] = -1;
                    pred_node[v] = -1;
                    popped[v] = false;
                }
                touched.clear();
                for b in buckets.iter_mut() {
                    b.clear();
                }
                let _scan_t = std::time::Instant::now();
                max_rc = max_reduced_cost_par(g, net);
                scan_ns += _scan_t.elapsed().as_nanos();
                continue;
            }
            // Track potential growth cheaply so the next source rarely overflows.
            if max_rc_seen > max_rc {
                max_rc = max_rc_seen;
            }
            dbg_cur_dist = cur_dist;
            dbg_k = k;
            break;
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
                if popped[v] {
                    net.potential[v] += d_sink - dist[v];
                }
            }
        } else {
            // No deficit reachable from this source - in a balanced problem this
            // is a reachability/SSP bug. Dump the first few so we can see WHY:
            // how much of the graph the source's Dijkstra reached, and whether
            // any deficit still exists globally (i.e. is it unreached, hence a
            // disconnect/traversal bug, vs genuinely none left).
            stranded += 1;
            if dbg && stranded <= 5 {
                let reached = touched.iter().filter(|&&v| popped[v]).count();
                let n_def = net.excess.iter().filter(|&&e| e < 0).count();
                let reached_def = touched
                    .iter()
                    .filter(|&&v| popped[v] && net.excess[v] < 0)
                    .count();
                eprintln!(
                    "[ssp1] STRANDED src={src} (idx {idx}): reached={reached} nodes, \
                     touched={}, global_deficit_nodes={n_def}, reached_deficits={reached_def}, \
                     last_cur_dist={dbg_cur_dist} k={dbg_k}",
                    touched.len()
                );
            }
        }

        // Reset scratch over touched only, then clear the buckets used this
        // source (early-exit can leave queued entries behind).
        for &v in &touched {
            dist[v] = i64::MAX;
            pred_arc[v] = -1;
            pred_node[v] = -1;
            popped[v] = false;
        }
        touched.clear();
        for b in buckets.iter_mut() {
            b.clear();
        }
    }
    if dbg {
        let ex: i64 = net
            .excess
            .iter()
            .filter(|&&e| e > 0)
            .map(|&e| e as i64)
            .sum();
        eprintln!(
            "[ssp1] DONE: stranded_sources={stranded} remaining_excess={ex} \
             max_reduced_cost_scan={:.1}ms",
            scan_ns as f64 / 1e6
        );
    }
}
