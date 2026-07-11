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
//!   heap). ~10x faster on whole-image graphs (a large single-tile frame,≈1472 s →
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
    let mut sp = ShortestPaths::new(net.num_nodes());
    run_scratch(g, net, &mut sp);
}

/// [`run`] with a caller-owned `ShortestPaths` buffer (the primal-dual driver
/// reuses its Dijkstra-phase allocation here instead of holding two).
pub fn run_scratch<G: ResidualGraph>(g: &G, net: &mut Network, sp: &mut ShortestPaths) {
    let dbg = crate::primal_dual::debug_enabled();
    let mut safety = 0;
    let safety_limit = 4 * net.num_nodes();
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
        dijkstra_multi_source_into(g, net, sp);
        let nearest_deficit = net
            .deficit_nodes()
            .filter(|&d| sp.was_reached(d))
            .min_by_key(|&d| sp.dist[d]);
        let Some(sink) = nearest_deficit else { return };

        // Walk the pred chain back to find the actual source seed (where
        // pred_arc<0). Predecessor node = tail of the predecessor arc.
        let mut arcs = Vec::new();
        let mut cur = sink;
        loop {
            let parc = sp.pred_arc[cur];
            if parc < 0 {
                break;
            }
            arcs.push(parc as usize);
            cur = net.arc_endpoints(g, parc as usize).0;
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
/// early-exits in a tiny neighborhood (~158 s vs ~1472 s for the
/// multi-source [`run`] on a large frame); on a heavily-masked frame the
/// **zero-cost masked sea**
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
/// unpopped node has `dist ≥ d_sink` by pop order. The invariant is checked at
/// runtime (release builds included): a negative reduced cost abandons that
/// source's search untouched, counts it for the BFS balance guard, and warns
/// loudly - it must never trigger on the full path. Do **not** use this after the early-exit
/// `run` (its `d_max`-capped potentials can be negative on frontier arcs); use
/// the multi-source [`run`] there.
pub fn run_single_source<G: ResidualGraph>(g: &G, net: &mut Network) {
    let mut sp = ShortestPaths::new(net.num_nodes());
    let mut buckets: Vec<VecDeque<u32>> = Vec::new();
    run_single_source_scratch(g, net, &mut sp, &mut buckets);
}

/// [`run_single_source`] with caller-owned scratch: the per-node
/// `dist`/`pred_arc`/`popped` triple lives in `sp` and the Dial FIFO buckets
/// in `buckets`, so the primal-dual driver can hand over the (identically
/// shaped) buffers its Dijkstra phase already allocated instead of holding
/// both sets alive - on a NISAR-scale frame that duplicate scratch was the
/// peak-RSS high-water mark of the whole unwrap.
pub fn run_single_source_scratch<G: ResidualGraph>(
    g: &G,
    net: &mut Network,
    sp: &mut ShortestPaths,
    buckets: &mut Vec<VecDeque<u32>>,
) {
    let dbg = crate::primal_dual::debug_enabled();
    let n_nodes = net.num_nodes();
    // Same initial state the old fresh allocations had (dist=MAX, pred=-1,
    // popped=false); per-source resets below only touch `touched` nodes.
    sp.reset(n_nodes);
    let dist = &mut sp.dist;
    let pred_arc = &mut sp.pred_arc;
    let popped = &mut sp.popped;
    let mut touched: Vec<usize> = Vec::new();
    let mut out_buf: Vec<(usize, usize)> = Vec::with_capacity(8);
    let has_ground = net.has_ground();
    // Dial buckets, reused across sources (grown to the largest k seen, cleared
    // between sources). Entries are bare node ids (u32): every entry pops at
    // exactly the distance it was queued with (rc < k + buckets drain before
    // the scan wraps), so staleness reduces to `dist[u] != cur_dist` - see
    // `shortest_path::dial::run_into` decl comment. FIFO (`VecDeque`, pop
    // FRONT) to match ww-orig's `std::queue` Dial buckets: equal-distance ties
    // across the cost-0 masked sea must resolve BFS/fewest-hops first (short
    // cuts), not LIFO/DFS (long cuts) - see `dial::run_full_into` decl comment.
    // Caller-owned: clear anything a previous user (the PD Dijkstra) left.
    for b in buckets.iter_mut() {
        b.clear();
    }

    let sources: Vec<usize> = net.excess_nodes().collect();
    let total = sources.len();
    // Total deficit UNITS remaining, maintained across augmentations. Without
    // this guard a one-sided network (excess left but every deficit already
    // drained - e.g. an imbalanced residue grid) sends EVERY remaining source
    // on a full-graph Dial flood that provably finds nothing: S sources x O(E)
    // of pure waste on the zero-cost masked sea. Bail up front and again the
    // moment the last deficit unit is paired. (The `stranded` diagnostic below
    // stays meaningful: it now only counts true unreachability while deficits
    // still exist.)
    let mut deficit_units: i64 = net
        .excess
        .iter()
        .filter(|&&e| e < 0)
        .map(|&e| -e as i64)
        .sum();
    if dbg {
        let ex: i64 = net
            .excess
            .iter()
            .filter(|&&e| e > 0)
            .map(|&e| e as i64)
            .sum();
        eprintln!(
            "[ssp1] single-source SSP (Dial): {total} sources; excess={ex} deficit={deficit_units} balanced={}",
            ex == deficit_units
        );
    }
    if deficit_units == 0 {
        return;
    }
    // Count of sources that exhausted their Dijkstra WITHOUT reaching any
    // deficit. In a BALANCED, connected residual graph every remaining excess
    // node provably has an augmenting path to some deficit (residual reverse
    // arcs included), so a stranded source is a REACHABILITY/SSP BUG, not
    // expected control flow - surfaced here instead of being silently skipped.
    let mut stranded = 0_usize;
    // Searches abandoned because a NEGATIVE reduced cost was seen (invalid
    // potentials). Always reported loudly at the end: it means an upstream
    // pass broke the potential invariant and must be investigated, even
    // though the BFS balance guard keeps the output usable.
    let mut neg_rc_sources = 0_usize;
    let mut scan_ns: u128 = 0; // time in max_reduced_cost_par (now only on k-overflow)
    // Maintained ACROSS sources (was rescanned every source - up to half the
    // runtime on large frames). A valid upper bound on every arc's reduced cost; grows lazily (a
    // source that hits rc >= k recomputes it tight + retries). One tight scan here.
    let mut max_rc = {
        let _scan_t = std::time::Instant::now();
        let v = max_reduced_cost_par(g, net);
        scan_ns += _scan_t.elapsed().as_nanos();
        v
    };

    for (idx, src) in sources.into_iter().enumerate() {
        if deficit_units == 0 {
            break; // every deficit paired - no remaining source can be served
        }
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
            buckets[0].push_back(src as u32);
            let mut pending = 1_usize;
            let mut cur_bucket = 0_usize;
            let mut cur_dist = 0_i64;
            let mut bucket_advances = 0_usize;
            let mut overflow = false;
            let mut neg_rc = false;
            let mut max_rc_seen = 0_i64;

            while pending > 0 {
                while buckets[cur_bucket].is_empty() {
                    cur_bucket = (cur_bucket + 1) % k;
                    cur_dist += 1;
                    bucket_advances += 1;
                    if bucket_advances > k {
                        break; // a full k-cycle of empty buckets: queue exhausted
                    }
                }
                if buckets[cur_bucket].is_empty() {
                    break;
                }
                // Reset on every non-empty bucket, exactly like the dial.rs
                // backends: the exit test above must mean "k CONSECUTIVE empty
                // buckets" (a full window with no work = exhausted), not "k
                // empty advances accumulated over the whole search". Without
                // this reset, any search whose distance range spans more than
                // k total empty skips self-terminates with live entries still
                // queued, stranding sources whose nearest deficit lies farther
                // than ~k in reduced cost (the Ridgecrest block-tear bug).
                bucket_advances = 0;
                let u = buckets[cur_bucket].pop_front().unwrap() as usize; // FIFO (see decl)
                pending -= 1;
                // Stale ⟺ dist[u] != cur_dist; covers already-popped too
                // (see `dial::run_into`) - no popped[] read on the pop path.
                if dist[u] != cur_dist {
                    debug_assert!(!popped[u] || dist[u] < cur_dist);
                    continue;
                }
                debug_assert!(!popped[u], "node {u} popped twice at dist {cur_dist}");
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
                // Ground arcs exist only on grounded networks; skip the
                // per-pop extra_outgoing call entirely on the (default)
                // ground-free path.
                if has_ground {
                    out_buf.extend(net.extra_outgoing(u));
                }
                for &(arc, v) in out_buf.iter() {
                    if net.is_arc_saturated(arc) {
                        continue;
                    }
                    let rc = (net.arc_cost(g, arc) as i64) - pot_u + net.potential[v];
                    // Non-negative reduced costs are the Dial validity
                    // invariant (valid entry potentials + the capped update
                    // below). Checked in RELEASE too - a debug_assert here is
                    // compiled out exactly where it matters, and a negative rc
                    // silently poisons distances (entries land in buckets the
                    // scan already passed and are discarded as stale). Recover
                    // by abandoning this source's search untouched: it is
                    // counted as stranded and the cost-ignoring BFS balance
                    // guard pairs it, so the failure is a bounded local error
                    // instead of a corrupt unwrap.
                    if rc < 0 {
                        neg_rc = true;
                        break;
                    }
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
                        pred_arc[v] = arc as i64;
                        buckets[(nd as usize) % k].push_back(v as u32);
                        pending += 1;
                    }
                }
                if overflow || neg_rc {
                    break;
                }
            }

            if neg_rc {
                // Invalid potentials: abandon this search with the network
                // state untouched (no augment, no potential update). The
                // source is counted as stranded below and the BFS balance
                // guard pairs it. Discard the partial Dijkstra scratch.
                for &v in &touched {
                    dist[v] = i64::MAX;
                    pred_arc[v] = -1;
                    popped[v] = false;
                }
                touched.clear();
                for b in buckets.iter_mut() {
                    b.clear();
                }
                neg_rc_sources += 1;
                sink_found = None;
                dbg_cur_dist = cur_dist;
                dbg_k = k;
                break;
            }
            if overflow {
                // Discard the partial Dijkstra, recompute a tight max_rc, retry.
                for &v in &touched {
                    dist[v] = i64::MAX;
                    pred_arc[v] = -1;
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
            deficit_units -= 1;
            let mut cur = sink;
            loop {
                let pa = pred_arc[cur];
                if pa < 0 {
                    break;
                }
                net.push_unit(g, pa as usize);
                // Predecessor node = tail of the predecessor arc.
                cur = net.arc_endpoints(g, pa as usize).0;
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
            popped[v] = false;
        }
        touched.clear();
        for b in buckets.iter_mut() {
            b.clear();
        }
    }
    if neg_rc_sources > 0 {
        eprintln!(
            "[ssp1] WARNING: {neg_rc_sources} source search(es) abandoned on a \
             negative reduced cost - the potential invariant broke upstream; \
             the BFS balance guard will pair the leftovers, but this should be \
             reported and investigated"
        );
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

/// Last-resort **cost-ignoring** residual BFS that pairs every remaining excess
/// unit with a deficit, guaranteeing the network is left balanced so the
/// integration never has to absorb a stranded residue as a full-width 2π tear.
///
/// # Why this exists
///
/// If any cost-aware pass (`primal_dual::run_full_dijkstra`, the Dial
/// single-source [`run_single_source`], the adaptive PD resume) ever leaves
/// residues unpaired, the row-major integration turns each unpaired ±1 pair
/// into a full-width 2π offset block - by far the worst possible failure, so
/// the network must NEVER be integrated unbalanced. This happened on the 2019
/// Ridgecrest coseismic belt (OPERA DISP-S1 F16941): [`run_single_source`]'s
/// bucket-advance exit tested *cumulative* rather than *consecutive* empty
/// advances, so long-range searches self-terminated with live entries still
/// queued and ~19-49 sources stranded. That root cause is fixed (the counter
/// now resets on every non-empty bucket, matching the `dial.rs` backends, and
/// a debug-assertion run confirmed reduced costs stay non-negative throughout,
/// so the potentials were never the problem). This guard stays as the safety
/// net for any future incomplete pass: a suboptimal pairing costs a bounded
/// local error, an unpaired residue costs a full-width tear.
///
/// This routine sidesteps all of that: a plain FIFO breadth-first search over
/// the residual graph, gated **only** on `is_arc_saturated` (never on cost or
/// potential). On a connected, balanced network it always finds a deficit for
/// every leftover excess unit, so the result is balanced by construction. The
/// pairing it picks is the fewest-hops augmenting path, which is not
/// cost-optimal - but a suboptimal pairing costs a bounded local error, whereas
/// leaving the pair unpaired costs a full-width tear. It is a guard, not the
/// primary solver: it only runs when excess survives the cost-aware passes, so
/// it is a no-op on the frames (e.g. the NISAR set) that already drain cleanly.
///
/// Returns the number of units it paired (0 when there was nothing to drain).
pub fn drain_residual_bfs<G: ResidualGraph>(g: &G, net: &mut Network) -> usize {
    let dbg = crate::primal_dual::debug_enabled();
    let n_nodes = net.num_nodes();

    let mut deficit_units: i64 = net
        .excess
        .iter()
        .filter(|&&e| e < 0)
        .map(|&e| -e as i64)
        .sum();
    if deficit_units == 0 {
        return 0;
    }

    let mut pred_arc: Vec<i64> = vec![-1; n_nodes];
    let mut visited = vec![false; n_nodes];
    let mut touched: Vec<usize> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut out_buf: Vec<(usize, usize)> = Vec::with_capacity(8);
    let has_ground = net.has_ground();

    let sources: Vec<usize> = net.excess_nodes().collect();
    let mut paired = 0_usize;
    let mut stranded = 0_usize;

    for &src in &sources {
        // A single source may hold more than one excess unit; drain them all.
        while net.excess[src] > 0 && deficit_units > 0 {
            // BFS from src over unsaturated residual arcs, early-exiting the
            // moment the first deficit node is popped (fewest-hops path).
            visited[src] = true;
            touched.push(src);
            queue.push_back(src as u32);
            let mut sink: Option<usize> = None;
            while let Some(u) = queue.pop_front() {
                let u = u as usize;
                if net.excess[u] < 0 {
                    sink = Some(u);
                    break;
                }
                out_buf.clear();
                if u < g.num_nodes() {
                    g.outgoing(u, &mut out_buf);
                }
                if has_ground {
                    out_buf.extend(net.extra_outgoing(u));
                }
                for &(arc, v) in out_buf.iter() {
                    if net.is_arc_saturated(arc) || visited[v] {
                        continue;
                    }
                    visited[v] = true;
                    touched.push(v);
                    pred_arc[v] = arc as i64;
                    queue.push_back(v as u32);
                }
            }

            if let Some(sink) = sink {
                net.increase_excess(sink, 1);
                net.decrease_excess(src, 1);
                deficit_units -= 1;
                paired += 1;
                let mut cur = sink;
                loop {
                    let pa = pred_arc[cur];
                    if pa < 0 {
                        break;
                    }
                    net.push_unit(g, pa as usize);
                    cur = net.arc_endpoints(g, pa as usize).0;
                }
            } else {
                // Genuinely no deficit reachable in the residual graph - only
                // possible if the network is unbalanced or truly disconnected,
                // neither of which should occur on the balanced `unwrap_linear`
                // grid. Report and abandon this source rather than spin.
                stranded += 1;
                // Unconditionally loud (first few): if the guard of last
                // resort itself cannot pair a residue, the output WILL carry
                // a full-width 2pi tear - the one failure this module exists
                // to prevent - and silence here would recreate the original
                // Ridgecrest bug with no trace in the logs.
                if stranded <= 5 {
                    eprintln!(
                        "[bfs_drain] WARNING: src={src} reached {} nodes, NO deficit \
                         reachable (network disconnected/unbalanced?) - the unwrap \
                         will contain a full-width 2pi tear",
                        touched.len()
                    );
                }
            }

            // Reset scratch over touched only.
            for &v in &touched {
                visited[v] = false;
                pred_arc[v] = -1;
            }
            touched.clear();
            queue.clear();

            if sink.is_none() {
                break; // this source can't be served; move on
            }
        }
    }

    let rem: i64 = net.excess.iter().map(|&e| (e as i64).abs()).sum();
    if stranded > 0 || rem > 0 {
        eprintln!(
            "[bfs_drain] WARNING: paired={paired} stranded_sources={stranded} \
             remaining_excess={rem} - unpaired residues integrate to full-width 2pi tears"
        );
    } else if dbg {
        eprintln!("[bfs_drain] paired={paired} stranded_sources={stranded} remaining_excess={rem}");
    }
    paired
}

#[cfg(test)]
mod single_source_tests {
    use super::*;
    use crate::grid::RectangularGridGraph;
    use ndarray::Array2;

    /// Regression test for the Ridgecrest block-tear bug: the Dial
    /// bucket-advance exit must test CONSECUTIVE empty advances (a full
    /// k-cycle with no work = queue exhausted), not CUMULATIVE advances over
    /// the whole search. With uniform arc cost 7 and zero potentials, k =
    /// max_rc + 1 = 8, and every hop of the wavefront costs 7 empty-bucket
    /// advances - so a sink more than one hop away needs cumulative advances
    /// beyond k while never exceeding 7 consecutively. Before the fix this
    /// search self-terminated after ~2 hops and stranded the source; each
    /// stranded pair then integrates into a full-width 2π tear.
    #[test]
    fn pairs_a_distant_sink_beyond_one_bucket_cycle() {
        // 4x16 residue grid, +1 and -1 planted 12 hops apart: shortest-path
        // distance 12 * 7 = 84 >> k = 8.
        let g = RectangularGridGraph::new(4, 16);
        let mut residues = Array2::<i32>::zeros((4, 16));
        residues[(1, 1)] = 1;
        residues[(1, 13)] = -1;
        let costs = vec![7_i32; g.num_forward];
        let mut net = Network::new(&g, residues.view(), &costs);
        assert!(net.is_balanced());

        run_single_source(&g, &mut net);
        let remaining: i64 = net.excess.iter().map(|&e| (e as i64).abs()).sum();
        assert_eq!(
            remaining, 0,
            "single-source SSP must pair a sink whose distance spans many bucket cycles"
        );
    }

    /// A NEGATIVE reduced cost (invalid potentials) must not panic, poison
    /// distances, or corrupt the network: the search is abandoned untouched,
    /// the source counts as stranded, and the BFS balance guard pairs it.
    /// Force the condition by corrupting one potential before the solve.
    #[test]
    fn negative_reduced_cost_is_abandoned_and_bfs_guard_recovers() {
        let g = RectangularGridGraph::new(4, 8);
        let mut residues = Array2::<i32>::zeros((4, 8));
        residues[(1, 1)] = 1;
        residues[(1, 6)] = -1;
        let costs = vec![7_i32; g.num_forward];
        let mut net = Network::new(&g, residues.view(), &costs);
        let src = net.excess_nodes().next().unwrap();
        // Every arc out of src now has rc = 7 - 1000 + 0 < 0.
        net.potential[src] = 1000;

        run_single_source(&g, &mut net);
        let remaining: i64 = net.excess.iter().map(|&e| (e as i64).abs()).sum();
        assert_eq!(
            remaining, 2,
            "abandoned search must leave the pair untouched"
        );

        let paired = drain_residual_bfs(&g, &mut net);
        assert_eq!(paired, 1, "BFS guard must pair what the SSP abandoned");
        let remaining: i64 = net.excess.iter().map(|&e| (e as i64).abs()).sum();
        assert_eq!(remaining, 0, "network must be balanced after the guard");
    }
}

#[cfg(test)]
mod bfs_drain_tests {
    use super::*;
    use crate::grid::RectangularGridGraph;
    use ndarray::Array2;

    /// The BFS drain must balance a connected network regardless of the
    /// potential/reduced-cost state - it never touches potentials. Here we hand
    /// it a raw, un-solved network (zero potentials, high arc costs so a
    /// cost-aware solver would still route, but we skip it) with a single
    /// +1/-1 residue pair and require it to pair them, leaving zero excess.
    #[test]
    fn drains_a_balanced_pair_with_no_prior_solve() {
        // 6x6 residue grid (5x5 pixel edges). Plant +1 and -1 at two interior
        // nodes; the rest zero → balanced.
        let g = RectangularGridGraph::new(6, 6);
        let mut residues = Array2::<i32>::zeros((6, 6));
        residues[(2, 1)] = 1;
        residues[(3, 4)] = -1;
        let costs = vec![7_i32; g.num_forward];
        let mut net = Network::new(&g, residues.view(), &costs);
        assert!(net.is_balanced());

        let paired = drain_residual_bfs(&g, &mut net);
        assert_eq!(paired, 1, "exactly one +1/-1 pair to route");
        let remaining: i64 = net.excess.iter().map(|&e| (e as i64).abs()).sum();
        assert_eq!(remaining, 0, "network must be balanced after the drain");
    }

    /// Multi-unit sources (an excess node holding more than one unit) must be
    /// fully drained, and several pairs paired in one call.
    #[test]
    fn drains_multi_unit_source() {
        let g = RectangularGridGraph::new(8, 8);
        let mut residues = Array2::<i32>::zeros((8, 8));
        residues[(2, 2)] = 2; // two units on one source
        residues[(5, 5)] = -1;
        residues[(1, 6)] = -1;
        let costs = vec![3_i32; g.num_forward];
        let mut net = Network::new(&g, residues.view(), &costs);
        assert!(net.is_balanced());

        let paired = drain_residual_bfs(&g, &mut net);
        assert_eq!(paired, 2, "two units to route from the +2 source");
        let remaining: i64 = net.excess.iter().map(|&e| (e as i64).abs()).sum();
        assert_eq!(remaining, 0);
    }

    /// A no-op when there is nothing to drain (already balanced with zero
    /// excess): returns 0 and leaves the network untouched. This is the state
    /// the guard sees on frames that already solved cleanly (e.g. NISAR).
    #[test]
    fn no_op_when_nothing_stranded() {
        let g = RectangularGridGraph::new(5, 5);
        let residues = Array2::<i32>::zeros((5, 5));
        let costs = vec![1_i32; g.num_forward];
        let mut net = Network::new(&g, residues.view(), &costs);
        assert_eq!(drain_residual_bfs(&g, &mut net), 0);
    }
}
