//! Dial's bucket-queue Dijkstra.
//!
//! Reduced costs in primal-dual are non-negative integers bounded by some
//! `C_max`. A circular bucket queue of size `K = C_max + 1` lets us pop the
//! minimum in amortized O(1) and gives O(V + E + max_dist) total work per call
//! - competitive with the binary heap when C_max is small relative to V.
//!
//! Two implementations live here:
//! - [`run`]: single-threaded reference version.
//! - [`run_parallel`]: each non-trivial bucket is relaxed in parallel using
//!   rayon. Phase 1 (parallel) collects proposed relaxations after each
//!   thread atomically claims `u` via `visited[u]`; phase 2 (serial) applies
//!   them. This avoids racing writes to `sp.dist / pred_arc / pred_node /
//!   source` and is provably equivalent to the serial Dial on any
//!   single-shortest-path ties - each (u, qd) pair is processed at most once.

use super::ShortestPaths;
use crate::network::Network;
use crate::residual_graph::ResidualGraph;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

/// Compute the per-arc bucket count `k = max_unsaturated_reduced_cost + 1`.
/// Parallelized via rayon - O(E) but trivially data-parallel and ~5x faster
/// at 4096² where E ≈ 32M.
pub(crate) fn max_reduced_cost_par<G: ResidualGraph>(g: &G, net: &Network) -> i64 {
    use rayon::prelude::*;
    (0..net.num_arcs())
        .into_par_iter()
        .map(|a| {
            if net.is_arc_saturated(a) {
                return 0;
            }
            if net.is_used(a) {
                return 0;
            }
            if net.convex_mode {
                // In convex mode the relax cost is the marginal cost, which
                // can be negative. After potential adjustment the *reduced*
                // marginal cost should be ≥ 0; assume that and use it.
                // Bellman-Ford pre-pass is the caller's responsibility.
                let (t, h) = net.arc_endpoints(g, a);
                (net.marginal_cost(a) - net.potential[t] + net.potential[h]).max(0)
            } else {
                net.reduced_cost(g, a)
            }
        })
        .max()
        .unwrap_or(0)
}

/// Count and mark deficit nodes (= sinks). Returns (is_sink, n_sinks).
fn collect_sinks(net: &Network) -> (Vec<bool>, usize) {
    let mut is_sink = vec![false; net.excess.len()];
    let mut n = 0;
    for (v, &e) in net.excess.iter().enumerate() {
        if e < 0 {
            is_sink[v] = true;
            n += 1;
        }
    }
    (is_sink, n)
}

/// Multi-source Dijkstra over the residual graph using Dial's bucket queue.
///
/// Early-exit: stops as soon as every deficit (sink) has been popped - any
/// further relaxation can't change a finalized distance. On scenes where
/// sinks cluster (real interferograms) this trims a large tail off late
/// primal-dual iterations.
pub fn run<G: ResidualGraph>(g: &G, net: &Network) -> ShortestPaths {
    let mut sp = ShortestPaths::new(net.num_nodes());
    run_into(g, net, &mut sp);
    sp
}

/// Reusable-buffer variant of [`run`].
pub fn run_into<G: ResidualGraph>(g: &G, net: &Network, sp: &mut ShortestPaths) {
    let n_nodes = net.num_nodes();
    sp.reset(n_nodes);
    let max_rc = max_reduced_cost_par(g, net);
    let k = (max_rc as usize).saturating_add(1).max(1);

    let (is_sink, total_sinks) = collect_sinks(net);
    let mut sinks_left = total_sinks;

    // Buckets store (node, queued_dist). queued_dist disambiguates stale
    // entries - when we pop a node whose sp.dist[u] no longer matches the
    // entry we pushed, we skip it.
    let mut buckets: Vec<Vec<(usize, i64)>> = vec![Vec::new(); k];
    let mut pending: usize = 0;

    // Seed every excess node at distance 0.
    for s in net.excess_nodes() {
        sp.dist[s] = 0;
        buckets[0].push((s, 0));
        pending += 1;
    }
    if pending == 0 || total_sinks == 0 {
        return;
    }

    let mut cur_bucket = 0_usize;
    let mut cur_dist: i64 = 0;
    let mut bucket_advances: usize = 0;
    let mut out_buf: Vec<(usize, usize)> = Vec::with_capacity(8);

    while pending > 0 {
        // Advance to the next non-empty bucket. After K consecutive empty
        // advances we know no work remains.
        while buckets[cur_bucket].is_empty() {
            cur_bucket = (cur_bucket + 1) % k;
            cur_dist += 1;
            bucket_advances += 1;
            if bucket_advances > k {
                return;
            }
        }
        bucket_advances = 0;

        let (u, qd) = buckets[cur_bucket].pop().unwrap();
        pending -= 1;
        if sp.popped[u] {
            continue;
        }
        // Stale: this entry was queued at qd but the node has since been
        // re-relaxed to a smaller dist.
        if sp.dist[u] != qd || qd != cur_dist {
            continue;
        }
        sp.popped[u] = true;
        if is_sink[u] {
            sinks_left -= 1;
            if sinks_left == 0 {
                // All sinks finalized - any further relaxation only affects
                // non-sinks and is wasted work for the augment phase.
                return;
            }
        }

        // Tail of every outgoing arc is u; pre-load π[u] once.
        let pot_u = net.potential[u];
        let mut relax = |arc: usize, v: usize, sp: &mut ShortestPaths, pending: &mut usize| {
            if net.is_arc_saturated(arc) {
                return;
            }
            // PHASS-style reuse: used arcs have reduced cost 0, independent
            // of cost/potential. is_used() returns false in MCF mode so this
            // branch only fires for the unwrap_reuse path.
            // Convex mode: use marginal cost instead of arc_cost.
            let rc = if net.is_used(arc) {
                0
            } else if net.convex_mode {
                net.marginal_cost(arc) - pot_u + net.potential[v]
            } else {
                net.arc_cost(g, arc) as i64 - pot_u + net.potential[v]
            };
            debug_assert!(rc >= 0, "negative reduced cost on arc {arc}: {rc}");
            let nd = cur_dist + rc;
            if nd < sp.dist[v] {
                sp.dist[v] = nd;
                sp.pred_arc[v] = arc as i32;
                let b = (nd as usize) % k;
                buckets[b].push((v, nd));
                *pending += 1;
            }
        };
        out_buf.clear();
        if u < g.num_nodes() {
            g.outgoing(u, &mut out_buf);
        }
        for &(arc, v) in out_buf.iter() {
            relax(arc, v, &mut *sp, &mut pending);
        }
        for &(arc, v) in net.extra_outgoing(u).iter() {
            relax(arc, v, &mut *sp, &mut pending);
        }
    }
}

/// Multi-source Dijkstra over the residual graph using Dial's bucket queue.
///
/// Full-completion variant: runs until the heap is empty - every reachable node
/// is popped and gets an exact finalized distance. Matches Python ww-orig's
/// `dijkstra_pd` which runs `while (!dijkstra.done())` with no early-exit.
///
/// Use this for the primal-dual loop when potential accuracy matters: the exact
/// `d[v]` for all nodes lets `update_potential_pd` compute tight reduced costs,
/// matching Python's MCF routing and closing the early-exit quality gap.
pub fn run_full<G: ResidualGraph>(g: &G, net: &Network) -> ShortestPaths {
    let mut sp = ShortestPaths::new(net.num_nodes());
    run_full_into(g, net, &mut sp);
    sp
}

/// Reusable-buffer variant of [`run_full`].
pub fn run_full_into<G: ResidualGraph>(g: &G, net: &Network, sp: &mut ShortestPaths) {
    let dbg = crate::primal_dual::debug_enabled();
    let n_nodes = net.num_nodes();
    sp.reset(n_nodes);

    let t_rc = std::time::Instant::now();
    let max_rc = max_reduced_cost_par(g, net);
    let k = (max_rc as usize).saturating_add(1).max(1);
    if dbg {
        eprintln!(
            "[run_full] max_rc={max_rc} k={k} t={:.3}s",
            t_rc.elapsed().as_secs_f64()
        );
    }

    // FIFO buckets (`VecDeque`, pop FRONT) - matches ww-orig's C++ Dial
    // (`std::queue` per bucket, pops `front()`). Equal-distance ties - which
    // dominate the cost-0 masked "sea" on heavily-masked frames - must resolve
    // BFS/fewest-hops first (pairing nearby residues, short branch cuts), NOT
    // LIFO/DFS (long snaking cuts that pair distant residues). Both are equally
    // cost-optimal, but only the FIFO one matches ww-orig / the correct unwrap;
    // it is scale-free (a tie-break, not a cost change). See ww
    // `ext/libwhirlwind/.../graph/dial.hpp`. NOTE: only this full-completion
    // (parity) Dijkstra is FIFO; the early-exit `run_into` (production reuse /
    // tiled) is intentionally left as-is.
    let mut buckets: Vec<VecDeque<(usize, i64)>> = vec![VecDeque::new(); k];
    let mut pending: usize = 0;

    for s in net.excess_nodes() {
        sp.dist[s] = 0;
        buckets[0].push_back((s, 0));
        pending += 1;
    }
    if pending == 0 {
        return;
    }

    let mut cur_bucket = 0_usize;
    let mut cur_dist: i64 = 0;
    let mut bucket_advances: usize = 0;
    let mut total_bucket_advances: usize = 0;
    let mut real_pops: usize = 0;
    let mut stale_pops: usize = 0;
    let mut out_buf: Vec<(usize, usize)> = Vec::with_capacity(8);

    while pending > 0 {
        while buckets[cur_bucket].is_empty() {
            cur_bucket = (cur_bucket + 1) % k;
            cur_dist += 1;
            bucket_advances += 1;
            total_bucket_advances += 1;
            if bucket_advances > k {
                if dbg {
                    eprintln!(
                        "[run_full] k-advance exit: cur_dist={cur_dist} real_pops={real_pops} stale_pops={stale_pops} total_ba={total_bucket_advances}"
                    );
                }
                return;
            }
        }
        bucket_advances = 0;

        let (u, qd) = buckets[cur_bucket].pop_front().unwrap(); // FIFO (see decl)
        pending -= 1;
        if sp.popped[u] {
            stale_pops += 1;
            continue;
        }
        if sp.dist[u] != qd || qd != cur_dist {
            stale_pops += 1;
            continue;
        }
        sp.popped[u] = true;
        real_pops += 1;
        // No early-exit - keep going until all reachable nodes are finalized.

        let pot_u = net.potential[u];
        let mut relax = |arc: usize, v: usize, sp: &mut ShortestPaths, pending: &mut usize| {
            if net.is_arc_saturated(arc) {
                return;
            }
            let rc = if net.is_used(arc) {
                0
            } else if net.convex_mode {
                net.marginal_cost(arc) - pot_u + net.potential[v]
            } else {
                net.arc_cost(g, arc) as i64 - pot_u + net.potential[v]
            };
            debug_assert!(rc >= 0, "negative reduced cost on arc {arc}: {rc}");
            let nd = cur_dist + rc;
            if nd < sp.dist[v] {
                sp.dist[v] = nd;
                sp.pred_arc[v] = arc as i32;
                let b = (nd as usize) % k;
                buckets[b].push_back((v, nd)); // FIFO (see decl)
                *pending += 1;
            }
        };
        out_buf.clear();
        if u < g.num_nodes() {
            g.outgoing(u, &mut out_buf);
        }
        for &(arc, v) in out_buf.iter() {
            relax(arc, v, &mut *sp, &mut pending);
        }
        for &(arc, v) in net.extra_outgoing(u).iter() {
            relax(arc, v, &mut *sp, &mut pending);
        }
    }
    if dbg {
        let popped_count = sp.popped.iter().filter(|&&p| p).count();
        eprintln!(
            "[run_full] done: cur_dist={cur_dist} real_pops={real_pops} stale_pops={stale_pops} total_ba={total_bucket_advances} popped={popped_count}/{n_nodes}"
        );
    }
}

/// Below this bucket size, parallel relaxation costs more in fork/join overhead
/// than it saves. (Set empirically; rayon overhead is ~10 µs/spawn on M-series.)
const PAR_THRESHOLD: usize = 256;

/// Proposal emitted by a phase-1 thread: `(v, nd, arc)`.
/// We collect into per-thread Vecs and reduce, so the constant matters less
/// than amortized.
type Proposal = (usize, i64, u32);

/// Parallel Dial's multi-source Dijkstra. Functionally equivalent to [`run`],
/// but each large-enough bucket is relaxed via rayon. See module docs for the
/// race-freedom argument.
pub fn run_parallel<G: ResidualGraph>(g: &G, net: &Network) -> ShortestPaths {
    let n_nodes = net.num_nodes();
    let mut sp = ShortestPaths::new(n_nodes);

    let max_rc = max_reduced_cost_par(g, net);
    let k = (max_rc as usize).saturating_add(1).max(1);

    let (is_sink, total_sinks) = collect_sinks(net);
    let mut sinks_left = total_sinks;

    let mut buckets: Vec<Vec<(usize, i64)>> = vec![Vec::new(); k];
    // AtomicBool per node - used to claim `u` in phase 1; one CAS per node
    // across the whole Dijkstra call.
    let visited: Vec<AtomicBool> = (0..n_nodes).map(|_| AtomicBool::new(false)).collect();
    let mut pending: usize = 0;

    for s in net.excess_nodes() {
        sp.dist[s] = 0;
        buckets[0].push((s, 0));
        pending += 1;
    }
    if pending == 0 || total_sinks == 0 {
        return sp;
    }

    let mut cur_bucket = 0_usize;
    let mut cur_dist: i64 = 0;
    let mut bucket_advances: usize = 0;

    while pending > 0 {
        while buckets[cur_bucket].is_empty() {
            cur_bucket = (cur_bucket + 1) % k;
            cur_dist += 1;
            bucket_advances += 1;
            if bucket_advances > k {
                return sp;
            }
        }
        bucket_advances = 0;

        let current = std::mem::take(&mut buckets[cur_bucket]);
        pending -= current.len();

        // Result of relaxing this bucket - either path mutates these.
        let mut sinks_popped_this_bucket: usize = 0;

        if current.len() < PAR_THRESHOLD {
            // Serial fast path - same logic as `run`.
            let mut out_buf: Vec<(usize, usize)> = Vec::with_capacity(8);
            for (u, qd) in current {
                if visited[u].load(Ordering::Relaxed) {
                    continue;
                }
                if sp.dist[u] != qd || qd != cur_dist {
                    continue;
                }
                if visited[u]
                    .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    continue;
                }
                sp.popped[u] = true;
                if is_sink[u] {
                    sinks_popped_this_bucket += 1;
                }
                let pot_u = net.potential[u];
                let mut relax =
                    |arc: usize, v: usize, sp: &mut ShortestPaths, pending: &mut usize| {
                        if net.is_arc_saturated(arc) {
                            return;
                        }
                        let rc = if net.is_used(arc) {
                            0
                        } else if net.convex_mode {
                            net.marginal_cost(arc) - pot_u + net.potential[v]
                        } else {
                            net.arc_cost(g, arc) as i64 - pot_u + net.potential[v]
                        };
                        let nd = cur_dist + rc;
                        if nd < sp.dist[v] {
                            sp.dist[v] = nd;
                            sp.pred_arc[v] = arc as i32;
                            let b = (nd as usize) % k;
                            buckets[b].push((v, nd));
                            *pending += 1;
                        }
                    };
                out_buf.clear();
                if u < g.num_nodes() {
                    g.outgoing(u, &mut out_buf);
                }
                for &(arc, v) in out_buf.iter() {
                    relax(arc, v, &mut sp, &mut pending);
                }
                for &(arc, v) in net.extra_outgoing(u).iter() {
                    relax(arc, v, &mut sp, &mut pending);
                }
            }
        } else {
            // Phase 1 (parallel): each thread claims its `u`s via
            // `visited[u].compare_exchange` and collects proposals against a
            // *snapshot* of sp.dist. Reads are safe because no thread writes
            // sp.dist in this phase.
            let visited_ref = &visited;
            let sp_dist_snap: &[i64] = &sp.dist;

            // Per-thread fold returns (proposals, popped_nodes, sinks_popped).
            // Popped nodes are accumulated in fold-local Vecs and applied to
            // sp.popped serially after the reduce.
            let (proposals, popped_nodes, sinks_popped, _) = current
                .par_iter()
                .fold(
                    || {
                        (
                            Vec::<Proposal>::new(),
                            Vec::<u32>::new(),
                            0_usize,
                            Vec::<(usize, usize)>::with_capacity(8),
                        )
                    },
                    |(mut props, mut pops, mut nsinks, mut out_buf): (
                        Vec<Proposal>,
                        Vec<u32>,
                        usize,
                        Vec<(usize, usize)>,
                    ),
                     &(u, qd)| {
                        if visited_ref[u].load(Ordering::Relaxed) {
                            return (props, pops, nsinks, out_buf);
                        }
                        if sp_dist_snap[u] != qd || qd != cur_dist {
                            return (props, pops, nsinks, out_buf);
                        }
                        if visited_ref[u]
                            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                            .is_err()
                        {
                            return (props, pops, nsinks, out_buf);
                        }
                        pops.push(u as u32);
                        if is_sink[u] {
                            nsinks += 1;
                        }
                        let pot_u = net.potential[u];
                        let consider = |arc: usize, v: usize, props: &mut Vec<Proposal>| {
                            if net.is_arc_saturated(arc) {
                                return;
                            }
                            let rc = if net.is_used(arc) {
                                0
                            } else if net.convex_mode {
                                net.marginal_cost(arc) - pot_u + net.potential[v]
                            } else {
                                net.arc_cost(g, arc) as i64 - pot_u + net.potential[v]
                            };
                            let nd = cur_dist + rc;
                            if nd < sp_dist_snap[v] {
                                props.push((v, nd, arc as u32));
                            }
                        };
                        out_buf.clear();
                        if u < g.num_nodes() {
                            g.outgoing(u, &mut out_buf);
                        }
                        for &(arc, v) in out_buf.iter() {
                            consider(arc, v, &mut props);
                        }
                        for &(arc, v) in net.extra_outgoing(u).iter() {
                            consider(arc, v, &mut props);
                        }
                        (props, pops, nsinks, out_buf)
                    },
                )
                .reduce(
                    || (Vec::new(), Vec::new(), 0_usize, Vec::new()),
                    |(mut pa, mut popa, sa, ob_a), (pb, popb, sb, _ob_b)| {
                        pa.extend(pb);
                        popa.extend(popb);
                        (pa, popa, sa + sb, ob_a)
                    },
                );

            for u in popped_nodes {
                sp.popped[u as usize] = true;
            }
            sinks_popped_this_bucket = sinks_popped;

            // Phase 2 (serial): re-check `nd < sp.dist[v]` because (a)
            // multiple threads may have proposed for the same v, (b) sp.dist
            // is now mutated as we apply.
            for (v, nd, arc) in proposals {
                if nd < sp.dist[v] {
                    sp.dist[v] = nd;
                    sp.pred_arc[v] = arc as i32;
                    let b = (nd as usize) % k;
                    buckets[b].push((v, nd));
                    pending += 1;
                }
            }
        }

        if sinks_popped_this_bucket > 0 {
            sinks_left = sinks_left.saturating_sub(sinks_popped_this_bucket);
            if sinks_left == 0 {
                return sp;
            }
        }
    }
    sp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost;
    use crate::residue;
    use crate::simulate;
    use ndarray::Array2;
    use num_complex::Complex32;
    use rand::SeedableRng;

    /// Serial and parallel Dial's must produce identical shortest-path
    /// *distances* for nodes both runs popped. (Pred chains can vary on
    /// ties; `dist[]` for unpopped nodes is implementation-defined because
    /// the early-exit triggers mid-bucket in `run` but only at end-of-bucket
    /// in `run_parallel`, so the parallel version may relax a few more
    /// non-sink nodes before exiting.)
    #[test]
    fn parallel_and_serial_agree_on_distances() {
        let m = 64;
        let n = 64;
        let truth = simulate::diagonal_ramp((m, n));
        let gamma = Array2::<f32>::from_elem((m, n), 0.4);
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let (igram, cor) = simulate::simulate_ifg(&truth, &gamma, 4, &mut rng);

        let wrapped = igram.mapv(|z: Complex32| z.arg());
        let residues = residue::compute(wrapped.view());
        let costs = cost::compute_carballo_costs(igram.view(), cor.view(), 4.0, None);
        let g = crate::grid::RectangularGridGraph::new(m + 1, n + 1);
        let net = crate::network::Network::new(&g, residues.view(), &costs);

        let sp_serial = run(&g, &net);
        let sp_parallel = run_parallel(&g, &net);

        assert_eq!(sp_serial.dist.len(), sp_parallel.dist.len());
        for i in 0..sp_serial.dist.len() {
            if sp_serial.popped[i] && sp_parallel.popped[i] {
                assert_eq!(
                    sp_serial.dist[i], sp_parallel.dist[i],
                    "dist mismatch at popped node {i}: serial={} parallel={}",
                    sp_serial.dist[i], sp_parallel.dist[i]
                );
            }
        }
    }
}
