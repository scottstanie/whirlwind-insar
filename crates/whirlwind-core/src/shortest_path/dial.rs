//! Dial's bucket-queue Dijkstra.
//!
//! Reduced costs in primal-dual are non-negative integers bounded by some
//! `C_max`. A circular bucket queue of size `K = C_max + 1` lets us pop the
//! minimum in amortized O(1) and gives O(V + E + max_dist) total work per call
//! — competitive with the binary heap when C_max is small relative to V.
//!
//! Two implementations live here:
//! - [`run`]: single-threaded reference version.
//! - [`run_parallel`]: each non-trivial bucket is relaxed in parallel using
//!   rayon. Phase 1 (parallel) collects proposed relaxations after each
//!   thread atomically claims `u` via `visited[u]`; phase 2 (serial) applies
//!   them. This avoids racing writes to `sp.dist / pred_arc / pred_node /
//!   source` and is provably equivalent to the serial Dial on any
//!   single-shortest-path ties — each (u, qd) pair is processed at most once.

use super::ShortestPaths;
use crate::grid::RectangularGridGraph;
use crate::network::Network;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

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

/// Below this bucket size, parallel relaxation costs more in fork/join overhead
/// than it saves. (Set empirically; rayon overhead is ~10 µs/spawn on M-series.)
const PAR_THRESHOLD: usize = 256;

/// Proposal emitted by a phase-1 thread: `(v, nd, arc, source_of_u, u)`.
/// All four fields fit in 24 bytes packed; we collect into per-thread Vecs and
/// reduce, so the constant matters less than amortized.
type Proposal = (usize, i64, u32, i32, u32);

/// Parallel Dial's multi-source Dijkstra. Functionally equivalent to [`run`],
/// but each large-enough bucket is relaxed via rayon. See module docs for the
/// race-freedom argument.
pub fn run_parallel(g: &RectangularGridGraph, net: &Network) -> ShortestPaths {
    let n_nodes = g.num_nodes();
    let mut sp = ShortestPaths::new(n_nodes);

    // Max reduced cost — sequential. O(E) and small relative to Dijkstra
    // itself; not worth parallelizing.
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

    let mut buckets: Vec<Vec<(usize, i64)>> = vec![Vec::new(); k];
    // AtomicBool per node — used to claim `u` in phase 1; one CAS per node
    // across the whole Dijkstra call.
    let visited: Vec<AtomicBool> = (0..n_nodes).map(|_| AtomicBool::new(false)).collect();
    let mut pending: usize = 0;

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

        if current.len() < PAR_THRESHOLD {
            // Serial fast path — same logic as `run`.
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
                let (ui, uj) = g.node_ij(u);
                let out = g.outgoing(ui, uj);
                for &(arc, v) in out.iter() {
                    if net.is_arc_saturated(arc) {
                        continue;
                    }
                    let rc = net.reduced_cost(g, arc);
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
        } else {
            // Phase 1 (parallel): each thread claims its `u`s via
            // `visited[u].compare_exchange` and collects proposals against a
            // *snapshot* of sp.dist. Reads are safe because no thread writes
            // sp.dist in this phase.
            let visited_ref = &visited;
            let sp_dist_snap: &[i64] = &sp.dist;
            let sp_source_snap: &[i32] = &sp.source;

            let proposals: Vec<Proposal> = current
                .par_iter()
                .fold(
                    Vec::new,
                    |mut local: Vec<Proposal>, &(u, qd)| {
                        if visited_ref[u].load(Ordering::Relaxed) {
                            return local;
                        }
                        if sp_dist_snap[u] != qd || qd != cur_dist {
                            return local;
                        }
                        if visited_ref[u]
                            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                            .is_err()
                        {
                            return local;
                        }
                        let (ui, uj) = g.node_ij(u);
                        let out = g.outgoing(ui, uj);
                        let src = sp_source_snap[u];
                        for &(arc, v) in out.iter() {
                            if net.is_arc_saturated(arc) {
                                continue;
                            }
                            let rc = net.reduced_cost(g, arc);
                            let nd = cur_dist + rc;
                            if nd < sp_dist_snap[v] {
                                local.push((v, nd, arc as u32, src, u as u32));
                            }
                        }
                        local
                    },
                )
                .reduce(
                    Vec::new,
                    |mut a, b| {
                        a.extend(b);
                        a
                    },
                );

            // Phase 2 (serial): re-check `nd < sp.dist[v]` because (a)
            // multiple threads may have proposed for the same v, (b) sp.dist
            // is now mutated as we apply.
            for (v, nd, arc, src, u) in proposals {
                if nd < sp.dist[v] {
                    sp.dist[v] = nd;
                    sp.pred_arc[v] = arc as i32;
                    sp.pred_node[v] = u as i32;
                    sp.source[v] = src;
                    let b = (nd as usize) % k;
                    buckets[b].push((v, nd));
                    pending += 1;
                }
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
    /// *distances* for the same network. (Pred chains can vary on ties.)
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
            assert_eq!(
                sp_serial.dist[i], sp_parallel.dist[i],
                "dist mismatch at node {i}: serial={} parallel={}",
                sp_serial.dist[i], sp_parallel.dist[i]
            );
        }
    }
}
