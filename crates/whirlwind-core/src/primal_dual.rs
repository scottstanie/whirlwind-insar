//! "Primal-dual" min-cost-flow loop: multi-source Dijkstra, augment, update
//! potentials, repeat. Falls back to per-source SSP after `max_iter` iters.
//!
//! Naming note: this is successive shortest paths with parallel augmentation
//! (one unit per source along the shortest-path forest each iteration), the
//! strategy PHASS introduced for phase unwrapping - not the primal-dual
//! method of Ahuja et al. §9.8, which additionally solves a max-flow
//! subproblem on the zero-reduced-cost network each iteration. The module
//! keeps the historical name.

use crate::network::Network;
use crate::residual_graph::ResidualGraph;
use crate::shortest_path::{
    ShortestPaths, dijkstra_multi_source_full_scratch_into, dijkstra_multi_source_into,
};
use crate::ssp;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::OnceLock;

/// If set, primal-dual prints per-iteration state to stderr. Cached after the
/// first read so 25+ Dijkstra iters don't each hit the env-var lookup.
pub fn debug_enabled() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("WHIRLWIND_DEBUG").is_ok())
}

/// Cumulative per-stage timings for one `run()` call. Useful for profiling.
#[derive(Default, Debug, Clone, Copy)]
pub struct PDTimings {
    pub dijkstra_ms: f64,
    pub augment_ms: f64,
    pub potential_ms: f64,
    pub iters: u32,
    pub ssp_calls: u32,
    pub ssp_iters: u32,
}

thread_local! {
    static LAST_TIMINGS: std::cell::RefCell<PDTimings> = const { std::cell::RefCell::new(PDTimings { dijkstra_ms: 0.0, augment_ms: 0.0, potential_ms: 0.0, iters: 0, ssp_calls: 0, ssp_iters: 0 }) };
}

pub fn last_timings() -> PDTimings {
    LAST_TIMINGS.with(|c| *c.borrow())
}

fn record_dijkstra(ms: f64) {
    LAST_TIMINGS.with(|c| c.borrow_mut().dijkstra_ms += ms);
}
fn record_augment(ms: f64) {
    LAST_TIMINGS.with(|c| c.borrow_mut().augment_ms += ms);
}
fn record_potential(ms: f64) {
    LAST_TIMINGS.with(|c| c.borrow_mut().potential_ms += ms);
}
fn record_iter() {
    LAST_TIMINGS.with(|c| c.borrow_mut().iters += 1);
}
pub(crate) fn record_ssp_call() {
    LAST_TIMINGS.with(|c| c.borrow_mut().ssp_calls += 1);
}
pub(crate) fn record_ssp_iter() {
    LAST_TIMINGS.with(|c| c.borrow_mut().ssp_iters += 1);
}
fn reset_timings() {
    LAST_TIMINGS.with(|c| *c.borrow_mut() = PDTimings::default());
}

pub fn run<G: ResidualGraph>(g: &G, net: &mut Network, max_iter: usize) {
    run_impl(g, net, max_iter, false);
}

/// Primal-dual loop using full-completion Dijkstra - matches Python ww-orig.
///
/// Python's `dijkstra_pd` runs until the heap is empty: every reachable node
/// is popped and gets an exact finalized distance. The subsequent potential
/// update `π[v] -= d[v]` for ALL nodes produces tight reduced costs, causing
/// each PD iteration to route significantly more flow than early-exit Dijkstra.
/// On large full-frame scenes this closes a several-percent quality gap.
pub fn run_full_dijkstra<G: ResidualGraph>(g: &G, net: &mut Network, max_iter: usize) {
    run_impl(g, net, max_iter, true);

    // GUARDED ADAPTIVE FALLBACK. ww-orig does PD(8)+SSP and its SSP completes;
    // ours can STRAND residues (the single-source SSP greedily fragments the
    // residual graph on heavily-masked frames → excess nodes trapped in tiny
    // residual components with no deficit). When that leaves the network
    // unbalanced, resume the multi-source PD (which converges to ww-orig's flow
    // - it doesn't fragment the same way) in chunks, retrying SSP
    // each round, up to a cap. This is a robustness lever, NOT literal ww-orig
    // parity (ww-orig fixes it in one 8-PD + SSP pass; the remaining trajectory
    // mismatch is still open). The final imbalance is always reported.
    let dbg = debug_enabled();
    let excess_now = |net: &Network| -> i64 {
        net.excess
            .iter()
            .filter(|&&e| e > 0)
            .map(|&e| e as i64)
            .sum()
    };
    let deficit_now = |net: &Network| -> i64 {
        net.excess
            .iter()
            .filter(|&&e| e < 0)
            .map(|&e| -e as i64)
            .sum()
    };
    // Gate on BOTH sides: a one-sided network (excess left, every deficit
    // drained - e.g. an imbalanced residue grid) has provably no augmenting
    // path left, so resuming PD / SSP would only burn whole-graph Dijkstra
    // floods discovering that. Report and return instead.
    if excess_now(net) > 0 && deficit_now(net) > 0 {
        let cap: usize = std::env::var("WHIRLWIND_LINEAR_PD_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        // One scratch set shared across every resume round's SSP (this loop
        // only runs on stranding frames, but when it does it can go tens of
        // rounds - reallocating ~n_nodes-sized buffers each round is waste).
        let mut sp = ShortestPaths::new(net.num_nodes());
        let mut fifo_buckets: Vec<VecDeque<u32>> = Vec::new();
        let mut total = max_iter;
        while excess_now(net) > 0 && deficit_now(net) > 0 && total < cap {
            let before = excess_now(net);
            let chunk = 16usize.min(cap - total);
            run_no_ssp(g, net, chunk);
            total += chunk;
            let after = excess_now(net);
            if dbg {
                eprintln!("[pd_full] adaptive resume: total_pd={total} excess {before}->{after}");
            }
            if after == 0 {
                break;
            }
            // SSP cleanup each round (drains anything PD's reached but didn't pair).
            record_ssp_call();
            ssp::run_single_source_scratch(g, net, &mut sp, &mut fifo_buckets);
            if excess_now(net) == 0 {
                break;
            }
            if after >= before {
                // PD made no progress this round AND SSP couldn't finish - the
                // residual is genuinely trapped for the cost-aware solvers; stop
                // here and let the cost-ignoring BFS guard below pair the rest.
                break;
            }
        }
    }
    if dbg {
        let rem: i64 = net.excess.iter().map(|&e| (e as i64).abs()).sum();
        eprintln!("[pd_full] ADAPTIVE FINAL remaining_excess={rem}");
    }

    // FINAL BALANCE GUARD. Integrating an unbalanced network turns every
    // unpaired residue pair into a full-width 2π tear, so if the cost-aware
    // passes above ever leave excess behind, pair it here no matter what.
    // (The one known way they did - the Ridgecrest block-tear, where the Dial
    // SSP's bucket-advance exit tested cumulative instead of consecutive empty
    // advances and cut long searches short - is fixed in `run_single_source`;
    // this guard remains as the safety net for any future incomplete pass.)
    // Pair any survivors with a cost-ignoring residual BFS, which cannot strand
    // on a connected balanced graph. It runs only when excess remains (a no-op
    // on frames that already drained, so ww-orig parity is untouched there) and
    // both sides are nonzero (a one-sided imbalance has no augmenting path left).
    // Opt out with WHIRLWIND_NO_BFS_DRAIN=1 to observe the raw stranding.
    if excess_now(net) > 0
        && deficit_now(net) > 0
        && std::env::var("WHIRLWIND_NO_BFS_DRAIN").is_err()
    {
        let paired = ssp::drain_residual_bfs(g, net);
        if dbg {
            let rem: i64 = net.excess.iter().map(|&e| (e as i64).abs()).sum();
            eprintln!("[pd_full] BFS DRAIN paired={paired} remaining_excess={rem}");
        }
    }
}

fn run_impl<G: ResidualGraph>(g: &G, net: &mut Network, max_iter: usize, full_dijkstra: bool) {
    reset_timings();
    let dbg = debug_enabled();
    let tag = if full_dijkstra { "pd_full" } else { "pd" };
    // Note: we do NOT require `net.is_balanced()` here. The boundary-zeroing
    // pass in `residue::compute` can leave a small charge imbalance for real
    // noisy data; the algorithm will route as much flow as it can and stop
    // when no more excess/deficit pairs are connected.

    // Reusable scratch buffers - kept alive across PD iterations so we don't
    // pay a ~n_nodes x 4-byte allocation per iter (on 8192² that's 268 MiB
    // each, churned tens of times across the run).
    let n_nodes = net.num_nodes();
    let mut visited_epoch: Vec<u32> = vec![0; n_nodes];
    let mut source_used: Vec<bool> = vec![false; n_nodes];
    let mut path_info: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    let mut deficits: Vec<usize> = Vec::new();
    let mut sp = ShortestPaths::new(n_nodes);
    // FIFO Dial buckets, reused across the full-completion Dijkstras (they
    // grow to ~n_nodes u32 entries; re-growing them every iteration was pure
    // allocator churn) and handed to the SSP tail below.
    let mut fifo_buckets: Vec<VecDeque<u32>> = Vec::new();
    // Epoch counter persists across PD iterations too - wrap-on-zero handles
    // the (vanishingly rare) ~4B-walk overflow.
    let mut epoch: u32 = 0;

    let mut iter = 0;
    let mut last_excess_total = i64::MAX;
    loop {
        let excess_total: i64 = net
            .excess
            .iter()
            .filter(|&&e| e > 0)
            .map(|&e| e as i64)
            .sum();
        let deficit_total: i64 = net
            .excess
            .iter()
            .filter(|&&e| e < 0)
            .map(|&e| -e as i64)
            .sum();
        if dbg {
            eprintln!("[{tag}] iter={iter} excess={excess_total} deficit={deficit_total}");
        }
        if excess_total == 0 || deficit_total == 0 {
            // Either fully balanced, or no remaining deficit to flow toward.
            return;
        }
        if excess_total >= last_excess_total {
            if dbg {
                eprintln!("[{tag}] no progress, falling to SSP");
            }
            // No progress this iter - give up to avoid spinning.
            break;
        }
        last_excess_total = excess_total;

        if dbg {
            eprintln!("[{tag}] iter={iter} running dijkstra");
        }
        let t0 = std::time::Instant::now();
        if full_dijkstra {
            dijkstra_multi_source_full_scratch_into(g, net, &mut sp, &mut fifo_buckets);
        } else {
            dijkstra_multi_source_into(g, net, &mut sp);
        }
        let dt = t0.elapsed().as_secs_f64();
        record_dijkstra(dt * 1000.0);
        record_iter();
        if dbg {
            eprintln!("[{tag}] iter={iter} dijkstra took {:.3}s", dt);
        }

        // Augment: each deficit node gets +1 from the *actual* source at the
        // end of its predecessor chain. The chain is walked via pred_arc only:
        // the predecessor node is the arc's tail (`net.arc_endpoints(..).0`).
        // (A per-relax `source` attribution used to exist but could go stale
        // relative to the pred chain when an upstream parent was re-relaxed
        // by a different source - the walk is the only trustworthy seed ID.)
        let t_aug = std::time::Instant::now();
        deficits.clear();
        deficits.extend(net.deficit_nodes());
        // Pre-compute actual source per sink by walking pred_node back to the
        // pred_arc<0 node (a seed source). Also remember the path arcs.
        //
        // Cycle detection: a `visited_epoch[v] == epoch` marks v as on the
        // current sink's path - replacing the per-sink HashSet (heap alloc
        // each sink) with one bumped `u32` counter that persists across
        // outer iterations.
        path_info.clear();
        for &sink in &deficits {
            if !sp.was_reached(sink) {
                continue;
            }
            epoch = epoch.wrapping_add(1);
            if epoch == 0 {
                // Counter wrap: reset the table (extremely rare; only after
                // ~4 billion deficit-walks).
                visited_epoch.fill(0);
                epoch = 1;
            }
            let mut arcs = Vec::new();
            let mut cur = sink;
            visited_epoch[cur] = epoch;
            loop {
                let parc = sp.pred_arc[cur];
                if parc < 0 {
                    break; // cur is a seed source
                }
                arcs.push(parc as usize);
                // Predecessor node = tail of the predecessor arc.
                cur = net.arc_endpoints(g, parc as usize).0;
                if visited_epoch[cur] == epoch {
                    if dbg && arcs.len() < 30 {
                        eprintln!(
                            "[{tag}] CYCLE in pred-chain from sink={sink}, revisits cur={cur} after {} hops, dist={}",
                            arcs.len(),
                            sp.dist[cur]
                        );
                    }
                    arcs.clear();
                    break;
                }
                visited_epoch[cur] = epoch;
            }
            if !arcs.is_empty() {
                path_info.push((sink, cur, arcs));
            }
        }
        // Augment order = which deficit each source serves. Match ww-orig's
        // `augment_flow_pd`: sort sinks by (source, key), keep the FIRST per
        // source (via `source_used` below) = the source's chosen deficit.
        //   * full-completion (parity) path: key = Dijkstra DISTANCE, exactly
        //     ww-orig (`distance_to_vertex(sink)`). With the FIFO Dial above,
        //     distances are true min-hop shortest-path distances, so "nearest by
        //     distance" pairs each residue with its nearest partner - the short
        //     branch cut that is the correct unwrap on masked frames. (Distance
        //     here is meaningful ONLY because the Dijkstra is FIFO; the two are a
        //     pair - hop-count sort or LIFO distances both diverge from ww-orig.)
        //   * early-exit path (default/convex/reuse production): keep hop count -
        //     distance there breaks the convex probe's SSP non-negativity
        //     invariant, and that path is not a parity path. (item.0 = sink,
        //     item.1 = source.)
        if full_dijkstra {
            path_info.sort_by_key(|item| (item.1, sp.dist[item.0]));
        } else {
            path_info.sort_by_key(|item| (item.1, item.2.len()));
        }
        // Reset the source_used scratch buffer in place; allocation is reused.
        source_used.iter_mut().for_each(|x| *x = false);
        let mut augmented = 0;

        for (sink, src, arcs) in path_info.drain(..) {
            if source_used[src] {
                continue;
            }
            source_used[src] = true;
            for arc in arcs {
                net.push_unit(g, arc);
            }
            net.increase_excess(sink, 1);
            net.decrease_excess(src, 1);
            augmented += 1;
        }
        record_augment(t_aug.elapsed().as_secs_f64() * 1000.0);
        if dbg {
            eprintln!("[{tag}] iter={iter} augmented {augmented}");
        }

        let t_pot = std::time::Instant::now();
        // Update potentials: π[v] -= dist[v] for nodes finalized by Dijkstra.
        // For NON-FINALIZED nodes (either truly unreached, or just relaxed
        // but never popped under early-exit), cap their effective dist at
        // D_max (the largest dist among popped nodes). Without this cap,
        // residual arcs that cross the boundary acquire negative reduced
        // cost on the next iteration → Dijkstra produces cyclic predecessor
        // chains. (Ahuja, Magnanti, Orlin §9: "valid potentials".)
        //
        // With full_dijkstra=true all nodes are popped so d_max is unused;
        // the `if popped { d } else { d_max }` always takes the `d` branch
        // and each node gets its exact shortest-path distance subtracted -
        // matching Python's `update_potential_pd`.
        let d_max = sp
            .dist
            .par_iter()
            .zip(sp.popped.par_iter())
            .filter_map(|(&d, &p)| if p { Some(d) } else { None })
            .max()
            .unwrap_or(0);
        net.potential
            .par_iter_mut()
            .zip(sp.dist.par_iter())
            .zip(sp.popped.par_iter())
            .for_each(|((pi, &d), &popped)| {
                let dv = if popped { d } else { d_max };
                *pi -= dv;
            });
        record_potential(t_pot.elapsed().as_secs_f64() * 1000.0);

        iter += 1;
        if iter >= max_iter {
            if dbg {
                eprintln!("[{tag}] hit max_iter, falling to SSP");
            }
            break;
        }
    }

    // Fall through to SSP for any remaining excess. Pick the variant that is
    // fast for this path's graph shape (ATBD §9.6): the full-completion path is
    // the single-tile whole-image solve, where the multi-source SSP is
    // catastrophic (a near-graph-wide Dijkstra per unit) - use single-source
    // there. The early-exit path is tiled / small-graph, where multi-source is
    // fast and robust to its d_max-capped potentials.
    //
    // The augment-phase scratch is dead from here on - free it first, and hand
    // the SSP the Dijkstra buffers (`sp`, the Dial buckets) instead of letting
    // it allocate an identically-shaped second set. Holding both sets alive
    // through the SSP tail used to be the peak-RSS high-water mark of a
    // NISAR-scale `unwrap_linear` (~390 MB of dead weight at 17.8M nodes).
    drop(visited_epoch);
    drop(source_used);
    drop(path_info);
    drop(deficits);
    record_ssp_call();
    if full_dijkstra {
        ssp::run_single_source_scratch(g, net, &mut sp, &mut fifo_buckets);
    } else {
        ssp::run_scratch(g, net, &mut sp);
    }

    if dbg {
        // FINAL MCF objective + leftover imbalance. total_cost is directly
        // comparable to ww-orig's `Network::total_cost()` (same Σ cost·flow over
        // forward arcs). If Rust's total_cost == ww-orig's but the unwrap
        // differs → equal-cost DEGENERATE optima (tie-break only); if Rust's is
        // HIGHER → Rust is sub-optimal (a real solver bug). remaining != 0 means
        // the flow never balanced (incomplete).
        let nf = net.num_forward();
        let total_cost: i64 = (0..nf)
            .map(|a| net.arc_flow(g, a) as i64 * net.arc_cost(g, a) as i64)
            .sum();
        let remaining: i64 = net.excess.iter().map(|&e| (e as i64).abs()).sum();
        eprintln!("[{tag}] FINAL total_cost={total_cost} remaining_excess={remaining}");
    }
}

/// Like [`run`] but stops after the primal-dual loop without SSP fallback.
///
/// Python `whirlwind_orig` does `primal_dual(network, maxiter=8)` and then
/// integrates immediately - it never calls SSP. On NISAR-scale problems
/// (71M arcs) SSP on remaining residues is catastrophically slow. Use this
/// when matching Python ww-orig behavior: run PD for a fixed number of
/// iterations, leave any unmatched residues as-is, and let the integration
/// absorb the small residual error.
pub fn run_no_ssp<G: ResidualGraph>(g: &G, net: &mut Network, max_iter: usize) {
    reset_timings();
    let dbg = debug_enabled();
    let n_nodes = net.num_nodes();
    let mut visited_epoch: Vec<u32> = vec![0; n_nodes];
    let mut source_used: Vec<bool> = vec![false; n_nodes];
    let mut path_info: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    let mut deficits: Vec<usize> = Vec::new();
    let mut sp = ShortestPaths::new(n_nodes);
    let mut epoch: u32 = 0;
    let mut iter = 0;
    let mut last_excess_total = i64::MAX;
    loop {
        let excess_total: i64 = net
            .excess
            .iter()
            .filter(|&&e| e > 0)
            .map(|&e| e as i64)
            .sum();
        let deficit_total: i64 = net
            .excess
            .iter()
            .filter(|&&e| e < 0)
            .map(|&e| -e as i64)
            .sum();
        if dbg {
            eprintln!("[pd_no_ssp] iter={iter} excess={excess_total} deficit={deficit_total}");
        }
        if excess_total == 0 || deficit_total == 0 {
            return;
        }
        if excess_total >= last_excess_total {
            if dbg {
                eprintln!("[pd_no_ssp] no progress, stopping (no SSP fallback)");
            }
            return; // stop here - no SSP
        }
        last_excess_total = excess_total;
        let t0 = std::time::Instant::now();
        dijkstra_multi_source_into(g, net, &mut sp);
        record_dijkstra(t0.elapsed().as_secs_f64() * 1000.0);
        record_iter();
        let t_aug = std::time::Instant::now();
        deficits.clear();
        deficits.extend(net.deficit_nodes());
        path_info.clear();
        for &sink in &deficits {
            if !sp.was_reached(sink) {
                continue;
            }
            epoch = epoch.wrapping_add(1);
            if epoch == 0 {
                visited_epoch.fill(0);
                epoch = 1;
            }
            let mut arcs = Vec::new();
            let mut cur = sink;
            visited_epoch[cur] = epoch;
            loop {
                let parc = sp.pred_arc[cur];
                if parc < 0 {
                    break;
                }
                arcs.push(parc as usize);
                // Predecessor node = tail of the predecessor arc.
                cur = net.arc_endpoints(g, parc as usize).0;
                if visited_epoch[cur] == epoch {
                    arcs.clear();
                    break;
                }
                visited_epoch[cur] = epoch;
            }
            if !arcs.is_empty() {
                path_info.push((sink, cur, arcs));
            }
        }
        path_info.sort_by_key(|item| (item.1, item.2.len()));
        source_used.iter_mut().for_each(|x| *x = false);
        for (sink, src, arcs) in path_info.drain(..) {
            if source_used[src] {
                continue;
            }
            source_used[src] = true;
            for arc in arcs {
                net.push_unit(g, arc);
            }
            net.increase_excess(sink, 1);
            net.decrease_excess(src, 1);
        }
        record_augment(t_aug.elapsed().as_secs_f64() * 1000.0);
        iter += 1;
        if iter >= max_iter {
            if dbg {
                eprintln!("[pd_no_ssp] hit max_iter, stopping (no SSP fallback)");
            }
            return; // stop here - no SSP
        }
    }
}
