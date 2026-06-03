//! Primal-dual min-cost-flow loop. Multi-source Dijkstra, augment, update
//! potentials, repeat. Falls back to per-source SSP after `max_iter` iters.

use crate::network::Network;
use crate::residual_graph::ResidualGraph;
use crate::shortest_path::{
    ShortestPaths, dijkstra_multi_source_full_into, dijkstra_multi_source_into,
};
use crate::ssp;
use rayon::prelude::*;
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

/// Primal-dual loop using full-completion Dijkstra — matches Python ww-orig.
///
/// Python's `dijkstra_pd` runs until the heap is empty: every reachable node
/// is popped and gets an exact finalized distance. The subsequent potential
/// update `π[v] -= d[v]` for ALL nodes produces tight reduced costs, causing
/// each PD iteration to route significantly more flow than early-exit Dijkstra.
/// On D_077 full-frame this closes a ~5.5% quality gap (94% → ~99%).
pub fn run_full_dijkstra<G: ResidualGraph>(g: &G, net: &mut Network, max_iter: usize) {
    run_impl(g, net, max_iter, true);
}

fn run_impl<G: ResidualGraph>(g: &G, net: &mut Network, max_iter: usize, full_dijkstra: bool) {
    reset_timings();
    let dbg = debug_enabled();
    let tag = if full_dijkstra { "pd_full" } else { "pd" };
    // Note: we do NOT require `net.is_balanced()` here. The boundary-zeroing
    // pass in `residue::compute` can leave a small charge imbalance for real
    // noisy data; the algorithm will route as much flow as it can and stop
    // when no more excess/deficit pairs are connected.

    // Reusable scratch buffers — kept alive across PD iterations so we don't
    // pay a ~n_nodes × 4-byte allocation per iter (on 8192² that's 268 MiB
    // each, churned tens of times across the run).
    let n_nodes = net.num_nodes();
    let mut visited_epoch: Vec<u32> = vec![0; n_nodes];
    let mut source_used: Vec<bool> = vec![false; n_nodes];
    let mut path_info: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    let mut deficits: Vec<usize> = Vec::new();
    let mut sp = ShortestPaths::new(n_nodes);
    // Epoch counter persists across PD iterations too — wrap-on-zero handles
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
            // No progress this iter — give up to avoid spinning.
            break;
        }
        last_excess_total = excess_total;

        if dbg {
            eprintln!("[{tag}] iter={iter} running dijkstra");
        }
        let t0 = std::time::Instant::now();
        if full_dijkstra {
            dijkstra_multi_source_full_into(g, net, &mut sp);
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
        // end of its predecessor chain. We don't trust sp.source[] for the
        // dedup key because Dijkstra relaxations can leave it stale relative
        // to pred_node[] (downstream nodes keep an old source attribution
        // after their upstream parent gets re-relaxed by a different source).
        let t_aug = std::time::Instant::now();
        deficits.clear();
        deficits.extend(net.deficit_nodes());
        // Pre-compute actual source per sink by walking pred_node back to the
        // pred_arc<0 node (a seed source). Also remember the path arcs.
        //
        // Cycle detection: a `visited_epoch[v] == epoch` marks v as on the
        // current sink's path — replacing the per-sink HashSet (heap alloc
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
                cur = sp.pred_node[cur] as usize;
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
        // Sort by source then hop count to sink. Sorting by Dijkstra distance
        // instead breaks the SSP non-negativity invariant for the convex solver
        // (the convex probe finds a negative residual cycle → non-optimal flow)
        // and has zero effect on D_077 linear quality, so we keep hop count.
        path_info.sort_by_key(|item| (item.1, item.2.len()));
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
        // and each node gets its exact shortest-path distance subtracted —
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

    // Fall through to SSP for any remaining excess.
    record_ssp_call();
    ssp::run(g, net);
}

/// Like [`run`] but stops after the primal-dual loop without SSP fallback.
///
/// Python `whirlwind_orig` does `primal_dual(network, maxiter=8)` and then
/// integrates immediately — it never calls SSP. On NISAR-scale problems
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
            return; // stop here — no SSP
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
                cur = sp.pred_node[cur] as usize;
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
            return; // stop here — no SSP
        }
    }
}
