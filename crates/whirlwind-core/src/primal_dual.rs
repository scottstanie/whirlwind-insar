//! Primal-dual min-cost-flow loop. Multi-source Dijkstra, augment, update
//! potentials, repeat. Falls back to per-source SSP after `max_iter` iters.

use crate::grid::RectangularGridGraph;
use crate::network::Network;
use crate::shortest_path::dijkstra_multi_source;
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

pub fn run(g: &RectangularGridGraph, net: &mut Network, max_iter: usize) {
    reset_timings();
    let dbg = debug_enabled();
    // Note: we do NOT require `net.is_balanced()` here. The boundary-zeroing
    // pass in `residue::compute` can leave a small charge imbalance for real
    // noisy data; the algorithm will route as much flow as it can and stop
    // when no more excess/deficit pairs are connected.

    let mut iter = 0;
    let mut last_excess_total = i64::MAX;
    loop {
        let excess_total: i64 = net.excess.iter().filter(|&&e| e > 0).map(|&e| e as i64).sum();
        let deficit_total: i64 = net.excess.iter().filter(|&&e| e < 0).map(|&e| -e as i64).sum();
        if dbg {
            eprintln!(
                "[pd] iter={iter} excess={excess_total} deficit={deficit_total}"
            );
        }
        if excess_total == 0 || deficit_total == 0 {
            // Either fully balanced, or no remaining deficit to flow toward.
            return;
        }
        if excess_total >= last_excess_total {
            if dbg { eprintln!("[pd] no progress, falling to SSP"); }
            // No progress this iter — give up to avoid spinning.
            break;
        }
        last_excess_total = excess_total;

        if dbg { eprintln!("[pd] iter={iter} running dijkstra"); }
        let t0 = std::time::Instant::now();
        let sp = dijkstra_multi_source(g, net);
        let dt = t0.elapsed().as_secs_f64();
        record_dijkstra(dt * 1000.0);
        record_iter();
        if dbg { eprintln!("[pd] iter={iter} dijkstra took {:.3}s", dt); }

        // Augment: each deficit node gets +1 from the *actual* source at the
        // end of its predecessor chain. We don't trust sp.source[] for the
        // dedup key because Dijkstra relaxations can leave it stale relative
        // to pred_node[] (downstream nodes keep an old source attribution
        // after their upstream parent gets re-relaxed by a different source).
        let t_aug = std::time::Instant::now();
        let deficits: Vec<usize> = net.deficit_nodes().collect();
        // Pre-compute actual source per sink by walking pred_node back to the
        // pred_arc<0 node (a seed source). Also remember the path arcs.
        //
        // Cycle detection: a `visited_epoch[v] == iter` marks v as on the
        // current sink's path — replacing the per-sink HashSet (heap alloc
        // each sink) with one bumped `u32` counter per outer iteration.
        let n_nodes = g.num_nodes();
        let mut path_info: Vec<(usize, usize, Vec<usize>)> = Vec::new(); // (sink, src, arcs)
        let mut visited_epoch = vec![0_u32; n_nodes];
        let mut epoch: u32 = 0;
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
                            "[pd] CYCLE in pred-chain from sink={sink}, revisits cur={cur} after {} hops, dist={}",
                            arcs.len(), sp.dist[cur]
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
        // Sort by source then path length so closest paths win.
        path_info.sort_by_key(|(_, src, arcs)| (*src, arcs.len()));
        let mut source_used = vec![false; n_nodes];
        let mut augmented = 0;

        for (sink, src, arcs) in path_info.into_iter() {
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
        if dbg { eprintln!("[pd] iter={iter} augmented {augmented}"); }

        let t_pot = std::time::Instant::now();
        // Update potentials: π[v] -= dist[v] for nodes reached.
        // For UNREACHED nodes, cap their effective dist at D_max (the largest
        // dist among reached nodes). Without this cap, residual arcs that
        // cross the reach/unreach boundary acquire negative reduced cost on
        // the next iteration → Dijkstra produces cyclic predecessor chains.
        // (Ahuja, Magnanti, Orlin §9: "valid potentials" after a Dijkstra pass.)
        let d_max = sp
            .dist
            .par_iter()
            .filter(|&&d| d < i64::MAX)
            .copied()
            .max()
            .unwrap_or(0);
        net.potential
            .par_iter_mut()
            .zip(sp.dist.par_iter())
            .for_each(|(pi, &d)| {
                let dv = if d == i64::MAX { d_max } else { d };
                *pi -= dv;
            });
        record_potential(t_pot.elapsed().as_secs_f64() * 1000.0);

        iter += 1;
        if iter >= max_iter {
            if dbg { eprintln!("[pd] hit max_iter, falling to SSP"); }
            break;
        }
    }

    // Fall through to SSP for any remaining excess.
    record_ssp_call();
    ssp::run(g, net);
}
