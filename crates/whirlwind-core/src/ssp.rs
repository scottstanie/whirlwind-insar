//! Successive Shortest Paths fallback. One source per iteration.
//!
//! Simpler but slower than primal-dual: pick one excess node, run single-source
//! Dijkstra, augment along shortest path to nearest deficit, repeat.

use crate::grid::RectangularGridGraph;
use crate::network::Network;
use crate::shortest_path::dijkstra_multi_source;

pub fn run(g: &RectangularGridGraph, net: &mut Network) {
    let dbg = crate::primal_dual::debug_enabled();
    let mut safety = 0;
    let safety_limit = 4 * g.num_nodes();
    while net.excess_nodes().next().is_some() {
        if dbg && safety % 50 == 0 {
            let ex: i64 = net.excess.iter().filter(|&&e| e > 0).map(|&e| e as i64).sum();
            let df: i64 = net.excess.iter().filter(|&&e| e < 0).map(|&e| -e as i64).sum();
            eprintln!("[ssp] iter={safety} excess={ex} deficit={df}");
        }
        crate::primal_dual::record_ssp_iter();
        let sp = dijkstra_multi_source(g, net);
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
            if parc < 0 { break; }
            arcs.push(parc as usize);
            cur = sp.pred_node[cur] as usize;
            if arcs.len() > g.num_nodes() {
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
        for v in 0..g.num_nodes() {
            let dv = if sp.popped[v] { sp.dist[v] } else { d_max };
            net.potential[v] -= dv;
        }

        safety += 1;
        assert!(safety <= safety_limit, "SSP did not converge");
    }
}
