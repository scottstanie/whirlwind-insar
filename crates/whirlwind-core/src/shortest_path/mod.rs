//! Multi-source shortest paths on the residue residual graph.

pub mod dial;
pub mod heap;

use crate::network::Network;
use crate::residual_graph::ResidualGraph;
use std::sync::OnceLock;

/// Backend selector for the multi-source Dijkstra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DijkstraBackend {
    /// Dial's bucket queue (default; fastest for bounded integer reduced costs).
    DialSerial,
    /// Parallel Dial's: process each bucket's nodes via rayon. Helps on
    /// residue-dense scenes where Dijkstra dominates.
    DialParallel,
    /// Binary-heap. Kept for comparison / debugging.
    Heap,
}

/// Read the backend from `WHIRLWIND_DIJKSTRA` (cached after first call).
/// Values: `heap`, `dial` / `serial` (default), `dial-par` / `parallel`.
///
/// Note on parallelism: the rayon-parallel Dial is a phase-1/phase-2 design
/// (parallel proposal collection + serial application). On the noisy-ramp
/// workloads we measured (M-series, 8 perf cores) it's *not* faster than
/// serial Dial because phase 2 dominates and is serial. It's kept as an
/// opt-in for further experimentation; for production use, leave the default.
pub fn backend() -> DijkstraBackend {
    static BE: OnceLock<DijkstraBackend> = OnceLock::new();
    *BE.get_or_init(
        || match std::env::var("WHIRLWIND_DIJKSTRA").ok().as_deref() {
            Some("heap") => DijkstraBackend::Heap,
            Some("dial-par") | Some("parallel") => DijkstraBackend::DialParallel,
            // Default is serial Dial.
            _ => DijkstraBackend::DialSerial,
        },
    )
}

/// Per-node (potential, distance) packed into one 16-byte struct.
///
/// Every Dijkstra relaxation reads these two values together (`rc = cost −
/// π_u + π_v`, then `nd < dist[v]`), and on a NISAR-scale frame those are the
/// two dominant random loads of the whole solve. Keeping them in separate
/// `Vec<i64>`s costs two cache-line fetches per neighbor; packed, one line
/// serves both (and the improvement write to `dist` hits the same line).
#[derive(Clone, Copy, Debug)]
pub struct PotDist {
    pub pot: i64,
    pub dist: i64,
}

/// Solver state for the fused (linear, full-completion) path: the node
/// potentials live HERE, not in `Network.potential`, for the duration of the
/// solve. Built by stealing `net.potential` (`FusedSolveState::take_from`) and
/// written back with `restore_into` at every solver exit - peak-RSS-neutral
/// relative to the split layout (`pd` replaces `potential` + `dist`).
pub struct FusedSolveState {
    pub pd: Vec<PotDist>,
    /// See [`ShortestPaths::pred_arc`].
    pub pred_arc: Vec<i64>,
    /// See [`ShortestPaths::popped`].
    pub popped: Vec<bool>,
}

impl FusedSolveState {
    /// Move `net.potential` into the fused layout (leaving it empty) with
    /// every distance at "unreached".
    pub fn take_from(net: &mut crate::network::Network) -> Self {
        let pot = std::mem::take(&mut net.potential);
        let n = pot.len();
        let pd = pot
            .into_iter()
            .map(|p| PotDist {
                pot: p,
                dist: i64::MAX,
            })
            .collect();
        Self {
            pd,
            pred_arc: vec![-1; n],
            popped: vec![false; n],
        }
    }

    /// Write the potentials back into `net.potential`. Call at every solver
    /// exit so code outside the fused path (adaptive resume, diagnostics,
    /// tests) keeps seeing valid potentials.
    ///
    /// Consumes the search state: `pred_arc`/`popped` are freed BEFORE the
    /// new potential vector is allocated, so the restore doesn't add a
    /// transient on top of the solve's peak RSS.
    pub fn restore_into(&mut self, net: &mut crate::network::Network) {
        self.pred_arc = Vec::new();
        self.popped = Vec::new();
        net.potential = self.pd.iter().map(|x| x.pot).collect();
    }

    /// Reset the per-Dijkstra fields (dist/pred/popped), preserving `pot`.
    pub fn reset_search(&mut self) {
        for x in self.pd.iter_mut() {
            x.dist = i64::MAX;
        }
        self.pred_arc.fill(-1);
        self.popped.fill(false);
    }
}

/// Result of a multi-source Dijkstra over the residual graph from every
/// `excess_node` simultaneously, using reduced costs as arc lengths.
pub struct ShortestPaths {
    pub dist: Vec<i64>,
    /// Arc id of the predecessor arc, -1 if none. The predecessor *node* is
    /// not stored - it is the tail of this arc, recoverable in O(1) via
    /// `net.arc_endpoints(g, arc).0`. (A stored `pred_node` and a per-relax
    /// `source` attribution used to live here; both were dead weight - 8
    /// bytes/node of RAM and two stores per relaxation in the hottest loop.
    /// The augment phase never trusted `source` anyway: it walks the pred
    /// chain to find the true seed, see `primal_dual::run_impl`.)
    ///
    /// `i64`, not `i32`: residual arc ids run to `2 * num_forward`, roughly
    /// 16 arcs per pixel, which overflows `i32` past ~268 Mpixel - and a
    /// NISAR frame at single-look posting is 3.6 Gpixel. A silent wrap here
    /// corrupts the pred chain (and the unwrap) with no error raised.
    pub pred_arc: Vec<i64>,
    /// True iff the node has been popped (i.e. `dist[node]` is finalized).
    /// With early-exit Dijkstra a node may have a finite `dist` after
    /// relaxation but not be finalized; callers must consult `popped`
    /// (or the `was_reached` helper) before trusting `dist[v]`.
    pub popped: Vec<bool>,
}

impl ShortestPaths {
    pub fn new(n_nodes: usize) -> Self {
        Self {
            dist: vec![i64::MAX; n_nodes],
            pred_arc: vec![-1; n_nodes],
            popped: vec![false; n_nodes],
        }
    }

    /// Clear the buffers for another Dijkstra run, preserving allocations when
    /// the graph size is unchanged.
    pub fn reset(&mut self, n_nodes: usize) {
        if self.dist.len() != n_nodes {
            *self = Self::new(n_nodes);
            return;
        }
        self.dist.fill(i64::MAX);
        self.pred_arc.fill(-1);
        self.popped.fill(false);
    }

    /// True iff this node was finalized by the Dijkstra (i.e. popped).
    /// Distinct from "merely relaxed" - with early-exit on, some relaxed
    /// nodes never get popped.
    pub fn was_reached(&self, node: usize) -> bool {
        self.popped[node]
    }
}

/// Run multi-source Dijkstra to FULL COMPLETION - every reachable node is
/// popped and gets an exact finalized distance. Matches Python ww-orig's
/// `dijkstra_pd` which runs `while (!dijkstra.done())`.
///
/// Use in primal-dual iterations when potential accuracy matters: exact `d[v]`
/// for all nodes makes the potential update `π[v] -= d[v]` produce tight
/// reduced costs, matching Python's MCF routing.
pub fn dijkstra_multi_source_full<G: ResidualGraph>(g: &G, net: &Network) -> ShortestPaths {
    let mut sp = ShortestPaths::new(net.num_nodes());
    dijkstra_multi_source_full_into(g, net, &mut sp);
    sp
}

/// Reusable-buffer variant of [`dijkstra_multi_source_full`].
pub fn dijkstra_multi_source_full_into<G: ResidualGraph>(
    g: &G,
    net: &Network,
    sp: &mut ShortestPaths,
) {
    let mut buckets = Vec::new();
    dijkstra_multi_source_full_scratch_into(g, net, sp, &mut buckets);
}

/// [`dijkstra_multi_source_full_into`] with caller-owned Dial bucket scratch,
/// so a primal-dual loop can reuse the ~n_nodes-sized bucket vectors across
/// its iterations instead of re-growing them every call.
pub fn dijkstra_multi_source_full_scratch_into<G: ResidualGraph>(
    g: &G,
    net: &Network,
    sp: &mut ShortestPaths,
    buckets: &mut Vec<std::collections::VecDeque<u32>>,
) {
    // Full completion only implemented for Dial serial; fall back to early-exit
    // heap for convex mode (marginal costs can be large → Dial bucket count
    // would explode).
    if net.convex_mode {
        heap::run_into(g, net, sp);
    } else {
        dial::run_full_scratch_into(g, net, sp, buckets);
    }
}

/// Run multi-source Dijkstra over the residual graph using reduced costs.
/// Sources are nodes with positive excess; distance 0 at each.
///
/// Backend is selected once per process via `backend()` (env-var
/// `WHIRLWIND_DIJKSTRA`).
pub fn dijkstra_multi_source<G: ResidualGraph>(g: &G, net: &Network) -> ShortestPaths {
    let mut sp = ShortestPaths::new(net.num_nodes());
    dijkstra_multi_source_into(g, net, &mut sp);
    sp
}

/// Reusable-buffer variant of [`dijkstra_multi_source`].
pub fn dijkstra_multi_source_into<G: ResidualGraph>(g: &G, net: &Network, sp: &mut ShortestPaths) {
    // Convex mode produces marginal costs up to ~weight · 200 · nshortcycle²
    // (1e6+ for typical high-coherence arcs); Dial's bucket vec would need
    // that many entries. Route convex networks to the binary-heap backend,
    // which scales O(E log V) without the bucket-count blow-up. The env-var
    // backend selector still controls the linear / reuse paths.
    if net.convex_mode {
        heap::run_into(g, net, sp);
        return;
    }
    match backend() {
        DijkstraBackend::Heap => heap::run_into(g, net, sp),
        DijkstraBackend::DialSerial => dial::run_into(g, net, sp),
        DijkstraBackend::DialParallel => {
            // The experimental parallel Dial backend still owns its buffers;
            // keep the default serial path allocation-free.
            *sp = dial::run_parallel(g, net);
        }
    }
}
