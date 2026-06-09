//! Network state for min-cost flow on the residue grid, optionally with a
//! single virtual *ground* node connected to every boundary residue.
//!
//! Layout (forward arc indexing):
//!     [0 .. nf_grid)              - grid forward arcs (unchanged from grid.rs)
//!     [nf_grid .. nf_total)       - ground forward arcs, one per boundary node
//! Reverse arcs sit at the corresponding offsets `+ nf_total`:
//!     [nf_total .. nf_total + nf_grid) - grid reverses
//!     [nf_total + nf_grid .. 2*nf_total) - ground reverses
//!
//! Centralising the forward-count in `Network::num_forward()` means callers
//! must use `net.transpose(arc)` (not `g.transpose(arc)`) and `net.num_forward()`
//! (not `g.num_forward`) whenever they need to flip arc direction. Grid arc
//! IDs themselves are unchanged, so integration's `g.right_arc(...)` /
//! `net.arc_flow(...)` flow happens to be backward-compatible - only the
//! transpose math shifts.

use crate::grid::RectangularGridGraph;
use crate::residual_graph::ResidualGraph;
use bitvec::prelude::*;
use ndarray::ArrayView2;

pub struct Network {
    pub excess: Vec<i32>,
    pub potential: Vec<i64>,
    /// Forward arc costs, length `nf_total` (grid + ground). Stored as `u16`
    /// (halves the biggest per-frame allocation; SNAPHU likewise stores costs
    /// as `short`): every cost builder emits values in `[0, 65535]` - the
    /// parity spline cost caps at `100·ln(1e30) ≈ 6,908`, the analytical
    /// Carballo LUT at 300, and the CRLB / sparse builders saturate at
    /// `u16::MAX` (an arc that expensive is already "never cut here").
    /// Construction asserts the range (`pack_cost`). Reverse-arc costs are
    /// the negation, applied at read time in [`Network::arc_cost`].
    pub cost_fwd: Vec<u16>,
    pub is_saturated: BitVec, // length = 2 * nf_total

    /// Signed flow on each forward arc; only meaningful in `reuse_mode` or
    /// `convex_mode`. Length `nf_total`. Positive = net forward flow,
    /// negative = net reverse.
    pub flow_count: Vec<i32>,
    /// PHASS-style flow-reuse mode: arcs never saturate (multi-unit capacity)
    /// and Dial overrides reduced cost to 0 on any arc with `flow_count != 0`.
    /// Prototype path - see `solver_reuse` history and `unwrap_reuse`.
    pub reuse_mode: bool,

    /// SNAPHU-style convex cost mode: cost is parabolic in flow per arc,
    /// `c_e(k) = weights[e] · (k · 100 − offsets[e])²`. Dial uses the
    /// *marginal* cost (cost of pushing one more unit at current flow)
    /// via [`Network::marginal_cost`]. Arcs are effectively multi-unit
    /// capacity (no saturation under convex_mode). Prototype path -
    /// see [`Network::new_convex_with_mask`] and `unwrap_convex`.
    pub convex_mode: bool,
    /// Per-forward-arc preferred-flow offsets (in units of nshortcycle=100).
    /// Filled by `cost::compute_snaphu_smooth_costs`; zero in non-convex mode.
    pub offsets: Vec<i32>,
    /// Per-forward-arc inverse-variance weights (in COST_SCALE units).
    /// Filled by `cost::compute_snaphu_smooth_costs`; zero in non-convex mode.
    pub weights: Vec<i32>,

    // Grid sub-layout (so we can transpose without holding a &Graph)
    num_grid_forward: usize,
    num_grid_nodes: usize,

    // Ground sub-layout (`num_ground == 0` ⇒ no ground node, no ground arcs).
    //
    // To allow bidirectional flow at cost `ground_cost` in *both* directions
    // (so MCF can route any + boundary residue → ground → any − boundary
    // residue with one Dijkstra pass), we instantiate TWO forward arcs per
    // boundary node: one (boundary → ground) and one (ground → boundary).
    // `num_ground` here therefore equals `2 * num_boundary_nodes`. The first
    // `num_boundary_nodes` ground forward arcs are boundary→ground; the
    // remaining `num_boundary_nodes` are ground→boundary.
    num_ground: usize,
    num_boundary: usize,
    /// Boundary residue node IDs. Indexed by `i ∈ [0, num_boundary)`.
    boundary_nodes: Vec<u32>,
    /// Per-node reverse lookup: `node_to_ground_idx[node] = i` if `node`
    /// is the i-th boundary node, else `-1`. Empty when ground is disabled.
    node_to_ground_idx: Vec<i32>,
}

// Encoding of `is_saturated[fwd]`, `is_saturated[rev = fwd + num_forward]`:
//
//   (fwd=false, rev=true)  : initial state, capacity 1 forward, 0 reverse
//   (fwd=true,  rev=false) : 1 unit of flow on forward arc
//   (fwd=true,  rev=true)  : arc is **forbidden** (mask said either endpoint
//                            pixel is invalid) - both directions saturated,
//                            never carry flow
//   (fwd=false, rev=false) : unreachable

/// Range-checked i32 → u16 cost conversion. Fails fast on out-of-range input
/// rather than silently wrapping: cost builders own the clamping policy.
#[inline]
fn pack_cost(c: i32) -> u16 {
    assert!(
        (0..=u16::MAX as i32).contains(&c),
        "arc cost {c} outside the u16 range [0, 65535] - the cost builder must clamp"
    );
    c as u16
}

impl Network {
    /// Build a network from a residue grid and per-arc grid costs. No ground.
    pub fn new(g: &RectangularGridGraph, residues: ArrayView2<i32>, costs: &[i32]) -> Self {
        Self::new_with_mask(g, residues, costs, None)
    }

    /// Build a network in PHASS-style flow-reuse mode (see `unwrap_reuse`).
    /// Same construction as [`new_with_mask`]; only the `reuse_mode` flag
    /// differs.
    pub fn new_reuse_with_mask(
        g: &RectangularGridGraph,
        residues: ArrayView2<i32>,
        costs: &[i32],
        mask: Option<ArrayView2<bool>>,
    ) -> Self {
        let mut net = Self::new_with_mask_and_ground(g, residues, costs, mask, None);
        net.reuse_mode = true;
        // flow_count is only used in reuse/convex mode (see arc_flow/push_unit),
        // so the shared constructor leaves it empty; allocate it here.
        net.flow_count = vec![0_i32; net.num_forward()];
        net
    }

    /// Build a network in SNAPHU-style convex cost mode (see `unwrap_convex`).
    ///
    /// `offsets[a]` and `weights[a]` together define the per-arc parabolic
    /// cost `c_e(k) = weights[a] · (k · 100 − offsets[a])²`, where `k` is
    /// the integer signed flow tracked in `flow_count[a]`. The `cost_fwd`
    /// field is left as a placeholder of zeros - Dial uses
    /// [`Network::marginal_cost`] instead in convex mode.
    ///
    /// Arcs are effectively multi-unit capacity in convex mode (no
    /// saturation bit toggling on push). At flow 0 the marginal cost is
    /// negative whenever `|offset| > 50` (the arc *wants* flow toward its
    /// parabola minimum `k* = round(offset/100)`), which would corrupt the
    /// Dijkstra/heap SSP. The caller MUST call [`Network::preload_convex_min`]
    /// first: it loads each arc to `k*` and adjusts node excess, after which
    /// every residual marginal is ≥0 and zero initial potentials are valid -
    /// no Bellman-Ford pre-pass needed (every subsequent push moves away from
    /// the minimum → non-negative, non-decreasing marginals).
    pub fn new_convex_with_mask(
        g: &RectangularGridGraph,
        residues: ArrayView2<i32>,
        offsets: &[i32],
        weights: &[i32],
        mask: Option<ArrayView2<bool>>,
    ) -> Self {
        assert_eq!(
            offsets.len(),
            g.num_forward,
            "offsets length must match num_forward"
        );
        assert_eq!(
            weights.len(),
            g.num_forward,
            "weights length must match num_forward"
        );
        // Build a zero-cost linear network just to reuse the topology /
        // mask-forbidding plumbing; we override cost lookups in dial.rs.
        let placeholder_costs = vec![0_i32; g.num_forward];
        let mut net = Self::new_with_mask_and_ground(g, residues, &placeholder_costs, mask, None);
        net.convex_mode = true;
        // flow_count is only used in reuse/convex mode; the shared constructor
        // leaves it empty, so allocate it here for the convex solve.
        net.flow_count = vec![0_i32; net.num_forward()];
        net.offsets = offsets.to_vec();
        net.weights = weights.to_vec();
        // Unit-capacity MCF starts reverse arcs as `is_saturated = true` so
        // they're unreachable until forward flow opens them. Convex_mode
        // wants reverse arcs available from t=0 (the cost is parabolic; the
        // cheaper push direction can be backward). Unsaturate reverse arcs
        // that aren't part of a masked-out edge - forbidden arcs (`fwd=true,
        // rev=true`) stay forbidden; only the (fwd=false, rev=true) entries
        // get unsaturated to (false, false).
        let nf_total = net.num_forward();
        for fwd in 0..nf_total {
            let rev = fwd + nf_total;
            if !net.is_saturated[fwd] {
                net.is_saturated.set(rev, false);
            }
        }
        net
    }

    /// Build a network and pre-saturate (forbid) arcs that cross a pixel
    /// edge with at least one masked-out endpoint. No ground.
    pub fn new_with_mask(
        g: &RectangularGridGraph,
        residues: ArrayView2<i32>,
        costs: &[i32],
        mask: Option<ArrayView2<bool>>,
    ) -> Self {
        Self::new_with_mask_and_ground(g, residues, costs, mask, None)
    }

    /// Topology-agnostic constructor. Use this when the residue graph is not
    /// a dense rectangular raster (e.g. the sparse triangulated graph used by
    /// `unwrap_sparse`).
    ///
    /// * `excess` - per-node integer winding count. Must satisfy
    ///   `excess.len() == num_nodes` and `excess.iter().sum() == 0`.
    /// * `costs` - per-forward-arc integer cost. Must satisfy
    ///   `costs.len() == num_forward`.
    /// * `forbidden_fwd` - optional bitset of forward arcs to pre-saturate
    ///   (both directions). Length `num_forward`. Useful when the caller
    ///   wants to disallow flow on specific edges before the solve.
    ///
    /// For warm-starting from a precomputed arc-level integer estimate
    /// (e.g. a spurt-style B_perp model), call [`Network::warm_start`] on
    /// the returned network before invoking the solver.
    pub fn from_topology(
        num_nodes: usize,
        num_forward: usize,
        excess: Vec<i32>,
        costs: Vec<i32>,
        forbidden_fwd: Option<&BitSlice>,
    ) -> Self {
        assert_eq!(
            excess.len(),
            num_nodes,
            "excess length must match num_nodes"
        );
        assert_eq!(
            costs.len(),
            num_forward,
            "costs length must match num_forward"
        );

        let nf = num_forward;
        let mut sat = bitvec![1; 2 * nf];
        sat[..nf].fill(false);

        if let Some(forbidden) = forbidden_fwd {
            assert_eq!(
                forbidden.len(),
                nf,
                "forbidden_fwd length must match num_forward"
            );
            for a in 0..nf {
                if forbidden[a] {
                    sat.set(a, true);
                    sat.set(a + nf, true);
                }
            }
        }

        Self {
            excess,
            potential: vec![0_i64; num_nodes],
            cost_fwd: costs.iter().map(|&c| pack_cost(c)).collect(),
            is_saturated: sat,
            // Empty in linear MCF mode (never read; see arc_flow/push_unit). The
            // reuse/convex constructors allocate it after flipping their flag.
            flow_count: Vec::new(),
            reuse_mode: false,
            convex_mode: false,
            offsets: Vec::new(),
            weights: Vec::new(),
            num_grid_forward: nf,
            num_grid_nodes: num_nodes,
            num_ground: 0,
            num_boundary: 0,
            boundary_nodes: Vec::new(),
            node_to_ground_idx: Vec::new(),
        }
    }

    /// **Experimental / not yet safe to call.** Toggles per-arc saturation
    /// from an integer flow vector, intended as a hook for spurt-style
    /// B⊥ ambiguity warm-starts. Entries must be in `{-1, 0, +1}`.
    ///
    /// # Known issues - do not use in production
    ///
    /// 1. **Excess is NOT adjusted.** Despite the "warm-start" framing, this
    ///    function only toggles `is_saturated`. To preserve the MCF
    ///    invariant `sum(excess) == 0`, the caller must already have pre-
    ///    balanced `excess` by `div(flow)` when building the network via
    ///    [`Network::from_topology`].
    /// 2. **Breaks Dial's reduced-cost invariant.** Even with pre-balanced
    ///    excess, the residual reverse of a warm-started forward arc has
    ///    cost `−fwd_cost`; under default zero potentials the reduced cost
    ///    is `−fwd_cost − 0 + 0 < 0`, which trips the `debug_assert!` in
    ///    `primal_dual::run::relax` on the first iteration. A correct
    ///    warm-start workflow needs an SPFA / Bellman-Ford pre-pass to
    ///    recompute valid potentials, plus Klein cycle-cancellation when
    ///    the warm-start flow contains negative residual cycles.
    ///
    /// Until that machinery lands, prefer the [`Network::from_topology`]
    /// `excess` parameter for "apply div(warm-start) without saturating
    /// arcs" - which is what `unwrap_sparse` currently does.
    #[doc(hidden)]
    pub fn warm_start<G: ResidualGraph>(&mut self, g: &G, flow: &[i32]) {
        assert_eq!(flow.len(), self.num_grid_forward);
        for (a, &f) in flow.iter().enumerate() {
            match f {
                0 => {}
                1 => self.push_unit(g, a),
                -1 => self.push_unit(g, self.transpose(a)),
                _ => panic!("warm_start entries must be in {{-1, 0, +1}}; got {f}"),
            }
        }
    }

    /// Build a network with optional `ground_cost`. When `Some(c)`, a virtual
    /// ground node is appended and connected to every boundary residue (rows
    /// 0/m, cols 0/n of the residue grid) via unit-capacity forward arcs of
    /// cost `c`. Lets MCF drain wrap-line termination charges at the image
    /// boundary without forcing them to pair with distant interior partners.
    ///
    /// `c == 0` makes the ground arc free - MCF then *always* prefers ground
    /// for boundary residues, which is desirable for clean wrapping inputs
    /// (no interior residues ⇒ Itoh integration alone recovers the unwrap)
    /// but for noisy data can pull interior residues toward boundary along
    /// non-physical paths. A moderate positive cost (e.g. comparable to the
    /// median grid arc cost) balances the two regimes.
    pub fn new_with_mask_and_ground(
        g: &RectangularGridGraph,
        residues: ArrayView2<i32>,
        costs: &[i32],
        mask: Option<ArrayView2<bool>>,
        ground_cost: Option<i32>,
    ) -> Self {
        assert_eq!(residues.dim(), (g.m, g.n));
        assert_eq!(costs.len(), g.num_forward);

        // Excess: copy residues; possibly extend by 1 for ground node.
        let excess_grid: Vec<i32> = if let Some(slice) = residues.as_slice() {
            slice.to_vec()
        } else {
            let mut v = Vec::with_capacity(g.num_nodes());
            for i in 0..g.m {
                for j in 0..g.n {
                    v.push(residues[(i, j)]);
                }
            }
            v
        };

        // Pre-collect boundary node IDs if ground is enabled. Each is unique;
        // the four image corners get exactly one entry.
        let mut boundary_nodes: Vec<u32> = Vec::new();
        let mut node_to_ground_idx: Vec<i32> = Vec::new();
        let mut num_ground = 0;
        let num_grid_nodes = g.num_nodes();
        if ground_cost.is_some() {
            node_to_ground_idx = vec![-1; num_grid_nodes + 1];
            for j in 0..g.n {
                let id = g.node_id(0, j) as u32;
                node_to_ground_idx[id as usize] = boundary_nodes.len() as i32;
                boundary_nodes.push(id);
            }
            for j in 0..g.n {
                let id = g.node_id(g.m - 1, j) as u32;
                if node_to_ground_idx[id as usize] < 0 {
                    node_to_ground_idx[id as usize] = boundary_nodes.len() as i32;
                    boundary_nodes.push(id);
                }
            }
            for i in 1..g.m - 1 {
                let id = g.node_id(i, 0) as u32;
                node_to_ground_idx[id as usize] = boundary_nodes.len() as i32;
                boundary_nodes.push(id);
                let id = g.node_id(i, g.n - 1) as u32;
                node_to_ground_idx[id as usize] = boundary_nodes.len() as i32;
                boundary_nodes.push(id);
            }
            // Two forward arcs per boundary node: boundary→ground and
            // ground→boundary. See struct doc-comment.
            num_ground = 2 * boundary_nodes.len();
        }
        let num_boundary = boundary_nodes.len();

        let nf_grid = g.num_forward;
        let nf_total = nf_grid + num_ground;

        // Costs: grid forwards then ground forwards.
        let mut cost_fwd = Vec::with_capacity(nf_total);
        cost_fwd.extend(costs.iter().map(|&c| pack_cost(c)));
        if let Some(c) = ground_cost {
            cost_fwd.extend(std::iter::repeat_n(pack_cost(c), num_ground));
        }

        // Saturation: 2*nf_total bits. Initial: forward unsaturated, reverse
        // saturated (capacity 1 forward only). Use bitvec! to set all then
        // clear the forward half.
        let mut sat = bitvec![1; 2 * nf_total];
        sat[..nf_total].fill(false);

        // Excess: extend by 1 entry for the ground node if enabled.
        let mut excess = excess_grid;
        let n_nodes_total = num_grid_nodes + (num_ground > 0) as usize;
        if num_ground > 0 {
            excess.resize(n_nodes_total, 0);
        }

        let mut net = Self {
            excess,
            potential: vec![0_i64; n_nodes_total],
            cost_fwd,
            is_saturated: sat,
            // Empty in linear MCF mode (never read; see arc_flow/push_unit). The
            // reuse/convex constructors allocate it after flipping their flag.
            flow_count: Vec::new(),
            reuse_mode: false,
            convex_mode: false,
            offsets: Vec::new(),
            weights: Vec::new(),
            num_grid_forward: nf_grid,
            num_grid_nodes,
            num_ground,
            num_boundary,
            boundary_nodes,
            node_to_ground_idx,
        };
        if let Some(mm) = mask {
            net.forbid_masked_arcs(g, mm);
        }
        net
    }

    // -- ground introspection ------------------------------------------------

    /// Total forward arc count (grid + ground).
    #[inline]
    pub fn num_forward(&self) -> usize {
        self.num_grid_forward + self.num_ground
    }

    /// Total arc count (2 * `num_forward()`).
    #[inline]
    pub fn num_arcs(&self) -> usize {
        2 * self.num_forward()
    }

    /// Total node count, including the ground node when enabled.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_grid_nodes + (self.num_ground > 0) as usize
    }

    /// ID of the ground node, if enabled. `None` for ground-disabled networks.
    #[inline]
    pub fn ground_node(&self) -> Option<usize> {
        if self.num_ground > 0 {
            Some(self.num_grid_nodes)
        } else {
            None
        }
    }

    /// Flip a forward arc to its reverse and vice versa. Replacement for
    /// `g.transpose(arc)` that accounts for the appended ground arcs.
    #[inline]
    pub fn transpose(&self, arc: usize) -> usize {
        let nf = self.num_forward();
        debug_assert!(arc < 2 * nf, "arc {arc} out of bounds (nf={nf})");
        if arc < nf { arc + nf } else { arc - nf }
    }

    /// For a boundary residue node, returns the index `i` such that the
    /// node-to-ground forward arc has index `num_grid_forward + i` and the
    /// ground-to-node forward arc has index `num_grid_forward + num_boundary + i`.
    /// `None` for non-boundary nodes / ground-disabled networks.
    #[inline]
    pub fn boundary_idx_of(&self, node: usize) -> Option<usize> {
        let i = *self.node_to_ground_idx.get(node)?;
        if i < 0 { None } else { Some(i as usize) }
    }

    /// Forward arc IDs for the two ground-arcs at boundary index `i`:
    /// `(node→ground, ground→node)`. Each is unit-capacity, cost
    /// `ground_cost`, so MCF can route both `node → ground` and
    /// `ground → node` independently in the same solve.
    #[inline]
    fn ground_arc_ids(&self, i: usize) -> (usize, usize) {
        let base = self.num_grid_forward;
        (base + i, base + self.num_boundary + i)
    }

    /// Return `(arc, head)` pairs for arcs leaving `node` that are NOT in the
    /// regular grid (i.e. the ground arcs).
    pub fn extra_outgoing(&self, node: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if self.num_ground == 0 {
            return out;
        }
        if Some(node) == self.ground_node() {
            // From ground: forward arcs ground→boundary[i] for every i, plus
            // residual reverses of the forward boundary→ground arcs.
            for i in 0..self.num_boundary {
                let (b2g, g2b) = self.ground_arc_ids(i);
                out.push((g2b, self.boundary_nodes[i] as usize));
                let g2b_residual = self.transpose(b2g); // forward boundary→ground reversed
                out.push((g2b_residual, self.boundary_nodes[i] as usize));
            }
        } else if let Some(i) = self.boundary_idx_of(node) {
            let (b2g, g2b) = self.ground_arc_ids(i);
            let g_node = self.ground_node().unwrap();
            out.push((b2g, g_node));
            // Residual reverse of the forward ground→boundary arc (so we can
            // "undo" a previously-pushed g→boundary flow if needed).
            let g2b_residual = self.transpose(g2b);
            out.push((g2b_residual, g_node));
        }
        out
    }

    /// Endpoints of a (possibly ground) arc, as `(tail, head)`.
    pub fn arc_endpoints<G: ResidualGraph>(&self, g: &G, arc: usize) -> (usize, usize) {
        let nf = self.num_forward();
        let (fwd_arc, is_reverse) = if arc < nf {
            (arc, false)
        } else {
            (arc - nf, true)
        };
        let (t, h) = if fwd_arc < self.num_grid_forward {
            g.arc_endpoints(fwd_arc)
        } else {
            // Ground forward arc. The first `num_boundary` IDs are
            // boundary→ground; the next `num_boundary` are ground→boundary.
            let g_node = self.ground_node().unwrap();
            let off = fwd_arc - self.num_grid_forward;
            if off < self.num_boundary {
                (self.boundary_nodes[off] as usize, g_node)
            } else {
                let i = off - self.num_boundary;
                (g_node, self.boundary_nodes[i] as usize)
            }
        };
        if is_reverse { (h, t) } else { (t, h) }
    }

    // -- forbidding -----------------------------------------------------------

    /// Forbid the arcs that run *along* the boundary frame of the residue
    /// grid. Kept for future flow-policy experiments - not called by default.
    #[allow(dead_code)]
    fn forbid_frame_along_arcs(&mut self, g: &RectangularGridGraph) {
        let m = g.m;
        let n = g.n;
        for j in 0..n - 1 {
            let r = g.right_arc(0, j).unwrap();
            let l = g.left_arc(0, j + 1).unwrap();
            let rt = self.transpose(r);
            let lt = self.transpose(l);
            self.is_saturated.set(r, true);
            self.is_saturated.set(l, true);
            self.is_saturated.set(rt, true);
            self.is_saturated.set(lt, true);
        }
        for j in 0..n - 1 {
            let r = g.right_arc(m - 1, j).unwrap();
            let l = g.left_arc(m - 1, j + 1).unwrap();
            let rt = self.transpose(r);
            let lt = self.transpose(l);
            self.is_saturated.set(r, true);
            self.is_saturated.set(l, true);
            self.is_saturated.set(rt, true);
            self.is_saturated.set(lt, true);
        }
        for i in 0..m - 1 {
            let d = g.down_arc(i, 0).unwrap();
            let u = g.up_arc(i + 1, 0).unwrap();
            let dt = self.transpose(d);
            let ut = self.transpose(u);
            self.is_saturated.set(d, true);
            self.is_saturated.set(u, true);
            self.is_saturated.set(dt, true);
            self.is_saturated.set(ut, true);
        }
        for i in 0..m - 1 {
            let d = g.down_arc(i, n - 1).unwrap();
            let u = g.up_arc(i + 1, n - 1).unwrap();
            let dt = self.transpose(d);
            let ut = self.transpose(u);
            self.is_saturated.set(d, true);
            self.is_saturated.set(u, true);
            self.is_saturated.set(dt, true);
            self.is_saturated.set(ut, true);
        }
    }

    fn forbid_masked_arcs(&mut self, g: &RectangularGridGraph, mask: ArrayView2<bool>) {
        let (m_phase, n_phase) = mask.dim();
        assert_eq!(m_phase + 1, g.m, "mask must be pixel-grid sized (g.m - 1)");
        assert_eq!(n_phase + 1, g.n, "mask must be pixel-grid sized (g.n - 1)");

        for ti in 0..m_phase {
            for tj in 1..n_phase {
                if !mask[(ti, tj - 1)] || !mask[(ti, tj)] {
                    let down = g.down_arc(ti, tj).unwrap();
                    let up = g.up_arc(ti + 1, tj).unwrap();
                    let dt = self.transpose(down);
                    let ut = self.transpose(up);
                    self.is_saturated.set(down, true);
                    self.is_saturated.set(up, true);
                    self.is_saturated.set(dt, true);
                    self.is_saturated.set(ut, true);
                }
            }
        }

        for ti in 1..m_phase {
            for tj in 0..n_phase {
                if !mask[(ti - 1, tj)] || !mask[(ti, tj)] {
                    let right = g.right_arc(ti, tj).unwrap();
                    let left = g.left_arc(ti, tj + 1).unwrap();
                    let rt = self.transpose(right);
                    let lt = self.transpose(left);
                    self.is_saturated.set(right, true);
                    self.is_saturated.set(left, true);
                    self.is_saturated.set(rt, true);
                    self.is_saturated.set(lt, true);
                }
            }
        }
    }

    // -- arc/flow queries -----------------------------------------------------

    #[inline]
    pub fn arc_cost<G: ResidualGraph>(&self, _g: &G, arc: usize) -> i32 {
        let nf = self.num_forward();
        if arc < nf {
            self.cost_fwd[arc] as i32
        } else {
            -(self.cost_fwd[arc - nf] as i32)
        }
    }

    #[inline]
    pub fn is_arc_saturated(&self, arc: usize) -> bool {
        self.is_saturated[arc]
    }

    /// In `reuse_mode`, an arc with `|flow_count| > 0` is "used" - Dial will
    /// override its reduced cost to 0 (PHASS `ASSP.cc:2034` behavior). In
    /// MCF mode, always returns false (no override).
    #[inline]
    pub fn is_used(&self, arc: usize) -> bool {
        if !self.reuse_mode {
            return false;
        }
        let nf = self.num_forward();
        let fwd = if arc < nf { arc } else { arc - nf };
        self.flow_count[fwd] != 0
    }

    /// SNAPHU-style convex marginal cost: cost of pushing one more unit of
    /// flow on `arc` given the current `flow_count` on its forward partner.
    ///
    /// For a forward arc, push moves `flow_count[fwd] : f → f + 1`. The
    /// total parabolic cost change is
    ///
    ///   Δc = w · ((f + 1) · 100 − O)² − w · (f · 100 − O)²
    ///      = w · (200 · (f · 100 − O) + 100²)
    ///
    /// For a reverse arc, push moves `flow_count[fwd] : f → f − 1`, giving
    /// the same formula with `+100` replaced by `−100`:
    ///
    ///   Δc = w · (−200 · (f · 100 − O) + 100²)
    ///      = w · (200 · (O − f · 100) + 100²)
    ///
    /// Returns `i64` because intermediate products can exceed `i32` range:
    /// with `nshortcycle = 100`, `f · 100 − O ∈ [-5000, +5000]` and the
    /// `· 200` factor takes that to `[-1e6, +1e6]` times weight (`w` up to
    /// `~1e4` at high coherence). Caller-side (dial.rs) keeps reduced costs
    /// in `i64`.
    ///
    /// Returns 0 in non-convex modes (caller should branch on `convex_mode`
    /// first; this exists as a safe fallback).
    #[inline]
    pub fn marginal_cost(&self, arc: usize) -> i64 {
        if !self.convex_mode {
            return 0;
        }
        let nf = self.num_forward();
        let (fwd, sign) = if arc < nf {
            (arc, 1_i64)
        } else {
            (arc - nf, -1_i64)
        };
        let f = self.flow_count[fwd] as i64;
        let o = self.offsets[fwd] as i64;
        let w = self.weights[fwd] as i64;
        let ns: i64 = 100; // NSHORTCYCLE
        let u = f * ns - o;
        // Δc = w · (sign · 200 · u + 100²) = w · (sign · 2 · ns · u + ns²)
        w * (sign * 2 * ns * u + ns * ns)
    }

    /// Pre-load each convex arc's flow to its per-arc parabola minimum
    /// `k* = round(offset / nshortcycle)`, then adjust node excess so the
    /// residual problem restores conservation.
    ///
    /// This is what makes the convex solve SOUND. The parabolic cost
    /// `w·(k·100 − offset)²` is minimized at integer `k*`; at `k*` BOTH the
    /// forward (`k*→k*+1`) and reverse (`k*→k*−1`) marginal costs are ≥0
    /// (a move in either direction climbs the parabola). So after pre-loading
    /// every arc to its own `k*`, all residual marginals are ≥0 - zero initial
    /// potentials are valid and the successive-shortest-path solver (Dijkstra /
    /// heap) stays sound for the rest of the run, since every subsequent push
    /// moves an arc *away* from its minimum (non-negative, non-decreasing
    /// marginal: the textbook ordered-parallel-arc reduction of convex-cost MCF,
    /// Ahuja–Magnanti–Orlin §14.5). No Bellman-Ford pre-pass and no negative
    /// cycles are needed - the negativity the old `unwrap_convex` tripped over
    /// (a forward push at `k=0` when `offset>50`, then a negative undo arc) is
    /// eliminated by starting at `k*` instead of `0`.
    ///
    /// A forward unit on arc (t→h) is `decrease_excess(t)+increase_excess(h)`
    /// (see `primal_dual::run` augment), so loading `k*` units gives
    /// `excess[t] -= k*; excess[h] += k*`. Masked/forbidden edges have
    /// `offset = 0 ⇒ k* = 0` and are skipped.
    pub fn preload_convex_min<G: ResidualGraph>(&mut self, g: &G) {
        assert!(self.convex_mode, "preload_convex_min requires convex_mode");
        const NS: i64 = 100; // NSHORTCYCLE, matches marginal_cost
        for fwd in 0..self.num_grid_forward {
            let o = self.offsets[fwd] as i64;
            let kstar = ((o as f64) / (NS as f64)).round() as i32;
            if kstar == 0 {
                continue;
            }
            let (t, h) = self.arc_endpoints(g, fwd);
            self.flow_count[fwd] = kstar;
            self.excess[t] -= kstar;
            self.excess[h] += kstar;
        }
    }

    /// Net signed flow on the forward partner of `arc`. In MCF mode this is
    /// 0 or 1 (forward saturated ⇒ 1, else 0). In reuse_mode / convex_mode
    /// this is `flow_count[fwd]` and can have any magnitude.
    #[inline]
    pub fn arc_flow<G: ResidualGraph>(&self, _g: &G, arc: usize) -> i32 {
        let nf = self.num_forward();
        let fwd = if arc < nf { arc } else { arc - nf };
        if self.reuse_mode || self.convex_mode {
            return self.flow_count[fwd];
        }
        let rev = fwd + nf;
        if self.is_saturated[fwd] && !self.is_saturated[rev] {
            1
        } else {
            0
        }
    }

    /// Reduced cost of an arc: `c - π_tail + π_head`.
    #[inline]
    pub fn reduced_cost<G: ResidualGraph>(&self, g: &G, arc: usize) -> i64 {
        let (t, h) = self.arc_endpoints(g, arc);
        self.arc_cost(g, arc) as i64 - self.potential[t] + self.potential[h]
    }

    /// Reduced cost when the caller already knows tail and head.
    #[inline]
    pub fn reduced_cost_with<G: ResidualGraph>(
        &self,
        g: &G,
        arc: usize,
        t: usize,
        h: usize,
    ) -> i64 {
        debug_assert_eq!(self.arc_endpoints(g, arc), (t, h));
        self.arc_cost(g, arc) as i64 - self.potential[t] + self.potential[h]
    }

    /// Push 1 unit of flow on `arc`.
    ///
    /// MCF mode: toggles saturation of arc and its transpose (capacity 1).
    ///
    /// Reuse mode: updates `flow_count` (signed by direction), unlocks the
    /// transpose if it was reverse-saturated, and *leaves the forward arc
    /// unsaturated* so subsequent demands can pile multiple units on the
    /// same edge. The override in `is_used` then forces Dial's reduced
    /// cost to 0 on this arc for the rest of the solve, mirroring PHASS
    /// `ASSP.cc:2034`.
    pub fn push_unit<G: ResidualGraph>(&mut self, _g: &G, arc: usize) {
        debug_assert!(!self.is_saturated[arc], "cannot push on saturated arc");
        let t = self.transpose(arc);
        if self.reuse_mode || self.convex_mode {
            let nf = self.num_forward();
            if arc < nf {
                self.flow_count[arc] += 1;
            } else {
                self.flow_count[arc - nf] -= 1;
            }
            // Unlock the transpose direction so the residual graph stays
            // traversable in both directions. Reduced cost on the now-flowing
            // arc is handled by either the `is_used` override (reuse_mode)
            // or by the next marginal_cost computation (convex_mode).
            self.is_saturated.set(t, false);
            // Deliberately do NOT set is_saturated[arc] = true. Multi-unit
            // capacity is what makes both reuse and convex modes work.
        } else {
            self.is_saturated.set(arc, true);
            self.is_saturated.set(t, false);
        }
    }

    pub fn excess_nodes(&self) -> impl Iterator<Item = usize> + '_ {
        self.excess
            .iter()
            .enumerate()
            .filter_map(|(i, &e)| if e > 0 { Some(i) } else { None })
    }
    pub fn deficit_nodes(&self) -> impl Iterator<Item = usize> + '_ {
        self.excess
            .iter()
            .enumerate()
            .filter_map(|(i, &e)| if e < 0 { Some(i) } else { None })
    }
    pub fn is_balanced(&self) -> bool {
        self.excess.iter().sum::<i32>() == 0
    }

    pub fn increase_excess(&mut self, node: usize, d: i32) {
        self.excess[node] += d;
    }
    pub fn decrease_excess(&mut self, node: usize, d: i32) {
        self.excess[node] -= d;
    }
}

#[cfg(test)]
mod convex_marginal_tests {
    use super::*;
    use ndarray::Array2;

    fn make_net_convex(offsets: Vec<i32>, weights: Vec<i32>) -> Network {
        // 3x3 residue grid (= 2x2 pixel-edge grid). num_forward = 2*n_v + 2*n_h.
        let g = RectangularGridGraph::new(3, 3);
        let nf = g.num_forward;
        assert_eq!(offsets.len(), nf);
        assert_eq!(weights.len(), nf);
        let residues = Array2::<i32>::zeros((3, 3));
        Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None)
    }

    /// Marginal cost at f=0 of pushing forward (f: 0 → 1) with offset O:
    ///   Δc = w · (200 · (-O) + 10000) = w · (10000 − 200·O)
    /// For O = 0:  Δc = 10000 · w
    /// For O = 50: Δc = 0        (arc indifferent at f=0)
    /// For O = -50: Δc = 20000·w (arc strongly resists +1)
    #[test]
    fn marginal_cost_at_zero_flow() {
        let g = RectangularGridGraph::new(3, 3);
        let nf = g.num_forward;
        // Set the first arc with offset=0, weight=1; second with offset=50, weight=1;
        // third with offset=-50, weight=1.
        let mut offsets = vec![0_i32; nf];
        let weights = vec![1_i32; nf];
        offsets[1] = 50;
        offsets[2] = -50;
        let net = make_net_convex(offsets, weights);

        assert_eq!(net.marginal_cost(0), 10_000, "offset=0 forward push");
        assert_eq!(
            net.marginal_cost(1),
            0,
            "offset=50 forward push (indifferent)"
        );
        assert_eq!(
            net.marginal_cost(2),
            20_000,
            "offset=-50 forward push (resist)"
        );
    }

    /// Pushing flow toward offset should be cheaper than pushing away.
    /// arc 0: offset=+50. At f=0, marginal forward (f → +1) = 0; reverse (f → -1) = 20000.
    /// arc 1: offset=-50. At f=0, marginal forward = 20000; reverse = 0.
    #[test]
    fn marginal_cost_direction_aware() {
        let g = RectangularGridGraph::new(3, 3);
        let nf = g.num_forward;
        let mut offsets = vec![0_i32; nf];
        offsets[0] = 50;
        offsets[1] = -50;
        let weights = vec![1_i32; nf];
        let net = make_net_convex(offsets, weights);
        // Reverse arc id = fwd + nf.
        assert_eq!(net.marginal_cost(0), 0, "fwd push toward +50 offset = 0");
        assert_eq!(
            net.marginal_cost(nf),
            20_000,
            "rev push away from +50 offset = 20000"
        );
        assert_eq!(
            net.marginal_cost(1),
            20_000,
            "fwd push away from -50 offset = 20000"
        );
        assert_eq!(
            net.marginal_cost(1 + nf),
            0,
            "rev push toward -50 offset = 0"
        );
    }

    /// MCF-mode networks have marginal_cost always returning 0 (no overhead).
    #[test]
    fn marginal_cost_zero_in_mcf_mode() {
        let g = RectangularGridGraph::new(3, 3);
        let residues = Array2::<i32>::zeros((3, 3));
        let costs = vec![10_i32; g.num_forward];
        let net = Network::new(&g, residues.view(), &costs);
        assert_eq!(net.marginal_cost(0), 0);
    }

    /// SOUNDNESS of `preload_convex_min`: a large offset (|offset|>50) makes the
    /// forward marginal NEGATIVE at f=0 - which would silently corrupt the
    /// release Dijkstra (its `debug_assert!(rc>=0)` is compiled out). After
    /// pre-loading each arc to k*=round(offset/100), EVERY arc's marginal in
    /// BOTH directions must be ≥0 (zero potentials are then valid), and the
    /// excess adjustment must preserve sum(excess)=0.
    #[test]
    fn preload_makes_all_marginals_nonnegative() {
        let g = RectangularGridGraph::new(3, 3);
        let nf = g.num_forward;
        let mut offsets = vec![0_i32; nf];
        let weights = vec![3_i32; nf];
        // Offsets that exceed the ±50 cap → negative forward marginal at f=0.
        offsets[0] = 90; // k* = 1
        offsets[1] = -90; // k* = -1
        offsets[2] = 55; // k* = 1 (round(0.55))
        let residues = Array2::<i32>::zeros((3, 3));
        let mut net = Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);

        // Pre-condition: arc 0's forward marginal is NEGATIVE at f=0.
        assert!(
            net.marginal_cost(0) < 0,
            "offset=90 should give negative f=0 marginal"
        );

        net.preload_convex_min(&g);

        assert_eq!(net.flow_count[0], 1, "offset 90 → k*=1");
        assert_eq!(net.flow_count[1], -1, "offset -90 → k*=-1");
        assert_eq!(net.flow_count[2], 1, "offset 55 → k*=1");

        // Soundness invariant: all reduced costs at zero potential (= marginals
        // at the pre-loaded flow) are ≥0 in both directions.
        for fwd in 0..nf {
            assert!(
                net.marginal_cost(fwd) >= 0,
                "fwd arc {fwd} marginal < 0 after preload"
            );
            assert!(
                net.marginal_cost(fwd + nf) >= 0,
                "rev arc {fwd} marginal < 0 after preload"
            );
        }
        // Conservation preserved.
        assert_eq!(
            net.excess.iter().sum::<i32>(),
            0,
            "preload must keep sum(excess)=0"
        );
    }
}
