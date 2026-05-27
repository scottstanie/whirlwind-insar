//! Network state for min-cost flow on the residue grid, optionally with a
//! single virtual *ground* node connected to every boundary residue.
//!
//! Layout (forward arc indexing):
//!     [0 .. nf_grid)              — grid forward arcs (unchanged from grid.rs)
//!     [nf_grid .. nf_total)       — ground forward arcs, one per boundary node
//! Reverse arcs sit at the corresponding offsets `+ nf_total`:
//!     [nf_total .. nf_total + nf_grid) — grid reverses
//!     [nf_total + nf_grid .. 2*nf_total) — ground reverses
//!
//! Centralising the forward-count in `Network::num_forward()` means callers
//! must use `net.transpose(arc)` (not `g.transpose(arc)`) and `net.num_forward()`
//! (not `g.num_forward`) whenever they need to flip arc direction. Grid arc
//! IDs themselves are unchanged, so integration's `g.right_arc(...)` /
//! `net.arc_flow(...)` flow happens to be backward-compatible — only the
//! transpose math shifts.

use crate::grid::RectangularGridGraph;
use bitvec::prelude::*;
use ndarray::ArrayView2;

pub struct Network {
    pub excess: Vec<i32>,
    pub potential: Vec<i64>,
    pub cost_fwd: Vec<i32>,   // length = nf_total (grid + ground forward costs)
    pub is_saturated: BitVec, // length = 2 * nf_total

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
//                            pixel is invalid) — both directions saturated,
//                            never carry flow
//   (fwd=false, rev=false) : unreachable

impl Network {
    /// Build a network from a residue grid and per-arc grid costs. No ground.
    pub fn new(g: &RectangularGridGraph, residues: ArrayView2<i32>, costs: &[i32]) -> Self {
        Self::new_with_mask(g, residues, costs, None)
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

    /// Build a network with optional `ground_cost`. When `Some(c)`, a virtual
    /// ground node is appended and connected to every boundary residue (rows
    /// 0/m, cols 0/n of the residue grid) via unit-capacity forward arcs of
    /// cost `c`. Lets MCF drain wrap-line termination charges at the image
    /// boundary without forcing them to pair with distant interior partners.
    ///
    /// `c == 0` makes the ground arc free — MCF then *always* prefers ground
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
        assert_eq!(costs.len(), g.num_arcs());

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
        cost_fwd.extend_from_slice(&costs[..nf_grid]);
        if let Some(c) = ground_cost {
            cost_fwd.extend(std::iter::repeat(c).take(num_ground));
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

    /// Like [`new_with_mask_and_ground`], but the ground node is connected to
    /// **every interior residue adjacent to a masked-out pixel** in addition
    /// to the image-boundary residues.
    ///
    /// Motivation: when the data region is interior to the image (e.g. a
    /// rotated NISAR GSLC inside a rectangular raster), image-boundary
    /// residues are all in masked-out territory and carry zero charge, so
    /// the standard image-only ground does nothing. Real wrap-lines that
    /// should terminate at the **data** boundary (where the unmasked region
    /// meets nodata) get forced to pair with distant interior partners,
    /// producing contiguous block-2π/4π offsets in the integrated unwrap.
    ///
    /// Connecting ground to the data boundary lets those wrap-line endings
    /// drain locally — the equivalent of SNAPHU's "ground = mask boundary"
    /// behavior.
    pub fn new_with_mask_and_full_ground(
        g: &RectangularGridGraph,
        residues: ArrayView2<i32>,
        costs: &[i32],
        mask: ArrayView2<bool>,
        ground_cost: i32,
    ) -> Self {
        assert_eq!(residues.dim(), (g.m, g.n));
        assert_eq!(costs.len(), g.num_arcs());
        let (m_phase, n_phase) = mask.dim();
        assert_eq!(m_phase + 1, g.m);
        assert_eq!(n_phase + 1, g.n);

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

        let num_grid_nodes = g.num_nodes();
        let mut boundary_nodes: Vec<u32> = Vec::new();
        let mut node_to_ground_idx = vec![-1_i32; num_grid_nodes + 1];

        // 1. Image-boundary residues (rows 0/m-1, cols 0/n-1 of residue grid).
        for j in 0..g.n {
            let id = g.node_id(0, j) as u32;
            if node_to_ground_idx[id as usize] < 0 {
                node_to_ground_idx[id as usize] = boundary_nodes.len() as i32;
                boundary_nodes.push(id);
            }
        }
        for j in 0..g.n {
            let id = g.node_id(g.m - 1, j) as u32;
            if node_to_ground_idx[id as usize] < 0 {
                node_to_ground_idx[id as usize] = boundary_nodes.len() as i32;
                boundary_nodes.push(id);
            }
        }
        for i in 1..g.m - 1 {
            for &col in &[0, g.n - 1] {
                let id = g.node_id(i, col) as u32;
                if node_to_ground_idx[id as usize] < 0 {
                    node_to_ground_idx[id as usize] = boundary_nodes.len() as i32;
                    boundary_nodes.push(id);
                }
            }
        }
        // 2. Data-boundary residues: any interior residue whose 2×2 pixel
        // block has at least one masked pixel.
        for r in 1..g.m - 1 {
            for c in 1..g.n - 1 {
                let i = r - 1;
                let j = c - 1;
                let any_masked = !mask[(i, j)]
                    || !mask[(i, j + 1)]
                    || !mask[(i + 1, j)]
                    || !mask[(i + 1, j + 1)];
                if !any_masked {
                    continue;
                }
                let id = g.node_id(r, c) as u32;
                if node_to_ground_idx[id as usize] < 0 {
                    node_to_ground_idx[id as usize] = boundary_nodes.len() as i32;
                    boundary_nodes.push(id);
                }
            }
        }
        let num_boundary = boundary_nodes.len();
        let num_ground = 2 * num_boundary;

        let nf_grid = g.num_forward;
        let nf_total = nf_grid + num_ground;

        let mut cost_fwd = Vec::with_capacity(nf_total);
        cost_fwd.extend_from_slice(&costs[..nf_grid]);
        cost_fwd.extend(std::iter::repeat(ground_cost).take(num_ground));

        let mut sat = bitvec![1; 2 * nf_total];
        sat[..nf_total].fill(false);

        let mut excess = excess_grid;
        let n_nodes_total = num_grid_nodes + 1;
        excess.resize(n_nodes_total, 0);

        let mut net = Self {
            excess,
            potential: vec![0_i64; n_nodes_total],
            cost_fwd,
            is_saturated: sat,
            num_grid_forward: nf_grid,
            num_grid_nodes,
            num_ground,
            num_boundary,
            boundary_nodes,
            node_to_ground_idx,
        };
        net.forbid_masked_arcs(g, mask);
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
        if arc < nf {
            arc + nf
        } else {
            arc - nf
        }
    }

    /// For a boundary residue node, returns the index `i` such that the
    /// node-to-ground forward arc has index `num_grid_forward + i` and the
    /// ground-to-node forward arc has index `num_grid_forward + num_boundary + i`.
    /// `None` for non-boundary nodes / ground-disabled networks.
    #[inline]
    pub fn boundary_idx_of(&self, node: usize) -> Option<usize> {
        let i = *self.node_to_ground_idx.get(node)?;
        if i < 0 {
            None
        } else {
            Some(i as usize)
        }
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
    pub fn arc_endpoints(&self, g: &RectangularGridGraph, arc: usize) -> (usize, usize) {
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
    /// grid. Kept for future flow-policy experiments — not called by default.
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
    pub fn arc_cost(&self, _g: &RectangularGridGraph, arc: usize) -> i32 {
        let nf = self.num_forward();
        if arc < nf {
            self.cost_fwd[arc]
        } else {
            -self.cost_fwd[arc - nf]
        }
    }

    #[inline]
    pub fn is_arc_saturated(&self, arc: usize) -> bool {
        self.is_saturated[arc]
    }

    /// 1 iff the forward partner of `arc` currently carries one unit of flow.
    #[inline]
    pub fn arc_flow(&self, _g: &RectangularGridGraph, arc: usize) -> i32 {
        let nf = self.num_forward();
        let fwd = if arc < nf { arc } else { arc - nf };
        let rev = fwd + nf;
        if self.is_saturated[fwd] && !self.is_saturated[rev] { 1 } else { 0 }
    }

    /// Reduced cost of an arc: `c - π_tail + π_head`.
    #[inline]
    pub fn reduced_cost(&self, g: &RectangularGridGraph, arc: usize) -> i64 {
        let (t, h) = self.arc_endpoints(g, arc);
        self.arc_cost(g, arc) as i64 - self.potential[t] + self.potential[h]
    }

    /// Reduced cost when the caller already knows tail and head.
    #[inline]
    pub fn reduced_cost_with(&self, g: &RectangularGridGraph, arc: usize, t: usize, h: usize) -> i64 {
        debug_assert_eq!(self.arc_endpoints(g, arc), (t, h));
        self.arc_cost(g, arc) as i64 - self.potential[t] + self.potential[h]
    }

    /// Push 1 unit of flow on `arc`. Toggles saturation of arc and its transpose.
    pub fn push_unit(&mut self, _g: &RectangularGridGraph, arc: usize) {
        debug_assert!(!self.is_saturated[arc], "cannot push on saturated arc");
        let t = self.transpose(arc);
        self.is_saturated.set(arc, true);
        self.is_saturated.set(t, false);
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
