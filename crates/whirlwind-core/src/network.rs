//! Network state for min-cost flow on the residue grid.

use crate::grid::RectangularGridGraph;
use bitvec::prelude::*;
use ndarray::ArrayView2;

pub struct Network {
    pub excess: Vec<i32>,
    pub potential: Vec<i64>,
    pub cost_fwd: Vec<i32>, // forward-arc costs only (length = num_forward)
    pub is_saturated: BitVec, // length = num_arcs; encoding below
}

// Encoding of `is_saturated[fwd]`, `is_saturated[rev = fwd + num_forward]`:
//
//   (fwd=false, rev=true)  : initial state, capacity 1 forward, 0 reverse
//   (fwd=true,  rev=false) : 1 unit of flow on forward arc
//   (fwd=true,  rev=true)  : arc is **forbidden** (mask said either endpoint
//                            pixel is invalid) — both directions saturated,
//                            never carry flow
//   (fwd=false, rev=false) : unreachable
//
// Dijkstra skips any arc with `is_saturated[arc] == true`, so forbidden arcs
// are naturally invisible. `arc_flow` distinguishes "carries flow" from
// "forbidden" by requiring `fwd && !rev`.

impl Network {
    /// Build a network from a residue grid and per-arc costs.
    ///
    /// `costs` has length `2 * num_forward`; only the forward half is stored —
    /// the reverse-arc cost is `-cost_fwd[transpose - num_forward]`.
    pub fn new(g: &RectangularGridGraph, residues: ArrayView2<i32>, costs: &[i32]) -> Self {
        Self::new_with_mask(g, residues, costs, None)
    }

    /// Build a network and pre-saturate (forbid) arcs that cross a pixel
    /// edge with at least one masked-out endpoint.
    ///
    /// `mask` has shape `(g.m - 1, g.n - 1)` (the pixel grid; `True` = valid,
    /// `False` = ignore). On Sentinel-1-scale frames with large invalid
    /// regions (e.g. ocean) this lets Dijkstra skip a substantial chunk of
    /// the residual graph.
    pub fn new_with_mask(
        g: &RectangularGridGraph,
        residues: ArrayView2<i32>,
        costs: &[i32],
        mask: Option<ArrayView2<bool>>,
    ) -> Self {
        assert_eq!(residues.dim(), (g.m, g.n));
        assert_eq!(costs.len(), g.num_arcs());

        // The residue ndarray is row-major and contiguous when freshly built;
        // when it is we can flatten via `as_slice()` for a single memcpy.
        let excess: Vec<i32> = if let Some(slice) = residues.as_slice() {
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
        let cost_fwd = costs[..g.num_forward].to_vec();

        // Saturation pattern: forward arcs start unsaturated (capacity 1, flow 0);
        // reverse arcs start saturated (capacity 0). `bitvec![1; n]` initializes
        // the whole thing in one pass and we then clear the forward half via
        // `fill` which is faster than the per-bit `set` loop.
        let mut sat = bitvec![1; g.num_arcs()];
        sat[..g.num_forward].fill(false);

        let mut net = Self {
            excess,
            potential: vec![0_i64; g.num_nodes()],
            cost_fwd,
            is_saturated: sat,
        };
        if let Some(mm) = mask {
            net.forbid_masked_arcs(g, mm);
        }
        net
    }

    /// For each pixel-edge whose endpoints aren't both valid, mark the two
    /// forward arcs (one per direction) that cross that edge as forbidden
    /// (`is_saturated[fwd] = true` AND `is_saturated[rev] = true`).
    ///
    /// Geometry (see `crate::grid`):
    /// - `DOWN(ti, tj)` and `UP(ti+1, tj)` both cross pixel-edge
    ///   `{(ti, tj-1), (ti, tj)}` (horizontal pixel-edge in row ti).
    /// - `RIGHT(ti, tj)` and `LEFT(ti, tj+1)` both cross pixel-edge
    ///   `{(ti-1, tj), (ti, tj)}` (vertical pixel-edge in column tj).
    fn forbid_masked_arcs(&mut self, g: &RectangularGridGraph, mask: ArrayView2<bool>) {
        let (m_phase, n_phase) = mask.dim();
        assert_eq!(m_phase + 1, g.m, "mask must be pixel-grid sized (g.m - 1)");
        assert_eq!(n_phase + 1, g.n, "mask must be pixel-grid sized (g.n - 1)");

        // Horizontal pixel-edges (vertical residue-edges = DOWN/UP arcs):
        for ti in 0..m_phase {
            for tj in 1..n_phase {
                if !mask[(ti, tj - 1)] || !mask[(ti, tj)] {
                    let down = g.down_arc(ti, tj).unwrap();
                    let up = g.up_arc(ti + 1, tj).unwrap();
                    self.is_saturated.set(down, true);
                    self.is_saturated.set(up, true);
                    self.is_saturated.set(g.transpose(down), true);
                    self.is_saturated.set(g.transpose(up), true);
                }
            }
        }

        // Vertical pixel-edges (horizontal residue-edges = RIGHT/LEFT arcs):
        for ti in 1..m_phase {
            for tj in 0..n_phase {
                if !mask[(ti - 1, tj)] || !mask[(ti, tj)] {
                    let right = g.right_arc(ti, tj).unwrap();
                    let left = g.left_arc(ti, tj + 1).unwrap();
                    self.is_saturated.set(right, true);
                    self.is_saturated.set(left, true);
                    self.is_saturated.set(g.transpose(right), true);
                    self.is_saturated.set(g.transpose(left), true);
                }
            }
        }
    }

    #[inline]
    pub fn arc_cost(&self, g: &RectangularGridGraph, arc: usize) -> i32 {
        if arc < g.num_forward {
            self.cost_fwd[arc]
        } else {
            -self.cost_fwd[arc - g.num_forward]
        }
    }

    #[inline]
    pub fn is_arc_saturated(&self, arc: usize) -> bool {
        self.is_saturated[arc]
    }

    /// "Arc flow" per the Whirlwind convention used by integration:
    /// 1 iff the forward partner currently carries one unit of flow.
    ///
    /// A forbidden arc (mask said one endpoint is invalid) has BOTH
    /// `is_saturated[fwd]` and `is_saturated[rev]` set, so this returns 0
    /// for it — it never contributes a cycle to the integration. Compare
    /// with a normally-flowing arc which has `fwd=true, rev=false`.
    #[inline]
    pub fn arc_flow(&self, g: &RectangularGridGraph, arc: usize) -> i32 {
        let fwd = if arc < g.num_forward { arc } else { arc - g.num_forward };
        let rev = fwd + g.num_forward;
        if self.is_saturated[fwd] && !self.is_saturated[rev] { 1 } else { 0 }
    }

    /// Reduced cost of an arc: `c - π_tail + π_head`.
    /// (Ahuja convention: ≥ 0 on arcs with residual capacity, by primal-dual invariant.)
    #[inline]
    pub fn reduced_cost(&self, g: &RectangularGridGraph, arc: usize) -> i64 {
        let (t, h) = g.arc_endpoints(arc);
        self.arc_cost(g, arc) as i64 - self.potential[t] + self.potential[h]
    }

    /// Reduced cost when the caller already knows tail and head of `arc`.
    /// Avoids re-deriving them via `g.arc_endpoints` — saves ~5–10 ns per arc
    /// in the Dijkstra inner loop, which scans ~1B arcs on residue-dense scenes.
    #[inline]
    pub fn reduced_cost_with(&self, g: &RectangularGridGraph, arc: usize, t: usize, h: usize) -> i64 {
        debug_assert_eq!(g.arc_endpoints(arc), (t, h));
        self.arc_cost(g, arc) as i64 - self.potential[t] + self.potential[h]
    }

    /// Push 1 unit of flow on `arc`. Toggles saturation of arc and its transpose.
    pub fn push_unit(&mut self, g: &RectangularGridGraph, arc: usize) {
        debug_assert!(!self.is_saturated[arc], "cannot push on saturated arc");
        let t = g.transpose(arc);
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
