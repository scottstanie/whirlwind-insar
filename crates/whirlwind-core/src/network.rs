//! Network state for min-cost flow on the residue grid.

use crate::grid::RectangularGridGraph;
use bitvec::prelude::*;
use ndarray::ArrayView2;

pub struct Network {
    pub excess: Vec<i32>,
    pub potential: Vec<i64>,
    pub cost_fwd: Vec<i32>, // forward-arc costs only (length = num_forward)
    pub is_saturated: BitVec, // length = num_arcs; forward arcs start FALSE, reverse start TRUE
}

impl Network {
    /// Build a network from a residue grid and per-arc costs.
    ///
    /// `costs` has length `2 * num_forward`; only the forward half is stored —
    /// the reverse-arc cost is `-cost_fwd[transpose - num_forward]`.
    pub fn new(g: &RectangularGridGraph, residues: ArrayView2<i32>, costs: &[i32]) -> Self {
        assert_eq!(residues.dim(), (g.m, g.n));
        assert_eq!(costs.len(), g.num_arcs());

        let mut excess = Vec::with_capacity(g.num_nodes());
        for i in 0..g.m {
            for j in 0..g.n {
                excess.push(residues[(i, j)]);
            }
        }
        let cost_fwd = costs[..g.num_forward].to_vec();

        // Saturation pattern: forward arcs start unsaturated (capacity 1, flow 0);
        // reverse arcs start saturated (capacity 0).
        let mut sat = bitvec![0; g.num_arcs()];
        for a in g.num_forward..g.num_arcs() {
            sat.set(a, true);
        }

        Self {
            excess,
            potential: vec![0_i64; g.num_nodes()],
            cost_fwd,
            is_saturated: sat,
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
    ///   forward arc:  1 if saturated, else 0
    ///   reverse arc:  saturation of the corresponding forward arc
    ///                 (= "capacity to undo")
    #[inline]
    pub fn arc_flow(&self, g: &RectangularGridGraph, arc: usize) -> i32 {
        if arc < g.num_forward {
            if self.is_saturated[arc] { 1 } else { 0 }
        } else {
            let fwd = arc - g.num_forward;
            if self.is_saturated[fwd] { 1 } else { 0 }
        }
    }

    /// Reduced cost of an arc: `c - π_tail + π_head`.
    /// (Ahuja convention: ≥ 0 on arcs with residual capacity, by primal-dual invariant.)
    #[inline]
    pub fn reduced_cost(&self, g: &RectangularGridGraph, arc: usize) -> i64 {
        let (t, h) = g.arc_endpoints(arc);
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
