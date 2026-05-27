//! Rectangular grid residual graph.
//!
//! For an `(m, n)` node grid (typically the residue grid, shape `m_phase+1` x
//! `n_phase+1`), the *residual* graph has 4 directional arcs per node where
//! applicable: down, up, right, left. Each base (forward) arc gets a residual
//! reverse partner (used by min-cost flow to undo decisions), giving 8 arc
//! slots per interior pair — but only because each pair has 2 *forward* arcs
//! (the two directions are independent Carballo decisions), not because we
//! invented extra arcs.
//!
//! Arc ID layout (compact, partitioned for O(1) transpose):
//! ```text
//!   [0, n_v)                                 forward DOWN  : (i,j)   -> (i+1,j)
//!   [n_v, 2*n_v)                             forward UP    : (i+1,j) -> (i,j)
//!   [2*n_v, 2*n_v + n_h)                     forward RIGHT : (i,j)   -> (i,j+1)
//!   [2*n_v + n_h, num_forward)               forward LEFT  : (i,j+1) -> (i,j)
//!   [num_forward, 2*num_forward)             reverse partner of forward i is i + num_forward
//! ```
//!
//! where `n_v = (m-1)*n` and `n_h = m*(n-1)`.

use crate::residual_graph::ResidualGraph;

/// Direction of a forward arc. Order matches the arc-ID partition above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    Down = 0,
    Up = 1,
    Right = 2,
    Left = 3,
}

#[derive(Debug, Clone)]
pub struct RectangularGridGraph {
    pub m: usize,
    pub n: usize,
    pub n_v: usize, // (m-1) * n  — vertical pairs
    pub n_h: usize, // m * (n-1)  — horizontal pairs
    pub num_forward: usize,
}

impl RectangularGridGraph {
    pub fn new(m: usize, n: usize) -> Self {
        assert!(m >= 2 && n >= 2, "grid must be at least 2x2");
        let n_v = (m - 1) * n;
        let n_h = m * (n - 1);
        let num_forward = 2 * n_v + 2 * n_h;
        Self {
            m,
            n,
            n_v,
            n_h,
            num_forward,
        }
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.m * self.n
    }

    #[inline]
    pub fn num_arcs(&self) -> usize {
        2 * self.num_forward
    }

    #[inline]
    pub fn node_id(&self, i: usize, j: usize) -> usize {
        i * self.n + j
    }

    #[inline]
    pub fn node_ij(&self, id: usize) -> (usize, usize) {
        (id / self.n, id % self.n)
    }

    #[inline]
    pub fn is_forward(&self, arc: usize) -> bool {
        arc < self.num_forward
    }

    #[inline]
    pub fn transpose(&self, arc: usize) -> usize {
        if arc < self.num_forward {
            arc + self.num_forward
        } else {
            arc - self.num_forward
        }
    }

    /// Forward arc IDs for each direction at node position (i, j).
    /// Returns None for arcs that would leave the grid.
    pub fn down_arc(&self, i: usize, j: usize) -> Option<usize> {
        if i + 1 < self.m {
            Some(i * self.n + j)
        } else {
            None
        }
    }
    pub fn up_arc(&self, i: usize, j: usize) -> Option<usize> {
        if i >= 1 {
            // UP arc tail = (i, j), head = (i-1, j). Indexed by (i-1, j).
            Some(self.n_v + (i - 1) * self.n + j)
        } else {
            None
        }
    }
    pub fn right_arc(&self, i: usize, j: usize) -> Option<usize> {
        if j + 1 < self.n {
            Some(2 * self.n_v + i * (self.n - 1) + j)
        } else {
            None
        }
    }
    pub fn left_arc(&self, i: usize, j: usize) -> Option<usize> {
        if j >= 1 {
            Some(2 * self.n_v + self.n_h + i * (self.n - 1) + (j - 1))
        } else {
            None
        }
    }

    /// Direction and (i,j) of the *tail* of a forward arc.
    pub fn forward_arc_info(&self, arc: usize) -> (Dir, usize, usize) {
        debug_assert!(arc < self.num_forward);
        if arc < self.n_v {
            let i = arc / self.n;
            let j = arc % self.n;
            (Dir::Down, i, j)
        } else if arc < 2 * self.n_v {
            let a = arc - self.n_v;
            let i = a / self.n;
            let j = a % self.n;
            // UP arc indexed by (i, j) goes from tail (i+1, j) to head (i, j).
            (Dir::Up, i + 1, j)
        } else if arc < 2 * self.n_v + self.n_h {
            let a = arc - 2 * self.n_v;
            let i = a / (self.n - 1);
            let j = a % (self.n - 1);
            (Dir::Right, i, j)
        } else {
            let a = arc - 2 * self.n_v - self.n_h;
            let i = a / (self.n - 1);
            let j = a % (self.n - 1);
            // LEFT arc indexed by (i, j) goes from tail (i, j+1) to head (i, j).
            (Dir::Left, i, j + 1)
        }
    }

    /// Get (tail_id, head_id) for any arc (forward or reverse).
    pub fn arc_endpoints(&self, arc: usize) -> (usize, usize) {
        let (fwd_arc, swap) = if arc < self.num_forward {
            (arc, false)
        } else {
            (arc - self.num_forward, true)
        };
        let (dir, ti, tj) = self.forward_arc_info(fwd_arc);
        let (hi, hj) = match dir {
            Dir::Down => (ti + 1, tj),
            Dir::Up => (ti - 1, tj),
            Dir::Right => (ti, tj + 1),
            Dir::Left => (ti, tj - 1),
        };
        let tail = self.node_id(ti, tj);
        let head = self.node_id(hi, hj);
        if swap { (head, tail) } else { (tail, head) }
    }

    /// Yield (arc_id, head_id) pairs for all outgoing arcs from `(i, j)`,
    /// including residual reverse arcs of forward arcs pointing INTO this node.
    pub fn outgoing_ij(&self, i: usize, j: usize) -> SmallVec8 {
        let mut out = SmallVec8::default();
        // Forward arcs out of (i, j):
        if let Some(a) = self.down_arc(i, j) {
            out.push((a, self.node_id(i + 1, j)));
        }
        if let Some(a) = self.up_arc(i, j) {
            out.push((a, self.node_id(i - 1, j)));
        }
        if let Some(a) = self.right_arc(i, j) {
            out.push((a, self.node_id(i, j + 1)));
        }
        if let Some(a) = self.left_arc(i, j) {
            out.push((a, self.node_id(i, j - 1)));
        }
        // Residual reverses of forward arcs pointing INTO (i, j):
        //   forward UP into (i, j) has tail (i+1, j) and head (i, j); its
        //   reverse points (i, j) -> (i+1, j).
        if i + 1 < self.m {
            let fwd = self.up_arc(i + 1, j).unwrap();
            out.push((self.transpose(fwd), self.node_id(i + 1, j)));
        }
        //   forward DOWN into (i, j) has tail (i-1, j); reverse heads back there.
        if i >= 1 {
            let down_into = self.down_arc(i - 1, j).unwrap();
            out.push((self.transpose(down_into), self.node_id(i - 1, j)));
        }
        if j + 1 < self.n {
            let left_into = self.left_arc(i, j + 1).unwrap();
            out.push((self.transpose(left_into), self.node_id(i, j + 1)));
        }
        if j >= 1 {
            let right_into = self.right_arc(i, j - 1).unwrap();
            out.push((self.transpose(right_into), self.node_id(i, j - 1)));
        }
        out
    }
}

impl ResidualGraph for RectangularGridGraph {
    #[inline]
    fn num_nodes(&self) -> usize {
        RectangularGridGraph::num_nodes(self)
    }
    #[inline]
    fn num_forward(&self) -> usize {
        self.num_forward
    }
    #[inline]
    fn arc_endpoints(&self, arc: usize) -> (usize, usize) {
        RectangularGridGraph::arc_endpoints(self, arc)
    }
    fn outgoing(&self, node: usize, out: &mut Vec<(usize, usize)>) {
        let (i, j) = self.node_ij(node);
        let buf = self.outgoing_ij(i, j);
        for &(a, h) in buf.iter() {
            out.push((a, h));
        }
    }
}

/// A tiny inline vector — outdegree of any residue node is ≤ 8 (4 forward + 4 reverse).
#[derive(Default)]
pub struct SmallVec8 {
    pub data: [(usize, usize); 8],
    pub len: usize,
}
impl SmallVec8 {
    pub fn push(&mut self, x: (usize, usize)) {
        self.data[self.len] = x;
        self.len += 1;
    }
    pub fn iter(&self) -> impl Iterator<Item = &(usize, usize)> {
        self.data[..self.len].iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_count_matches_4_per_pair() {
        let g = RectangularGridGraph::new(4, 5);
        assert_eq!(g.num_nodes(), 20);
        // pairs: vertical = 3*5 = 15, horizontal = 4*4 = 16, total = 31
        // 2 forward arcs per pair + 2 reverse = 4 arcs per pair = 124
        assert_eq!(g.num_arcs(), 4 * 31);
        assert_eq!(g.num_forward, 2 * 31);
    }

    #[test]
    fn transpose_is_involution() {
        let g = RectangularGridGraph::new(3, 4);
        for a in 0..g.num_arcs() {
            assert_eq!(g.transpose(g.transpose(a)), a);
        }
    }

    #[test]
    fn transpose_swaps_endpoints() {
        let g = RectangularGridGraph::new(3, 4);
        for a in 0..g.num_arcs() {
            let (t, h) = g.arc_endpoints(a);
            let (t2, h2) = g.arc_endpoints(g.transpose(a));
            assert_eq!((t, h), (h2, t2));
        }
    }

    #[test]
    fn outgoing_makes_sense_for_corner() {
        let g = RectangularGridGraph::new(3, 3);
        // Top-left (0, 0): forward DOWN + RIGHT (2). Reverses of forward UP
        // from (1, 0) and forward LEFT from (0, 1) — both exist (they're the
        // ones pointing INTO (0, 0)). So total = 4.
        let out = g.outgoing_ij(0, 0);
        assert_eq!(out.len, 4);
        // Bottom-right (2, 2): symmetric — also 4.
        let out = g.outgoing_ij(2, 2);
        assert_eq!(out.len, 4);
    }

    #[test]
    fn outgoing_makes_sense_for_interior() {
        let g = RectangularGridGraph::new(5, 5);
        let out = g.outgoing_ij(2, 2);
        // 4 forward (D, U, R, L) + 4 reverse partners-of-forward-into = 8
        assert_eq!(out.len, 8);
    }
}
