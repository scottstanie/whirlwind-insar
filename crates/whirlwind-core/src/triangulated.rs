//! Sparse residual graph over a Delaunay triangulation of valid pixels.
//!
//! Used for irregular-grid unwrapping (à la `spurt`): when fewer than ~10% of
//! pixels in a raster are coherent enough to unwrap, picking them out and
//! solving MCF only on those pixels is more accurate and far cheaper than
//! solving the full dense problem and discarding low-quality pixels later.
//!
//! ## Topology
//!
//! Input: N "good" pixels with 2D coordinates. Delaunator gives back T
//! triangles + a convex hull. The MCF dual graph has:
//!
//!   * `T + 1` nodes: one per triangle, plus one for the *outer face* (the
//!     unbounded region outside the convex hull). Node IDs `0..T` are
//!     triangles; node `T` is the outer face.
//!   * One *triangulation edge* per pair of adjacent half-edges, plus one
//!     per hull edge (between a hull triangle and the outer face). Each
//!     triangulation edge becomes two forward arcs (canonical + reversed
//!     direction) + two residual-reverse partners, mirroring the rect-grid
//!     layout. That gives the future Carballo-style direction-dependent
//!     costs a place to live, even though the CRLB cost we ship today is
//!     direction-symmetric.
//!
//! Arc ID layout (with `E = num_edges` triangulation edges):
//! ```text
//!   [0,    E)         forward canonical : node tail[e] -> node head[e]
//!   [E,   2E)         forward reversed  : node head[e] -> node tail[e]
//!   [2E,  3E)         residual reverse of canonical  (transpose of [0, E))
//!   [3E,  4E)         residual reverse of reversed   (transpose of [E, 2E))
//! ```
//! so `num_forward = 2E`, `num_arcs = 4E`, and `transpose(a) = a XOR (^2E)`
//! follows the trait default.
//!
//! For every dual-graph arc we also expose the *primal* pixel-pair endpoints
//! (`edge_pixel_pair`) - the two input pixel indices that this triangulation
//! edge connects. That's what downstream code (e.g. a spurt-style B_perp
//! integer-ambiguity fit) needs to plug per-edge models in via
//! `Network::warm_start`.

use crate::residual_graph::ResidualGraph;
use delaunator::{EMPTY, Point, triangulate};

/// Sparse dual graph of a Delaunay triangulation over a set of input pixels.
pub struct TriangulatedGraph {
    /// Number of triangles in the triangulation.
    pub num_triangles: usize,

    /// Per-edge tail node ID in the canonical direction (a triangle, or
    /// `num_triangles` for the outer face). Length = `num_edges`.
    tail: Vec<u32>,

    /// Per-edge head node ID in the canonical direction. Length = `num_edges`.
    head: Vec<u32>,

    /// For each edge, the two original pixel indices it spans, in the order
    /// `(pixel_origin, pixel_destination)` of the canonical-direction
    /// half-edge. Length = `num_edges`.
    edge_pixels: Vec<(u32, u32)>,

    /// CSR adjacency: for each node, list of (edge_idx, is_canonical_tail)
    /// for every triangulation edge it touches. `adj_offsets[node..node+1]`
    /// slices into `adj_data`.
    adj_offsets: Vec<u32>,
    adj_data: Vec<(u32, bool)>,

    /// CCW triangle vertex triples (3 entries per triangle). Indices refer
    /// to the *input* pixel array. Length = `3 * num_triangles`.
    triangles: Vec<u32>,

    /// True iff this edge was carved out as a boundary edge because it
    /// exceeded the user-supplied `max_edge_length`. Long edges always go
    /// to the outer face; integration BFS skips them. Length = `num_edges`.
    is_long: Vec<bool>,
}

impl TriangulatedGraph {
    /// Build a dual graph from input pixel positions. Coordinates are
    /// arbitrary 2D points - typically `(row as f64, col as f64)`. Returns
    /// `None` if the triangulation is degenerate (fewer than 3 unique points
    /// or all collinear).
    pub fn new(points: &[Point]) -> Option<Self> {
        Self::with_max_edge_length(points, f64::INFINITY)
    }

    /// Build a dual graph, treating any triangulation edge longer than
    /// `max_edge_length` as a boundary edge (i.e. one that connects only to
    /// the outer face). This is the standard sparse-unwrap workaround for
    /// the fact that MCF here has unit per-arc capacity: a single arc can
    /// carry at most one cycle correction, so an edge whose true phase
    /// gradient exceeds 2π (multi-wrap edge) cannot be balanced by flow.
    /// Marking such edges as boundary "carves them out" of the interior -
    /// MCF avoids them and integration won't cross them (pixels separated
    /// only by long edges come back as NaN).
    ///
    /// For typical sparse workflows (10% of pixels of a dense raster,
    /// median-neighbor distance ~3-5 px), set `max_edge_length` to a few
    /// times the median pixel spacing.
    pub fn with_max_edge_length(points: &[Point], max_edge_length: f64) -> Option<Self> {
        if points.len() < 3 {
            return None;
        }
        let tri = triangulate(points);
        if tri.is_empty() {
            return None;
        }

        let num_triangles = tri.len();
        let outer = num_triangles as u32;
        let num_nodes = num_triangles + 1;
        let max_len_sq = max_edge_length * max_edge_length;

        // Walk half-edges. For each, decide: (a) twin doesn't exist (pure
        // hull edge → boundary), (b) edge is long (carve out → boundary on
        // both sides), or (c) normal interior edge.
        let n_he = tri.triangles.len();
        let mut tail: Vec<u32> = Vec::new();
        let mut head: Vec<u32> = Vec::new();
        let mut edge_pixels: Vec<(u32, u32)> = Vec::new();
        let mut is_long: Vec<bool> = Vec::new();

        let next_he = |he: usize| -> usize { if he % 3 == 2 { he - 2 } else { he + 1 } };
        let pixel_dist_sq = |a: usize, b: usize| -> f64 {
            let pa = &points[a];
            let pb = &points[b];
            let dx = pa.x - pb.x;
            let dy = pa.y - pb.y;
            dx * dx + dy * dy
        };

        for he in 0..n_he {
            let twin = tri.halfedges[he];
            let pa = tri.triangles[he];
            let pb = tri.triangles[next_he(he)];
            let too_long = pixel_dist_sq(pa, pb) > max_len_sq;

            if twin == EMPTY {
                // Pure convex-hull edge - always a boundary edge.
                tail.push((he / 3) as u32);
                head.push(outer);
                edge_pixels.push((pa as u32, pb as u32));
                is_long.push(too_long);
            } else if too_long {
                // Long interior edge: emit TWO boundary edges, one per side.
                // Visit this branch from both half-edges (no `he < twin`
                // guard) - each emits one boundary edge for its own face.
                tail.push((he / 3) as u32);
                head.push(outer);
                edge_pixels.push((pa as u32, pb as u32));
                is_long.push(true);
            } else {
                // Normal interior edge. Emit only on the lower-indexed
                // half-edge so the pair isn't double-counted.
                if twin < he {
                    continue;
                }
                tail.push((he / 3) as u32);
                head.push((twin / 3) as u32);
                edge_pixels.push((pa as u32, pb as u32));
                is_long.push(false);
            }
        }

        // CSR adjacency.
        let num_edges = tail.len();
        let mut degree = vec![0u32; num_nodes];
        for e in 0..num_edges {
            degree[tail[e] as usize] += 1;
            degree[head[e] as usize] += 1;
        }
        let mut adj_offsets = vec![0u32; num_nodes + 1];
        for i in 0..num_nodes {
            adj_offsets[i + 1] = adj_offsets[i] + degree[i];
        }
        let total_adj = adj_offsets[num_nodes] as usize;
        let mut adj_data = vec![(0u32, false); total_adj];
        let mut cursor = adj_offsets.clone();
        for e in 0..num_edges {
            let t = tail[e] as usize;
            let h = head[e] as usize;
            adj_data[cursor[t] as usize] = (e as u32, true);
            cursor[t] += 1;
            adj_data[cursor[h] as usize] = (e as u32, false);
            cursor[h] += 1;
        }

        let triangles: Vec<u32> = tri.triangles.iter().map(|&v| v as u32).collect();

        Some(Self {
            num_triangles,
            tail,
            head,
            edge_pixels,
            adj_offsets,
            adj_data,
            triangles,
            is_long,
        })
    }

    /// Node ID of the outer face (everything outside the convex hull).
    #[inline]
    pub fn outer_face(&self) -> usize {
        self.num_triangles
    }

    /// Number of triangulation edges (= forward arcs in the canonical
    /// direction = half of `num_forward()`).
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.tail.len()
    }

    /// CCW vertex triple of a triangle, as pixel indices into the input
    /// point array.
    #[inline]
    pub fn triangle_vertices(&self, t: usize) -> (u32, u32, u32) {
        let base = 3 * t;
        (
            self.triangles[base],
            self.triangles[base + 1],
            self.triangles[base + 2],
        )
    }

    /// Pixel indices `(origin, destination)` for the canonical direction of
    /// triangulation edge `e`.
    #[inline]
    pub fn edge_pixel_pair(&self, e: usize) -> (u32, u32) {
        self.edge_pixels[e]
    }

    /// True iff edge `e` was carved out as a boundary (long) edge.
    #[inline]
    pub fn edge_is_long(&self, e: usize) -> bool {
        self.is_long[e]
    }

    /// For an arc ID, return the pixel pair `(origin, destination)` in the
    /// *primal* graph, ordered by arc direction. Useful for downstream code
    /// that wants to attach per-edge models (e.g. B_perp ambiguity).
    pub fn arc_pixel_pair(&self, arc: usize) -> (u32, u32) {
        let e_idx = self.edge_index_of_arc(arc);
        let (a, b) = self.edge_pixels[e_idx];
        if self.arc_is_canonical_direction(arc) {
            (a, b)
        } else {
            (b, a)
        }
    }

    /// True if `arc` runs in the canonical direction (`tail[e] -> head[e]`).
    ///
    /// `[0, E)` and `[3E, 4E)` are canonical-direction; `[E, 2E)` and
    /// `[2E, 3E)` are the reversed direction.
    #[inline]
    pub fn arc_is_canonical_direction(&self, arc: usize) -> bool {
        let e = self.num_edges();
        arc < e || arc >= 3 * e
    }

    /// Edge index of a (possibly residual) arc.
    #[inline]
    pub fn edge_index_of_arc(&self, arc: usize) -> usize {
        let e = self.num_edges();
        match arc {
            a if a < e => a,
            a if a < 2 * e => a - e,
            a if a < 3 * e => a - 2 * e,
            a => a - 3 * e,
        }
    }
}

impl ResidualGraph for TriangulatedGraph {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.num_triangles + 1
    }

    #[inline]
    fn num_forward(&self) -> usize {
        2 * self.num_edges()
    }

    fn arc_endpoints(&self, arc: usize) -> (usize, usize) {
        let e = self.num_edges();
        let (eidx, swap) = match arc {
            a if a < e => (a, false),
            a if a < 2 * e => (a - e, true),
            a if a < 3 * e => (a - 2 * e, true),
            a => (a - 3 * e, false),
        };
        let t = self.tail[eidx] as usize;
        let h = self.head[eidx] as usize;
        if swap { (h, t) } else { (t, h) }
    }

    fn outgoing(&self, node: usize, out: &mut Vec<(usize, usize)>) {
        let e = self.num_edges();
        let start = self.adj_offsets[node] as usize;
        let end = self.adj_offsets[node + 1] as usize;
        for &(eidx_u32, is_tail) in &self.adj_data[start..end] {
            let eidx = eidx_u32 as usize;
            if is_tail {
                // node == tail[eidx]. Forward canonical arc out + residual
                // reverse of the reversed-direction forward arc.
                let head_node = self.head[eidx] as usize;
                out.push((eidx, head_node)); // canonical fwd
                out.push((eidx + 3 * e, head_node)); // residual reverse of reversed fwd
            } else {
                // node == head[eidx]. Forward reversed arc out + residual
                // reverse of the canonical-direction forward arc.
                let tail_node = self.tail[eidx] as usize;
                out.push((eidx + e, tail_node)); // reversed fwd
                out.push((eidx + 2 * e, tail_node)); // residual reverse of canonical fwd
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual_graph::ResidualGraph;

    fn square() -> Vec<Point> {
        vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 0.0, y: 1.0 },
        ]
    }

    #[test]
    fn square_triangulation_has_two_triangles_and_five_edges() {
        let g = TriangulatedGraph::new(&square()).unwrap();
        assert_eq!(g.num_triangles, 2);
        // 4 hull edges + 1 interior diagonal = 5 triangulation edges.
        assert_eq!(g.num_edges(), 5);
        // 1 node per triangle + outer face = 3 nodes.
        assert_eq!(g.num_nodes(), 3);
    }

    #[test]
    fn transpose_is_involution() {
        let g = TriangulatedGraph::new(&square()).unwrap();
        for a in 0..g.num_arcs() {
            assert_eq!(g.transpose(g.transpose(a)), a);
        }
    }

    #[test]
    fn transpose_swaps_endpoints() {
        let g = TriangulatedGraph::new(&square()).unwrap();
        for a in 0..g.num_arcs() {
            let (t, h) = g.arc_endpoints(a);
            let (t2, h2) = g.arc_endpoints(g.transpose(a));
            assert_eq!((t, h), (h2, t2), "arc {a}");
        }
    }

    #[test]
    fn outgoing_arcs_have_correct_tail() {
        let g = TriangulatedGraph::new(&square()).unwrap();
        let mut buf = Vec::new();
        for node in 0..g.num_nodes() {
            buf.clear();
            g.outgoing(node, &mut buf);
            for &(arc, head) in &buf {
                let (t, h) = g.arc_endpoints(arc);
                assert_eq!(t, node, "arc {arc} from node {node}: tail mismatch {t}");
                assert_eq!(h, head, "arc {arc} from node {node}: head mismatch");
            }
        }
    }

    #[test]
    fn arc_pixel_pair_matches_direction() {
        let pts = square();
        let g = TriangulatedGraph::new(&pts).unwrap();
        for e in 0..g.num_edges() {
            let (a, b) = g.edge_pixel_pair(e);
            // Canonical forward arc (id = e): same direction.
            assert_eq!(g.arc_pixel_pair(e), (a, b));
            // Forward reverse arc (id = e + E): swapped.
            assert_eq!(g.arc_pixel_pair(e + g.num_edges()), (b, a));
            // Residual reverse of canonical (id = e + 2E): swapped.
            assert_eq!(g.arc_pixel_pair(e + 2 * g.num_edges()), (b, a));
            // Residual reverse of reversed (id = e + 3E): canonical.
            assert_eq!(g.arc_pixel_pair(e + 3 * g.num_edges()), (a, b));
        }
    }
}
