//! Abstraction over the *residual graph* that MCF flows live on.
//!
//! Both the dense rectangular grid (4-connected residues over a 2D pixel
//! raster) and the sparse Delaunay-triangulated graph (one residue per triangle
//! face + one outer face) implement this trait. Dijkstra, the primal-dual loop,
//! and `Network` are all generic over `G: ResidualGraph`, so swapping topology
//! is a matter of plugging in a different impl - no fork of the solver.
//!
//! Indexing convention (shared by all impls):
//!
//!   * Nodes are `usize` in `[0, num_nodes())`.
//!   * Forward arcs are `usize` in `[0, num_forward())`.
//!   * Reverse arcs are `usize` in `[num_forward(), 2 * num_forward())`, with
//!     `transpose(fwd) = fwd + num_forward()`.
//!
//! Networks may *append* extra nodes/arcs on top (e.g. a ground node) - those
//! live outside the trait and are managed by `Network` itself.

/// Residual-graph topology. Sole reason for the trait: lets one MCF solver
/// drive both dense raster grids and sparse triangulations.
pub trait ResidualGraph: Sync {
    /// Total node count owned by the graph (excluding any extras the Network
    /// appends, e.g. a ground node).
    fn num_nodes(&self) -> usize;

    /// Total *forward* arc count. Total arc count is `2 * num_forward()`.
    fn num_forward(&self) -> usize;

    #[inline]
    fn num_arcs(&self) -> usize {
        2 * self.num_forward()
    }

    /// Flip a forward arc to its reverse partner and vice versa.
    #[inline]
    fn transpose(&self, arc: usize) -> usize {
        let nf = self.num_forward();
        if arc < nf { arc + nf } else { arc - nf }
    }

    /// Endpoints of `arc` as `(tail, head)`. Works for both forward and reverse.
    fn arc_endpoints(&self, arc: usize) -> (usize, usize);

    /// Append `(arc_id, head_node)` for every outgoing arc from `node` -
    /// forward arcs out of `node` and residual reverses of forward arcs
    /// into `node`. The caller is responsible for clearing `out` first when
    /// reusing the buffer across nodes.
    fn outgoing(&self, node: usize, out: &mut Vec<(usize, usize)>);
}
