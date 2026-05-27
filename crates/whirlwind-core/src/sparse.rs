//! Sparse / irregular-grid phase unwrapping over a Delaunay triangulation.
//!
//! End-to-end pipeline for the "select <10% of valid pixels and unwrap only
//! there" workflow (the same idea as `isce-framework/spurt`, but here the
//! MCF solver is whirlwind's own primal-dual rather than a generic LP).
//!
//! Caller supplies:
//!   * `points`: 2D coordinates of the valid pixels (row, col — units only
//!     affect edge-length terms in optional cost weights).
//!   * `wrapped_phase`: per-pixel wrapped phase, length matches `points`.
//!   * `variance`: per-pixel CRLB phase variance, length matches `points`.
//!
//! Output:
//!   * `Vec<f32>` of unwrapped phase, one entry per input pixel. Pixels not
//!     reachable from the integration seed (e.g. disconnected components in
//!     a degenerate triangulation) come back as `NaN`.
//!
//! The triangulation, MCF dual graph, residues, and per-edge costs are
//! computed internally. The dual graph and arc layout are exposed in
//! [`triangulated::TriangulatedGraph`] for callers who want to bypass the
//! one-shot entry point (e.g. to inject a B_perp warm-start via
//! `Network::warm_start`).

use crate::cost::COST_SCALE;
use crate::network::Network;
use crate::primal_dual;
use crate::residual_graph::ResidualGraph;
use crate::triangulated::TriangulatedGraph;
use delaunator::Point;
use std::f32::consts::TAU;

/// Top-level sparse phase unwrap.
///
/// * `points`: `(x, y)` of each valid pixel; length `n`. Coordinates can be
///   floats — they only enter the Delaunay triangulation, not the cost.
/// * `wrapped_phase`: wrapped phase per pixel, length `n`.
/// * `variance`: CRLB phase variance per pixel (rad²), length `n`. Used to
///   build per-edge integer arc costs via the same `(σ²_a + σ²_b) / 2`
///   recipe the dense CRLB unwrap uses.
/// * `max_edge_length`: optional cutoff. Triangulation edges longer than
///   this are treated as boundary edges (to the outer face) and integration
///   does not cross them. Required when the input point set has long convex-
///   hull spans relative to the phase gradient — without it, multi-wrap
///   edges can't be handled by the unit-capacity MCF and the unwrap is
///   garbage. Pass `None` (or a very large value) only when you've already
///   filtered the input to ensure bounded edge lengths.
///
/// Pixels that end up disconnected from the integration seed by the
/// short-edge subgraph come back as `NaN`. Returns `Err` if shapes don't
/// match or the triangulation is degenerate (fewer than 3 unique points, or
/// all collinear).
pub fn unwrap_sparse(
    points: &[(f64, f64)],
    wrapped_phase: &[f32],
    variance: &[f32],
    max_edge_length: Option<f64>,
) -> Result<Vec<f32>, SparseUnwrapError> {
    if points.len() != wrapped_phase.len() || points.len() != variance.len() {
        return Err(SparseUnwrapError::ShapeMismatch);
    }
    if points.len() < 3 {
        return Err(SparseUnwrapError::TooFewPoints);
    }

    let pts: Vec<Point> = points.iter().map(|&(x, y)| Point { x, y }).collect();
    let g = TriangulatedGraph::with_max_edge_length(
        &pts,
        max_edge_length.unwrap_or(f64::INFINITY),
    ).ok_or(SparseUnwrapError::DegenerateTriangulation)?;

    let residues = compute_triangle_residues(&g, wrapped_phase);
    let costs = compute_edge_costs(&g, variance);

    let mut net = Network::from_topology(
        g.num_nodes(),
        g.num_forward(),
        residues,
        costs,
        None,
    );

    primal_dual::run(&g, &mut net, 50);

    Ok(integrate_triangulated(&g, &net, wrapped_phase))
}

/// Errors that can come out of the sparse unwrap path.
#[derive(Debug, thiserror::Error)]
pub enum SparseUnwrapError {
    #[error("points, wrapped_phase, and variance must have the same length")]
    ShapeMismatch,
    #[error("triangulation needs at least 3 input points")]
    TooFewPoints,
    #[error("triangulation is degenerate (collinear or near-duplicate points)")]
    DegenerateTriangulation,
}

/// Per-triangle winding count (residue), plus the outer-face winding so the
/// sum over all nodes balances to zero. Layout: indices `0..T` are triangles
/// in delaunator order; index `T` is the outer face.
pub fn compute_triangle_residues(
    g: &TriangulatedGraph,
    wrapped_phase: &[f32],
) -> Vec<i32> {
    let mut excess = vec![0_i32; g.num_nodes()];
    let mut total = 0_i32;
    for t in 0..g.num_triangles {
        let (a, b, c) = g.triangle_vertices(t);
        let pa = wrapped_phase[a as usize];
        let pb = wrapped_phase[b as usize];
        let pc = wrapped_phase[c as usize];
        let s = cycle_diff(pb, pa) + cycle_diff(pc, pb) + cycle_diff(pa, pc);
        excess[t] = s;
        total += s;
    }
    // Outer face absorbs the rest so total residue == 0 (the MCF invariant).
    // Conservation: by Stokes, the integer winding around the convex hull
    // equals the sum of interior triangle windings — so outer = -total puts
    // the boundary deposit where wrap lines exiting the hull terminate.
    excess[g.outer_face()] = -total;
    excess
}

/// Per-edge integer cost vector for the MCF. Length = `num_forward = 2 * E`,
/// with the canonical and reversed forward arcs each getting the same cost
/// (symmetric CRLB recipe — direction-dependent costs are reserved for a
/// future Carballo-style implementation that needs the smoothed phase
/// gradient on each triangulation edge).
///
/// Recipe: `cost(edge) = round((σ²_a + σ²_b) / 2 * COST_SCALE)`, clamped
/// non-negative.
pub fn compute_edge_costs(g: &TriangulatedGraph, variance: &[f32]) -> Vec<i32> {
    let e = g.num_edges();
    let mut costs = vec![0_i32; 2 * e];
    for i in 0..e {
        let (pa, pb) = g.edge_pixel_pair(i);
        let va = variance[pa as usize];
        let vb = variance[pb as usize];
        // Treat NaN / non-finite as "noisy" (zero cost) so MCF freely flows
        // across edges touching bad pixels — same convention as the dense
        // CRLB path treats NaN nodata.
        let va = if va.is_finite() { va } else { 0.0 };
        let vb = if vb.is_finite() { vb } else { 0.0 };
        let c = ((0.5 * (va + vb)) * COST_SCALE).round().max(0.0) as i32;
        costs[i] = c;
        costs[i + e] = c;
    }
    costs
}

/// Integrate the flow-corrected wrapped gradients into per-pixel unwrapped
/// phase by walking a spanning tree of the *primal* triangulation graph
/// (the graph whose vertices are pixels and edges are triangulation edges).
///
/// Seed = pixel 0. Pixels not reachable from the seed (shouldn't happen with
/// a single Delaunay triangulation of a single connected point set, but
/// guarded against) come back as NaN.
pub fn integrate_triangulated(
    g: &TriangulatedGraph,
    net: &Network,
    wrapped_phase: &[f32],
) -> Vec<f32> {
    let n_pix = wrapped_phase.len();
    let mut unw = vec![f32::NAN; n_pix];

    // Build pixel-level adjacency from the dual graph's edge list, skipping
    // long edges (those that were carved out as outer-face boundary).
    // Integration BFS must not cross long edges because the unit-capacity
    // MCF can carry at most one cycle per arc — and long edges typically
    // span multiple wraps.
    let n_edges = g.num_edges();
    let mut pix_degree = vec![0_u32; n_pix];
    for i in 0..n_edges {
        if g.edge_is_long(i) {
            continue;
        }
        let (a, b) = g.edge_pixel_pair(i);
        pix_degree[a as usize] += 1;
        pix_degree[b as usize] += 1;
    }
    let mut pix_off = vec![0_u32; n_pix + 1];
    for i in 0..n_pix {
        pix_off[i + 1] = pix_off[i] + pix_degree[i];
    }
    let total_adj = pix_off[n_pix] as usize;
    let mut pix_adj: Vec<(u32, u32, bool)> = vec![(0, 0, false); total_adj];
    let mut cursor = pix_off.clone();
    for i in 0..n_edges {
        if g.edge_is_long(i) {
            continue;
        }
        let (a, b) = g.edge_pixel_pair(i);
        let ai = a as usize;
        let bi = b as usize;
        pix_adj[cursor[ai] as usize] = (b, i as u32, true);
        cursor[ai] += 1;
        pix_adj[cursor[bi] as usize] = (a, i as u32, false);
        cursor[bi] += 1;
    }

    // BFS from pixel 0.
    use std::collections::VecDeque;
    let mut queue: VecDeque<u32> = VecDeque::new();
    unw[0] = wrapped_phase[0];
    queue.push_back(0);

    while let Some(u_u32) = queue.pop_front() {
        let u = u_u32 as usize;
        let phi_u = unw[u] as f64;
        let start = pix_off[u] as usize;
        let end = pix_off[u + 1] as usize;
        for &(v_u32, eidx_u32, u_is_canonical) in &pix_adj[start..end] {
            let v = v_u32 as usize;
            if !unw[v].is_nan() {
                continue;
            }
            let eidx = eidx_u32 as usize;
            let dpsi = wrapped_diff(wrapped_phase[v], wrapped_phase[u]);
            // Sign convention: for the canonical edge direction (pixel a → b),
            // flow on the canonical forward arc (dual: tail_face → head_face)
            // is the integer cycle correction applied to the primal gradient
            // (phase[b] − phase[a]) with sign chosen so that integrating in
            // the canonical direction adds (net_flow · 2π). When walking the
            // edge in the reversed direction (b → a), flip the sign.
            let net_flow = arc_pair_net_flow(g, net, eidx);
            let signed_flow = if u_is_canonical { net_flow } else { -net_flow };
            let dphi = dpsi + TAU * (signed_flow as f32);
            unw[v] = (phi_u + dphi as f64) as f32;
            queue.push_back(v_u32);
        }
    }

    unw
}

/// Net integer cycle correction on a triangulation edge: difference between
/// flow in the canonical direction vs. the reversed direction.
#[inline]
fn arc_pair_net_flow(g: &TriangulatedGraph, net: &Network, edge_idx: usize) -> i32 {
    let e = g.num_edges();
    // Forward canonical arc id = edge_idx; forward reversed arc id = edge_idx + E.
    net.arc_flow(g, edge_idx) - net.arc_flow(g, edge_idx + e)
}

#[inline]
fn cycle_diff(a: f32, b: f32) -> i32 {
    ((a - b) / TAU).round() as i32
}

#[inline]
fn wrapped_diff(a: f32, b: f32) -> f32 {
    let d = a - b;
    d - TAU * (d / TAU).round()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn wrap(x: f32) -> f32 {
        let y = x % TAU;
        if y > PI {
            y - TAU
        } else if y <= -PI {
            y + TAU
        } else {
            y
        }
    }

    /// Smooth ramp on a sparse random set of pixels: no residues anywhere,
    /// MCF does nothing, integration recovers the ramp up to a constant.
    #[test]
    fn smooth_ramp_recovered_on_random_subset() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let n = 200;
        let mut points: Vec<(f64, f64)> = Vec::with_capacity(n);
        let mut truth: Vec<f32> = Vec::with_capacity(n);
        let mut wrapped: Vec<f32> = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.gen_range(0.0..100.0_f64);
            let y = rng.gen_range(0.0..100.0_f64);
            // Smooth phase gradient < 2π/grid so no wraps between adjacent pts.
            let t = 0.02_f32 * (x as f32) + 0.01_f32 * (y as f32);
            truth.push(t);
            wrapped.push(wrap(t));
            points.push((x, y));
        }
        let variance = vec![0.1_f32; n];
        let unw = unwrap_sparse(&points, &wrapped, &variance, None).unwrap();
        // Match truth up to a constant (seed pixel sets the zero).
        let offset = unw[0] as f64 - truth[0] as f64;
        let mut max_err = 0.0_f64;
        for i in 0..n {
            if unw[i].is_nan() {
                continue;
            }
            let err = (unw[i] as f64 - truth[i] as f64 - offset).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 1e-3, "smooth ramp recovery error too large: {max_err}");
    }

    /// Conservation: total residue (sum over all triangles + outer face) is 0
    /// by construction.
    #[test]
    fn residue_total_is_zero() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xABCD);
        let n = 100;
        let mut pts = Vec::with_capacity(n);
        let mut phase = Vec::with_capacity(n);
        for _ in 0..n {
            pts.push(Point {
                x: rng.gen_range(0.0..50.0),
                y: rng.gen_range(0.0..50.0),
            });
            phase.push(rng.gen_range(-PI..PI));
        }
        let g = TriangulatedGraph::new(&pts).unwrap();
        let res = compute_triangle_residues(&g, &phase);
        let total: i32 = res.iter().sum();
        assert_eq!(total, 0, "residues must balance to zero (outer face included)");
    }

    /// Wrapping ramp on a regular dense grid sampled as "points": MCF should
    /// drain interior residues (none here, since smooth wraps generate only
    /// hull-residues against the outer face) and integration should recover
    /// the original ramp up to a constant. This is the strongest correctness
    /// check — sign convention in the integrator has to match the dual-graph
    /// flow direction or this test fails by 2π·k.
    #[test]
    fn wrapping_ramp_recovers_truth_on_dense_subset() {
        // 20x20 jittered grid is dense enough that adjacent triangulation
        // edges are short, but the ramp slope still produces multiple wraps.
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xBEEF);
        let g_side = 20;
        let n = g_side * g_side;
        let mut pts = Vec::with_capacity(n);
        let mut truth = Vec::with_capacity(n);
        let mut wrapped = Vec::with_capacity(n);
        for i in 0..g_side {
            for j in 0..g_side {
                let jitter_x = rng.gen_range(-0.1..0.1_f64);
                let jitter_y = rng.gen_range(-0.1..0.1_f64);
                let x = i as f64 + jitter_x;
                let y = j as f64 + jitter_y;
                // 4 full wraps across the grid.
                let t = 1.3_f32 * (x as f32) + 0.7_f32 * (y as f32);
                truth.push(t);
                wrapped.push(wrap(t));
                pts.push((x, y));
            }
        }
        let variance = vec![0.05_f32; n];
        let unw = unwrap_sparse(&pts, &wrapped, &variance, Some(3.0)).unwrap();
        // Match truth up to a constant offset (seed pixel sets the zero).
        let mut offset_sum = 0.0_f64;
        let mut n_finite = 0;
        for i in 0..n {
            if unw[i].is_finite() {
                offset_sum += unw[i] as f64 - truth[i] as f64;
                n_finite += 1;
            }
        }
        assert!(n_finite > 9 * n / 10, "expected most pixels finite, got {n_finite}/{n}");
        let offset = offset_sum / n_finite as f64;
        let mut max_err = 0.0_f64;
        for i in 0..n {
            if unw[i].is_finite() {
                let err = (unw[i] as f64 - truth[i] as f64 - offset).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        // Tolerance: integer-cycle errors would be ≥ 2π ≈ 6.28, so anything
        // under 1 rad means MCF + integration recovered the ramp correctly.
        assert!(max_err < 0.5, "wrapping-ramp recovery error: {max_err}");
    }
}
