//! Closure-constrained integer correction on a stack of unwrapped interferograms.
//!
//! Reframing: the unwrapped phase of E interferograms over D acquisitions lives
//! in a (D-1)-dimensional subspace of per-date acquisition phases. Any observed
//! IG that disagrees with that subspace by an integer multiple of 2π has an
//! unwrapping error that closure can detect *and correct*.
//!
//! ## Algorithm: tree-based integer assignment
//!
//! 1. Pick a spanning tree T over the temporal graph. Edges with high quality
//!    (e.g. short baseline, low CRLB variance) form the tree; remaining edges
//!    are "loop-closing".
//! 2. Per pixel: walk the tree from the reference date, propagating unwrapped
//!    phase. The tree gives a clean θ ∈ R^D (one phase per acquisition).
//! 3. For each non-tree edge e = (i, j): residual r_e = y_e − (θ_j − θ_i).
//!    Round r_e/(2π) to the nearest integer k_e and subtract 2π·k_e from y_e.
//!
//! This is the GNSS / LAMBDA-style trick: trust a spanning subset of "best"
//! observations, let everything else absorb the integer ambiguities. It is
//! O(E + D) per pixel — orders of magnitude faster than the per-pixel global
//! LS — and avoids the symmetric-degeneracy failure where a global LS smears
//! a single-edge integer error across every edge in a cycle.
//!
//! The price: tree edges with unwrapping errors propagate to all dates
//! connected through them. Picking a high-quality tree (lowest median CRLB
//! variance per edge) is therefore the place where the noise model from
//! phase-linking earns its keep.

use ndarray::{Array2, Array3, ArrayView3};
use rayon::prelude::*;
use std::f32::consts::TAU;

/// One temporal-graph edge: an interferogram between two acquisition dates.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub from: u32,
    pub to: u32,
}

/// Temporal acquisition graph.
#[derive(Clone, Debug)]
pub struct TemporalGraph {
    pub n_dates: usize,
    pub edges: Vec<Edge>,
    /// Reference acquisition index. θ at this index is fixed to 0.
    pub reference: usize,
}

impl TemporalGraph {
    pub fn new(n_dates: usize, edges: Vec<Edge>, reference: usize) -> Self {
        assert!(reference < n_dates, "reference {reference} ≥ n_dates {n_dates}");
        for e in &edges {
            assert!((e.from as usize) < n_dates);
            assert!((e.to as usize) < n_dates);
            assert_ne!(e.from, e.to, "self-loop in temporal graph");
        }
        Self { n_dates, edges, reference }
    }
}

/// Output of one pass of closure correction.
pub struct ClosureOutput {
    /// Corrected unwrapped IG stack, shape (n_edges, m, n).
    pub corrected: Array3<f32>,
    /// Integer corrections applied, in units of cycles: corrected = original − 2π·k.
    /// Shape (n_edges, m, n).
    pub corrections: Array3<i16>,
    /// Per-date recovered acquisition phases. Shape (n_dates, m, n).
    /// Reference index is identically 0.
    pub date_phases: Array3<f32>,
    /// Per-pixel RMS closure residual after correction, in radians. Shape (m, n).
    /// Large values indicate noisy / unresolvable pixels — physical non-closure,
    /// noise, or unwrapping errors that the spanning tree pulled into θ.
    pub closure_rms: Array2<f32>,
}

/// Closure-correct an unwrapped IG stack.
///
/// * `unw_stack` — baseline unwrapped IGs, shape (E, m, n).
/// * `graph`    — temporal graph; `edges.len()` must equal E (matched by index).
/// * `tree_edge_priority` — optional per-edge "quality" score (lower = better);
///   the spanning tree is built by Prim's algorithm using these as edge weights.
///   If `None`, edges are taken in their natural order (which works for many
///   short-baseline networks). For phase-linked inputs the right value is the
///   median CRLB-derived variance per edge across coherent pixels.
pub fn correct(
    unw_stack: ArrayView3<f32>,
    graph: &TemporalGraph,
    tree_edge_priority: Option<&[f32]>,
) -> ClosureOutput {
    let (n_edges, m, n) = unw_stack.dim();
    assert_eq!(
        n_edges,
        graph.edges.len(),
        "stack edge count {n_edges} != graph edge count {}",
        graph.edges.len()
    );

    // Build the spanning tree once for the whole scene.
    let tree = build_spanning_tree(graph, tree_edge_priority);
    assert_eq!(
        tree.tree_edges.len(),
        graph.n_dates - 1,
        "graph is not connected: tree has {} edges, expected {}",
        tree.tree_edges.len(),
        graph.n_dates - 1
    );

    // Storage.
    let mut corrected = Array3::<f32>::zeros((n_edges, m, n));
    let mut corrections = Array3::<i16>::zeros((n_edges, m, n));
    let mut date_phases = Array3::<f32>::zeros((graph.n_dates, m, n));
    let mut closure_rms = Array2::<f32>::zeros((m, n));

    // Parallel over rows. Each row computes per-pixel θ + per-edge corrections.
    let row_results: Vec<RowResult> = (0..m)
        .into_par_iter()
        .map(|i| solve_row(unw_stack, i, graph, &tree, n_edges, n))
        .collect();

    for (i, row) in row_results.into_iter().enumerate() {
        for j in 0..n {
            for e in 0..n_edges {
                corrected[(e, i, j)] = row.corrected[e * n + j];
                corrections[(e, i, j)] = row.corrections[e * n + j];
            }
            for d in 0..graph.n_dates {
                date_phases[(d, i, j)] = row.date_phases[d * n + j];
            }
            closure_rms[(i, j)] = row.closure_rms[j];
        }
    }

    ClosureOutput {
        corrected,
        corrections,
        date_phases,
        closure_rms,
    }
}

// ---- spanning tree -------------------------------------------------------

#[derive(Debug)]
struct SpanningTree {
    /// Indices into `graph.edges` of the edges chosen for the tree. Length D-1.
    tree_edges: Vec<usize>,
    /// BFS-order traversal of dates starting from `reference`. Length D.
    /// `bfs_order[0] = reference`. For each subsequent date, the edge that
    /// links it to its parent in the tree is `parent_edge[date]`.
    bfs_order: Vec<u32>,
    /// For each date d != reference, `(parent_date, edge_index, signed_dir)`:
    /// signed_dir is +1 if the edge goes parent→d (i.e. edge.from == parent),
    /// −1 if it goes d→parent. Used to integrate along the tree.
    parent: Vec<Option<(u32, usize, f32)>>,
}

fn build_spanning_tree(graph: &TemporalGraph, priority: Option<&[f32]>) -> SpanningTree {
    let d = graph.n_dates;

    // Adjacency: for each date, list of (neighbour_date, edge_index, signed_dir).
    let mut adj: Vec<Vec<(u32, usize, f32)>> = vec![Vec::new(); d];
    for (idx, e) in graph.edges.iter().enumerate() {
        adj[e.from as usize].push((e.to, idx, 1.0));   // travelling from→to gives +ψ_e
        adj[e.to as usize].push((e.from, idx, -1.0));  // travelling to→from gives −ψ_e
    }

    let mut parent: Vec<Option<(u32, usize, f32)>> = vec![None; d];
    let mut in_tree = vec![false; d];
    let mut bfs_order: Vec<u32> = Vec::with_capacity(d);
    let mut tree_edges: Vec<usize> = Vec::with_capacity(d - 1);

    // Prim's algorithm with a simple linear-scan priority queue. d ≤ ~100 in
    // any realistic case so we don't need a heap.
    in_tree[graph.reference] = true;
    bfs_order.push(graph.reference as u32);

    // (best_priority, parent_date, edge_index, signed_dir) for each date not yet in tree
    let mut best: Vec<Option<(f32, u32, usize, f32)>> = vec![None; d];

    // Seed: relax edges out of the reference.
    relax(graph.reference, &adj, priority, &in_tree, &mut best);

    while tree_edges.len() < d - 1 {
        // Pick the not-yet-in-tree date with the smallest best-priority.
        let mut pick: Option<usize> = None;
        let mut pick_pri = f32::INFINITY;
        for v in 0..d {
            if in_tree[v] {
                continue;
            }
            if let Some((p, _, _, _)) = best[v] {
                if p < pick_pri {
                    pick = Some(v);
                    pick_pri = p;
                }
            }
        }
        let v = pick.expect("graph not connected");
        let (_, pdate, eidx, sgn) = best[v].unwrap();
        parent[v] = Some((pdate, eidx, sgn));
        tree_edges.push(eidx);
        in_tree[v] = true;
        bfs_order.push(v as u32);
        relax(v, &adj, priority, &in_tree, &mut best);
    }

    SpanningTree { tree_edges, bfs_order, parent }
}

fn relax(
    u: usize,
    adj: &[Vec<(u32, usize, f32)>],
    priority: Option<&[f32]>,
    in_tree: &[bool],
    best: &mut [Option<(f32, u32, usize, f32)>],
) {
    for &(v, eidx, sgn) in &adj[u] {
        let v = v as usize;
        if in_tree[v] {
            continue;
        }
        let pri = priority.map(|p| p[eidx]).unwrap_or(1.0);
        let take = match best[v] {
            None => true,
            Some((b, _, _, _)) => pri < b,
        };
        if take {
            best[v] = Some((pri, u as u32, eidx, sgn));
        }
    }
}

// ---- per-row inner loop --------------------------------------------------

struct RowResult {
    // Flattened (E, n)
    corrected: Vec<f32>,
    corrections: Vec<i16>,
    // Flattened (D, n)
    date_phases: Vec<f32>,
    // Length n
    closure_rms: Vec<f32>,
}

fn solve_row(
    unw_stack: ArrayView3<f32>,
    i: usize,
    graph: &TemporalGraph,
    tree: &SpanningTree,
    n_edges: usize,
    n: usize,
) -> RowResult {
    let d = graph.n_dates;
    let mut corrected = vec![0.0_f32; n_edges * n];
    let mut corrections = vec![0_i16; n_edges * n];
    let mut date_phases = vec![0.0_f32; d * n];
    let mut closure_rms = vec![0.0_f32; n];

    // Mark which edges are in the tree for the fast inner check.
    let mut is_tree_edge = vec![false; n_edges];
    for &ei in &tree.tree_edges {
        is_tree_edge[ei] = true;
    }

    for j in 0..n {
        // 1) Propagate θ along the tree in BFS order.
        // θ[reference] = 0. For every other date d, θ[d] = θ[parent] ± y[edge].
        for &dv in &tree.bfs_order {
            let dv = dv as usize;
            if dv == graph.reference {
                date_phases[dv * n + j] = 0.0;
                continue;
            }
            let (pd, ei, sgn) = tree.parent[dv].expect("non-reference node has no parent");
            let pd = pd as usize;
            let y = unw_stack[(ei, i, j)];
            date_phases[dv * n + j] = date_phases[pd * n + j] + sgn * y;
        }

        // 2) Tree edges: by construction y_e − (θ_to − θ_from) = 0 (mod sign).
        //    Copy them through unchanged.
        // 3) Non-tree edges: round their residual to the nearest 2π multiple.
        let mut sumsq = 0.0_f32;
        let mut n_loop = 0;
        for (e, edge) in graph.edges.iter().enumerate() {
            let from = edge.from as usize;
            let to = edge.to as usize;
            let y = unw_stack[(e, i, j)];
            let expected = date_phases[to * n + j] - date_phases[from * n + j];
            if is_tree_edge[e] {
                corrected[e * n + j] = y;
                corrections[e * n + j] = 0;
            } else {
                let r = y - expected;
                let k = (r / TAU).round();
                let k_i = k as i32;
                let k_i = k_i.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                corrected[e * n + j] = y - TAU * k;
                corrections[e * n + j] = k_i;
                let r_after = r - TAU * k;
                sumsq += r_after * r_after;
                n_loop += 1;
            }
        }
        closure_rms[j] = if n_loop > 0 {
            (sumsq / n_loop as f32).sqrt()
        } else {
            0.0
        };
    }

    RowResult {
        corrected,
        corrections,
        date_phases,
        closure_rms,
    }
}

// ---- tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 3-date triangle. Inject +1-cycle error on edge (0,1). The tree-based
    /// corrector picks edges (0,1) and (1,2) for the tree (natural order),
    /// so the integer error on (0,1) propagates into θ and shows up as a
    /// closure residue on the loop edge (0,2). Closure must hit zero after
    /// correction, regardless of which edge "absorbs" the cycle.
    #[test]
    fn triangle_closes_after_correction() {
        let graph = TemporalGraph::new(
            3,
            vec![
                Edge { from: 0, to: 1 },
                Edge { from: 1, to: 2 },
                Edge { from: 0, to: 2 },
            ],
            0,
        );
        let theta = [0.0_f32, std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2];
        let mut psi = Vec::<f32>::new();
        for e in &graph.edges {
            psi.push(theta[e.to as usize] - theta[e.from as usize]);
        }
        psi[0] += TAU;

        let mut stack = Array3::<f32>::zeros((3, 1, 1));
        for e in 0..3 {
            stack[(e, 0, 0)] = psi[e];
        }
        let out = correct(stack.view(), &graph, None);

        // After correction, edge (0,1) + edge (1,2) − edge (0,2) must equal 0.
        let c = out.corrected[(0, 0, 0)] + out.corrected[(1, 0, 0)] - out.corrected[(2, 0, 0)];
        assert!(c.abs() < 1e-4, "closure {c} not zero");
        assert!(out.closure_rms[(0, 0)] < 1e-4);

        // Sum of integer corrections, weighted by sign in the cycle, must be ±1.
        // (Some edge absorbs the +1 cycle; the tree-based picker may put it
        //  on a different edge than the original injection.)
        let total_k: i32 = out.corrections[(0, 0, 0)] as i32
            + out.corrections[(1, 0, 0)] as i32
            - out.corrections[(2, 0, 0)] as i32;
        assert_eq!(total_k.abs(), 1, "total cycle correction magnitude should be 1, got {total_k}");
    }

    /// Noiseless dense network of 4 dates and 6 IGs: corrections should be all
    /// zero and recovered θ should match truth (to numerical precision).
    #[test]
    fn noiseless_stack_is_unchanged() {
        let graph = TemporalGraph::new(
            4,
            vec![
                Edge { from: 0, to: 1 },
                Edge { from: 0, to: 2 },
                Edge { from: 0, to: 3 },
                Edge { from: 1, to: 2 },
                Edge { from: 1, to: 3 },
                Edge { from: 2, to: 3 },
            ],
            0,
        );
        let theta = [0.0_f32, 0.3, -0.7, 1.2];
        let mut stack = Array3::<f32>::zeros((6, 2, 2));
        for (e, edge) in graph.edges.iter().enumerate() {
            let v = theta[edge.to as usize] - theta[edge.from as usize];
            for i in 0..2 {
                for j in 0..2 {
                    stack[(e, i, j)] = v;
                }
            }
        }
        let out = correct(stack.view(), &graph, None);
        for e in 0..6 {
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(out.corrections[(e, i, j)], 0);
                    assert!((out.corrected[(e, i, j)] - stack[(e, i, j)]).abs() < 1e-5);
                }
            }
        }
        for d in 0..4 {
            assert!((out.date_phases[(d, 0, 0)] - theta[d]).abs() < 1e-4);
        }
    }

    /// Inject multiple integer errors and check that every triangle closes.
    #[test]
    fn multiple_errors_all_triangles_close() {
        let graph = TemporalGraph::new(
            4,
            vec![
                Edge { from: 0, to: 1 },
                Edge { from: 0, to: 2 },
                Edge { from: 0, to: 3 },
                Edge { from: 1, to: 2 },
                Edge { from: 1, to: 3 },
                Edge { from: 2, to: 3 },
            ],
            0,
        );
        let theta = [0.0_f32, 0.3, -0.7, 1.2];
        let mut psi = Vec::<f32>::new();
        for e in &graph.edges {
            psi.push(theta[e.to as usize] - theta[e.from as usize]);
        }
        // Inject errors on non-tree edges (the tree will be 0,1,2 for natural
        // BFS order from 0). Edges 3, 4, 5 are loops.
        psi[3] += TAU;     // (1,2) +1 cycle
        psi[5] -= 2.0 * TAU; // (2,3) -2 cycles

        let mut stack = Array3::<f32>::zeros((6, 1, 1));
        for e in 0..6 {
            stack[(e, 0, 0)] = psi[e];
        }
        let out = correct(stack.view(), &graph, None);

        // All four triangles of the 4-clique must close:
        //   (0,1,2): e0 + e3 - e1 = 0
        //   (0,1,3): e0 + e4 - e2 = 0
        //   (0,2,3): e1 + e5 - e2 = 0
        //   (1,2,3): e3 + e5 - e4 = 0
        let c = &out.corrected;
        let g = |a: usize, b: usize, cc: usize, s: f32| {
            let v = c[(a, 0, 0)] + s * c[(b, 0, 0)] - c[(cc, 0, 0)];
            v.abs()
        };
        assert!(g(0, 3, 1, 1.0) < 1e-4);
        assert!(g(0, 4, 2, 1.0) < 1e-4);
        assert!(g(1, 5, 2, 1.0) < 1e-4);
        assert!((c[(3, 0, 0)] + c[(5, 0, 0)] - c[(4, 0, 0)]).abs() < 1e-4);

        assert!(out.closure_rms[(0, 0)] < 1e-3);
    }

    /// With per-edge priority (e.g. CRLB), the spanning tree should pick the
    /// cheapest edges. We feed a 4-date network with one cheap edge per date
    /// and confirm the tree uses them.
    #[test]
    fn priority_drives_tree_selection() {
        let graph = TemporalGraph::new(
            4,
            vec![
                Edge { from: 0, to: 1 },  // edge 0  — cheap
                Edge { from: 0, to: 2 },  // edge 1  — expensive
                Edge { from: 0, to: 3 },  // edge 2  — expensive
                Edge { from: 1, to: 2 },  // edge 3  — cheap
                Edge { from: 1, to: 3 },  // edge 4  — expensive
                Edge { from: 2, to: 3 },  // edge 5  — cheap
            ],
            0,
        );
        let priority = [0.1, 5.0, 5.0, 0.1, 5.0, 0.1];
        let tree = build_spanning_tree(&graph, Some(&priority));
        let mut tree_edges = tree.tree_edges.clone();
        tree_edges.sort();
        assert_eq!(tree_edges, vec![0, 3, 5]);
    }
}
