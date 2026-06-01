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

/// One signed edge step in a cycle traversal. `sign = +1` if we walk the
/// underlying IG in its native (from→to) direction, `-1` otherwise.
#[derive(Clone, Copy, Debug)]
pub struct CycleStep {
    pub edge_idx: u32,
    pub sign: i8,
}

/// A fundamental cycle basis: one cycle per non-tree edge of the spanning
/// tree, expressed as a signed sequence of original-graph edges. Each cycle
/// has length ≤ 2·(tree depth) + 1.
pub struct CycleBasis {
    /// Flat storage of all cycle steps.
    steps: Vec<CycleStep>,
    /// `cycle_offsets[c .. c+1]` gives the steps for cycle c.
    cycle_offsets: Vec<u32>,
    /// For diagnostics: the underlying non-tree edge that defines each cycle.
    pub defining_edge: Vec<u32>,
}

impl CycleBasis {
    pub fn num_cycles(&self) -> usize {
        self.cycle_offsets.len() - 1
    }
    pub fn cycle(&self, c: usize) -> &[CycleStep] {
        let lo = self.cycle_offsets[c] as usize;
        let hi = self.cycle_offsets[c + 1] as usize;
        &self.steps[lo..hi]
    }
}

/// Build the fundamental cycle basis induced by a spanning tree.
///
/// For each non-tree edge e_nt = (a, b), the unique fundamental cycle is
/// e_nt followed by the reverse of the tree path from a to b — equivalently,
/// "go from a to b via the tree, then take e_nt back to close the loop."
/// The orientation convention here is: traverse each cycle so that summing
/// signed unwrapped phases around it gives ψ̃(tree-path) − ψ̃(e_nt). A
/// non-zero closure residue means the cycle is integer-violated by that
/// many cycles of 2π.
pub fn build_cycle_basis(graph: &TemporalGraph, tree: &SpanningTree) -> CycleBasis {
    let d = graph.n_dates;
    let mut is_tree_edge = vec![false; graph.edges.len()];
    for &ei in &tree.tree_edges {
        is_tree_edge[ei] = true;
    }

    // Compute depth from the reference along the tree (for LCA finding).
    let mut depth = vec![0_u32; d];
    // bfs_order has reference first; we can derive depths by walking parents.
    for &v in &tree.bfs_order {
        let v = v as usize;
        if v == graph.reference {
            depth[v] = 0;
        } else {
            let (pd, _, _) = tree.parent[v].expect("non-reference has parent");
            depth[v] = depth[pd as usize] + 1;
        }
    }

    let mut steps: Vec<CycleStep> = Vec::new();
    let mut cycle_offsets: Vec<u32> =
        Vec::with_capacity(graph.edges.len() - tree.tree_edges.len() + 1);
    let mut defining_edge: Vec<u32> = Vec::new();
    cycle_offsets.push(0);

    // Reusable scratch.
    let mut path_from_a: Vec<CycleStep> = Vec::new();
    let mut path_from_b: Vec<CycleStep> = Vec::new();

    for (e_idx, edge) in graph.edges.iter().enumerate() {
        if is_tree_edge[e_idx] {
            continue;
        }
        // Walk from a and b up to their LCA in the tree. Each step we take
        // along the tree contributes one CycleStep with the appropriate sign.
        // We orient the cycle as: a → b along the tree, then b → a via e_nt
        // (negative orientation of e_nt). The closure residue is then:
        //     Σ_steps sign · ψ̃ = ψ̃(tree path a→b) − ψ̃(e_nt)
        path_from_a.clear();
        path_from_b.clear();

        let mut x = edge.from as usize;
        let mut y = edge.to as usize;
        while depth[x] > depth[y] {
            // Walk x up one step. The tree edge from x leads to its parent;
            // moving x→parent in our "a→b" direction has sign that depends on
            // how the original IG is oriented. tree.parent[x] = (pd, eidx, sgn)
            // where sgn was set such that θ_x = θ_pd + sgn · ψ̃_eidx, i.e.
            // walking the IG in the direction pd→x picks up +sgn·ψ̃. Going
            // the other way (x→pd) picks up −sgn·ψ̃.
            let (pd, eidx, sgn) = tree.parent[x].expect("non-reference has parent");
            path_from_a.push(CycleStep {
                edge_idx: eidx as u32,
                sign: -(sgn as i8),
            });
            x = pd as usize;
        }
        while depth[y] > depth[x] {
            let (pd, eidx, sgn) = tree.parent[y].expect("non-reference has parent");
            // Walking y→pd then reversing later: in the final a→b traversal
            // this segment is pd→y, which contributes +sgn·ψ̃.
            path_from_b.push(CycleStep {
                edge_idx: eidx as u32,
                sign: sgn as i8,
            });
            y = pd as usize;
        }
        while x != y {
            let (pdx, eidxx, sgnx) = tree.parent[x].expect("non-reference has parent");
            path_from_a.push(CycleStep {
                edge_idx: eidxx as u32,
                sign: -(sgnx as i8),
            });
            x = pdx as usize;
            let (pdy, eidxy, sgny) = tree.parent[y].expect("non-reference has parent");
            path_from_b.push(CycleStep {
                edge_idx: eidxy as u32,
                sign: sgny as i8,
            });
            y = pdy as usize;
        }
        // Tree path a→b = path_from_a (in order) ++ reverse(path_from_b).
        for s in &path_from_a {
            steps.push(*s);
        }
        for s in path_from_b.iter().rev() {
            steps.push(*s);
        }
        // Closing edge: travel back b→a via e_nt. Sign of −1 since the IG is
        // oriented from edge.from → edge.to.
        steps.push(CycleStep {
            edge_idx: e_idx as u32,
            sign: -1,
        });

        defining_edge.push(e_idx as u32);
        cycle_offsets.push(steps.len() as u32);
    }

    CycleBasis {
        steps,
        cycle_offsets,
        defining_edge,
    }
}

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
        assert!(
            reference < n_dates,
            "reference {reference} ≥ n_dates {n_dates}"
        );
        for e in &edges {
            assert!((e.from as usize) < n_dates);
            assert!((e.to as usize) < n_dates);
            assert_ne!(e.from, e.to, "self-loop in temporal graph");
        }
        Self {
            n_dates,
            edges,
            reference,
        }
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

    // Stripe-based parallel processing: each stripe of STRIPE_H rows is
    // computed in parallel (rayon over rows within the stripe), then scattered
    // into the output arrays serially. This bounds the peak intermediate
    // memory to STRIPE_H × per_row_bytes instead of m × per_row_bytes — the
    // latter is multi-GB on full scenes and was the cause of severe thrashing.
    const STRIPE_H: usize = 64;
    for stripe_start in (0..m).step_by(STRIPE_H) {
        let stripe_end = (stripe_start + STRIPE_H).min(m);
        let rows: Vec<RowResult> = (stripe_start..stripe_end)
            .into_par_iter()
            .map(|i| solve_row(unw_stack, i, graph, &tree, n_edges, n))
            .collect();
        for (offset, row) in rows.into_iter().enumerate() {
            let i = stripe_start + offset;
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
pub struct SpanningTree {
    /// Indices into `graph.edges` of the edges chosen for the tree. Length D-1.
    pub tree_edges: Vec<usize>,
    /// BFS-order traversal of dates starting from `reference`. Length D.
    /// `bfs_order[0] = reference`. For each subsequent date, the edge that
    /// links it to its parent in the tree is `parent_edge[date]`.
    pub bfs_order: Vec<u32>,
    /// For each date d != reference, `(parent_date, edge_index, signed_dir)`:
    /// signed_dir is +1 if the edge goes parent→d (i.e. edge.from == parent),
    /// −1 if it goes d→parent. Used to integrate along the tree.
    pub parent: Vec<Option<(u32, usize, f32)>>,
}

pub fn build_spanning_tree(graph: &TemporalGraph, priority: Option<&[f32]>) -> SpanningTree {
    let d = graph.n_dates;

    // Adjacency: for each date, list of (neighbour_date, edge_index, signed_dir).
    let mut adj: Vec<Vec<(u32, usize, f32)>> = vec![Vec::new(); d];
    for (idx, e) in graph.edges.iter().enumerate() {
        adj[e.from as usize].push((e.to, idx, 1.0)); // travelling from→to gives +ψ_e
        adj[e.to as usize].push((e.from, idx, -1.0)); // travelling to→from gives −ψ_e
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
            if let Some((p, _, _, _)) = best[v]
                && p < pick_pri
            {
                pick = Some(v);
                pick_pri = p;
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

    SpanningTree {
        tree_edges,
        bfs_order,
        parent,
    }
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

// =========================================================================
// Per-pixel quality map from fundamental-cycle residuals
// =========================================================================
//
// Phase linking guarantees that the *wrapped* sum around any temporal cycle
// is identically zero (wrap respects the algebraic identity). After per-IG
// 2D unwrapping each IG independently picks an integer ambiguity k_e, so the
// *unwrapped* cycle sum is exactly 2π · (Σ_e ε_e k_e) — i.e. always an
// integer multiple of 2π, with the integer being the per-cycle unwrap
// mistake count.
//
// `quality_max_integer_cycles` returns, per pixel, the max |K| over the
// fundamental cycle basis (E - D + 1 cycles, defined by the spanning tree).
// 0 = perfectly consistent across all cycles; ≥1 = at least one cycle has
// an integer-ambiguity mismatch. Use as a "trust this pixel" gate: water
// and decorrelated regions naturally produce high K because their per-IG
// unwraps are arbitrary and don't cohere over loops.
//
// This is a heuristic (max |K| over fundamental cycles), not a true
// Bayesian posterior. A more principled per-pixel posterior would solve
// a weighted integer LS (LAMBDA / closest-vector-in-lattice) over the
// full integer-correction vector; deferred — see ATBD-3d §10.5.

/// Per-pixel max |K| over fundamental cycles. Returns shape (m, n).
pub fn quality_max_integer_cycles(
    unw_stack: ArrayView3<f32>,
    graph: &TemporalGraph,
    tree_edge_priority: Option<&[f32]>,
) -> Array2<u16> {
    let (n_edges, m, n) = unw_stack.dim();
    assert_eq!(
        n_edges,
        graph.edges.len(),
        "stack edge count {n_edges} != graph edge count {}",
        graph.edges.len()
    );
    let tree = build_spanning_tree(graph, tree_edge_priority);
    assert_eq!(
        tree.tree_edges.len(),
        graph.n_dates - 1,
        "graph is not connected for quality map"
    );
    let mut is_tree_edge = vec![false; n_edges];
    for &ei in &tree.tree_edges {
        is_tree_edge[ei] = true;
    }

    let mut out = Array2::<u16>::zeros((m, n));
    const STRIPE_H: usize = 64;
    for stripe_start in (0..m).step_by(STRIPE_H) {
        let stripe_end = (stripe_start + STRIPE_H).min(m);
        let rows: Vec<Vec<u16>> = (stripe_start..stripe_end)
            .into_par_iter()
            .map(|i| quality_row(unw_stack, i, graph, &tree, &is_tree_edge, n))
            .collect();
        for (offset, row) in rows.into_iter().enumerate() {
            let i = stripe_start + offset;
            for j in 0..n {
                out[(i, j)] = row[j];
            }
        }
    }
    out
}

fn quality_row(
    unw_stack: ArrayView3<f32>,
    i: usize,
    graph: &TemporalGraph,
    tree: &SpanningTree,
    is_tree_edge: &[bool],
    n: usize,
) -> Vec<u16> {
    let d = graph.n_dates;
    let mut theta = vec![0.0_f32; d];
    let mut out = vec![0_u16; n];
    for j in 0..n {
        // Propagate θ along the tree in BFS order.
        for &dv in &tree.bfs_order {
            let dv = dv as usize;
            if dv == graph.reference {
                theta[dv] = 0.0;
                continue;
            }
            let (pd, ei, sgn) = tree.parent[dv].expect("non-reference node has no parent");
            theta[dv] = theta[pd as usize] + sgn * unw_stack[(ei, i, j)];
        }
        // For each non-tree edge, compute the cycle residual integer.
        let mut max_k: u16 = 0;
        for (e, edge) in graph.edges.iter().enumerate() {
            if is_tree_edge[e] {
                continue;
            }
            let y = unw_stack[(e, i, j)];
            let expected = theta[edge.to as usize] - theta[edge.from as usize];
            let k_abs = ((y - expected) / TAU).round().abs() as u32;
            let k_u16 = k_abs.min(u16::MAX as u32) as u16;
            if k_u16 > max_k {
                max_k = k_u16;
            }
        }
        out[j] = max_k;
    }
    out
}

/// Per-pixel max |K| over all temporal *triangles* (3-cycles) in the graph.
///
/// A triangle is a triple (a, b, c) where IGs (a, b), (b, c), and (a, c)
/// all exist in the temporal graph. The cycle sum
/// `ψ_{a,b} + ψ_{b,c} − ψ_{a,c}` is exactly 0 (mod 2π) for phase-linked
/// inputs, so any nonzero residual after per-IG unwrapping is exactly
/// 2π·K with K integer.
///
/// Compared to `quality_max_integer_cycles` (which uses the fundamental
/// cycle basis, with cycles of length up to D−1 through the spanning
/// tree), triangles are *local*: only three IGs participate per cycle, so
/// errors don't accumulate over long tree paths. Recommended for the
/// "reliable region" gate on phase-linked stacks where short-baseline
/// triangles are the natural redundancy structure.
///
/// Returns shape (m, n); each value is the per-pixel max |K| over all
/// triangles that pass through this pixel. K = 0 means every triangle
/// agrees on its integer ambiguities here.
pub fn quality_from_triangles(unw_stack: ArrayView3<f32>, graph: &TemporalGraph) -> Array2<u16> {
    let (n_edges, m, n) = unw_stack.dim();
    assert_eq!(n_edges, graph.edges.len());

    // Enumerate triangles. For each ordered triple (a, b, c) with a<b<c we
    // need edges (a,b), (b,c), and (a,c). Each `graph.edges[e]` is a directed
    // edge `from → to` and `unw_stack[e]` represents `θ(to) − θ(from)`; the
    // sign of its contribution to a cycle depends on whether it was stored
    // in the canonical (low → high) or reversed (high → low) direction.
    //
    // Lookup yields `(edge_index, +1)` if stored canonically, `(edge_index,
    // −1)` if reversed. Reading `s[e] = sign · unw_stack[e]` then always
    // gives `θ(high) − θ(low)`, so the cycle sum `s_ab + s_bc − s_ac` is
    // closed (mod 2π) regardless of how dolphin chose to orient each edge.
    use std::collections::HashMap;
    let mut edge_lookup: HashMap<(u32, u32), (usize, i8)> = HashMap::new();
    for (idx, e) in graph.edges.iter().enumerate() {
        if e.from < e.to {
            edge_lookup.insert((e.from, e.to), (idx, 1));
        } else {
            edge_lookup.insert((e.to, e.from), (idx, -1));
        }
    }
    let mut triangles: Vec<(usize, i8, usize, i8, usize, i8)> = Vec::new();
    let d = graph.n_dates as u32;
    for a in 0..d {
        for b in (a + 1)..d {
            let Some(&(e_ab, s_ab)) = edge_lookup.get(&(a, b)) else {
                continue;
            };
            for c in (b + 1)..d {
                let Some(&(e_bc, s_bc)) = edge_lookup.get(&(b, c)) else {
                    continue;
                };
                let Some(&(e_ac, s_ac)) = edge_lookup.get(&(a, c)) else {
                    continue;
                };
                triangles.push((e_ab, s_ab, e_bc, s_bc, e_ac, s_ac));
            }
        }
    }

    let mut out = Array2::<u16>::zeros((m, n));
    if triangles.is_empty() {
        return out;
    }

    const STRIPE_H: usize = 64;
    for stripe_start in (0..m).step_by(STRIPE_H) {
        let stripe_end = (stripe_start + STRIPE_H).min(m);
        let rows: Vec<Vec<u16>> = (stripe_start..stripe_end)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0_u16; n];
                for j in 0..n {
                    let mut max_k: u16 = 0;
                    for &(e_ab, s_ab, e_bc, s_bc, e_ac, s_ac) in &triangles {
                        let s = (s_ab as f32) * unw_stack[(e_ab, i, j)]
                            + (s_bc as f32) * unw_stack[(e_bc, i, j)]
                            - (s_ac as f32) * unw_stack[(e_ac, i, j)];
                        let k_abs = (s / TAU).round().abs() as u32;
                        let k_u16 = k_abs.min(u16::MAX as u32) as u16;
                        if k_u16 > max_k {
                            max_k = k_u16;
                        }
                    }
                    row[j] = max_k;
                }
                row
            })
            .collect();
        for (offset, row) in rows.into_iter().enumerate() {
            let i = stripe_start + offset;
            for j in 0..n {
                out[(i, j)] = row[j];
            }
        }
    }
    out
}

#[cfg(test)]
mod quality_tests {
    use super::*;
    use ndarray::Array3;

    fn make_triangle_graph() -> TemporalGraph {
        TemporalGraph {
            n_dates: 3,
            edges: vec![
                Edge { from: 0, to: 1 },
                Edge { from: 1, to: 2 },
                Edge { from: 0, to: 2 },
            ],
            reference: 0,
        }
    }

    #[test]
    fn quality_zero_on_consistent_stack() {
        // A triangle (0→1, 1→2, 0→2) with ψ_0→1 + ψ_1→2 - ψ_0→2 = 0 → K=0 everywhere.
        let g = make_triangle_graph();
        let m = 4;
        let n = 4;
        let mut stack = Array3::<f32>::zeros((3, m, n));
        for i in 0..m {
            for j in 0..n {
                let a = 0.3 * i as f32;
                let b = 0.7 * j as f32;
                let c = a + b;
                stack[(0, i, j)] = a;
                stack[(1, i, j)] = b;
                stack[(2, i, j)] = c;
            }
        }
        let q = quality_max_integer_cycles(stack.view(), &g, None);
        assert!(
            q.iter().all(|&v| v == 0),
            "consistent stack should give K=0 everywhere"
        );
    }

    #[test]
    fn quality_detects_integer_mismatch() {
        // Plant a +2π on edge 2 (the non-tree edge in a sensibly-chosen tree)
        // at one pixel; that pixel's K should be exactly 1.
        let g = make_triangle_graph();
        let m = 4;
        let n = 4;
        let mut stack = Array3::<f32>::zeros((3, m, n));
        for i in 0..m {
            for j in 0..n {
                let a = 0.1 * i as f32;
                let b = 0.2 * j as f32;
                stack[(0, i, j)] = a;
                stack[(1, i, j)] = b;
                stack[(2, i, j)] = a + b;
            }
        }
        // Plant the mismatch at (2, 1).
        stack[(2, 2, 1)] += TAU;
        let q = quality_max_integer_cycles(stack.view(), &g, None);
        assert_eq!(q[(2, 1)], 1, "planted +2π should yield K=1");
        // Other pixels unchanged.
        for i in 0..m {
            for j in 0..n {
                if (i, j) != (2, 1) {
                    assert_eq!(q[(i, j)], 0, "unplanted pixel ({i},{j}) should be 0");
                }
            }
        }
    }

    #[test]
    fn quality_triangles_detect_mismatch() {
        let g = make_triangle_graph();
        let m = 4;
        let n = 4;
        let mut stack = Array3::<f32>::zeros((3, m, n));
        for i in 0..m {
            for j in 0..n {
                let a = 0.1 * i as f32;
                let b = 0.2 * j as f32;
                stack[(0, i, j)] = a; // 0→1
                stack[(1, i, j)] = b; // 1→2
                stack[(2, i, j)] = a + b; // 0→2
            }
        }
        // Plant +2π on edge 0 (0→1) at (1, 2). Triangle sum becomes ±2π.
        stack[(0, 1, 2)] += TAU;
        let q = quality_from_triangles(stack.view(), &g);
        assert_eq!(q[(1, 2)], 1, "planted +2π on (0→1) should yield K=1");
        for i in 0..m {
            for j in 0..n {
                if (i, j) != (1, 2) {
                    assert_eq!(q[(i, j)], 0, "unplanted pixel ({i},{j}) should be 0");
                }
            }
        }
    }

    /// Regression: graphs with edges stored "high date → low date" are valid
    /// dolphin output and must yield the same triangle closure as the
    /// canonical orientation. Earlier the lookup canonicalized the key but
    /// not the sign of the stack value, producing false K≠0 readings.
    #[test]
    fn quality_triangles_respect_reversed_edges() {
        // Same logical (0,1,2) triangle but edge 1 stored as 2→1 (reversed).
        let g = TemporalGraph {
            n_dates: 3,
            edges: vec![
                Edge { from: 0, to: 1 },
                Edge { from: 2, to: 1 }, // reversed: stack[1] = θ(1) - θ(2)
                Edge { from: 0, to: 2 },
            ],
            reference: 0,
        };
        let m = 4;
        let n = 4;
        let mut stack = Array3::<f32>::zeros((3, m, n));
        for i in 0..m {
            for j in 0..n {
                let theta1 = 0.1 * i as f32;
                let theta2 = theta1 + 0.2 * j as f32;
                stack[(0, i, j)] = theta1; // θ(1) - θ(0); θ(0)=0
                stack[(1, i, j)] = theta1 - theta2; // θ(1) - θ(2) (reversed)
                stack[(2, i, j)] = theta2; // θ(2) - θ(0)
            }
        }
        // Consistent stack ⇒ closure must be 0 even with reversed edge.
        let q = quality_from_triangles(stack.view(), &g);
        assert!(
            q.iter().all(|&v| v == 0),
            "consistent stack with reversed edge should give K=0 everywhere"
        );
        // Now plant +2π on the reversed edge; closure should detect K=1.
        stack[(1, 2, 1)] += TAU;
        let q2 = quality_from_triangles(stack.view(), &g);
        assert_eq!(q2[(2, 1)], 1, "+2π on reversed edge should yield K=1");
    }
}

// =========================================================================
// Cycle-greedy MCF refinement (Joint 3D MCF, MVP)
// =========================================================================
//
// The tree-based corrector forces k_tree ≡ 0. That's the limitation it has;
// the joint MCF removes it. Here we run a per-pixel greedy minimum-cost flow
// on the fundamental cycle basis: each cycle with a nonzero integer closure
// residue routes its correction to the *highest-variance* (noisiest) edge in
// the cycle — i.e. the edge for which absorbing an integer ambiguity is
// cheapest under the L2-weighted-flow cost ∑ w_e · k_e² with w_e = 1/σ²_e.
//
// This is a heuristic, not provably optimal. But for sparse integer demands
// (the typical case after a good tree-based pass), it lands on or near the
// MCF optimum. Real LAMBDA / closest-vector-in-lattice can replace this if
// pathological pixels surface.

pub struct RefineOutput {
    /// Refined unwrapped stack (n_edges, m, n).
    pub corrected: Array3<f32>,
    /// Total integer corrections (n_edges, m, n) — relative to the *input*
    /// stack (i.e. additive on top of whatever the caller passed in).
    pub corrections: Array3<i16>,
    /// Per-pixel count of cycles still violated after refinement. Should be
    /// zero on convergence; nonzero on pathological pixels.
    pub residual_violations: Array2<u16>,
    /// Max iterations of cycle resolution actually used at each pixel.
    pub iterations: Array2<u8>,
}

/// Refine an already-unwrapped stack via greedy cycle MCF.
///
/// * `unw_stack`     — starting point, shape (E, m, n). Usually the output of
///   [`correct`], but raw per-IG unwraps work too.
/// * `graph`         — temporal graph, must match `unw_stack`'s edge layout.
/// * `crlb_per_date` — per-acquisition CRLB σ²_d(p), shape (D, m, n) in rad².
/// * `tree_edge_priority` — same as in [`correct`]. Used only to pick the
///   spanning tree that defines the cycle basis; the choice of tree does
///   not affect convergence as long as it's valid, but a lowest-variance
///   tree tends to give a basis where most cycles are already closed.
/// * `max_iter`      — cap on greedy iterations per pixel. 32 is plenty.
pub fn refine_mcf(
    unw_stack: ArrayView3<f32>,
    graph: &TemporalGraph,
    crlb_per_date: ArrayView3<f32>,
    tree_edge_priority: Option<&[f32]>,
    max_iter: u8,
) -> RefineOutput {
    let (n_edges, m, n) = unw_stack.dim();
    assert_eq!(n_edges, graph.edges.len(), "stack/graph edge mismatch");
    assert_eq!(
        crlb_per_date.dim().0,
        graph.n_dates,
        "CRLB date axis mismatch"
    );
    assert_eq!(crlb_per_date.dim().1, m);
    assert_eq!(crlb_per_date.dim().2, n);

    let tree = build_spanning_tree(graph, tree_edge_priority);
    let basis = build_cycle_basis(graph, &tree);

    // Build edge→cycle inverted index for efficient "update coupled cycles"
    // when a routed flow lands on an edge shared by multiple cycles.
    let mut edge_to_cycles: Vec<Vec<(u32, i8)>> = vec![Vec::new(); n_edges];
    for c in 0..basis.num_cycles() {
        for step in basis.cycle(c) {
            edge_to_cycles[step.edge_idx as usize].push((c as u32, step.sign));
        }
    }

    let mut corrected = Array3::<f32>::zeros((n_edges, m, n));
    let mut corrections = Array3::<i16>::zeros((n_edges, m, n));
    let mut residual_violations = Array2::<u16>::zeros((m, n));
    let mut iterations = Array2::<u8>::zeros((m, n));

    // Stripe-based: see closure::correct for rationale.
    const STRIPE_H: usize = 64;
    for stripe_start in (0..m).step_by(STRIPE_H) {
        let stripe_end = (stripe_start + STRIPE_H).min(m);
        let rows: Vec<RefineRow> = (stripe_start..stripe_end)
            .into_par_iter()
            .map(|i| {
                refine_row(
                    unw_stack,
                    i,
                    graph,
                    crlb_per_date,
                    &basis,
                    &edge_to_cycles,
                    max_iter,
                    n_edges,
                    n,
                )
            })
            .collect();
        for (offset, row) in rows.into_iter().enumerate() {
            let i = stripe_start + offset;
            for j in 0..n {
                for e in 0..n_edges {
                    corrected[(e, i, j)] = row.corrected[e * n + j];
                    corrections[(e, i, j)] = row.corrections[e * n + j];
                }
                residual_violations[(i, j)] = row.violations[j];
                iterations[(i, j)] = row.iters[j];
            }
        }
    }

    RefineOutput {
        corrected,
        corrections,
        residual_violations,
        iterations,
    }
}

struct RefineRow {
    corrected: Vec<f32>,   // (E, n)
    corrections: Vec<i16>, // (E, n)
    violations: Vec<u16>,  // (n,)
    iters: Vec<u8>,        // (n,)
}

fn refine_row(
    unw_stack: ArrayView3<f32>,
    i: usize,
    graph: &TemporalGraph,
    crlb_per_date: ArrayView3<f32>,
    basis: &CycleBasis,
    edge_to_cycles: &[Vec<(u32, i8)>],
    max_iter: u8,
    n_edges: usize,
    n: usize,
) -> RefineRow {
    let mut corrected = vec![0.0_f32; n_edges * n];
    let mut corrections = vec![0_i16; n_edges * n];
    let mut violations = vec![0_u16; n];
    let mut iters = vec![0_u8; n];

    // Per-pixel scratch (reused across columns in this row).
    let mut y = vec![0.0_f32; n_edges];
    let mut k = vec![0_i32; n_edges];
    let mut sigma2 = vec![0.0_f32; n_edges]; // per-edge IG variance
    let mut demand = vec![0_i32; basis.num_cycles()]; // residue per cycle / 2π, rounded

    for j in 0..n {
        // Initialise y from the input stack and σ²_e from per-date CRLB.
        for e in 0..n_edges {
            y[e] = unw_stack[(e, i, j)];
            k[e] = 0;
            let edge = graph.edges[e];
            sigma2[e] =
                crlb_per_date[(edge.from as usize, i, j)] + crlb_per_date[(edge.to as usize, i, j)];
        }

        // Compute initial demands per cycle.
        for c in 0..basis.num_cycles() {
            let mut s = 0.0_f32;
            for step in basis.cycle(c) {
                s += (step.sign as f32) * y[step.edge_idx as usize];
            }
            demand[c] = (s / TAU).round() as i32;
        }

        // Greedy iteration.
        let mut it: u8 = 0;
        while it < max_iter {
            // Pick the cycle with largest |demand|.
            let mut pick_c: Option<usize> = None;
            let mut pick_abs: i32 = 0;
            for (c, &d) in demand.iter().enumerate() {
                let a = d.abs();
                if a > pick_abs {
                    pick_abs = a;
                    pick_c = Some(c);
                }
            }
            let Some(c) = pick_c else { break };

            // For this cycle, pick the edge with maximum σ²_e (noisiest =
            // cheapest to absorb correction under L2 cost).
            let mut best_step_idx: usize = 0;
            let mut best_sigma2: f32 = f32::NEG_INFINITY;
            let cycle_steps = basis.cycle(c);
            for (idx, step) in cycle_steps.iter().enumerate() {
                let s2 = sigma2[step.edge_idx as usize];
                if s2 > best_sigma2 {
                    best_sigma2 = s2;
                    best_step_idx = idx;
                }
            }
            let routed_step = cycle_steps[best_step_idx];
            // Demand d means the cycle sum is +2π·d; to zero it out we want
            // to subtract +2π·d from the cycle sum. If the routed step's sign
            // in the cycle is +1, we *increase* k_e by d (so y_e ← y_e − 2π·d
            // changes the contribution to the cycle sum by −2π·d). If sign is
            // −1, we decrease k_e by d.
            let routed_k: i32 = demand[c] * (routed_step.sign as i32);
            let e = routed_step.edge_idx as usize;
            y[e] -= TAU * (routed_k as f32);
            k[e] += routed_k;

            // Update demands of every cycle that touches edge e.
            for &(c_other, sign_other) in &edge_to_cycles[e] {
                let co = c_other as usize;
                demand[co] -= routed_k * (sign_other as i32);
            }
            it += 1;
        }

        // Tally remaining violations & write outputs.
        let mut viol: u16 = 0;
        for &d in &demand {
            if d != 0 {
                viol += 1;
            }
        }
        violations[j] = viol;
        iters[j] = it;
        for e in 0..n_edges {
            corrected[e * n + j] = y[e];
            // Clamp k into i16. With max_iter = 32 and per-cycle |demand| ≤
            // small ints, |k_e| ≪ 32k in practice.
            let kc = k[e].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            corrections[e * n + j] = kc;
        }
    }

    RefineRow {
        corrected,
        corrections,
        violations,
        iters,
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
        let theta = [
            0.0_f32,
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_2,
        ];
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
        let total_k: i32 = out.corrections[(0, 0, 0)] as i32 + out.corrections[(1, 0, 0)] as i32
            - out.corrections[(2, 0, 0)] as i32;
        assert_eq!(
            total_k.abs(),
            1,
            "total cycle correction magnitude should be 1, got {total_k}"
        );
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
        psi[3] += TAU; // (1,2) +1 cycle
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

    /// Cycle-greedy MCF should recover a +1-cycle injection on a TREE edge —
    /// exactly the case the tree-based corrector cannot fix.
    #[test]
    fn mcf_recovers_tree_edge_error() {
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
        // With natural-order priority, the tree picks edges 0, 1, 2 (the
        // three star edges from reference). Inject +1 cycle on TREE edge 1.
        psi[1] += TAU;

        let mut stack = Array3::<f32>::zeros((6, 1, 1));
        for e in 0..6 {
            stack[(e, 0, 0)] = psi[e];
        }
        // Construct a CRLB cube where edge 1 (between dates 0 and 2) has the
        // higher per-date variance — so it's correctly identified as the
        // noisiest edge to absorb a correction. Reverse the natural priority:
        // dates 0, 2 are noisier so edge 1 is noisiest; dates 1, 3 are quiet.
        let mut crlb = Array3::<f32>::zeros((4, 1, 1));
        crlb[(0, 0, 0)] = 0.5;
        crlb[(1, 0, 0)] = 0.05;
        crlb[(2, 0, 0)] = 0.5;
        crlb[(3, 0, 0)] = 0.05;

        // Tree priority based on summed variance (matches what unwrap_stack.py does).
        let mut prio = Vec::with_capacity(6);
        for e in &graph.edges {
            prio.push(crlb[(e.from as usize, 0, 0)] + crlb[(e.to as usize, 0, 0)]);
        }
        let out = refine_mcf(stack.view(), &graph, crlb.view(), Some(&prio), 16);
        // After MCF, the +1 cycle on edge 1 should be removed.
        assert_eq!(
            out.residual_violations[(0, 0)],
            0,
            "all cycles should close"
        );
        // Sign convention: corrected = original − 2π·k. Injection was +2π, so
        // the correction that removes it is k = +1.
        assert_eq!(
            out.corrections[(1, 0, 0)],
            1,
            "MCF should record k=+1 on edge 1, got {}",
            out.corrections[(1, 0, 0)]
        );
        // The corrected value of edge 1 should now match the truth.
        assert!((out.corrected[(1, 0, 0)] - (theta[2] - theta[0])).abs() < 1e-4);
    }

    /// Cycle basis size matches E - (D-1).
    #[test]
    fn cycle_basis_size_matches() {
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
        let tree = build_spanning_tree(&graph, None);
        let basis = build_cycle_basis(&graph, &tree);
        assert_eq!(basis.num_cycles(), graph.edges.len() - (graph.n_dates - 1));
        // Each cycle's edges sum to a closed loop — verify by traversing.
        // We just check each cycle has at least 3 edges (triangle minimum).
        for c in 0..basis.num_cycles() {
            assert!(basis.cycle(c).len() >= 3);
        }
    }

    /// With per-edge priority (e.g. CRLB), the spanning tree should pick the
    /// cheapest edges. We feed a 4-date network with one cheap edge per date
    /// and confirm the tree uses them.
    #[test]
    fn priority_drives_tree_selection() {
        let graph = TemporalGraph::new(
            4,
            vec![
                Edge { from: 0, to: 1 }, // edge 0  — cheap
                Edge { from: 0, to: 2 }, // edge 1  — expensive
                Edge { from: 0, to: 3 }, // edge 2  — expensive
                Edge { from: 1, to: 2 }, // edge 3  — cheap
                Edge { from: 1, to: 3 }, // edge 4  — expensive
                Edge { from: 2, to: 3 }, // edge 5  — cheap
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
