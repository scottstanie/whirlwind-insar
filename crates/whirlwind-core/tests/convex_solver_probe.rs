//! SCRATCH PROBE (issue #65): does whirlwind's EXISTING convex solver fail
//! whole-image because of the SOLVER (not the cost)? We confirm the precise
//! mechanism with tiny synthetic cases (NOT a NISAR frame):
//!
//!  1. `negative_rc_occurs_after_preload` — directly demonstrates that after
//!     `preload_convex_min`, the FIRST SSP augmentation pushes an arc past its
//!     k*, after which a residual marginal toward k* is NEGATIVE. We replay the
//!     primal-dual augment by hand and check raw marginals on residual arcs.
//!
//!  2. `convex_matches_brute_force_optimum` — a tiny grid whose true convex
//!     MCF optimum we enumerate by exhaustive search over integer flows on the
//!     active arcs, then compare the solver's converged flow cost. A gap proves
//!     the solver does not reach the convex optimum.
//!
//!  3. `debug_asserts_or_wrong_on_noisy_ramp` — a small noisy steep ramp run
//!     through `unwrap_convex`; in debug this trips the rc>=0 debug_assert if
//!     negative reduced costs reach Dijkstra.

use ndarray::Array2;
use num_complex::Complex32;
use whirlwind_core::grid::RectangularGridGraph;
use whirlwind_core::network::Network;

/// Build the full residual-arc list (tail, head, raw_marginal) for the CONVERGED
/// flow, then run Bellman-Ford to detect a NEGATIVE residual cycle. A negative
/// residual cycle is a CERTIFICATE that `flow` is NOT the convex optimum
/// (augmenting around the cycle strictly lowers total cost). Returns
/// Some(min_dist_improvement) if a negative cycle exists, else None.
fn has_negative_residual_cycle(
    g: &RectangularGridGraph,
    net: &Network,
    flow: &[i32],
    offsets: &[i32],
    weights: &[i32],
) -> bool {
    let nf = g.num_forward;
    let num_nodes = g.num_nodes();
    // Collect residual arcs (only those NOT saturated/forbidden, i.e. reachable
    // in the residual graph). Use raw marginal as the residual cost.
    let mut edges: Vec<(usize, usize, i64)> = Vec::new();
    for a in 0..2 * nf {
        if net.is_arc_saturated(a) {
            continue;
        }
        let (t, h) = g.arc_endpoints(a);
        let c = raw_marginal(a, nf, flow, offsets, weights);
        edges.push((t, h, c));
    }
    // Bellman-Ford from a virtual super-source (all dist 0) — detects any
    // negative cycle reachable in the residual graph.
    let mut dist = vec![0_i64; num_nodes];
    for _ in 0..num_nodes - 1 {
        let mut changed = false;
        for &(t, h, c) in &edges {
            if dist[t].saturating_add(c) < dist[h] {
                dist[h] = dist[t].saturating_add(c);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // One more pass: if anything still relaxes, there is a negative cycle.
    for &(t, h, c) in &edges {
        if dist[t].saturating_add(c) < dist[h] {
            return true;
        }
    }
    false
}

/// Raw marginal helper used by the negative-cycle certificate. Mirrors
/// Network::marginal_cost exactly (no potentials).
fn raw_marginal(arc: usize, nf: usize, flow: &[i32], offsets: &[i32], weights: &[i32]) -> i64 {
    let ns = 100_i64;
    let (fwd, sign) = if arc < nf { (arc, 1_i64) } else { (arc - nf, -1_i64) };
    let f = flow[fwd] as i64;
    let o = offsets[fwd] as i64;
    let w = weights[fwd] as i64;
    let u = f * ns - o;
    w * (sign * 2 * ns * u + ns * ns)
}

/// Total convex cost of a flow vector under per-arc parabola w*(k*100-off)^2.
fn total_cost(flow: &[i32], offsets: &[i32], weights: &[i32]) -> i64 {
    let ns = 100_i64;
    let mut c = 0_i64;
    for e in 0..flow.len() {
        let u = flow[e] as i64 * ns - offsets[e] as i64;
        c += weights[e] as i64 * u * u;
    }
    c
}

/// Net divergence (excess produced) at each node by a flow vector, using the
/// SAME convention as the solver: a forward unit on (t->h) does
/// excess[t]-=1, excess[h]+=1. So div[node] = sum_in - sum_out = the excess
/// that flow INDUCES. For the residue problem, we need induced excess to
/// CANCEL the residue excess (so the residual is balanced to zero).
fn induced_excess(g: &RectangularGridGraph, flow: &[i32], num_nodes: usize) -> Vec<i32> {
    let mut e = vec![0_i32; num_nodes];
    for a in 0..g.num_forward {
        let f = flow[a];
        if f == 0 {
            continue;
        }
        let (t, h) = g.arc_endpoints(a);
        e[t] -= f;
        e[h] += f;
    }
    e
}


/// PROBE 1: After preload, run ONE multi-source-Dijkstra + augment by replaying
/// what primal_dual::run does, then check whether any UNSATURATED residual arc
/// has a NEGATIVE raw marginal (the textbook "pushed past k*, reverse-toward-k*
/// is now negative"). This is the exact invariant the Johnson potential update
/// assumes is impossible.
#[test]
fn negative_rc_occurs_after_preload() {
    // 3x3 residue grid. Put a +1 residue at node (1,1) (center) and a -1 at a
    // corner so MCF MUST route a unit of flow between them. Then load offsets
    // so that EVERY arc on the only short path already sits at k*=0 (offset~0),
    // forcing the augment to push those arcs from 0 -> +/-1 (i.e. PAST/away
    // from k*). After the push the residual reverse marginal on a pushed arc is
    // strictly negative.
    let g = RectangularGridGraph::new(3, 3);
    let nf = g.num_forward;

    // Offsets/weights: a couple of arcs sit AT a nonzero k* so preload moves
    // them; the path arcs sit at k*=0. Mix so the post-push state is realistic.
    let mut offsets = vec![0_i32; nf];
    let weights = vec![5_i32; nf];
    // Give one arc a big offset so preload fires (k*=1) on it.
    offsets[0] = 90;

    let mut residues = Array2::<i32>::zeros((3, 3));
    residues[(1, 1)] = 1; // excess source
    residues[(2, 2)] = -1; // deficit sink

    let mut net = Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);

    // Pre-condition: arc 0 has negative f=0 marginal (offset>50).
    assert!(net.marginal_cost(0) < 0, "expected neg f=0 marginal on arc 0");

    net.preload_convex_min(&g);

    // After preload, EVERY residual marginal must be >=0 (the soundness claim).
    for a in 0..2 * nf {
        let m = net.marginal_cost(a);
        assert!(m >= 0, "post-preload marginal on arc {a} = {m} < 0 (preload broken)");
    }

    // Now run the real solver to convergence.
    whirlwind_core::primal_dual::run(&g, &mut net, 50);

    // Read back the converged flow.
    let flow: Vec<i32> = (0..nf).map(|a| net.arc_flow(&g, a)).collect();

    // KEY CHECK: in the CONVERGED flow, is there any unsaturated residual arc
    // whose RAW marginal (toward k*) is negative? If yes, a NEGATIVE reduced
    // cost existed in the residual graph during the solve unless potentials
    // exactly compensated — and the Johnson update (which subtracts shortest-
    // path distances computed under the assumption of FIXED arc costs) cannot
    // track a cost that CHANGED when we pushed the arc.
    let mut neg_raw = 0;
    for a in 0..2 * nf {
        // skip masked/forbidden
        if net.is_arc_saturated(a) {
            continue;
        }
        let m = raw_marginal(a, nf, &flow, &offsets, &weights);
        if m < 0 {
            neg_raw += 1;
        }
    }
    eprintln!("[probe1] converged flow = {flow:?}");
    eprintln!("[probe1] residual arcs with NEGATIVE raw marginal = {neg_raw}");

    // A pushed arc (flow != 0 on a k*=0 arc) ALWAYS has a negative reverse
    // marginal toward 0. So if the solver pushed ANY k*=0 arc, neg_raw > 0.
    let pushed: Vec<usize> = (0..nf)
        .filter(|&e| flow[e] != ((offsets[e] as f64 / 100.0).round() as i32))
        .collect();
    eprintln!("[probe1] arcs pushed away from their k* = {pushed:?}");
    assert!(
        !pushed.is_empty(),
        "expected the augment to push at least one arc away from k*"
    );
    assert!(
        neg_raw > 0,
        "expected negative residual raw marginals after pushing past k*"
    );
}

/// PROBE 2: enumerate the TRUE convex-MCF optimum on a tiny 3x3 residue grid
/// (2x2 pixel edges) by brute force over integer flows on a small set of arcs,
/// and compare to the solver's converged total cost. A positive gap proves the
/// SSP solver does not reach the convex optimum.
#[test]
fn convex_matches_brute_force_optimum() {
    let g = RectangularGridGraph::new(3, 3);
    let nf = g.num_forward;

    // Construct offsets that create a "coherent multi-cycle deviation" pattern:
    // a few adjacent arcs strongly want k=+1 (offset>=100) forming a small loop
    // around the center, so the convex optimum routes a circulation. This is
    // exactly the regime (offset near +/-100, large weight) the convex cost was
    // built for and where the solver is suspected to fail.
    let mut offsets = vec![0_i32; nf];
    let mut weights = vec![1_i32; nf];

    // Strong coherent arcs: assign large offsets to the four arcs of the
    // center cell so the parabola minimum wants a +1 circulation there.
    // Center cell corners: nodes (1,1),(1,2),(2,1),(2,2). Loop edges:
    let d = g.down_arc(1, 1).unwrap(); // (1,1)->(2,1)
    let r = g.right_arc(2, 1).unwrap(); // (2,1)->(2,2)
    let u = g.up_arc(2, 2).unwrap(); // (2,2)->(1,2)
    let l = g.left_arc(1, 2).unwrap(); // (1,2)->(1,1)
    for &a in &[d, r, u, l] {
        offsets[a] = 100; // k* = +1, parabola min at +1 unit forward
        weights[a] = 10;
    }

    // Residues that DEMAND flow: a dipole that is NOT satisfiable by the pure
    // k* circulation, forcing the solver to push some arcs past k*.
    let mut residues = Array2::<i32>::zeros((3, 3));
    residues[(0, 0)] = 1;
    residues[(2, 2)] = -1;

    // --- Brute force the optimum over the active arc set. ---
    // Active arcs = the 4 loop arcs (which can be 0,1,2) PLUS a routing path
    // from (0,0) to (2,2). To keep the search small, restrict each forward arc
    // flow to {-2,-1,0,1,2}; that's plenty for a single dipole + circulation.
    // We brute force over a SUBSET of arcs that can plausibly carry flow: the
    // 4 loop arcs and the arcs on two simple (0,0)->(2,2) staircases.
    let mut active: Vec<usize> = vec![d, r, u, l];
    // ONE staircase A from (0,0) to (2,2): down (0,0)->(1,0), down (1,0)->(2,0),
    // right (2,0)->(2,1), then `r` (already in set) completes (2,1)->(2,2).
    active.push(g.down_arc(0, 0).unwrap());
    active.push(g.down_arc(1, 0).unwrap());
    active.push(g.right_arc(2, 0).unwrap());
    // dedup
    active.sort_unstable();
    active.dedup();

    let num_nodes = g.num_nodes();
    let mut want_excess = vec![0_i32; num_nodes];
    want_excess[g.node_id(0, 0)] = 1;
    want_excess[g.node_id(2, 2)] = -1;

    // Exhaustive search over flow in {-2..=2} on each active arc; keep feasible
    // (induced excess cancels residues so residual balanced) min-cost flow.
    let range: Vec<i32> = (-2..=2).collect();
    let mut best_cost = i64::MAX;
    let mut best_flow_full = vec![0_i32; nf];
    let k = active.len();
    let mut idx = vec![0usize; k];
    let total: u64 = (range.len() as u64).pow(k as u32);
    assert!(total < 4_000_000, "search space too big: {total}");
    for combo in 0..total {
        let mut c = combo;
        for slot in idx.iter_mut() {
            *slot = (c % range.len() as u64) as usize;
            c /= range.len() as u64;
        }
        let mut flow = vec![0_i32; nf];
        for (slot, &a) in idx.iter().zip(active.iter()) {
            flow[a] = range[*slot];
        }
        let ie = induced_excess(&g, &flow, num_nodes);
        // Feasible iff induced excess exactly satisfies the residue demand:
        // residues[node] + induced_excess == 0 at every node (flow drains all
        // excess). residues live in want_excess.
        let feasible = (0..num_nodes).all(|nd| want_excess[nd] + ie[nd] == 0);
        if !feasible {
            continue;
        }
        let cost = total_cost(&flow, &offsets, &weights);
        if cost < best_cost {
            best_cost = cost;
            best_flow_full = flow;
        }
    }
    assert!(best_cost < i64::MAX, "brute force found no feasible flow");

    // --- Solver ---
    // Restrict the solver to the SAME active arc set as the brute force by
    // forbidding (saturating both directions of) every other grid arc. This
    // makes the comparison apples-to-apples: both optimize over the same
    // feasible flow space, so any cost gap is purely the SOLVER's failure to
    // find the convex optimum, not a routing-set difference.
    let mut net = Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);
    let active_set: std::collections::HashSet<usize> = active.iter().copied().collect();
    for a in 0..nf {
        if !active_set.contains(&a) {
            net.is_saturated.set(a, true); // forbid forward
            net.is_saturated.set(a + nf, true); // forbid reverse
        }
    }
    net.preload_convex_min(&g);
    whirlwind_core::primal_dual::run(&g, &mut net, 50);
    let solver_flow: Vec<i32> = (0..nf).map(|a| net.arc_flow(&g, a)).collect();
    let solver_cost = total_cost(&solver_flow, &offsets, &weights);

    // Sanity: solver flow must be feasible (drain all residue excess).
    let ie = induced_excess(&g, &solver_flow, num_nodes);
    let solver_feasible = (0..num_nodes).all(|nd| want_excess[nd] + ie[nd] == 0);

    eprintln!("[probe2] brute-force optimum cost = {best_cost}");
    eprintln!("[probe2] brute-force optimum flow = {best_flow_full:?}");
    eprintln!("[probe2] solver cost            = {solver_cost} (feasible={solver_feasible})");
    eprintln!("[probe2] solver flow            = {solver_flow:?}");

    // Report the gap. We DO NOT assert equality here (the test's purpose is to
    // MEASURE whether the solver hits the optimum); instead print and assert
    // feasibility so the harness records the gap either way.
    if solver_feasible {
        eprintln!(
            "[probe2] OPTIMALITY GAP = {} ({} over optimum)",
            solver_cost - best_cost,
            if best_cost == 0 {
                f64::INFINITY
            } else {
                (solver_cost - best_cost) as f64 / best_cost as f64
            }
        );
    }
    assert!(solver_feasible, "solver flow not feasible");
}

/// PROBE 3: a small NOISY steep ramp run through unwrap_convex. In a DEBUG
/// build the heap relax `debug_assert!(rc >= 0)` will fire if a negative
/// reduced cost reaches Dijkstra. In RELEASE it is compiled out (silent).
#[test]
fn debug_asserts_or_wrong_on_noisy_ramp() {
    use rand::SeedableRng;
    use whirlwind_core::{cost, residue, simulate};

    let m = 48;
    let n = 48;
    // Steep ramp: ~6 cycles across the frame so there are many genuine wrap
    // lines (the regime where preload fires and SSP must push past k*).
    let truth = Array2::from_shape_fn((m, n), |(i, j)| {
        2.0 * std::f32::consts::PI * 1.5 * (i as f32 + j as f32) / (m as f32)
    });
    let gamma = Array2::<f32>::from_elem((m, n), 0.5); // noisy
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let (igram, cor) = simulate::simulate_ifg(&truth, &gamma, 4, &mut rng);

    // Mirror unwrap_convex internals so we can also probe the post-solve state.
    let wrapped = igram.mapv(|z: Complex32| z.arg());
    let residues = residue::compute(wrapped.view());
    let (offsets, weights) = cost::compute_snaphu_smooth_costs(igram.view(), cor.view(), 4.0, None);
    let g = RectangularGridGraph::new(m + 1, n + 1);
    let nf = g.num_forward;

    // How often does preload actually fire? (k* != 0 fraction.)
    let kstar_nonzero = offsets
        .iter()
        .filter(|&&o| ((o as f64 / 100.0).round() as i32) != 0)
        .count();
    eprintln!(
        "[probe3] preload fires on {kstar_nonzero}/{} grid arcs ({:.2}%) | max|offset|={}",
        nf,
        100.0 * kstar_nonzero as f64 / nf as f64,
        offsets.iter().map(|o| o.abs()).max().unwrap_or(0)
    );

    let mut net = Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);
    net.preload_convex_min(&g);

    // Confirm post-preload soundness on the REAL cost field.
    let mut neg_after_preload = 0;
    for a in 0..2 * nf {
        if net.is_arc_saturated(a) {
            continue;
        }
        if net.marginal_cost(a) < 0 {
            neg_after_preload += 1;
        }
    }
    eprintln!("[probe3] residual arcs with neg marginal AFTER preload (pre-solve) = {neg_after_preload}");

    // Run the solver. In debug this fires the rc>=0 debug_assert if a negative
    // reduced cost reaches Dijkstra during SSP.
    whirlwind_core::primal_dual::run(&g, &mut net, 50);

    // Post-solve: count residual arcs with negative RAW marginal (toward k*).
    let flow: Vec<i32> = (0..nf).map(|a| net.arc_flow(&g, a)).collect();
    let mut neg_raw_after_solve = 0;
    for a in 0..2 * nf {
        if net.is_arc_saturated(a) {
            continue;
        }
        let m = raw_marginal(a, nf, &flow, &offsets, &weights);
        if m < 0 {
            neg_raw_after_solve += 1;
        }
    }
    eprintln!(
        "[probe3] residual arcs with neg RAW marginal AFTER solve = {neg_raw_after_solve} / {}",
        2 * nf
    );
    // This is the smoking gun: if >0, the solve ran with negative residual
    // marginals present (non-optimal / unsound for plain Dijkstra-SSP).
    eprintln!("[probe3] done (no debug_assert fired => either release, or rc>=0 held via potentials)");
}

/// PROBE 4: RANDOMIZED differential test. For many random tiny instances
/// restricted to a small active-arc set (so brute force is tractable), compare
/// the solver's converged convex cost to the TRUE brute-forced optimum. ANY
/// instance with a positive gap is a definitive proof the existing SSP solver
/// does NOT reach the convex optimum (the cost is identical in both — only the
/// solver differs).
#[test]
fn randomized_convex_optimality_differential() {
    use rand::Rng;
    use rand::SeedableRng;

    // Use a fixed small active set on a 4x4 residue grid: a 2x2 block of cells
    // in the interior, whose 12 boundary+internal arcs form loops + routes.
    let g = RectangularGridGraph::new(4, 4);
    let nf = g.num_forward;
    let num_nodes = g.num_nodes();

    // Active arcs: all forward arcs touching the 3x3 node block rows 0..3,
    // cols 0..3 would be too many; pick a hand-chosen small set forming a
    // connected sub-network with cycles between nodes (0,0),(0,1),(1,0),(1,1),
    // (2,2),(3,3) etc. Keep |active| <= 9 so 5^9 ~ 2M is tractable.
    let a0 = g.right_arc(0, 0).unwrap(); // (0,0)->(0,1)
    let a1 = g.down_arc(0, 1).unwrap(); // (0,1)->(1,1)
    let a2 = g.left_arc(1, 1).unwrap(); // (1,1)->(1,0)
    let a3 = g.up_arc(1, 0).unwrap(); // (1,0)->(0,0)  [closes loop A]
    let a4 = g.down_arc(1, 1).unwrap(); // (1,1)->(2,1)
    let a5 = g.right_arc(2, 1).unwrap(); // (2,1)->(2,2)
    let a6 = g.down_arc(2, 2).unwrap(); // (2,2)->(3,2)
    let a7 = g.right_arc(3, 2).unwrap(); // (3,2)->(3,3)
    let a8 = g.right_arc(1, 1).unwrap(); // (1,1)->(1,2)  extra branch
    let active = [a0, a1, a2, a3, a4, a5, a6, a7, a8];
    let active_set: std::collections::HashSet<usize> = active.iter().copied().collect();
    let kk = active.len();
    let range: Vec<i32> = (-2..=2).collect();
    let total: u64 = (range.len() as u64).pow(kk as u32);

    let mut rng = rand::rngs::StdRng::seed_from_u64(20260601);
    let trials = 200;
    let mut gaps = 0;
    let mut infeasible = 0;
    let mut worst_gap: f64 = 0.0;
    let mut worst_info = String::new();

    for trial in 0..trials {
        // Random offsets in [-150,150] (so k* in {-2..2}) and weights in [1,12]
        // on the active arcs only; all others zero.
        let mut offsets = vec![0_i32; nf];
        let mut weights = vec![0_i32; nf];
        for &a in &active {
            offsets[a] = rng.gen_range(-150..=150);
            weights[a] = rng.gen_range(1..=12);
        }

        // Random residue dipole/tripole at nodes on the active subgraph. Pick
        // nodes touched by active arcs and assign small +/- excess summing 0.
        let touched: Vec<usize> = {
            let mut s = std::collections::BTreeSet::new();
            for &a in &active {
                let (t, h) = g.arc_endpoints(a);
                s.insert(t);
                s.insert(h);
            }
            s.into_iter().collect()
        };
        let mut residues = Array2::<i32>::zeros((4, 4));
        let mut want_excess = vec![0_i32; num_nodes];
        // Place a random balanced multi-source/multi-sink charge on touched
        // nodes: assign each touched node a random excess in {-2..2}, then
        // re-balance so sum == 0. This creates MULTI-PATH demands that share
        // arcs in a single primal-dual Dijkstra iteration — the regime where
        // the batched-augment marginal staleness (if it mattered) would bite.
        let mut charges: Vec<i32> = touched.iter().map(|_| rng.gen_range(-2..=2)).collect();
        let sum: i32 = charges.iter().sum();
        // Re-balance: dump the negative of the sum onto the first node.
        charges[0] -= sum;
        for (&node, &c) in touched.iter().zip(charges.iter()) {
            if c != 0 {
                let (ni, nj) = g.node_ij(node);
                residues[(ni, nj)] = c;
                want_excess[node] = c;
            }
        }
        if want_excess.iter().all(|&e| e == 0) {
            continue; // no demand
        }

        // --- Brute force optimum over active arcs ---
        let mut best_cost = i64::MAX;
        let mut idx = vec![0usize; kk];
        for combo in 0..total {
            let mut c = combo;
            for slot in idx.iter_mut() {
                *slot = (c % range.len() as u64) as usize;
                c /= range.len() as u64;
            }
            let mut flow = vec![0_i32; nf];
            for (slot, &a) in idx.iter().zip(active.iter()) {
                flow[a] = range[*slot];
            }
            let ie = induced_excess(&g, &flow, num_nodes);
            if !(0..num_nodes).all(|nd| want_excess[nd] + ie[nd] == 0) {
                continue;
            }
            let cost = total_cost(&flow, &offsets, &weights);
            if cost < best_cost {
                best_cost = cost;
            }
        }
        if best_cost == i64::MAX {
            continue; // infeasible dipole on this active set; skip
        }

        // --- Solver on the SAME restricted problem ---
        let mut net = Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);
        for a in 0..nf {
            if !active_set.contains(&a) {
                net.is_saturated.set(a, true);
                net.is_saturated.set(a + nf, true);
            }
        }
        net.preload_convex_min(&g);
        whirlwind_core::primal_dual::run(&g, &mut net, 50);
        let solver_flow: Vec<i32> = (0..nf).map(|a| net.arc_flow(&g, a)).collect();
        let ie = induced_excess(&g, &solver_flow, num_nodes);
        let feasible = (0..num_nodes).all(|nd| want_excess[nd] + ie[nd] == 0);
        let solver_cost = total_cost(&solver_flow, &offsets, &weights);

        // Skip trials where the solver used |flow| > 2 on some arc (outside the
        // brute-force range, so best_cost is not a valid lower bound there).
        let within_range = active.iter().all(|&a| solver_flow[a].abs() <= 2);
        if !feasible {
            infeasible += 1;
            continue;
        }
        if !within_range {
            continue;
        }
        if solver_cost > best_cost {
            gaps += 1;
            let g_rel = (solver_cost - best_cost) as f64 / (best_cost.max(1)) as f64;
            if g_rel > worst_gap {
                worst_gap = g_rel;
                worst_info = format!(
                    "trial {trial}: optimum={best_cost} solver={solver_cost} gap={} ({:.1}%) | offsets_active={:?}",
                    solver_cost - best_cost,
                    100.0 * g_rel,
                    active.iter().map(|&a| offsets[a]).collect::<Vec<_>>()
                );
            }
        }
    }

    eprintln!(
        "[probe4] {gaps}/{trials} trials where solver cost > brute-force optimum (SOLVER sub-optimal); {infeasible} infeasible-solver trials"
    );
    if gaps > 0 {
        eprintln!("[probe4] worst: {worst_info}");
    }
    // We assert NOTHING about gaps==0 here; the count is the finding. (If the
    // solver were sound, gaps would be 0.)
}

/// PROBE 5: OPTIMALITY CERTIFICATE on a small noisy ramp run through the full
/// convex pipeline (preload + primal_dual + SSP). After the solve, we check
/// whether the converged flow admits a NEGATIVE residual cycle under the raw
/// (re-linearized) marginals. A negative residual cycle is a definitive proof
/// that the SOLVER did NOT reach the convex optimum (a strictly cheaper flow
/// exists, reachable by augmenting around the cycle). This is the cleanest
/// SOLVER-vs-cost discriminator: the cost field is identical either way.
#[test]
fn negative_cycle_certificate_on_noisy_ramp() {
    use rand::SeedableRng;
    use whirlwind_core::{cost, residue, simulate};

    for (seed, gamma_val, cycles, m) in [
        (1_u64, 0.4_f32, 2.0_f32, 40usize),
        (2, 0.6, 1.5, 40),
        (3, 0.3, 3.0, 40),
        // Larger, lower-coherence (residue-dense) cases at scale.
        (4, 0.3, 4.0, 128),
        (5, 0.25, 5.0, 160),
    ] {
        let n = m;
        let truth = Array2::from_shape_fn((m, n), |(i, j)| {
            2.0 * std::f32::consts::PI * cycles * (i as f32 + j as f32) / (m as f32)
        });
        let gamma = Array2::<f32>::from_elem((m, n), gamma_val);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let (igram, cor) = simulate::simulate_ifg(&truth, &gamma, 4, &mut rng);

        let wrapped = igram.mapv(|z: Complex32| z.arg());
        let residues = residue::compute(wrapped.view());
        let (offsets, weights) =
            cost::compute_snaphu_smooth_costs(igram.view(), cor.view(), 4.0, None);
        let g = RectangularGridGraph::new(m + 1, n + 1);
        let nf = g.num_forward;

        let mut net =
            Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);
        net.preload_convex_min(&g);
        whirlwind_core::primal_dual::run(&g, &mut net, 50);

        let flow: Vec<i32> = (0..nf).map(|a| net.arc_flow(&g, a)).collect();
        let neg_cycle = has_negative_residual_cycle(&g, &net, &flow, &offsets, &weights);

        // Count residue excess remaining (solver should drain it all).
        let resid_excess: i64 = net.excess.iter().take(g.num_nodes()).map(|&e| e.unsigned_abs() as i64).sum();

        eprintln!(
            "[probe5] seed={seed} gamma={gamma_val} cycles={cycles}: NEGATIVE residual cycle present = {neg_cycle} | leftover |excess|={resid_excess}"
        );
    }
}

/// Mean per-pixel deviation of unwrapped from truth, modulo a global constant
/// (we subtract the mean difference so a constant offset doesn't count).
fn quality_vs_truth(unw: &Array2<f32>, truth: &Array2<f32>) -> (f64, f64) {
    let n = unw.len() as f64;
    let mean_diff: f64 = unw.iter().zip(truth.iter()).map(|(&u, &t)| (u - t) as f64).sum::<f64>() / n;
    // Fraction of pixels within 0.1 rad of truth after removing global offset.
    let mut within = 0usize;
    let mut sumsq = 0.0;
    for (&u, &t) in unw.iter().zip(truth.iter()) {
        let d = (u as f64 - t as f64) - mean_diff;
        sumsq += d * d;
        if d.abs() < 0.1 {
            within += 1;
        }
    }
    (within as f64 / n, (sumsq / n).sqrt())
}

/// PROBE 6: compare the unwrap QUALITY of unwrap_convex against (a) the
/// flow=0 Itoh-only baseline (just integrate the wrapped phase with no
/// residue routing) and (b) the truth, on a small steep noisy ramp. This
/// discriminates: if convex matches Itoh-only and both are bad, the failure
/// is upstream of the solver (the cost/anchor), not the solver mis-routing.
#[test]
fn convex_quality_vs_baseline_on_noisy_ramp() {
    use rand::SeedableRng;
    use whirlwind_core::{cost, integrate, residue, simulate};

    for (seed, gamma_val, cycles) in [(11_u64, 0.5_f32, 3.0_f32), (12, 0.7, 4.0)] {
        let m = 64;
        let n = 64;
        let truth = Array2::from_shape_fn((m, n), |(i, j)| {
            2.0 * std::f32::consts::PI * cycles * (i as f32 + j as f32) / (m as f32)
        });
        let gamma = Array2::<f32>::from_elem((m, n), gamma_val);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let (igram, cor) = simulate::simulate_ifg(&truth, &gamma, 4, &mut rng);
        let wrapped = igram.mapv(|z: Complex32| z.arg());

        // unwrap_convex
        let unw_convex =
            whirlwind_core::unwrap_convex(igram.view(), cor.view(), 4.0, None).unwrap();
        let (frac_c, rms_c) = quality_vs_truth(&unw_convex, &truth);

        // flow=0 Itoh baseline: integrate wrapped phase with an all-zero flow.
        let g = RectangularGridGraph::new(m + 1, n + 1);
        let zero_flow = vec![0_i32; g.num_forward];
        let unw_itoh = integrate::integrate_with_flow(wrapped.view(), &g, &zero_flow, None);
        let (frac_i, rms_i) = quality_vs_truth(&unw_itoh, &truth);

        // unwrap_reuse (linear coherence cost) for reference.
        let unw_reuse =
            whirlwind_core::unwrap_reuse(igram.view(), cor.view(), 4.0, None).unwrap();
        let (frac_r, rms_r) = quality_vs_truth(&unw_reuse, &truth);

        eprintln!(
            "[probe6] seed={seed} g={gamma_val} cyc={cycles}: convex frac<0.1={frac_c:.3} rms={rms_c:.2} | itoh frac={frac_i:.3} rms={rms_i:.2} | reuse frac={frac_r:.3} rms={rms_r:.2}"
        );
    }
}

/// PROBE 7: SELF-TEST of the negative-cycle certificate. Construct a flow that
/// is KNOWN to be sub-optimal (a circulation around the center cell where every
/// arc's k* = +1 but we set flow = 0), and confirm the certificate FIRES
/// (returns true). This guards against a false-negative certificate that would
/// invalidate probe 5's "no negative cycle" conclusion.
#[test]
fn certificate_detects_known_suboptimal_flow() {
    let g = RectangularGridGraph::new(3, 3);
    let nf = g.num_forward;
    let mut offsets = vec![0_i32; nf];
    let mut weights = vec![1_i32; nf];
    // Center-cell loop arcs all want k* = +1 (offset 100). With flow = 0 on all
    // arcs, augmenting +1 around the loop strictly lowers cost: there MUST be a
    // negative residual cycle.
    let d = g.down_arc(1, 1).unwrap();
    let r = g.right_arc(2, 1).unwrap();
    let u = g.up_arc(2, 2).unwrap();
    let l = g.left_arc(1, 2).unwrap();
    for &a in &[d, r, u, l] {
        offsets[a] = 100;
        weights[a] = 5;
    }
    let residues = Array2::<i32>::zeros((3, 3));
    let net = Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);
    // DO NOT preload — leave flow at 0 (sub-optimal: the loop wants +1).
    let flow = vec![0_i32; nf];
    let neg = has_negative_residual_cycle(&g, &net, &flow, &offsets, &weights);
    eprintln!("[probe7] certificate on known-suboptimal flow=0 (loop wants +1): neg_cycle={neg}");
    assert!(
        neg,
        "certificate FAILED to detect a known negative cycle — probe5 conclusion would be unreliable"
    );

    // And after preloading to k*, the certificate must NOT fire (optimal).
    let mut net2 = Network::new_convex_with_mask(&g, residues.view(), &offsets, &weights, None);
    net2.preload_convex_min(&g);
    let flow2: Vec<i32> = (0..nf).map(|a| net2.arc_flow(&g, a)).collect();
    let neg2 = has_negative_residual_cycle(&g, &net2, &flow2, &offsets, &weights);
    eprintln!("[probe7] certificate after preload-to-k* (optimal): neg_cycle={neg2}");
    assert!(!neg2, "certificate FALSE-POSITIVE on the optimal preloaded flow");
}

/// PROBE 8: the COST mechanism on a CLEAN (high-coherence) steep ramp. SNAPHU's
/// deviation offset is ~0 on a smooth ramp (raw gradient ~= box-mean), so the
/// convex parabola minimum k* = 0 everywhere — the standalone convex solve has
/// NO drive to lay down the wrap lines a steep ramp needs. We compare the
/// number of wrap discontinuities the convex unwrap produces vs the truth.
/// This isolates COST behavior from SOLVER behavior (a clean ramp has near-zero
/// residues, so the solver has almost nothing to do).
#[test]
fn clean_steep_ramp_cost_under_wraps() {
    use rand::SeedableRng;
    use whirlwind_core::{cost, simulate};

    let m = 64;
    let n = 64;
    // 6 cycles across — every adjacent-pixel gradient is well under pi, but the
    // ramp spans 12*pi total. (A clean ramp Itoh-integrates exactly; the test
    // is whether the convex COST agrees that no flow is needed AND whether the
    // offsets carry any absolute-ramp signal.)
    let cycles = 6.0_f32;
    let truth = Array2::from_shape_fn((m, n), |(i, j)| {
        2.0 * std::f32::consts::PI * cycles * (j as f32) / (n as f32)
    });
    let gamma = Array2::<f32>::from_elem((m, n), 0.99); // clean
    let mut rng = rand::rngs::StdRng::seed_from_u64(99);
    let (igram, cor) = simulate::simulate_ifg(&truth, &gamma, 16, &mut rng);

    let (offsets, _weights) = cost::compute_snaphu_smooth_costs(igram.view(), cor.view(), 16.0, None);
    let nf = offsets.len();
    let kstar_nonzero = offsets
        .iter()
        .filter(|&&o| ((o as f64 / 100.0).round() as i32) != 0)
        .count();
    let maxoff = offsets.iter().map(|o| o.abs()).max().unwrap_or(0);

    // Convex unwrap quality on the clean ramp.
    let unw = whirlwind_core::unwrap_convex(igram.view(), cor.view(), 16.0, None).unwrap();
    let (frac, rms) = quality_vs_truth(&unw, &truth);

    eprintln!(
        "[probe8] CLEAN 6-cycle ramp: preload fires {kstar_nonzero}/{nf} arcs, max|offset|={maxoff} | convex unwrap frac<0.1={frac:.3} rms={rms:.3}"
    );
}
