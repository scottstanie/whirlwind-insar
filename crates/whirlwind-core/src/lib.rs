//! Whirlwind: Bayesian minimum-cost-flow phase unwrapper.
//!
//! Pipeline:
//! 1. Compute residues from wrapped phase.
//! 2. Compute Carballo-style Bayesian edge costs from the interferogram + coherence.
//! 3. Solve a min-cost flow problem on the residue grid (primal-dual SSP).
//! 4. Integrate the flow-corrected wrapped gradients to recover unwrapped phase.

// Opinion lints that fire on top-level API surfaces and clear range loops.
// Splitting public unwrap_* into builder-style would hurt callers more than
// it helps clippy; range loops where the index is genuinely needed (also
// passed to helper functions) read more clearly than zip/enumerate juggling.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::doc_overindented_list_items,
    clippy::neg_cmp_op_on_partial_ord
)]

pub mod bridge;
pub mod closure;
pub mod conncomp;
pub mod cost;
pub mod cycle_cancel;
pub mod goldstein;
pub mod grid;
pub mod integrate;
pub mod interpolate;
pub mod network;
pub mod primal_dual;
pub mod residual_graph;
pub mod residue;
pub mod shortest_path;
pub mod simulate;
pub mod sparse;
pub mod ssp;
pub mod tile;
pub mod triangulated;

pub use bridge::bridge_components;
pub use conncomp::ConnCompParams;
pub use residual_graph::ResidualGraph;
pub use triangulated::TriangulatedGraph;

/// Scale factor applied to the Carballo log-likelihood ratio when forming the
/// integer connected-component cost. Mirrors `CONNCOMP_COST_SCALE` in the Python
/// wrapper.
pub const CONNCOMP_COST_SCALE: f64 = 6.0;

/// Connected-component `cost_threshold` for a target per-edge one-cycle
/// probability.
///
/// An edge is cut (a component boundary) when its cost is `<= cost_threshold`,
/// which happens when its local one-cycle-correction probability is at least
/// `cycle_prob`. A lower `cycle_prob` raises the threshold and cuts more edges
/// (stricter). The default `cost_threshold = 50` corresponds to a `cycle_prob`
/// of about 2.4e-4. Mirrors `whirlwind.cost_threshold_from_cycle_prob`.
pub fn cost_threshold_from_cycle_prob(cycle_prob: f64) -> i32 {
    let p = cycle_prob.clamp(1e-12, 1.0 - 1e-12);
    libm::rint(CONNCOMP_COST_SCALE * ((1.0 - p) / p).ln()) as i32
}

/// Connected-component `cost_threshold` from a Gaussian-equivalent noise level:
/// an edge is cut when its one-cycle-correction probability exceeds
/// `0.5 * erfc(sigma / sqrt(2))`. A higher sigma is stricter. `sigma` of about
/// 3.5 reproduces the default `cost_threshold = 50`.
pub fn cost_threshold_from_sigma(sigma: f64) -> i32 {
    let cycle_prob = 0.5 * libm::erfc(sigma / std::f64::consts::SQRT_2);
    cost_threshold_from_cycle_prob(cycle_prob)
}

use ndarray::{Array2, ArrayView2};
use num_complex::Complex32;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum UnwrapError {
    #[error("igram and coherence shape mismatch: {0:?} vs {1:?}")]
    ShapeMismatch((usize, usize), (usize, usize)),
    #[error("array too small: at least 2x2 required, got {0:?}")]
    TooSmall((usize, usize)),
}

/// Whole-image default phase kernel, read once from `WHIRLWIND_UNWRAP_SOLVER`
/// ∈ {`linear` (default), `tiled`, `reuse`, `convex`}.
///
/// **`linear` is the default** - the verified ww-orig-parity single-tile solver
/// (`unwrap_linear`; its adaptive PD/SSP fallback drains heavily-masked frames,
/// see §7.6.1). It is the default *until the tiled / reuse paths are validated
/// across the full NISAR frame set*: the tiled robustness layer can produce
/// artifacts on fragmented scenes and the reuse (PHASS) whole-image solver is
/// not yet validated either, so neither is trusted as the default. Override with
/// `WHIRLWIND_UNWRAP_SOLVER=tiled` (old behavior), `=reuse`, or `=convex`.
fn unwrap_solver() -> String {
    use std::sync::OnceLock;
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| std::env::var("WHIRLWIND_UNWRAP_SOLVER").unwrap_or_else(|_| "linear".into()))
        .clone()
}

pub fn unwrap_coherence(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    tile_size: usize,
    tile_overlap: usize,
    multilook: usize,
) -> Result<Array2<f32>, UnwrapError> {
    let solver = unwrap_solver();
    let explicit_tile = tile_size >= 4 && tile_overlap >= 2 && tile_overlap < tile_size;
    // Tiled path is now OPT-IN: an explicit tile request, a multilook downlook
    // (noisy / moderate-coherence scenes coherently down-look then unwrap the
    // coarse frame), or `WHIRLWIND_UNWRAP_SOLVER=tiled`. The tiled robustness
    // layer is still empirically tuned and not validated across all NISAR
    // frames, so it is no longer the silent default for large frames.
    if multilook > 1 || explicit_tile || solver == "tiled" {
        let (ts, to) = if tile_size == 0 {
            (512, 64)
        } else {
            (tile_size, tile_overlap)
        };
        return tile::unwrap_tiled_robust(igram, corr, nlooks, mask, ts, to, multilook);
    }
    // Whole-image default kernel.
    match solver.as_str() {
        "reuse" => unwrap_reuse(igram, corr, nlooks, mask),
        "convex" => unwrap_convex(igram, corr, nlooks, mask),
        // DEFAULT: the verified single-tile ww-orig-parity linear solver.
        _ => unwrap_linear(igram, corr, nlooks, mask),
    }
}

/// SNAPHU-style connected components grown from the global Carballo cost grid
/// **without running the MCF solve**.
///
/// Component labels depend only on (a) mask-forbidden arcs and (b) raw arc
/// costs - both fixed at [`network::Network`] construction (see
/// [`conncomp::grow_components`] / `edge_is_cut`: "MCF flow placement is
/// deliberately not a cut signal"). They are therefore independent of how - or
/// whether - the phase was solved, so this composes with the tiled/robust phase
/// path. Peak memory is one global cost grid (`O(pixels)`); there is no
/// per-source Dijkstra / solve state.
pub fn components_only(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    params: ConnCompParams,
) -> Result<Array2<u32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != corr.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), corr.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let costs = cost::compute_carballo_costs(igram, corr, nlooks, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let net = network::Network::new_with_mask(&graph, residues.view(), &costs, mask);
    Ok(conncomp::grow_components(&graph, &net, mask, &params))
}

/// Robust coherence-cost unwrap returning `(phase, conn_components)` - the
/// engine behind the public `unwrap`.
///
/// Phase comes from the robust tiled pipeline ([`unwrap_coherence`]); components
/// are grown globally and solve-free ([`components_only`]). This replaces the
/// old whole-image solve-then-grow path: the conncomp path now inherits the
/// same tiling/robustness as phase, and skips the global solve entirely
/// (strictly less memory than the old path).
pub fn unwrap_coherence_with_components(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    tile_size: usize,
    tile_overlap: usize,
    multilook: usize,
    params: ConnCompParams,
) -> Result<(Array2<f32>, Array2<u32>), UnwrapError> {
    let dbg = std::env::var("WHIRLWIND_TIMING").is_ok();
    let t = std::time::Instant::now();
    let phase = unwrap_coherence(
        igram,
        corr,
        nlooks,
        mask,
        tile_size,
        tile_overlap,
        multilook,
    )?;
    if dbg {
        eprintln!(
            "[ww] phase (unwrap_coherence): {:.2}s",
            t.elapsed().as_secs_f64()
        );
    }
    let t = std::time::Instant::now();
    let comps = components_only(igram, corr, nlooks, mask, params)?;
    if dbg {
        eprintln!(
            "[ww] conncomp (components_only, global no-solve): {:.2}s",
            t.elapsed().as_secs_f64()
        );
    }
    Ok((phase, comps))
}

/// **Specialized - not a general substitute for [`unwrap_reuse`].**
///
/// [`unwrap_reuse`] with a virtual ground node, the coherence-cost twin of
/// [`unwrap_crlb_grounded`]. Adds a single ground node connected to every
/// boundary residue with a unit-capacity arc of `ground_cost`, so
/// wrap-line endpoints can terminate at the image boundary independently
/// of each other. This fixes the capacity-1 stacking failure documented
/// in the ignored `diagonal_ramp_512` regression test: clean smooth ramps
/// whose wrap-lines all exit at the same boundary segment.
///
/// **Do not use on noisy real-world IGs.** In empirical sweeps, K-agreement
/// vs SNAPHU collapses at every `ground_cost` tested. Real data has dense
/// interior residue pairs that *want* to pair internally; routing them to
/// ground along non-physical paths corrupts the unwrap.
///
/// Use [`unwrap_reuse`] for real data. This function is exported only for the
/// boundary-stacking regression and for callers who have verified their
/// scene is in that regime.
pub fn unwrap_grounded(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    ground_cost: i32,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != corr.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), corr.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let costs = cost::compute_carballo_costs(igram, corr, nlooks, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new_with_mask_and_ground(
        &graph,
        residues.view(),
        &costs,
        mask,
        Some(ground_cost),
    );
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mask.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mask)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    Ok(unw)
}

/// **Prototype - SNAPHU-style convex (quadratic) cost solver.**
///
/// Same coherence input as [`unwrap_reuse`], but the per-arc cost is parabolic
/// in flow rather than linear: `c_e(k) = w_e · (k · 100 − offset_e)²`,
/// where `offset_e` encodes the local smoothed phase gradient as a
/// preferred integer flow direction, and `w_e` is the inverse noise
/// variance from the Just/Bamler 1994 approximation. The Dial bucket-
/// queue Dijkstra reads the *marginal* cost (cost of pushing one more
/// unit at current flow) via [`Network::marginal_cost`] in convex mode.
///
/// Built to address the residual gap from the reuse prototype: with
/// path-dependence ruled out, the residual error is the genuine cost-optimum
/// of the linear coherence-cost model. Quadratic curvature should make large
/// coherent multi-cycle deviations structurally expensive in a way linear cost
/// cannot.
pub fn unwrap_convex(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != corr.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), corr.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let (offsets, weights) = cost::compute_snaphu_smooth_costs(igram, corr, nlooks, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net =
        network::Network::new_convex_with_mask(&graph, residues.view(), &offsets, &weights, mask);
    // Pre-load each arc to its parabola minimum k* = round(offset/100) so all
    // residual marginals are ≥0 and the SSP solve is sound (see
    // Network::preload_convex_min). Without this the solve silently corrupts in
    // release once any |offset| > 50 (which the deviation offset now produces).
    net.preload_convex_min(&graph);
    // Batched primal-dual augment: feasible, but lands far from the convex
    // optimum at NISAR scale. The convex path is experimental; see
    // [`cycle_cancel`] for negative-cycle canceling that drives the feasible
    // flow toward the optimum.
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mask.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mask)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    Ok(unw)
}

/// Capacity-1 (linear) MCF solver - exact replica of Python `whirlwind_orig`.
///
/// Uses standard unit-capacity arcs (no reuse, no multi-unit) and only 8
/// primal-dual iterations (matching `primal_dual(network, maxiter=8)` in
/// Python). Residues are computed from the full phase array (not mask-gated),
/// matching `ww_orig._unwrap.unwrap`. This is the verified default kernel; use
/// the public `whirlwind.unwrap` for conncomp + bridge.
pub fn unwrap_linear(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != corr.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), corr.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let mut residues = residue::compute(wrapped_phase.view());
    // Match Python ww-orig: zero the boundary frame of the residue grid.
    // Python's `_unwrap.py` does exactly:
    //   residue[0, :] = 0; residue[-1, :] = 0
    //   residue[:, 0] = 0; residue[:, -1] = 0
    // These are "artifacts of the finite image where wrap lines cross the
    // boundary, not actual phase singularities" (Python comment). Without
    // this, Rust routes interior residues to boundary frame nodes while
    // Python routes them only to other interior nodes - completely different
    // MCF solutions and ~45% quality loss on masked scenes.
    {
        let (rm, rn) = residues.dim();
        residues.row_mut(0).fill(0);
        residues.row_mut(rm - 1).fill(0);
        residues.column_mut(0).fill(0);
        residues.column_mut(rn - 1).fill(0);
    }
    // Python ww-orig does NOT forbid masked arcs - it only sets their cost to 0.
    // Rust's new_with_mask explicitly forbids them, isolating residues in masked
    // regions and degrading quality on ~50%-masked NISAR scenes. Use new() here
    // (no mask forbidding) to match Python. Masked arcs have cost=0 so MCF
    // routes through them freely, then we NaN masked pixels post-integration.
    // Parity cost mode: 100x scale + zero only where both endpoints are
    // invalid - matches Python _cost.compute_carballo_costs exactly.
    let costs = cost::compute_carballo_costs_parity(igram, corr, nlooks, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new(&graph, residues.view(), &costs);
    // The network has copied both into its own buffers; free them now (~1.5 i32
    // arc/node vectors) so they don't sit alive through the Dijkstra peak.
    drop(costs);
    drop(residues);
    // Use full-completion Dijkstra to match Python ww-orig's `dijkstra_pd`
    // which runs `while (!dijkstra.done())`. Early-exit Dijkstra leaves
    // unpopped nodes with conservative d_max potentials instead of exact
    // distances, causing looser reduced costs and ~5.5% quality loss over
    // 8 PD iterations on masked NISAR scenes.
    //
    // 8 PD iterations, matching ww-orig `primal_dual(maxiter=8)`.
    primal_dual::run_full_dijkstra(&graph, &mut net, 8);
    let mut unw = integrate::integrate(wrapped_phase.view(), &graph, &net);
    if let Some(mm) = mask {
        unw.zip_mut_with(&mm, |u, &v| {
            if !v {
                *u = f32::NAN;
            }
        });
    }
    Ok(unw)
}

/// Diagnostic: capacity-1 MCF with externally-supplied arc costs.
///
/// Accepts precomputed costs in Rust arc order: [DOWN(n_v), UP(n_v), RIGHT(n_h), LEFT(n_h)].
/// Convert from Python `_cost.compute_carballo_costs` layout [UP, LEFT, DOWN, RIGHT] as:
///   rust_costs = concat([py[n_v+n_h:2n_v+n_h], py[0:n_v], py[2n_v+n_h:], py[n_v:n_v+n_h]])
/// where n_v = m*(n+1), n_h = (m+1)*n for an (m,n) phase image.
pub fn unwrap_linear_ext_costs(
    igram: ArrayView2<Complex32>,
    mask: Option<ArrayView2<bool>>,
    ext_costs: &[i32],
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let mut residues = residue::compute(wrapped_phase.view());
    {
        let (rm, rn) = residues.dim();
        residues.row_mut(0).fill(0);
        residues.row_mut(rm - 1).fill(0);
        residues.column_mut(0).fill(0);
        residues.column_mut(rn - 1).fill(0);
    }
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    assert_eq!(
        ext_costs.len(),
        graph.num_forward,
        "cost length must equal num_forward arcs"
    );
    let mut net = network::Network::new(&graph, residues.view(), ext_costs);
    primal_dual::run_full_dijkstra(&graph, &mut net, 8);
    let mut unw = integrate::integrate(wrapped_phase.view(), &graph, &net);
    if let Some(mm) = mask {
        unw.zip_mut_with(&mm, |u, &v| {
            if !v {
                *u = f32::NAN;
            }
        });
    }
    Ok(unw)
}

/// PHASS-style flow-reuse solver - the default whole-image coherence solver
/// (and the per-tile default; see [`tile`]). The corner-safe replacement for
/// the removed capacity-1 solver.
///
/// Same Carballo coherence cost and primal-dual driver as the tiled coherence
/// path, same Dial bucket-queue Dijkstra. The difference: the underlying
/// `Network` runs in `reuse_mode`, which makes every arc multi-unit (no
/// saturation), and Dial overrides reduced cost to 0 on any arc with prior flow
/// (PHASS `ASSP.cc:2034`). After one wrap-line is laid down, subsequent demands
/// route through the same arcs for free - which fixes the capacity-1
/// boundary-stacking bug on steep clean ramps.
pub fn unwrap_reuse(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != corr.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), corr.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute(wrapped_phase.view());
    // Pass mask=None to network construction: Python ww-orig does NOT forbid
    // masked arcs - it only sets their cost to 0. Forbidding masked arcs
    // isolates residues inside masked regions, preventing cross-mask routing
    // and degrading quality from ~99% to ~42% on ~50%-masked NISAR scenes.
    // Masked arcs have cost=0 so MCF routes through them freely; post-integration
    // we NaN masked pixels.
    let costs = cost::compute_carballo_costs(igram, corr, nlooks, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new_reuse_with_mask(&graph, residues.view(), &costs, None);
    primal_dual::run(&graph, &mut net, 50);
    // Full integration (through masked areas) so all valid pixels are reached
    // even those adjacent to masked regions; NaN masked pixels afterward.
    let mut unw = integrate::integrate(wrapped_phase.view(), &graph, &net);
    if let Some(mm) = mask {
        unw.zip_mut_with(&mm, |u, &v| {
            if !v {
                *u = f32::NAN;
            }
        });
    }
    Ok(unw)
}

/// Corner-safe CRLB unwrap (CRLB cost + PHASS flow-reuse network) - the default
/// whole-image CRLB path, the CRLB twin of [`unwrap_reuse`].
///
/// A plain unit-capacity network mis-routes the corners of smooth steep signals
/// (the capacity-1 boundary-stacking limit). The reuse network lets arcs carry
/// multiple units of flow at zero marginal cost after the first push, which
/// fixes that. The public `unwrap_crlb` binding and the CRLB tiler route through
/// this.
pub fn unwrap_crlb_reuse(
    igram: ArrayView2<Complex32>,
    variance: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != variance.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), variance.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let costs = cost::compute_crlb_costs(igram, variance, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new_reuse_with_mask(&graph, residues.view(), &costs, mask);
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mask.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mask)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    Ok(unw)
}

/// CRLB-cost connected components from the global cost grid **without running
/// the MCF solve** - the CRLB twin of [`components_only`]. Labels are
/// solve-independent (see that function); memory is one global cost grid,
/// `O(pixels)`.
pub fn crlb_components_only(
    igram: ArrayView2<Complex32>,
    variance: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    params: ConnCompParams,
) -> Result<Array2<u32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != variance.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), variance.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let costs = cost::compute_crlb_costs(igram, variance, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let net = network::Network::new_with_mask(&graph, residues.view(), &costs, mask);
    Ok(conncomp::grow_components(&graph, &net, mask, &params))
}

/// CRLB-cost unwrap returning `(phase, conn_components)` - the engine behind the
/// public `unwrap_crlb`.
///
/// **Experimental / not validated.** Phase uses the tiled CRLB pipeline
/// [`tile::unwrap_crlb_tiled_robust`] (`tile_size == 0` tiles frames larger than
/// 512 px + a gated multi-shift winding fix); components are grown globally and
/// solve-free ([`crlb_components_only`]). This rides the same experimental tiling
/// as the coherence tiled path. Porting the CRLB path to the verified
/// single-tile kernel (the coherence default) is future work.
pub fn unwrap_crlb_robust_with_components(
    igram: ArrayView2<Complex32>,
    variance: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    tile_size: usize,
    tile_overlap: usize,
    confidence: Option<ArrayView2<f32>>,
    params: ConnCompParams,
) -> Result<(Array2<f32>, Array2<u32>), UnwrapError> {
    let (m, n) = igram.dim();
    let (ts, to) = if tile_size == 0 {
        if m > 512 || n > 512 {
            (512, 64)
        } else {
            (0, 0)
        }
    } else {
        (tile_size, tile_overlap)
    };
    let use_tiling = ts >= 4 && to >= 2 && to < ts;
    let phase = if use_tiling {
        tile::unwrap_crlb_tiled_robust(igram, variance, mask, ts, to, confidence)?
    } else {
        unwrap_crlb_reuse(igram, variance, mask)?
    };
    let comps = crlb_components_only(igram, variance, mask, params)?;
    Ok((phase, comps))
}

/// Top-level CRLB-weighted phase unwrap with a virtual ground node.
///
/// Adds a single ground node connected to every boundary residue with a
/// unit-capacity forward arc of cost `ground_cost`. Wrap-line endpoints
/// can then terminate at the image boundary independently of each other,
/// fixing the capacity-1 stacking limitation of a unit-capacity network.
///
/// * `ground_cost = 0` - ground is free. Best for clean inputs whose
///   wrap-lines all exit at the boundary (e.g. smooth ramps with no
///   interior residues): MCF drains every boundary residue to ground
///   independently and places no spurious flow on interior arcs, leaving
///   the Itoh integration alone to recover the unwrap.
/// * `ground_cost > 0` - ground is preferred only when it's cheaper than
///   pairing with an opposite-sign interior residue along an internal
///   path. For data with dense interior residues (real noisy IGs), a
///   moderate positive cost keeps internal routing for the bulk of
///   residues while still draining boundary-only wrap-lines to ground.
pub fn unwrap_crlb_grounded(
    igram: ArrayView2<Complex32>,
    variance: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    ground_cost: i32,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != variance.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), variance.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let costs = cost::compute_crlb_costs(igram, variance, mask);
    let graph = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new_with_mask_and_ground(
        &graph,
        residues.view(),
        &costs,
        mask,
        Some(ground_cost),
    );
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mask.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mask)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    Ok(unw)
}

#[cfg(test)]
mod threshold_tests {
    use super::*;

    /// Reference values captured from `whirlwind.cost_threshold_from_cycle_prob`
    /// and the `0.5*erfc(sigma/sqrt2)` sigma mapping in the Python wrapper.
    #[test]
    fn cost_threshold_matches_python() {
        assert_eq!(cost_threshold_from_cycle_prob(2.4e-4), 50);
        assert_eq!(cost_threshold_from_cycle_prob(1e-3), 41);
        assert_eq!(cost_threshold_from_cycle_prob(1e-5), 69);
        assert_eq!(cost_threshold_from_sigma(3.5), 50);
        assert_eq!(cost_threshold_from_sigma(3.0), 40);
        assert_eq!(cost_threshold_from_sigma(4.0), 62);
    }
}
