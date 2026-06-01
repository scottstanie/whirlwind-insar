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

pub mod closure;
pub mod conncomp;
pub mod cost;
pub mod goldstein;
pub mod grid;
pub mod integrate;
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

pub use conncomp::ConnCompParams;
pub use residual_graph::ResidualGraph;
pub use triangulated::TriangulatedGraph;

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

/// Robust coherence-cost phase unwrap with auto-tiling.
///
/// `tile_size == 0` (and `multilook <= 1`) auto-tiles frames larger than
/// 512 px at 512 / overlap-64 — the empirically best universal size (a
/// whole-image solve runs away to ~80% on NISAR; bigger tiles regress clean
/// scenes). Otherwise the explicit tile params are honored. Tiled frames go
/// through [`tile::unwrap_tiled_robust`] (gated multi-shift + global anchor +
/// multi-scale cascade + seam-repair); frames that fit one tile use the
/// corner-safe reuse solver. This is the phase engine behind the public
/// `unwrap`.
pub fn unwrap_coherence(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    tile_size: usize,
    tile_overlap: usize,
    multilook: usize,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    let (ts, to) = if tile_size == 0 && multilook <= 1 {
        if m > 512 || n > 512 {
            (512, 64)
        } else {
            (0, 0)
        }
    } else {
        (tile_size, tile_overlap)
    };
    let use_tiling = multilook > 1 || (ts >= 4 && to >= 2 && to < ts);
    if use_tiling {
        tile::unwrap_tiled_robust(igram, corr, nlooks, mask, ts, to, multilook)
    } else {
        unwrap_reuse(igram, corr, nlooks, mask)
    }
}

/// SNAPHU-style connected components grown from the global Carballo cost grid
/// **without running the MCF solve**.
///
/// Component labels depend only on (a) mask-forbidden arcs and (b) raw arc
/// costs — both fixed at [`network::Network`] construction (see
/// [`conncomp::grow_components`] / `edge_is_cut`: "MCF flow placement is
/// deliberately not a cut signal"). They are therefore independent of how — or
/// whether — the phase was solved, so this composes with the tiled/robust phase
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

/// Robust coherence-cost unwrap returning `(phase, conn_components)` — the
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

/// **Specialized — not a general substitute for [`unwrap_reuse`].**
///
/// [`unwrap_reuse`] with a virtual ground node, the coherence-cost twin of
/// [`unwrap_crlb_grounded`]. Adds a single ground node connected to every
/// boundary residue with a unit-capacity arc of `ground_cost`, so
/// wrap-line endpoints can terminate at the image boundary independently
/// of each other. This fixes the capacity-1 stacking failure documented
/// in the ignored `diagonal_ramp_512` regression test: clean smooth ramps
/// whose wrap-lines all exit at the same boundary segment.
///
/// **Do not use on noisy real-world IGs.** Empirical sweep on a Capella
/// Palos Verdes scene (`paper/phass_experiments.md`, 2026-05-28 follow-up):
/// K-agreement vs SNAPHU drops from 90.7 % (baseline) to ~20 % at every
/// `ground_cost ∈ {0, 50, 100, 200}` tested. Real data has dense interior
/// residue pairs that *want* to pair internally; routing them to ground
/// along non-physical paths corrupts the unwrap. Same direction on a NISAR
/// scene (80 % → 42 %).
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

/// **Prototype — SNAPHU-style convex (quadratic) cost solver.**
///
/// Same coherence input as [`unwrap_reuse`], but the per-arc cost is parabolic
/// in flow rather than linear: `c_e(k) = w_e · (k · 100 − offset_e)²`,
/// where `offset_e` encodes the local smoothed phase gradient as a
/// preferred integer flow direction, and `w_e` is the inverse noise
/// variance from the Just/Bamler 1994 approximation. The Dial bucket-
/// queue Dijkstra reads the *marginal* cost (cost of pushing one more
/// unit at current flow) via [`Network::marginal_cost`] in convex mode.
///
/// Built to address the residual NISAR gap from the reuse prototype:
/// path-dependence ruled out (`paper/phass_experiments.md` 2026-05-28
/// follow-up), the 7 % residual error is the genuine cost-optimum of
/// the linear coherence-cost model. Quadratic curvature should make
/// large coherent multi-cycle deviations structurally expensive in a
/// way linear cost cannot. See `paper/convex_cost_design.md`.
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
    primal_dual::run(&graph, &mut net, 50);
    let unw = if mask.is_some() {
        integrate::integrate_with_mask(wrapped_phase.view(), &graph, &net, mask)
    } else {
        integrate::integrate(wrapped_phase.view(), &graph, &net)
    };
    Ok(unw)
}

/// PHASS-style flow-reuse solver — the default whole-image coherence solver
/// (and the per-tile default; see [`tile`]). The corner-safe replacement for
/// the removed capacity-1 solver.
///
/// Same Carballo coherence cost and primal-dual driver as the tiled coherence
/// path, same Dial bucket-queue Dijkstra. The difference: the underlying
/// `Network` runs in `reuse_mode`, which makes every arc multi-unit (no
/// saturation), and Dial overrides reduced cost to 0 on any arc with prior flow
/// (PHASS `ASSP.cc:2034`). After one wrap-line is laid down, subsequent demands
/// route through the same arcs for free — which fixes the capacity-1
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
    let residues = residue::compute_with_mask(wrapped_phase.view(), mask);
    let costs = cost::compute_carballo_costs(igram, corr, nlooks, mask);
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

/// Corner-safe CRLB unwrap (CRLB cost + PHASS flow-reuse network) — the default
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
/// the MCF solve** — the CRLB twin of [`components_only`]. Labels are
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

/// Robust CRLB-cost unwrap returning `(phase, conn_components)` — the engine
/// behind the public `unwrap_crlb`.
///
/// Phase uses [`tile::unwrap_crlb_tiled_robust`] (auto-tile + gated multi-shift
/// winding fix); components are grown globally and solve-free
/// ([`crlb_components_only`]). `tile_size == 0` auto-tiles frames larger than
/// 512 px. (Anchor + cascade parity with the coherence path is pending — see
/// issue #35.)
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
/// * `ground_cost = 0` — ground is free. Best for clean inputs whose
///   wrap-lines all exit at the boundary (e.g. smooth ramps with no
///   interior residues): MCF drains every boundary residue to ground
///   independently and places no spurious flow on interior arcs, leaving
///   the Itoh integration alone to recover the unwrap.
/// * `ground_cost > 0` — ground is preferred only when it's cheaper than
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
