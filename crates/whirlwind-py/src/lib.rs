//! Python bindings for whirlwind-core.

// PyO3 function signatures naturally have many args (one per Python kwarg)
// and complex tuple return types - clippy's defaults don't fit them.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use ndarray::Array2;
use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::VecDeque;

/// Compute the integer residue grid from a wrapped-phase array.
///
/// For an ``(m, n)`` wrapped-phase image, returns an
/// ``(m + 1, n + 1)`` int32 grid placed on pixel corners. Each entry is
/// the closed-loop residue (in cycles) of the four wrapped-gradient steps
/// around that corner; non-zero values are the residues MCF must pair.
#[pyfunction]
fn compute_residues<'py>(
    py: Python<'py>,
    wrapped_phase: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray2<i32>> {
    let phase = wrapped_phase.as_array();
    let res = whirlwind_core::residue::compute(phase);
    res.into_pyarray(py)
}

/// Diagonal phase ramp ``(i + j) / max(m, n) * 2π * cycles`` for testing.
///
/// Returns an ``(m, n)`` float32 unwrapped-phase array suitable for use
/// as ground truth in unwrapping benchmarks.
#[pyfunction]
fn diagonal_ramp<'py>(py: Python<'py>, m: usize, n: usize) -> Bound<'py, PyArray2<f32>> {
    whirlwind_core::simulate::diagonal_ramp((m, n)).into_pyarray(py)
}

/// Wrap an unwrapped-phase array into ``(-π, π]``.
///
/// Convenience for tests / synthetic generation: returns
/// ``angle(exp(1j * unw))`` element-wise as float32, same shape as input.
#[pyfunction]
fn wrap_phase<'py>(py: Python<'py>, unw: PyReadonlyArray2<'py, f32>) -> Bound<'py, PyArray2<f32>> {
    let arr = unw.as_array().to_owned();
    whirlwind_core::simulate::wrap_phase(&arr).into_pyarray(py)
}

/// 4-connected connected-component labels of a boolean mask.
///
/// Returns ``(labels, n_components)`` where ``labels`` is an ``(m, n)`` int32
/// array with ``0`` for masked/background pixels and ``1..=n_components`` for
/// each connected valid region (raster-order seeding - the same partition the
/// MCF integrator walks in ``integrate_with_mask``). A dependency-free
/// replacement for ``scipy.ndimage.label`` used by the bridging post-pass.
#[pyfunction]
fn label_components<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray2<'py, bool>,
) -> (Bound<'py, PyArray2<i32>>, usize) {
    let mask = mask.as_array();
    let (m, n) = mask.dim();
    let mut labels = Array2::<i32>::zeros((m, n));
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    let mut next: i32 = 0;
    for si in 0..m {
        for sj in 0..n {
            if !mask[(si, sj)] || labels[(si, sj)] != 0 {
                continue;
            }
            next += 1;
            labels[(si, sj)] = next;
            queue.clear();
            queue.push_back((si, sj));
            while let Some((i, j)) = queue.pop_front() {
                if j + 1 < n && mask[(i, j + 1)] && labels[(i, j + 1)] == 0 {
                    labels[(i, j + 1)] = next;
                    queue.push_back((i, j + 1));
                }
                if j >= 1 && mask[(i, j - 1)] && labels[(i, j - 1)] == 0 {
                    labels[(i, j - 1)] = next;
                    queue.push_back((i, j - 1));
                }
                if i + 1 < m && mask[(i + 1, j)] && labels[(i + 1, j)] == 0 {
                    labels[(i + 1, j)] = next;
                    queue.push_back((i + 1, j));
                }
                if i >= 1 && mask[(i - 1, j)] && labels[(i - 1, j)] == 0 {
                    labels[(i - 1, j)] = next;
                    queue.push_back((i - 1, j));
                }
            }
        }
    }
    (labels.into_pyarray(py), next as usize)
}

/// Spiral persistent-scatterer phase interpolator - the Rust port of dolphin's
/// ``interpolation.interpolate``.
///
/// For each valid pixel (``ifg != 0``) with ``weights < weight_cutoff``, replaces
/// the phase with a Gaussian-distance-weighted average of the nearest
/// ``num_neighbors`` high-weight pixels' unit phasors (searched in concentric
/// circles out to ``max_radius``); the amplitude is preserved. High-weight and
/// masked pixels pass through. Returns a complex64 ``(m, n)`` array.
#[pyfunction]
#[pyo3(signature = (
    ifg, weights, weight_cutoff = 0.5, num_neighbors = 20,
    max_radius = 51, min_radius = 0, alpha = 0.75,
))]
fn interpolate<'py>(
    py: Python<'py>,
    ifg: PyReadonlyArray2<'py, Complex32>,
    weights: PyReadonlyArray2<'py, f32>,
    weight_cutoff: f32,
    num_neighbors: usize,
    max_radius: usize,
    min_radius: usize,
    alpha: f64,
) -> Bound<'py, PyArray2<Complex32>> {
    let ig = ifg.as_array();
    let w = weights.as_array();
    let out = py.detach(|| {
        whirlwind_core::interpolate::interpolate(
            ig,
            w,
            weight_cutoff,
            num_neighbors,
            max_radius,
            min_radius,
            alpha,
        )
    });
    out.into_pyarray(py)
}

/// Simulate a multilook complex interferogram + sample coherence.
///
/// Draws Lee-PDF phase noise around ``truth`` at per-pixel coherence
/// ``gamma`` with ``nlooks`` looks, returning ``(igram, corr)`` where:
///
/// * ``igram`` - complex64 ``exp(1j * (truth + noise))``, shape of truth.
/// * ``corr`` - float32 sample coherence (biased upward at low γ; matches
///   the multilook estimator). Shape of truth.
///
/// Reproducible: same ``seed`` ⇒ same outputs.
#[pyfunction]
fn simulate_ifg<'py>(
    py: Python<'py>,
    truth: PyReadonlyArray2<'py, f32>,
    gamma: PyReadonlyArray2<'py, f32>,
    nlooks: usize,
    seed: u64,
) -> (Bound<'py, PyArray2<Complex32>>, Bound<'py, PyArray2<f32>>) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let t: Array2<f32> = truth.as_array().to_owned();
    let g: Array2<f32> = gamma.as_array().to_owned();
    let (igram, cor) = whirlwind_core::simulate::simulate_ifg(&t, &g, nlooks, &mut rng);
    (igram.into_pyarray(py), cor.into_pyarray(py))
}

/// CRLB-weighted unwrap returning ``(unwrapped_phase, conn_components)`` - the
/// phase-linked (Dolphin/EVD/EMI) twin of :func:`unwrap`.
///
/// **EXPERIMENTAL / WIP - not validated.** Phase uses the tiled CRLB pipeline
/// (per-tile CRLB solve + coarse anchor + cascade + gated multi-shift winding
/// fix) - the same tiling that was mid-implementation and never brought to useful
/// results for either coherence or CRLB. Components are grown globally from the
/// CRLB cost grid, independent of the solve. A verified single-tile CRLB path is
/// future work (#35).
///
/// * ``igram`` - complex64, shape (m, n).
/// * ``variance`` - float32 σ²_IG = σ²_a + σ²_b in rad², shape (m, n).
/// * ``mask`` - optional bool, shape (m, n).
/// * ``coherence`` - optional float32 ``[0, 1]`` confidence map for the
///   anchor/cascade region-vote + seam stitch (e.g. the dolphin ``.cor``).
///   Defaults to a variance-derived pseudo-coherence, which is low-dynamic-
///   range; passing a real coherence raster markedly improves tile-block-offset
///   pinning (#58). It is NOT used as the cost - the cost stays CRLB variance.
/// * ``tile_size`` - 0 (default) auto-tiles frames > 512 px; ``≥ 4`` forces it.
/// * ``cost_threshold`` / ``min_size_px`` / ``max_ncomps`` - conncomp params.
#[pyfunction]
#[pyo3(signature = (
    igram, variance, mask = None, coherence = None, tile_size = 0, tile_overlap = 0,
    cost_threshold = 50, min_size_px = 100, max_ncomps = 1024,
))]
fn unwrap_crlb<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    variance: PyReadonlyArray2<'py, f32>,
    mask: Option<PyReadonlyArray2<'py, bool>>,
    coherence: Option<PyReadonlyArray2<'py, f32>>,
    tile_size: usize,
    tile_overlap: usize,
    cost_threshold: i32,
    min_size_px: usize,
    max_ncomps: u32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>)> {
    let ig = igram.as_array();
    let v = variance.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let c = coherence.as_ref().map(|c| c.as_array());
    let params = whirlwind_core::ConnCompParams {
        cost_threshold,
        min_size_px,
        min_size_frac: 0.0001,
        max_ncomps,
    };
    let out = py.detach(|| {
        whirlwind_core::unwrap_crlb_robust_with_components(
            ig,
            v,
            m,
            tile_size,
            tile_overlap,
            c,
            params,
        )
    });
    let (unw, comps) = out.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok((unw.into_pyarray(py), comps.into_pyarray(py)))
}

/// Closure-correct a stack of unwrapped interferograms.
///
/// **WARNING - opt-in, currently regresses.** Enforces exact temporal closure
/// (Σ ε·ψ ≡ 0 mod 2π) but degrades per-IG accuracy on real data: median
/// absolute RMS vs SNAPHU is 2.29 rad (raw 2D + reference anchor) vs 5.61 rad
/// (+ tree closure). `scripts/unwrap_stack.py` defaults to `closure_mode="off"`
/// for this reason. Use only when you require exact temporal consistency. See
/// `ATBD-3d.md §10.2`.
///
/// Inputs:
///   unw_stack       : float32 (n_edges, m, n) - baseline unwrapped IGs
///   edges_from      : uint32 (n_edges,) - reference-date index per IG
///   edges_to        : uint32 (n_edges,) - secondary-date index per IG
///   n_dates         : int    - number of unique acquisitions
///   reference       : int    - date index whose phase is fixed to 0
///   tree_priority   : float32 (n_edges,) or None - Prim weights (lower=better)
///
/// Returns a dict with:
///   corrected       : float32 (n_edges, m, n)
///   corrections     : int16   (n_edges, m, n)
///   date_phases     : float32 (n_dates, m, n)
///   closure_rms     : float32 (m, n)
#[pyfunction]
#[pyo3(signature = (unw_stack, edges_from, edges_to, n_dates, reference, tree_priority = None))]
fn closure_correct<'py>(
    py: Python<'py>,
    unw_stack: PyReadonlyArray3<'py, f32>,
    edges_from: PyReadonlyArray1<'py, u32>,
    edges_to: PyReadonlyArray1<'py, u32>,
    n_dates: usize,
    reference: usize,
    tree_priority: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let ef = edges_from.as_array();
    let et = edges_to.as_array();
    if ef.len() != et.len() {
        return Err(PyValueError::new_err(
            "edges_from and edges_to must be same length",
        ));
    }
    let n_edges = ef.len();
    let edges: Vec<whirlwind_core::closure::Edge> = ef
        .iter()
        .zip(et.iter())
        .map(|(&a, &b)| whirlwind_core::closure::Edge { from: a, to: b })
        .collect();
    let graph = whirlwind_core::closure::TemporalGraph::new(n_dates, edges, reference);

    let stack = unw_stack.as_array();
    if stack.shape()[0] != n_edges {
        return Err(PyValueError::new_err(format!(
            "stack shape[0]={} != n_edges {n_edges}",
            stack.shape()[0]
        )));
    }

    let priority_owned: Option<Vec<f32>> = tree_priority.as_ref().map(|p| p.as_array().to_vec());
    let priority_slice = priority_owned.as_deref();

    let out = py.detach(|| whirlwind_core::closure::correct(stack, &graph, priority_slice));

    let dict = PyDict::new(py);
    dict.set_item("corrected", out.corrected.into_pyarray(py))?;
    dict.set_item("corrections", out.corrections.into_pyarray(py))?;
    dict.set_item("date_phases", out.date_phases.into_pyarray(py))?;
    dict.set_item("closure_rms", out.closure_rms.into_pyarray(py))?;
    Ok(dict)
}

/// Per-pixel quality map: max |K| over the fundamental cycle basis, where
/// K = round(cycle_residual / 2π) is the integer mismatch count per cycle.
///
/// Because phase linking guarantees the *wrapped* sum around any temporal
/// cycle is exactly zero, any post-unwrap cycle residual is exactly 2π · K
/// with K an integer. K=0 means all fundamental cycles through this pixel
/// agree on the integer ambiguity choices; K≥1 means at least one cycle
/// disagrees - typically water or decorrelated regions where per-IG
/// unwraps were arbitrary.
///
/// Inputs:
///   unw_stack       : float32 (E, m, n)
///   edges_from      : uint32  (E,)
///   edges_to        : uint32  (E,)
///   n_dates         : int
///   reference       : int
///   tree_priority   : float32 (E,) or None - same semantics as closure_correct
///
/// Returns: uint16 (m, n).
#[pyfunction]
#[pyo3(signature = (unw_stack, edges_from, edges_to, n_dates, reference, tree_priority = None))]
fn quality_map<'py>(
    py: Python<'py>,
    unw_stack: PyReadonlyArray3<'py, f32>,
    edges_from: PyReadonlyArray1<'py, u32>,
    edges_to: PyReadonlyArray1<'py, u32>,
    n_dates: usize,
    reference: usize,
    tree_priority: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, numpy::PyArray2<u16>>> {
    let ef = edges_from.as_array();
    let et = edges_to.as_array();
    if ef.len() != et.len() {
        return Err(PyValueError::new_err(
            "edges_from and edges_to must be same length",
        ));
    }
    let n_edges = ef.len();
    let edges: Vec<whirlwind_core::closure::Edge> = ef
        .iter()
        .zip(et.iter())
        .map(|(&a, &b)| whirlwind_core::closure::Edge { from: a, to: b })
        .collect();
    let graph = whirlwind_core::closure::TemporalGraph::new(n_dates, edges, reference);

    let stack = unw_stack.as_array();
    if stack.shape()[0] != n_edges {
        return Err(PyValueError::new_err(format!(
            "stack shape[0]={} != n_edges {n_edges}",
            stack.shape()[0]
        )));
    }

    let priority_owned: Option<Vec<f32>> = tree_priority.as_ref().map(|p| p.as_array().to_vec());
    let priority_slice = priority_owned.as_deref();

    let out = py.detach(|| {
        whirlwind_core::closure::quality_max_integer_cycles(stack, &graph, priority_slice)
    });
    Ok(out.into_pyarray(py))
}

/// Unwrap an interferogram with a virtual ground node attached to every
/// boundary residue.
///
/// Fixes the unit-capacity stacking limitation of `unwrap_crlb` for clean
/// wrapping inputs (smooth ramps; tile boundaries). `ground_cost = 0` makes
/// ground free, which is right for inputs with no interior residues. For
/// noisy real data a moderate positive cost (≈ median grid arc cost) keeps
/// internal routing for the bulk of residues while still letting
/// boundary-only wrap-lines drain to ground.
#[pyfunction]
#[pyo3(signature = (igram, variance, mask = None, ground_cost = 0))]
fn unwrap_crlb_grounded<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    variance: PyReadonlyArray2<'py, f32>,
    mask: Option<PyReadonlyArray2<'py, bool>>,
    ground_cost: i32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ig = igram.as_array();
    let v = variance.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let unw = py.detach(|| whirlwind_core::unwrap_crlb_grounded(ig, v, m, ground_cost));
    let unw = unw.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(unw.into_pyarray(py))
}

/// PHASS-style flow-reuse solver - the default whole-image tile solver. Same
/// coherence cost as `unwrap`, but arcs carry multiple units of flow at zero
/// marginal cost after the first push (the corner-safe behaviour that replaced
/// the removed capacity-1 solver). See ``paper/phass_experiments.md``.
#[pyfunction]
#[pyo3(signature = (igram, corr, nlooks = 1.0, mask = None))]
fn unwrap_reuse<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    corr: PyReadonlyArray2<'py, f32>,
    nlooks: f32,
    mask: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ig = igram.as_array();
    let co = corr.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let unw = py.detach(|| whirlwind_core::unwrap_reuse(ig, co, nlooks, m));
    let unw = unw.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(unw.into_pyarray(py))
}

/// Capacity-1 (linear) MCF solver - exact replica of Python `whirlwind_orig`.
///
/// Uses unit-capacity arcs and only 8 primal-dual iterations, matching
/// `primal_dual(network, maxiter=8)` in `ww_orig._unwrap`. Diagnostic function
/// for validating Rust/Python parity; use `unwrap_reuse` for production.
#[pyfunction]
#[pyo3(signature = (igram, corr, nlooks = 1.0, mask = None))]
fn unwrap_linear<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    corr: PyReadonlyArray2<'py, f32>,
    nlooks: f32,
    mask: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ig = igram.as_array();
    let co = corr.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let unw = py.detach(|| whirlwind_core::unwrap_linear(ig, co, nlooks, m));
    let unw = unw.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(unw.into_pyarray(py))
}

/// Diagnostic: capacity-1 MCF with externally-supplied arc costs.
///
/// Accepts costs in Rust arc order: ``[DOWN(n_v), UP(n_v), RIGHT(n_h), LEFT(n_h)]``.
/// Convert from Python ``_cost.compute_carballo_costs`` output (layout
/// ``[UP, LEFT, DOWN, RIGHT]``) with the mapping documented in
/// ``whirlwind_core::unwrap_linear_ext_costs``.
#[pyfunction]
#[pyo3(signature = (igram, mask, costs))]
fn unwrap_linear_ext_costs<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    mask: Option<PyReadonlyArray2<'py, bool>>,
    costs: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ig = igram.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let c = costs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("costs must be C-contiguous: {e}")))?;
    let unw = py.detach(|| whirlwind_core::unwrap_linear_ext_costs(ig, m, c));
    let unw = unw.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(unw.into_pyarray(py))
}

/// Per-pixel quality from temporal triangles (3-cycles).
///
/// Same idea as `quality_map` but uses only triangles instead of the
/// fundamental cycle basis. Triangles are *local* (3 IGs per cycle), so
/// errors don't accumulate over long tree paths - the recommended default
/// for "reliable region" gating on phase-linked stacks where short-baseline
/// triangle redundancy is the natural network structure.
#[pyfunction]
fn quality_triangles<'py>(
    py: Python<'py>,
    unw_stack: PyReadonlyArray3<'py, f32>,
    edges_from: PyReadonlyArray1<'py, u32>,
    edges_to: PyReadonlyArray1<'py, u32>,
    n_dates: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<u16>>> {
    let ef = edges_from.as_array();
    let et = edges_to.as_array();
    if ef.len() != et.len() {
        return Err(PyValueError::new_err(
            "edges_from and edges_to must be same length",
        ));
    }
    let edges: Vec<whirlwind_core::closure::Edge> = ef
        .iter()
        .zip(et.iter())
        .map(|(&a, &b)| whirlwind_core::closure::Edge { from: a, to: b })
        .collect();
    let graph = whirlwind_core::closure::TemporalGraph::new(n_dates, edges, 0);
    let stack = unw_stack.as_array();
    if stack.shape()[0] != ef.len() {
        return Err(PyValueError::new_err(format!(
            "stack shape[0]={} != n_edges {}",
            stack.shape()[0],
            ef.len()
        )));
    }
    let out = py.detach(|| whirlwind_core::closure::quality_from_triangles(stack, &graph));
    Ok(out.into_pyarray(py))
}

/// Cycle-greedy MCF refinement on an already-unwrapped stack.
///
/// Unlike `closure_correct`, this does NOT trust the spanning tree -
/// integer corrections can land on any edge (including tree edges), routed
/// to whichever edge has the largest per-pixel CRLB variance in each
/// closure-violated cycle.
///
/// Inputs:
///   unw_stack       : float32 (E, m, n) - usually closure_correct's output
///   edges_from      : uint32 (E,) - reference-date index per IG
///   edges_to        : uint32 (E,) - secondary-date index per IG
///   n_dates         : int
///   reference       : int
///   crlb_per_date   : float32 (D, m, n) - σ²_d(p) per acquisition, in rad²
///   tree_priority   : float32 (E,) or None - for cycle-basis selection
///   max_iter        : int - cap on greedy iterations per pixel (32 is plenty)
///
/// Returns a dict with:
///   corrected            : float32 (E, m, n)
///   corrections          : int16   (E, m, n) - additive on top of input
///   residual_violations  : uint16  (m, n)   - cycles still open per pixel
///   iterations           : uint8   (m, n)
#[pyfunction]
#[pyo3(signature = (
    unw_stack, edges_from, edges_to, n_dates, reference,
    crlb_per_date, tree_priority = None, max_iter = 32,
))]
fn closure_refine_mcf<'py>(
    py: Python<'py>,
    unw_stack: PyReadonlyArray3<'py, f32>,
    edges_from: PyReadonlyArray1<'py, u32>,
    edges_to: PyReadonlyArray1<'py, u32>,
    n_dates: usize,
    reference: usize,
    crlb_per_date: PyReadonlyArray3<'py, f32>,
    tree_priority: Option<PyReadonlyArray1<'py, f32>>,
    max_iter: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let ef = edges_from.as_array();
    let et = edges_to.as_array();
    if ef.len() != et.len() {
        return Err(PyValueError::new_err(
            "edges_from and edges_to must be same length",
        ));
    }
    let edges: Vec<whirlwind_core::closure::Edge> = ef
        .iter()
        .zip(et.iter())
        .map(|(&a, &b)| whirlwind_core::closure::Edge { from: a, to: b })
        .collect();
    let graph = whirlwind_core::closure::TemporalGraph::new(n_dates, edges, reference);

    let stack = unw_stack.as_array();
    let crlb = crlb_per_date.as_array();
    if stack.shape()[0] != ef.len() {
        return Err(PyValueError::new_err("stack edge count != edges length"));
    }
    if crlb.shape()[0] != n_dates {
        return Err(PyValueError::new_err("crlb date axis != n_dates"));
    }

    let prio_owned: Option<Vec<f32>> = tree_priority.as_ref().map(|p| p.as_array().to_vec());
    let prio_slice = prio_owned.as_deref();

    let out = py
        .detach(|| whirlwind_core::closure::refine_mcf(stack, &graph, crlb, prio_slice, max_iter));

    let dict = PyDict::new(py);
    dict.set_item("corrected", out.corrected.into_pyarray(py))?;
    dict.set_item("corrections", out.corrections.into_pyarray(py))?;
    dict.set_item(
        "residual_violations",
        out.residual_violations.into_pyarray(py),
    )?;
    dict.set_item("iterations", out.iterations.into_pyarray(py))?;
    Ok(dict)
}

/// Engine behind the public Python ``unwrap``: single-tile linear coherence-cost
/// unwrap returning ``(unwrapped_phase, conn_components)``.
///
/// Phase DEFAULTS to the verified single-tile linear MCF solver (ww-orig-parity
/// Carballo Lee-1994 cost, capacity-1 min-cost-flow with an adaptive PD→SSP
/// fallback for masked frames). Components are grown globally from the Carballo
/// cost grid: a pixel edge is a *cut* when one underlying arc is mask-forbidden,
/// or the min raw forward cost across the two underlying arcs is ≤
/// ``cost_threshold``; BFS through non-cut edges labels components, those below
/// ``min_size_px`` are dropped, and the largest ``max_ncomps`` (by size) are
/// kept and renumbered ``1..=N``. Component labels are solve-independent, so
/// they compose with the phase. The integration-component gauge "bridge"
/// post-pass and the K-transfer back onto the original phase live in the Python
/// ``unwrap`` wrapper (Goldstein pre-filtering is OFF by default there).
///
/// * ``tile_size`` - 0 (default) is single-tile linear on the whole frame (NOT
///   auto-tiled); ``≥ 4`` opts in to the unvalidated tiled pipeline at that
///   tile size.
/// * ``multilook`` - > 1 routes through the coherent-downlook-first path.
/// * ``cost_threshold`` - Carballo units (``COST_SCALE = 100``); ≈ γ̂ 0.3 at 50.
/// * ``min_size_px`` - absolute component floor in pixels.
/// * ``max_ncomps`` - keep at most this many components (largest by size).
#[pyfunction]
#[pyo3(name = "_unwrap_native", signature = (
    igram, corr, nlooks, mask = None,
    tile_size = 0, tile_overlap = 0, multilook = 1,
    cost_threshold = 50, min_size_px = 100, max_ncomps = 1024,
))]
fn unwrap_native<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    corr: PyReadonlyArray2<'py, f32>,
    nlooks: f32,
    mask: Option<PyReadonlyArray2<'py, bool>>,
    tile_size: usize,
    tile_overlap: usize,
    multilook: usize,
    cost_threshold: i32,
    min_size_px: usize,
    max_ncomps: u32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>)> {
    let ig = igram.as_array();
    let co = corr.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let params = whirlwind_core::ConnCompParams {
        cost_threshold,
        min_size_px,
        // Vestigial fractional cap; the absolute px floor is the real control.
        min_size_frac: 0.0001,
        max_ncomps,
    };
    let out = py.detach(|| {
        whirlwind_core::unwrap_coherence_with_components(
            ig,
            co,
            nlooks,
            m,
            tile_size,
            tile_overlap,
            multilook,
            params,
        )
    });
    let (unw, comps) = out.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok((unw.into_pyarray(py), comps.into_pyarray(py)))
}

/// Whole-image MCF unwrap with caller-supplied integer arc costs.
///
/// For testing/research: pass precomputed costs (e.g. Python Carballo spline)
/// directly to the Rust reuse-network primal-dual solver, bypassing the
/// built-in cost computation. Costs must be packed in the same arc-id order
/// as `whirlwind_core::cost::compute_carballo_costs` returns.
///
/// * ``igram`` - complex64 (m, n); used for residues and integration only.
/// * ``costs`` - int32 flat vector of length ``num_forward_arcs``.
/// * ``mask`` - optional bool (m, n) validity mask.
#[pyfunction]
#[pyo3(signature = (igram, costs, mask = None))]
fn _unwrap_with_costs<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    costs: PyReadonlyArray1<'py, i32>,
    mask: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    use whirlwind_core::{grid, integrate, network, primal_dual, residue};
    let ig = igram.as_array();
    let cost_slice = costs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let m_view = mask.as_ref().map(|m| m.as_array());
    let out = py.detach(|| {
        let (m, n) = ig.dim();
        let wrapped_phase = ig.mapv(|z| z.arg());
        let residues = residue::compute(wrapped_phase.view());
        let g = grid::RectangularGridGraph::new(m + 1, n + 1);
        let costs_vec = cost_slice.to_vec();
        let mut net =
            network::Network::new_reuse_with_mask(&g, residues.view(), &costs_vec, m_view);
        primal_dual::run(&g, &mut net, 50);
        if m_view.is_some() {
            integrate::integrate_with_mask(wrapped_phase.view(), &g, &net, m_view)
        } else {
            integrate::integrate(wrapped_phase.view(), &g, &net)
        }
    });
    Ok(out.into_pyarray(py))
}

/// Goldstein adaptive phase filter (Goldstein & Werner 1998).
///
/// Block-parallel Rust port of the Python helper. See
/// :func:`whirlwind.goldstein` for the documentation; this version
/// is bit-identical to the Python one but typically 10x–30x faster on
/// large scenes thanks to rustfft + rayon over independent FFT blocks.
///
/// * ``igram`` - complex64, shape ``(m, n)``.
/// * ``alpha`` - filter strength in ``[0, 1]``. 0 disables filtering.
/// * ``psize`` - square FFT patch size (must be even, ≥ 4).
#[pyfunction]
#[pyo3(signature = (igram, alpha = 0.7, psize = 64))]
fn goldstein<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    alpha: f32,
    psize: usize,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    let ig = igram.as_array();
    let out = py.detach(|| whirlwind_core::goldstein::goldstein(ig, alpha, psize));
    Ok(out.into_pyarray(py))
}

/// Set the number of threads used by ww's internal parallel work.
///
/// Initialises rayon's global thread pool. **Must be called before the
/// first parallel ww function** (`unwrap*`, `goldstein`, etc.) -
/// rayon's global pool can only be initialised once per process.
///
/// Raises ``RuntimeError`` if the pool is already initialised (either
/// by a prior call, by the ``WHIRLWIND_NUM_THREADS`` /
/// ``RAYON_NUM_THREADS`` env vars read at module import, or by an
/// earlier rayon-using call).
///
/// Precedence at process startup, highest to lowest:
///   1. ``WHIRLWIND_NUM_THREADS`` env var
///   2. ``RAYON_NUM_THREADS`` env var
///   3. ``whirlwind.set_num_threads(n)`` (if neither env var was set)
///   4. rayon default (= all logical CPUs)
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Return the size of the rayon thread pool whirlwind uses.
///
/// If called before any parallel work has happened *and* no env var was
/// set, this returns rayon's default (= all logical CPUs).
#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// Read `WHIRLWIND_NUM_THREADS` then `RAYON_NUM_THREADS` (first match
/// wins) and initialise rayon's global pool to that thread count.
/// Failures are silently dropped: rayon's `build_global` returns Err
/// when the pool is already set (e.g. another rayon-using extension
/// beat us to it), which is fine - we just defer to whoever did.
fn maybe_init_thread_pool_from_env() {
    let n = std::env::var("WHIRLWIND_NUM_THREADS")
        .ok()
        .or_else(|| std::env::var("RAYON_NUM_THREADS").ok())
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0);
    if let Some(n) = n {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }
}

/// Sparse / irregular-grid unwrap over a Delaunay triangulation of the
/// "good" pixels. Useful for spurt-style workflows where you've selected
/// <10% of pixels by coherence and want to unwrap only those, getting
/// better results than dense unwrap-then-mask.
///
/// Inputs:
///   points: float64 (n, 2) - `(x, y)` of each valid pixel.
///   wrapped_phase: float32 (n,) - wrapped phase per pixel.
///   variance: float32 (n,) - CRLB phase variance σ² per pixel (rad²).
///   max_edge_length: float or None - see `unwrap_sparse` rustdocs. Set this
///     to a few times the median nearest-neighbor distance; without it,
///     long convex-hull edges produce garbage.
///
/// Returns: float32 (n,) unwrapped phase. NaN at pixels disconnected from
/// the seed (pixel 0) by the short-edge subgraph.
#[pyfunction]
#[pyo3(signature = (points, wrapped_phase, variance, max_edge_length = None))]
fn unwrap_sparse<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    wrapped_phase: PyReadonlyArray1<'py, f32>,
    variance: PyReadonlyArray1<'py, f32>,
    max_edge_length: Option<f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    let pts_view = points.as_array();
    if pts_view.shape()[1] != 2 {
        return Err(PyValueError::new_err("points must have shape (n, 2)"));
    }
    let n = pts_view.shape()[0];
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        pts.push((pts_view[(i, 0)], pts_view[(i, 1)]));
    }
    let wp = wrapped_phase.as_array().to_vec();
    let v = variance.as_array().to_vec();
    let out = py.detach(|| whirlwind_core::sparse::unwrap_sparse(&pts, &wp, &v, max_edge_length));
    let out = out.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(numpy::PyArray1::from_vec(py, out))
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    maybe_init_thread_pool_from_env();
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_crlb, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_crlb_grounded, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_reuse, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_linear, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_linear_ext_costs, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_native, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(compute_residues, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal_ramp, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_phase, m)?)?;
    m.add_function(wrap_pyfunction!(label_components, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_ifg, m)?)?;
    m.add_function(wrap_pyfunction!(closure_correct, m)?)?;
    m.add_function(wrap_pyfunction!(closure_refine_mcf, m)?)?;
    m.add_function(wrap_pyfunction!(quality_map, m)?)?;
    m.add_function(wrap_pyfunction!(quality_triangles, m)?)?;
    m.add_function(wrap_pyfunction!(goldstein, m)?)?;
    m.add_function(wrap_pyfunction!(_unwrap_with_costs, m)?)?;
    Ok(())
}
