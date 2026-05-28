//! Python bindings for whirlwind-core.

// PyO3 function signatures naturally have many args (one per Python kwarg)
// and complex tuple return types — clippy's defaults don't fit them.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use ndarray::Array2;
use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Unwrap an interferogram with the Carballo/SNAPHU-style coherence cost.
///
/// Suitable for boxcar-multilooked IGs where ``corr`` is the sample
/// coherence γ̂. For phase-linked IGs (Dolphin/EVD/EMI), use
/// :func:`unwrap_crlb` instead — the CRLB variance is the proper per-pixel
/// noise weight there.
///
/// * ``igram`` — complex64, shape ``(m, n)``.
/// * ``corr`` — float32 sample coherence in ``[0, 1]``, shape ``(m, n)``.
/// * ``nlooks`` — effective number of looks (≥ 1) used to estimate ``corr``.
/// * ``mask`` — optional bool, shape ``(m, n)``. ``False`` pixels are
///   excluded (their incident arcs are forbidden).
///
/// Returns the unwrapped phase as float32 ``(m, n)``.
///
/// * ``tile_size`` — if ≥ 4 and < min(m, n), tile the image into
///   ``tile_size × tile_size`` sub-images with ``tile_overlap`` overlap,
///   unwrap each in parallel, and stitch with a coherence-weighted
///   overlap-median 2π reconciliation. Bounds per-IG MCF memory to
///   tile-size scale and keeps flow local (prevents whole-frame runaway).
#[pyfunction]
#[pyo3(signature = (igram, corr, nlooks, mask = None, tile_size = 0, tile_overlap = 0))]
fn unwrap<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    corr: PyReadonlyArray2<'py, f32>,
    nlooks: f32,
    mask: Option<PyReadonlyArray2<'py, bool>>,
    tile_size: usize,
    tile_overlap: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ig = igram.as_array();
    let co = corr.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let use_tiling = tile_size >= 4 && tile_overlap >= 2 && tile_overlap < tile_size;

    let unw = py.detach(|| {
        if use_tiling {
            whirlwind_core::tile::unwrap_tiled(ig, co, nlooks, m, tile_size, tile_overlap)
        } else {
            whirlwind_core::unwrap(ig, co, nlooks, m)
        }
    });
    let unw = unw.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(unw.into_pyarray(py))
}

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
fn diagonal_ramp<'py>(
    py: Python<'py>,
    m: usize,
    n: usize,
) -> Bound<'py, PyArray2<f32>> {
    whirlwind_core::simulate::diagonal_ramp((m, n)).into_pyarray(py)
}

/// Wrap an unwrapped-phase array into ``(-π, π]``.
///
/// Convenience for tests / synthetic generation: returns
/// ``angle(exp(1j * unw))`` element-wise as float32, same shape as input.
#[pyfunction]
fn wrap_phase<'py>(
    py: Python<'py>,
    unw: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray2<f32>> {
    let arr = unw.as_array().to_owned();
    whirlwind_core::simulate::wrap_phase(&arr).into_pyarray(py)
}

/// Simulate a multilook complex interferogram + sample coherence.
///
/// Draws Lee-PDF phase noise around ``truth`` at per-pixel coherence
/// ``gamma`` with ``nlooks`` looks, returning ``(igram, corr)`` where:
///
/// * ``igram`` — complex64 ``exp(1j * (truth + noise))``, shape of truth.
/// * ``corr`` — float32 sample coherence (biased upward at low γ; matches
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

/// Unwrap an interferogram using CRLB-derived per-pixel variance.
///
/// * `igram` — complex64, shape (m, n)
/// * `variance` — float32 σ²_IG = σ²_a + σ²_b in rad², shape (m, n)
/// * `mask` — optional bool, shape (m, n)
/// * `tile_size` — if > 0 and < min(m, n), tile the image into
///   `tile_size × tile_size` sub-images with `tile_overlap` overlap,
///   unwrap each in parallel, and stitch with CRLB-weighted overlap-median
///   2π reconciliation. Bounds per-IG MCF memory to tile-size scale.
#[pyfunction]
#[pyo3(signature = (igram, variance, mask = None, tile_size = 0, tile_overlap = 0))]
fn unwrap_crlb<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    variance: PyReadonlyArray2<'py, f32>,
    mask: Option<PyReadonlyArray2<'py, bool>>,
    tile_size: usize,
    tile_overlap: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ig = igram.as_array();
    let v = variance.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let use_tiling = tile_size >= 4 && tile_overlap >= 2 && tile_overlap < tile_size;
    let unw = py.detach(|| {
        if use_tiling {
            whirlwind_core::tile::unwrap_crlb_tiled(ig, v, m, tile_size, tile_overlap)
        } else {
            whirlwind_core::unwrap_crlb(ig, v, m)
        }
    });
    let unw = unw.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(unw.into_pyarray(py))
}

/// Closure-correct a stack of unwrapped interferograms.
///
/// Inputs:
///   unw_stack       : float32 (n_edges, m, n) — baseline unwrapped IGs
///   edges_from      : uint32 (n_edges,) — reference-date index per IG
///   edges_to        : uint32 (n_edges,) — secondary-date index per IG
///   n_dates         : int    — number of unique acquisitions
///   reference       : int    — date index whose phase is fixed to 0
///   tree_priority   : float32 (n_edges,) or None — Prim weights (lower=better)
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
        return Err(PyValueError::new_err("edges_from and edges_to must be same length"));
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
        whirlwind_core::closure::correct(stack, &graph, priority_slice)
    });

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
/// disagrees — typically water or decorrelated regions where per-IG
/// unwraps were arbitrary.
///
/// Inputs:
///   unw_stack       : float32 (E, m, n)
///   edges_from      : uint32  (E,)
///   edges_to        : uint32  (E,)
///   n_dates         : int
///   reference       : int
///   tree_priority   : float32 (E,) or None — same semantics as closure_correct
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
        return Err(PyValueError::new_err("edges_from and edges_to must be same length"));
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

/// **Prototype.** SNAPHU-style convex (quadratic) per-arc cost.
/// Per-arc preferred offset from the smoothed phase gradient + inverse-
/// variance weight; cost grows quadratically away from offset. Tests
/// whether convex curvature closes the residual NISAR K gap that reuse
/// couldn't. See ``paper/convex_cost_design.md``.
#[pyfunction]
#[pyo3(signature = (igram, corr, nlooks = 1.0, mask = None))]
fn unwrap_convex<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    corr: PyReadonlyArray2<'py, f32>,
    nlooks: f32,
    mask: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ig = igram.as_array();
    let co = corr.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let unw = py.detach(|| whirlwind_core::unwrap_convex(ig, co, nlooks, m));
    let unw = unw.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(unw.into_pyarray(py))
}

/// **Prototype.** PHASS-style flow-reuse solver — same coherence cost
/// as `unwrap`, but arcs can carry multiple units of flow at zero
/// marginal cost after the first push. Tests whether flow-reuse alone
/// (the load-bearing piece of dolphin PHASS per ASSP.cc:2034) closes
/// whirlwind's no-Goldstein gap. See ``paper/phass_experiments.md``.
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

/// **Specialized — not a general substitute for `unwrap`.**
///
/// Coherence-cost unwrap with a virtual ground node, the Carballo twin
/// of `unwrap_crlb_grounded`. Fixes the boundary-stacking failure for
/// clean smooth ramps whose wrap-lines all exit at the same image edge.
///
/// Do not use on noisy real-world interferograms — see the Rust-side
/// docs and ``paper/phass_experiments.md`` for the empirical reason
/// (K-agreement drops sharply at every ``ground_cost`` tested).
#[pyfunction]
#[pyo3(signature = (igram, corr, nlooks = 1.0, mask = None, ground_cost = 0))]
fn unwrap_grounded<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    corr: PyReadonlyArray2<'py, f32>,
    nlooks: f32,
    mask: Option<PyReadonlyArray2<'py, bool>>,
    ground_cost: i32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ig = igram.as_array();
    let co = corr.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let unw = py.detach(|| whirlwind_core::unwrap_grounded(ig, co, nlooks, m, ground_cost));
    let unw = unw.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(unw.into_pyarray(py))
}

/// Per-pixel quality from temporal triangles (3-cycles).
///
/// Same idea as `quality_map` but uses only triangles instead of the
/// fundamental cycle basis. Triangles are *local* (3 IGs per cycle), so
/// errors don't accumulate over long tree paths — the recommended default
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
        return Err(PyValueError::new_err("edges_from and edges_to must be same length"));
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
            "stack shape[0]={} != n_edges {}", stack.shape()[0], ef.len()
        )));
    }
    let out = py.detach(|| {
        whirlwind_core::closure::quality_from_triangles(stack, &graph)
    });
    Ok(out.into_pyarray(py))
}

/// Cycle-greedy MCF refinement on an already-unwrapped stack.
///
/// Unlike `closure_correct`, this does NOT trust the spanning tree —
/// integer corrections can land on any edge (including tree edges), routed
/// to whichever edge has the largest per-pixel CRLB variance in each
/// closure-violated cycle.
///
/// Inputs:
///   unw_stack       : float32 (E, m, n) — usually closure_correct's output
///   edges_from      : uint32 (E,) — reference-date index per IG
///   edges_to        : uint32 (E,) — secondary-date index per IG
///   n_dates         : int
///   reference       : int
///   crlb_per_date   : float32 (D, m, n) — σ²_d(p) per acquisition, in rad²
///   tree_priority   : float32 (E,) or None — for cycle-basis selection
///   max_iter        : int — cap on greedy iterations per pixel (32 is plenty)
///
/// Returns a dict with:
///   corrected            : float32 (E, m, n)
///   corrections          : int16   (E, m, n) — additive on top of input
///   residual_violations  : uint16  (m, n)   — cycles still open per pixel
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
        return Err(PyValueError::new_err("edges_from and edges_to must be same length"));
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

    let out = py.detach(|| {
        whirlwind_core::closure::refine_mcf(stack, &graph, crlb, prio_slice, max_iter)
    });

    let dict = PyDict::new(py);
    dict.set_item("corrected", out.corrected.into_pyarray(py))?;
    dict.set_item("corrections", out.corrections.into_pyarray(py))?;
    dict.set_item("residual_violations", out.residual_violations.into_pyarray(py))?;
    dict.set_item("iterations", out.iterations.into_pyarray(py))?;
    Ok(dict)
}

/// Carballo-cost unwrap returning ``(unwrapped_phase, conn_components)``.
///
/// SNAPHU-style components grown from the same MCF solve. A pixel edge is
/// treated as a *cut* when (a) one of its underlying arcs is forbidden by
/// the input mask, or (b) the minimum raw forward cost across the two
/// underlying arcs is ≤ ``cost_threshold``. BFS through non-cut edges
/// labels components; components covering less than ``min_size_frac`` of
/// valid pixels are dropped, and the largest ``max_ncomps`` (by size) are
/// kept and renumbered ``1..=N``. Background / dropped pixels are ``0``.
///
/// Defaults are SNAPHU-equivalent. ``cost_threshold = 50`` (in integer
/// Carballo units with ``COST_SCALE = 100``) corresponds roughly to
/// γ̂ ≈ 0.3 at average local phase smoothness — i.e. "cut clearly
/// decorrelated noise, keep everything else." Set higher (≈100 ≈ γ̂ ≈ 0.6)
/// for an aggressive spurt-style mask; set to 0 for connectivity from the
/// input mask alone (no cost-based cutting).
///
/// * ``igram`` — complex64, shape ``(m, n)``.
/// * ``corr`` — float32 sample coherence in ``[0, 1]``, shape ``(m, n)``.
/// * ``nlooks`` — effective number of looks (≥ 1).
/// * ``mask`` — optional bool, shape ``(m, n)``.
/// * ``cost_threshold`` — see above.
/// * ``min_size_frac`` — drop components below this fraction of valid pixels.
/// * ``max_ncomps`` — keep at most this many components.
#[pyfunction]
#[pyo3(signature = (
    igram, corr, nlooks, mask = None,
    cost_threshold = 50, min_size_frac = 0.01, max_ncomps = 64,
))]
fn unwrap_with_conncomp<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    corr: PyReadonlyArray2<'py, f32>,
    nlooks: f32,
    mask: Option<PyReadonlyArray2<'py, bool>>,
    cost_threshold: i32,
    min_size_frac: f32,
    max_ncomps: u32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>)> {
    let ig = igram.as_array();
    let co = corr.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let params = whirlwind_core::ConnCompParams {
        cost_threshold,
        min_size_frac,
        max_ncomps,
    };
    let out = py.detach(|| whirlwind_core::unwrap_with_components(ig, co, nlooks, m, params));
    let (unw, comps) = out.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok((unw.into_pyarray(py), comps.into_pyarray(py)))
}

/// CRLB-cost unwrap returning ``(unwrapped_phase, conn_components)``.
///
/// CRLB-path twin of :func:`unwrap_with_conncomp` for phase-linked IGs
/// (Dolphin/EVD/EMI). Uses per-pixel CRLB variance σ²_IG = σ²_a + σ²_b
/// (typically ``crlb_<date_a>.tif + crlb_<date_b>.tif``) as the noise
/// weight instead of sample coherence.
///
/// Cut rule and parameter semantics match :func:`unwrap_with_conncomp`.
/// Note that the cost units differ from the Carballo path: CRLB-derived
/// costs are scaled differently, so ``cost_threshold`` is not directly
/// comparable to the Carballo path's threshold. For typical Dolphin
/// outputs the default ``50`` is still a reasonable "exclude clearly
/// decorrelated" cutoff.
///
/// * ``igram`` — complex64, shape ``(m, n)``.
/// * ``variance`` — float32 σ²_IG in rad², shape ``(m, n)``.
/// * ``mask`` — optional bool, shape ``(m, n)``.
/// * ``cost_threshold``, ``min_size_frac``, ``max_ncomps`` — see
///   :func:`unwrap_with_conncomp`.
#[pyfunction]
#[pyo3(signature = (
    igram, variance, mask = None,
    cost_threshold = 50, min_size_frac = 0.01, max_ncomps = 64,
))]
fn unwrap_crlb_with_conncomp<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    variance: PyReadonlyArray2<'py, f32>,
    mask: Option<PyReadonlyArray2<'py, bool>>,
    cost_threshold: i32,
    min_size_frac: f32,
    max_ncomps: u32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>)> {
    let ig = igram.as_array();
    let v = variance.as_array();
    let m = mask.as_ref().map(|m| m.as_array());
    let params = whirlwind_core::ConnCompParams {
        cost_threshold,
        min_size_frac,
        max_ncomps,
    };
    let out = py.detach(|| whirlwind_core::unwrap_crlb_with_components(ig, v, m, params));
    let (unw, comps) = out.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok((unw.into_pyarray(py), comps.into_pyarray(py)))
}

/// Goldstein adaptive phase filter (Goldstein & Werner 1998).
///
/// Block-parallel Rust port of the Python helper. See
/// :func:`whirlwind.goldstein` for the documentation; this version
/// is bit-identical to the Python one but typically 10×–30× faster on
/// large scenes thanks to rustfft + rayon over independent FFT blocks.
///
/// * ``igram`` — complex64, shape ``(m, n)``.
/// * ``alpha`` — filter strength in ``[0, 1]``. 0 disables filtering.
/// * ``psize`` — square FFT patch size (must be even, ≥ 4).
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
/// first parallel ww function** (`unwrap*`, `goldstein`, etc.) —
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

/// Return the size of the rayon thread pool whirlwind-rs uses.
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
/// beat us to it), which is fine — we just defer to whoever did.
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
///   points: float64 (n, 2) — `(x, y)` of each valid pixel.
///   wrapped_phase: float32 (n,) — wrapped phase per pixel.
///   variance: float32 (n,) — CRLB phase variance σ² per pixel (rad²).
///   max_edge_length: float or None — see `unwrap_sparse` rustdocs. Set this
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
    let out = py.detach(|| {
        whirlwind_core::sparse::unwrap_sparse(&pts, &wp, &v, max_edge_length)
    });
    let out = out.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(numpy::PyArray1::from_vec(py, out))
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    maybe_init_thread_pool_from_env();
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_crlb, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_crlb_grounded, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_grounded, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_convex, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_reuse, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_with_conncomp, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_crlb_with_conncomp, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(compute_residues, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal_ramp, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_phase, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_ifg, m)?)?;
    m.add_function(wrap_pyfunction!(closure_correct, m)?)?;
    m.add_function(wrap_pyfunction!(closure_refine_mcf, m)?)?;
    m.add_function(wrap_pyfunction!(quality_map, m)?)?;
    m.add_function(wrap_pyfunction!(quality_triangles, m)?)?;
    m.add_function(wrap_pyfunction!(goldstein, m)?)?;
    Ok(())
}
