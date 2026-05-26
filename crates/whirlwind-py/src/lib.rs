//! Python bindings for whirlwind-core.

use ndarray::Array2;
use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (igram, corr, nlooks, mask = None))]
fn unwrap<'py>(
    py: Python<'py>,
    igram: PyReadonlyArray2<'py, Complex32>,
    corr: PyReadonlyArray2<'py, f32>,
    nlooks: f32,
    mask: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ig = igram.as_array();
    let co = corr.as_array();
    let m = mask.as_ref().map(|m| m.as_array());

    let unw = py.detach(|| whirlwind_core::unwrap(ig, co, nlooks, m));
    let unw = unw.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(unw.into_pyarray(py))
}

#[pyfunction]
fn compute_residues<'py>(
    py: Python<'py>,
    wrapped_phase: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray2<i32>> {
    let phase = wrapped_phase.as_array();
    let res = whirlwind_core::residue::compute(phase);
    res.into_pyarray(py)
}

#[pyfunction]
fn diagonal_ramp<'py>(
    py: Python<'py>,
    m: usize,
    n: usize,
) -> Bound<'py, PyArray2<f32>> {
    whirlwind_core::simulate::diagonal_ramp((m, n)).into_pyarray(py)
}

#[pyfunction]
fn wrap_phase<'py>(
    py: Python<'py>,
    unw: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray2<f32>> {
    let arr = unw.as_array().to_owned();
    whirlwind_core::simulate::wrap_phase(&arr).into_pyarray(py)
}

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

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unwrap, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_crlb, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_crlb_grounded, m)?)?;
    m.add_function(wrap_pyfunction!(compute_residues, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal_ramp, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_phase, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_ifg, m)?)?;
    m.add_function(wrap_pyfunction!(closure_correct, m)?)?;
    m.add_function(wrap_pyfunction!(closure_refine_mcf, m)?)?;
    m.add_function(wrap_pyfunction!(quality_map, m)?)?;
    m.add_function(wrap_pyfunction!(quality_triangles, m)?)?;
    Ok(())
}
