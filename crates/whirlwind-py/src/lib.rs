//! Python bindings for whirlwind-core.

use ndarray::Array2;
use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unwrap, m)?)?;
    m.add_function(wrap_pyfunction!(compute_residues, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal_ramp, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_phase, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_ifg, m)?)?;
    Ok(())
}
