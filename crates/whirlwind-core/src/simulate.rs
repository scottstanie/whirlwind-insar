//! Minimal synthetic interferogram generator for testing.
//!
//! Goodman complex Gaussian model:
//!   c = γ · exp(iφ_true) + √(1 - γ²) · n
//! where n ~ CN(0, 1). Multilook by averaging L independent realizations to
//! get sample coherence ≈ γ.

use ndarray::Array2;
use num_complex::Complex32;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// A diagonal phase ramp going from -3π to +3π (the SNAPHU-style test).
pub fn diagonal_ramp(shape: (usize, usize)) -> Array2<f32> {
    let (m, n) = shape;
    let mut out = Array2::<f32>::zeros(shape);
    for i in 0..m {
        for j in 0..n {
            let x = -3.0 + 6.0 * (j as f32) / ((n - 1).max(1) as f32);
            let y = -3.0 + 6.0 * (i as f32) / ((m - 1).max(1) as f32);
            out[(i, j)] = std::f32::consts::PI * (x + y);
        }
    }
    out
}

/// A radially symmetric Gaussian "bump" deformation field.
pub fn gaussian_bump(shape: (usize, usize), amp: f32, sigma: f32) -> Array2<f32> {
    let (m, n) = shape;
    let mut out = Array2::<f32>::zeros(shape);
    let ci = (m as f32 - 1.0) * 0.5;
    let cj = (n as f32 - 1.0) * 0.5;
    let s2 = 2.0 * sigma * sigma;
    for i in 0..m {
        for j in 0..n {
            let dy = i as f32 - ci;
            let dx = j as f32 - cj;
            out[(i, j)] = amp * (-(dy * dy + dx * dx) / s2).exp();
        }
    }
    out
}

/// Wrap unwrapped phase to [-π, π).
pub fn wrap_phase(unw: &Array2<f32>) -> Array2<f32> {
    let two_pi = std::f32::consts::TAU;
    unw.mapv(|x| {
        let y = x - two_pi * (x / two_pi).round();
        if y == -std::f32::consts::PI { -std::f32::consts::PI } else { y }
    })
}

/// Generate a complex interferogram + sample coherence from a true unwrapped
/// phase, coherence map γ, and `nlooks`. Each pixel is the average of `nlooks`
/// independent Goodman realizations.
pub fn simulate_ifg<R: Rng + ?Sized>(
    truth: &Array2<f32>,
    gamma: &Array2<f32>,
    nlooks: usize,
    rng: &mut R,
) -> (Array2<Complex32>, Array2<f32>) {
    assert_eq!(truth.dim(), gamma.dim());
    let (m, n) = truth.dim();
    let mut ig = Array2::<Complex32>::zeros((m, n));
    let mut cor = Array2::<f32>::zeros((m, n));
    let normal = StandardNormal;
    let sqrt_half: f32 = (0.5_f32).sqrt();

    for i in 0..m {
        for j in 0..n {
            let g = gamma[(i, j)].clamp(0.0, 0.999);
            let phi = truth[(i, j)];
            let signal = Complex32::new(phi.cos(), phi.sin()) * g;
            let noise_scale = (1.0 - g * g).sqrt();

            // Accumulate L looks.
            let mut acc = Complex32::new(0.0, 0.0);
            for _ in 0..nlooks {
                let re: f32 = normal.sample(rng);
                let im: f32 = normal.sample(rng);
                let n_l = Complex32::new(re, im) * sqrt_half * noise_scale;
                acc += signal + n_l;
            }
            acc /= nlooks as f32;
            ig[(i, j)] = acc;
            cor[(i, j)] = acc.norm();
        }
    }
    (ig, cor)
}
