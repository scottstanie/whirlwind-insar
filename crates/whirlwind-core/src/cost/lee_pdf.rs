//! Lee 1994 multilook interferometric phase PDF.
//!
//! The PDF of the phase noise φ_N (or phase difference) given true coherence γ
//! and number of looks L. We use the equivalent phase-difference form so
//! `eval(alpha, gamma, L)` returns f(alpha | gamma, L), peaked at 0 for high γ.
//!
//! Reference: Lee, J-S. et al. (1994), "Intensity and phase statistics of
//! multilook polarimetric and interferometric SAR imagery."
//! Form:
//!     f(φ; L, γ) = ((1 - γ²)^L / (2π)) · ₂F₁(L, 1; 1/2; β²)
//!                + (Γ(L + 1/2) · (1 - γ²)^L · β)
//!                  / (2 · √π · Γ(L) · (1 - β²)^(L + 1/2))
//! where β = γ · cos(φ).

use super::hyp2f1::hyp2f1_f64;
use std::f64::consts::PI as PI64;

/// Lanczos approximation for ln Γ(x), x > 0. Good to ~1e-13.
fn lanczos_lgamma(x: f64) -> f64 {
    // g = 7, n = 9 coefficients.
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        // Reflection
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - lanczos_lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = COEF[0];
    for (i, &c) in COEF.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Lee multilook phase PDF f(φ; L, γ).
/// `alpha` is the phase (radians, any real); `gamma` ∈ [0, 1); `nlooks` ≥ 1.
pub fn pdf(alpha: f32, gamma: f32, nlooks: f32) -> f32 {
    assert!((0.0..1.0).contains(&gamma), "gamma must be in [0, 1)");
    assert!(
        nlooks >= 1.0,
        "nlooks must be >= 1 for the Lee (1994) multilook phase PDF, got {nlooks}"
    );
    let g = gamma as f64;
    let a = alpha as f64;
    let n = nlooks as f64;

    let g2 = g * g;
    let beta = g * a.cos();
    let b2 = beta * beta;
    let one_minus_b2 = (1.0 - b2).max(1e-300);

    // Term 1 = (1 - γ²)^L / (2π) · ₂F₁(L, 1; 0.5; β²)
    // For β² near 1, evaluate in log-space using the Euler transform:
    //     ₂F₁(L, 1; 0.5; b²) = (1-b²)^(-L-0.5) · ₂F₁(0.5-L, -0.5; 0.5; b²)
    let t1 = if b2 < 0.5 {
        (1.0 - g2).powf(n) / (2.0 * PI64) * hyp2f1_f64(n, 1.0, 0.5, b2)
    } else {
        let log_pref = n * (1.0 - g2).ln() - (n + 0.5) * one_minus_b2.ln();
        log_pref.exp() / (2.0 * PI64) * hyp2f1_f64(0.5 - n, -0.5, 0.5, b2)
    };

    // Term 2: (Γ(L+0.5)/Γ(L)) · β · (1-γ²)^L / (2 √π · (1-β²)^(L+0.5))
    // Evaluate in log-space too.
    let log_t2_mag = lanczos_lgamma(n + 0.5) - lanczos_lgamma(n) + n * (1.0 - g2).ln()
        - (n + 0.5) * one_minus_b2.ln();
    let t2 = beta.signum() * beta.abs() * (log_t2_mag.exp()) / (2.0 * PI64.sqrt());

    (t1 + t2) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// PDF must integrate to ~1 over [-π, π) for typical (γ, L) in InSAR.
    /// (Extreme cases like γ=0.99, L=50 produce delta-spike PDFs that need
    /// adaptive quadrature; not worth testing here.)
    #[test]
    fn pdf_integrates_to_one() {
        for (gamma, nlooks) in [(0.1, 1.0), (0.5, 5.0), (0.9, 10.0)] {
            let n = 8192;
            let dx = 2.0 * PI / (n as f32);
            let mut s = 0.0_f64;
            for k in 0..n {
                let x = -PI + (k as f32) * dx + 0.5 * dx;
                s += pdf(x, gamma, nlooks) as f64 * dx as f64;
            }
            let s = s as f32;
            assert!(
                (s - 1.0).abs() < 5e-3,
                "PDF integral for γ={gamma}, L={nlooks} = {s}, expected ~1.0"
            );
        }
    }

    #[test]
    fn pdf_peaks_at_zero_for_high_coherence() {
        let p0 = pdf(0.0, 0.9, 10.0);
        let p1 = pdf(1.0, 0.9, 10.0);
        assert!(p0 > p1, "high-γ PDF should peak near 0");
    }

    #[test]
    fn pdf_uniform_for_zero_coherence() {
        let p0 = pdf(0.0, 0.001, 1.0);
        let p1 = pdf(2.0, 0.001, 1.0);
        let uniform = 1.0 / (2.0 * PI);
        assert!((p0 - uniform).abs() < 0.05);
        assert!((p1 - uniform).abs() < 0.05);
    }
}
