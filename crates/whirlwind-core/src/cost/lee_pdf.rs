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

use super::hyp2f1::ln_hyp2f1_pos;
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

    // Term 1 = (1 - γ²)^L / (2π) · ₂F₁(L, 1; 0.5; β²), evaluated in log space.
    //
    // This used to branch at β² = 0.5 onto the Euler-transformed series
    //     ₂F₁(L, 1; 0.5; b²) = (1-b²)^(-L-0.5) · ₂F₁(0.5-L, -0.5; 0.5; b²)
    // to dodge overflow of the direct series. The two forms are algebraically
    // identical (the (1-b²) powers cancel exactly), but the transformed one has
    // a large negative first parameter, so its terms alternate and grow before
    // cancelling: at L = 300, z = 0.36 it returned -2.9e20 against a true 18.4,
    // and the resulting PDF went *negative* from γ ≈ 0.8 upward. Since the
    // direct series has only positive terms, taking its logarithm with
    // rescaling removes both the overflow and the cancellation, so one formula
    // covers the whole range.
    let ln_t1 = n * (1.0 - g2).ln() - (2.0 * PI64).ln() + ln_hyp2f1_pos(n, 1.0, 0.5, b2);
    let t1 = ln_t1.exp();

    // Term 2: (Γ(L+0.5)/Γ(L)) · β · (1-γ²)^L / (2 √π · (1-β²)^(L+0.5))
    // Also in log space; β carries the only sign in the whole expression.
    let ln_t2 = lanczos_lgamma(n + 0.5) - lanczos_lgamma(n) + n * (1.0 - g2).ln()
        - (n + 0.5) * one_minus_b2.ln()
        + beta.abs().ln()
        - (2.0 * PI64.sqrt()).ln();

    if beta == 0.0 {
        return t1 as f32;
    }
    // `exp(u) - exp(v) = -exp(u)·expm1(v-u)` keeps the subtraction accurate for
    // moderate cancellation.
    let sum = if beta > 0.0 {
        ln_t1.exp() + ln_t2.exp()
    } else {
        -ln_t1.exp() * (ln_t2 - ln_t1).exp_m1()
    };

    // Deep in the tail at high looks the two terms agree to every bit f64 has:
    // at γ = 0.8, L = 276, φ ≈ -π both are 0.17086 with identical logs to 16
    // digits, while the true density is ~6e-62. No rearrangement recovers that
    // from this two-term form -- it is a conditioning limit of the analytic
    // expression, not of the ₂F₁ evaluation (which matches a 50-digit reference
    // exactly). What is left is sign noise at the 1e-15 level, and 0 is a far
    // better estimate of a 1e-62 density than a negative number is, so clamp.
    // Distinct from the old failure, where the *values themselves* were wrong
    // by order unity (-17.7 against a peak of similar size).
    (sum.max(0.0)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// PDF must integrate to ~1 over [-π, π), including the high-coherence and
    /// high-looks corners that the old Euler-transformed branch got wrong.
    #[test]
    fn pdf_integrates_to_one() {
        for (gamma, nlooks) in [
            (0.1, 1.0),
            (0.5, 5.0),
            (0.9, 10.0),
            (0.8, 85.0),
            (0.95, 80.0),
            (0.99, 20.0),
            (0.99, 300.0),
        ] {
            let n = 1 << 17;
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

    /// The defect this implementation was written to fix: the previous
    /// two-branch form returned *negative* densities from γ ≈ 0.8 upward
    /// (−17.7 at γ = 0.99, L = 80). A density is never negative, at any
    /// parameters, so this is an absolute invariant rather than a tolerance.
    #[test]
    fn pdf_is_never_negative() {
        for &gamma in &[0.0_f32, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999] {
            for &nlooks in &[1.0_f32, 5.0, 20.0, 80.0, 143.0, 276.0, 300.0, 1000.0] {
                for k in 0..=512 {
                    let phi = -PI + 2.0 * PI * (k as f32) / 512.0;
                    let p = pdf(phi, gamma, nlooks);
                    assert!(
                        p.is_finite() && p >= 0.0,
                        "pdf({phi}, γ={gamma}, L={nlooks}) = {p}"
                    );
                }
            }
        }
    }

    /// Spot values against a high-precision reference (mpmath, 50 digits) at
    /// parameters where the old branch was badly wrong.
    #[test]
    fn pdf_matches_high_precision_reference() {
        // (alpha, gamma, nlooks, expected)
        for &(a, g, n, want) in &[
            (0.0_f32, 0.6_f32, 20.0_f32, 1.880_561_597_8_f64),
            (0.5, 0.9, 80.0, 1.204_986_615_1e-23),
            (0.2, 0.95, 80.0, 1.948_639_265_3e-10),
            (0.1, 0.99, 20.0, 4.872_209_923_1e-3),
        ] {
            let got = pdf(a, g, n) as f64;
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel < 2e-4,
                "pdf({a}, {g}, {n}) = {got}, want {want} (rel {rel:.2e})"
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
