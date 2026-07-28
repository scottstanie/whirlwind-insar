//! Gauss hypergeometric ₂F₁(a, b; c; z) for the small parameter set we need.
//!
//! Lee's multilook phase PDF needs `₂F₁(L, 1; ½; β²)` with `β = γ cos(φ)`.
//! `β² ∈ [0, 1)`. Strategy:
//!  - |z| < 0.7  → Gauss series.
//!  - |z| ≥ 0.7 and z > 0 → Pfaff transform w = z/(z-1) ∈ (-∞, 0]; for our
//!    case w ∈ (-∞, -1], Gauss series still diverges. So we use the Euler
//!    transform first to move the singularity:
//!        ₂F₁(L, 1; ½; z) = (1−z)^(−L−½) · ₂F₁(½−L, −½; ½; z)
//!    The new series has |z| < 1 and parameters where it converges OK.
//!
//! We work entirely in f64 because (1−z)^(−L−½) can be huge.

/// Gauss series. `max_terms` is enforced for safety.
fn gauss_series(a: f64, b: f64, c: f64, z: f64, max_terms: usize) -> f64 {
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for k in 0..max_terms {
        let kf = k as f64;
        let num = (a + kf) * (b + kf) * z;
        let den = (c + kf) * (kf + 1.0);
        term *= num / den;
        sum += term;
        // Stop when relative term is tiny.
        if term.abs() <= 1e-15 * sum.abs() {
            break;
        }
    }
    sum
}

/// ₂F₁(a, b; c; z) for `z ∈ (-∞, 1)`. Uses Euler transform when z is large.
pub fn hyp2f1_f64(a: f64, b: f64, c: f64, z: f64) -> f64 {
    if z < 0.7 {
        return gauss_series(a, b, c, z, 1000);
    }
    // Euler: ₂F₁(a, b; c; z) = (1-z)^(c-a-b) · ₂F₁(c-a, c-b; c; z)
    let pref = (1.0 - z).powf(c - a - b);
    pref * gauss_series(c - a, c - b, c, z, 1000)
}

/// `ln ₂F₁(a, b; c; z)` for `a, b, c > 0` and `z ∈ [0, 1)`.
///
/// In this parameter range every term of the Gauss series is positive, so there
/// is no cancellation and the only hazard is overflow of the partial sum. That
/// is handled by rescaling and carrying the scale in the logarithm, which makes
/// the result usable far beyond the point where the value itself overflows f64
/// (e.g. `₂F₁(80, 1; ½; 0.98) ≈ 9e137`).
///
/// This is what lets [`super::lee_pdf`] use a single formula instead of
/// switching to an Euler-transformed series whose terms alternate: that series
/// is what returned negative densities at high coherence.
///
/// Convergence is geometric with ratio `z`, so `z` very close to 1 needs many
/// terms. Callers are LUT builders, which are cached per `nlooks`, so the cost
/// is paid once.
pub fn ln_hyp2f1_pos(a: f64, b: f64, c: f64, z: f64) -> f64 {
    debug_assert!(a > 0.0 && b > 0.0 && c > 0.0, "positive-term form only");
    debug_assert!((0.0..1.0).contains(&z), "z must be in [0, 1)");
    if z <= 0.0 {
        return 0.0;
    }
    const RESCALE: f64 = 1e250;
    let ln_rescale = RESCALE.ln();
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    let mut ln_scale = 0.0_f64;
    for k in 0..200_000 {
        let kf = k as f64;
        term *= (a + kf) * (b + kf) * z / ((c + kf) * (kf + 1.0));
        sum += term;
        if sum > RESCALE {
            sum /= RESCALE;
            term /= RESCALE;
            ln_scale += ln_rescale;
        }
        if term <= 1e-16 * sum {
            break;
        }
    }
    sum.ln() + ln_scale
}

/// f32 convenience wrapper.
pub fn hyp2f1(a: f32, b: f32, c: f32, z: f32) -> f32 {
    hyp2f1_f64(a as f64, b as f64, c as f64, z as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference identities (no scipy needed):
    //   ₂F₁(a, b; b; z) = (1-z)^(-a)   → independent of b.
    //   ₂F₁(1, 1; 2; z) = -ln(1-z)/z

    #[test]
    fn matches_known_identity_pow() {
        // ₂F₁(2, 1; 1; 0.5) = (0.5)^(-2) = 4
        let v = hyp2f1_f64(2.0, 1.0, 1.0, 0.5);
        assert!((v - 4.0).abs() < 1e-9, "got {v}, expected 4");
        // ₂F₁(3, 5; 5; 0.3) = (0.7)^(-3) = 1/0.343 ≈ 2.915...
        let v = hyp2f1_f64(3.0, 5.0, 5.0, 0.3);
        assert!((v - (0.7_f64).powi(-3)).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn matches_known_log_identity() {
        // ₂F₁(1, 1; 2; z) = -ln(1-z)/z
        for &z in &[0.1, 0.3, 0.6, 0.85] {
            let v = hyp2f1_f64(1.0, 1.0, 2.0, z);
            let expected = -(1.0_f64 - z).ln() / z;
            assert!(
                (v - expected).abs() / expected.abs() < 1e-9,
                "z={z}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn z_zero_returns_one() {
        assert!((hyp2f1_f64(3.0, 1.0, 0.5, 0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn large_z_finite() {
        // ₂F₁(5, 1; 0.5; 0.81) should be finite and large.
        let v = hyp2f1_f64(5.0, 1.0, 0.5, 0.81);
        assert!(v.is_finite() && v > 0.0, "got {v}");
    }
}
