//! 2D LUT in (α, γ) for Lee's PDF at a fixed `nlooks`. Built lazily; one LUT
//! per (rounded) nlooks. Bilinear interpolation.
//!
//! Sidesteps the per-pixel cost of evaluating `₂F₁` (the original reason for
//! Carballo's pre-baked splines).

use std::collections::HashMap;
use std::f32::consts::TAU;
use std::sync::{Mutex, OnceLock};

const N_ALPHA: usize = 257; // odd → exact zero sample
const N_GAMMA: usize = 129;
const ALPHA_LO: f32 = -3.0 * TAU; // wide enough to accommodate ±2π shifts
const ALPHA_HI: f32 = 3.0 * TAU;
const GAMMA_LO: f32 = 0.0;
const GAMMA_HI: f32 = 0.999;

pub struct Lut {
    nlooks: f32,
    values: Vec<f32>,
}

impl Lut {
    pub fn build(nlooks: f32) -> Self {
        let mut values = vec![0.0_f32; N_ALPHA * N_GAMMA];
        for i in 0..N_GAMMA {
            let gamma = GAMMA_LO + (GAMMA_HI - GAMMA_LO) * (i as f32) / ((N_GAMMA - 1) as f32);
            for j in 0..N_ALPHA {
                let alpha = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * (j as f32) / ((N_ALPHA - 1) as f32);
                values[i * N_ALPHA + j] = super::lee_pdf::pdf(alpha, gamma, nlooks);
            }
        }
        Self { nlooks, values }
    }

    #[inline]
    pub fn nlooks(&self) -> f32 {
        self.nlooks
    }

    /// Bilinear sample. Clamps inputs to LUT bounds.
    pub fn eval(&self, alpha: f32, gamma: f32) -> f32 {
        let a = alpha.clamp(ALPHA_LO, ALPHA_HI);
        let g = gamma.clamp(GAMMA_LO, GAMMA_HI);

        let ai = (a - ALPHA_LO) / (ALPHA_HI - ALPHA_LO) * ((N_ALPHA - 1) as f32);
        let gi = (g - GAMMA_LO) / (GAMMA_HI - GAMMA_LO) * ((N_GAMMA - 1) as f32);

        let a0 = ai.floor() as usize;
        let g0 = gi.floor() as usize;
        let a1 = (a0 + 1).min(N_ALPHA - 1);
        let g1 = (g0 + 1).min(N_GAMMA - 1);
        let fa = ai - (a0 as f32);
        let fg = gi - (g0 as f32);

        let v00 = self.values[g0 * N_ALPHA + a0];
        let v01 = self.values[g0 * N_ALPHA + a1];
        let v10 = self.values[g1 * N_ALPHA + a0];
        let v11 = self.values[g1 * N_ALPHA + a1];
        let v0 = v00 * (1.0 - fa) + v01 * fa;
        let v1 = v10 * (1.0 - fa) + v11 * fa;
        v0 * (1.0 - fg) + v1 * fg
    }
}

type LutMap = Mutex<HashMap<u32, &'static Lut>>;

fn cache() -> &'static LutMap {
    static CACHE: OnceLock<LutMap> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get a LUT for `nlooks`, building (and caching) one if needed. We round
/// nlooks to one decimal so e.g. 9.97 ≈ 10.0 shares a cache entry.
pub fn get_or_build(nlooks: f32) -> &'static Lut {
    let key = (nlooks * 10.0).round() as u32;
    let mut map = cache().lock().unwrap();
    if let Some(l) = map.get(&key) {
        return l;
    }
    // Leak the LUT - it's a ~512 KiB one-time cost, and we want 'static.
    let lut: &'static Lut = Box::leak(Box::new(Lut::build(nlooks)));
    map.insert(key, lut);
    lut
}

// =============================================================================
// γ → σ² lookup (wrapped phase variance from the full Lee 1994 PDF).
// =============================================================================

const N_GAMMA_VAR: usize = 1024;
/// Sample γ across `[0, 0.999]` - same upper bound as the 2D PDF LUT.
const GAMMA_VAR_LO: f32 = 0.0;
const GAMMA_VAR_HI: f32 = 0.999;

pub struct VarianceLut {
    nlooks: f32,
    sigma_sq: Vec<f32>, // length N_GAMMA_VAR
}

impl VarianceLut {
    /// Build a γ → σ² lookup by numerically integrating
    /// `σ² = ∫_{-π}^{π} α² · p(α | γ, L) dα` for each γ sample. The PDF is
    /// symmetric around α=0, so the mean integral is 0 and the second
    /// moment equals the variance directly.
    pub fn build(nlooks: f32) -> Self {
        use std::f32::consts::PI;
        const N_ALPHA: usize = 1024; // mid-point samples over (-π, π]
        let dalpha = 2.0 * PI / (N_ALPHA as f32);
        let mut sigma_sq = vec![0.0_f32; N_GAMMA_VAR];
        for i in 0..N_GAMMA_VAR {
            let gamma = GAMMA_VAR_LO
                + (GAMMA_VAR_HI - GAMMA_VAR_LO) * (i as f32) / ((N_GAMMA_VAR - 1) as f32);
            // Mid-point rule: α_k = -π + (k + 0.5) · dα for k in 0..N_ALPHA.
            // PDF is symmetric so the mean ∫ α p(α) dα = 0; ∫ α² p dα is the
            // second central moment = variance.
            let mut s2 = 0.0_f32;
            let mut norm = 0.0_f32;
            for k in 0..N_ALPHA {
                let alpha = -PI + (k as f32 + 0.5) * dalpha;
                let p = super::lee_pdf::pdf(alpha, gamma, nlooks);
                s2 += alpha * alpha * p;
                norm += p;
            }
            // Normalize in case the PDF doesn't integrate to exactly 1
            // (numerical error from finite samples). norm·dα ≈ 1.
            sigma_sq[i] = if norm > 1e-12 {
                s2 / norm
            } else {
                (PI * PI) / 3.0
            };
        }
        Self { nlooks, sigma_sq }
    }

    #[inline]
    pub fn nlooks(&self) -> f32 {
        self.nlooks
    }

    /// Lookup σ² at γ via linear interpolation. Clamps γ to `[0, 0.999]`.
    pub fn eval(&self, gamma: f32) -> f32 {
        let g = gamma.clamp(GAMMA_VAR_LO, GAMMA_VAR_HI);
        let gi = (g - GAMMA_VAR_LO) / (GAMMA_VAR_HI - GAMMA_VAR_LO) * ((N_GAMMA_VAR - 1) as f32);
        let i0 = gi.floor() as usize;
        let i1 = (i0 + 1).min(N_GAMMA_VAR - 1);
        let f = gi - (i0 as f32);
        self.sigma_sq[i0] * (1.0 - f) + self.sigma_sq[i1] * f
    }
}

type VarLutMap = Mutex<HashMap<u32, &'static VarianceLut>>;

fn var_cache() -> &'static VarLutMap {
    static CACHE: OnceLock<VarLutMap> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get a variance LUT for `nlooks`. Same cache-rounding convention as
/// [`get_or_build`].
pub fn get_or_build_variance(nlooks: f32) -> &'static VarianceLut {
    let key = (nlooks * 10.0).round() as u32;
    let mut map = var_cache().lock().unwrap();
    if let Some(l) = map.get(&key) {
        return l;
    }
    let lut: &'static VarianceLut = Box::leak(Box::new(VarianceLut::build(nlooks)));
    map.insert(key, lut);
    lut
}

// =============================================================================
// Carballo LLR cost LUT: -log(p1/p0) from Lee 1994 CDF
// =============================================================================
//
// p0(α) = ∫_{α−π}^{π}  Lee_PDF(t | γ, L) dt  (for α > 0, upper=π, lower=α−π)
// p1(α) = ∫_{−π}^{α−π} Lee_PDF(t | γ, L) dt  (for α > 0; = CDF(α−π))
// cost(α, γ) = min(−log(p1/p0), MAX_CARBALLO_COST)
//
// At α = π  (wrap line): CDF(0) = 0.5 → p1=p0=0.5 → cost = 0.
// At α → 0+ (smooth):    CDF(−π)→0   → p1→0         → cost = MAX_CARBALLO_COST.
// For α ≤ 0:             p1 = 0 by construction        → cost = MAX_CARBALLO_COST.
//
// The LUT is indexed by (gamma, alpha) with bilinear interpolation. Built once
// per nlooks (rounded to 0.1) and leaked to 'static - ~160 KB per nlooks.

const N_ALPHA_CARB: usize = 501;
const N_GAMMA_CARB: usize = 101;
/// Maximum Carballo LLR cost, in nats. Arcs with p1 ≈ 0 are capped here.
pub const MAX_CARBALLO_COST: f32 = 50.0;

pub struct CarballoLut {
    values: Vec<f32>, // [N_GAMMA_CARB x N_ALPHA_CARB], row = gamma, col = alpha
}

impl CarballoLut {
    /// Build the LUT by integrating the Lee PDF CDF for each γ sample.
    pub fn build(nlooks: f32) -> Self {
        use std::f32::consts::PI;
        const N_CDF: usize = 2001; // trapezoidal nodes over [−π, π]
        let dt = 2.0 * PI as f64 / ((N_CDF - 1) as f64);

        let mut values = vec![MAX_CARBALLO_COST; N_GAMMA_CARB * N_ALPHA_CARB];

        for ig in 0..N_GAMMA_CARB {
            let gamma = (ig as f32) / ((N_GAMMA_CARB - 1) as f32) * 0.999_f32;

            // 1. Build CDF[k] = ∫_{-π}^{t_k} Lee_PDF(s) ds via trapezoidal rule.
            let mut cdf = vec![0.0_f64; N_CDF];
            for k in 1..N_CDF {
                let t0 = -PI as f64 + ((k - 1) as f64) * dt;
                let t1 = -PI as f64 + (k as f64) * dt;
                let p0 = super::lee_pdf::pdf(t0 as f32, gamma, nlooks) as f64;
                let p1 = super::lee_pdf::pdf(t1 as f32, gamma, nlooks) as f64;
                cdf[k] = cdf[k - 1] + 0.5 * (p0 + p1) * dt;
            }
            let total = cdf[N_CDF - 1].max(1e-12);
            for c in cdf.iter_mut() {
                *c /= total;
            }

            // Linearly interpolated CDF lookup at t ∈ [-π, π].
            let cdf_at = |t: f32| -> f64 {
                let tc = (t as f64).clamp(-PI as f64, PI as f64);
                let ki = (tc - (-PI as f64)) / dt;
                let k0 = (ki.floor() as usize).min(N_CDF - 2);
                let f = ki - k0 as f64;
                cdf[k0] * (1.0 - f) + cdf[k0 + 1] * f
            };

            // 2. For each alpha sample compute the Carballo cost.
            for ia in 0..N_ALPHA_CARB {
                let alpha = -PI + (ia as f32) / ((N_ALPHA_CARB - 1) as f32) * 2.0 * PI;
                let cost = if alpha <= 0.0 {
                    MAX_CARBALLO_COST
                } else {
                    let p1 = cdf_at(alpha - PI);
                    let p0 = 1.0 - p1;
                    if p1 < 1e-30 {
                        MAX_CARBALLO_COST
                    } else {
                        (-(p1 / p0).ln() as f32).clamp(0.0, MAX_CARBALLO_COST)
                    }
                };
                values[ig * N_ALPHA_CARB + ia] = cost;
            }
        }
        Self { values }
    }

    /// Bilinear lookup. Returns cost ∈ [0, MAX_CARBALLO_COST].
    #[inline]
    pub fn eval(&self, alpha: f32, gamma: f32) -> f32 {
        use std::f32::consts::PI;
        let a = alpha.clamp(-PI, PI);
        let g = gamma.clamp(0.0, 0.999);

        let ai = (a + PI) / (2.0 * PI) * ((N_ALPHA_CARB - 1) as f32);
        let gi = g / 0.999 * ((N_GAMMA_CARB - 1) as f32);

        let a0 = (ai.floor() as usize).min(N_ALPHA_CARB - 2);
        let g0 = (gi.floor() as usize).min(N_GAMMA_CARB - 2);
        let fa = ai - a0 as f32;
        let fg = gi - g0 as f32;

        let v00 = self.values[g0 * N_ALPHA_CARB + a0];
        let v01 = self.values[g0 * N_ALPHA_CARB + a0 + 1];
        let v10 = self.values[(g0 + 1) * N_ALPHA_CARB + a0];
        let v11 = self.values[(g0 + 1) * N_ALPHA_CARB + a0 + 1];
        let v0 = v00 * (1.0 - fa) + v01 * fa;
        let v1 = v10 * (1.0 - fa) + v11 * fa;
        v0 * (1.0 - fg) + v1 * fg
    }
}

type CarbLutMap = Mutex<HashMap<u32, &'static CarballoLut>>;

fn carb_cache() -> &'static CarbLutMap {
    static CACHE: OnceLock<CarbLutMap> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get the Carballo cost LUT for `nlooks`, building and caching on first use.
/// nlooks is rounded to one decimal (e.g., 9.97 → 10.0) for cache sharing.
pub fn get_or_build_carballo(nlooks: f32) -> &'static CarballoLut {
    let key = (nlooks * 10.0).round() as u32;
    let mut map = carb_cache().lock().unwrap();
    if let Some(l) = map.get(&key) {
        return l;
    }
    let lut: &'static CarballoLut = Box::leak(Box::new(CarballoLut::build(nlooks)));
    map.insert(key, lut);
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_matches_direct_pdf() {
        let l = Lut::build(10.0);
        // Spot-check several (alpha, gamma) pairs against direct PDF eval.
        for (a, g) in [(0.0, 0.5), (1.0, 0.8), (-2.5, 0.3), (3.0, 0.9)] {
            let lut_v = l.eval(a, g);
            let ref_v = super::super::lee_pdf::pdf(a, g, 10.0);
            assert!(
                (lut_v - ref_v).abs() < 0.01 * ref_v.max(0.01),
                "LUT vs direct mismatch at ({a}, {g}): {lut_v} vs {ref_v}"
            );
        }
    }

    #[test]
    fn carballo_lut_wrap_line_has_zero_cost() {
        use std::f32::consts::PI;
        // At alpha=π, cost must be 0 regardless of gamma: CDF(0)=0.5 → p1=p0.
        let lut = CarballoLut::build(10.0);
        for ig in 0..N_GAMMA_CARB {
            let gamma = (ig as f32) / ((N_GAMMA_CARB - 1) as f32) * 0.999;
            let c = lut.eval(PI, gamma);
            assert!(
                c < 1e-2,
                "cost at alpha=π, gamma={gamma} should be ~0, got {c}"
            );
        }
    }

    #[test]
    fn carballo_lut_smooth_interior_has_max_cost() {
        // At alpha=0.0 (or negative), cost must equal MAX_CARBALLO_COST.
        let lut = CarballoLut::build(10.0);
        let c_zero = lut.eval(0.0, 0.8);
        let c_neg = lut.eval(-1.0, 0.8);
        assert_eq!(
            c_zero, MAX_CARBALLO_COST,
            "cost at alpha=0 should be MAX, got {c_zero}"
        );
        assert_eq!(
            c_neg, MAX_CARBALLO_COST,
            "cost at alpha<0 should be MAX, got {c_neg}"
        );
    }

    #[test]
    fn carballo_lut_monotone_decreasing_in_alpha() {
        use std::f32::consts::PI;
        // For fixed gamma > 0, cost should decrease monotonically from 0 to π.
        let lut = CarballoLut::build(16.0);
        let gamma = 0.7;
        let alphas: Vec<f32> = (1..=50).map(|k| k as f32 / 50.0 * PI).collect();
        let costs: Vec<f32> = alphas.iter().map(|&a| lut.eval(a, gamma)).collect();
        for i in 1..costs.len() {
            assert!(
                costs[i] <= costs[i - 1] + 1e-3,
                "cost should decrease in alpha: cost[{i}]={} > cost[{}]={}",
                costs[i],
                i - 1,
                costs[i - 1]
            );
        }
    }
}
