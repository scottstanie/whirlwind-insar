//! 2D LUT in (α, γ) for Lee's PDF at a fixed `nlooks`. Built lazily; one LUT
//! per (rounded) nlooks. Bilinear interpolation.
//!
//! Sidesteps the per-pixel cost of evaluating `₂F₁` (the original reason for
//! Carballo's pre-baked splines).

use std::collections::HashMap;
use std::f32::consts::TAU;
use std::sync::{Mutex, OnceLock};

const N_ALPHA: usize = 257;   // odd → exact zero sample
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
    // Leak the LUT — it's a ~512 KiB one-time cost, and we want 'static.
    let lut: &'static Lut = Box::leak(Box::new(Lut::build(nlooks)));
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
}
