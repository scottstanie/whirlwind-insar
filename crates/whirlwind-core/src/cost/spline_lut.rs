//! Trilinear interpolation of the ww-orig Carballo PDF tables.
//!
//! The two binary blobs embed the same grid data as Python's
//! `carballo-pdf-0-spline.npz` / `carballo-pdf-1-spline.npz`.
//! The tables store `p0(α, γ, L)` and `p1(α, γ, L)`:
//!   p0 = P(Δk = 0 | observation)
//!   p1 = P(Δk = ±1 | observation)   [note: p0 + p1 ≠ 1 in general]
//!
//! Grid dimensions (all in C-row-major order in the stored arrays):
//!   axis 0 — phase_diff α : 31 samples, uniformly spaced in [-π, π]
//!   axis 1 — coherence γ  : 11 samples, [0.0, 0.1, …, 1.0]
//!   axis 2 — nlooks L     : 11 samples, log-spaced [1.0, …, 80.0]

use std::sync::OnceLock;

static GRID_PHASE: &[u8] = include_bytes!("carballo_grid_phase.bin");
static GRID_CORR: &[u8] = include_bytes!("carballo_grid_corr.bin");
static GRID_NLOOKS: &[u8] = include_bytes!("carballo_grid_nlooks.bin");
static P0_BYTES: &[u8] = include_bytes!("carballo_p0.bin");
static P1_BYTES: &[u8] = include_bytes!("carballo_p1.bin");

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub struct CarballoSplineLut {
    phase: Vec<f32>,    // length 31
    corr: Vec<f32>,     // length 11
    nlooks: Vec<f32>,   // length 11
    p0: Vec<f32>,       // shape [31][11][11], row-major
    p1: Vec<f32>,
    n_corr: usize,
    n_nlooks: usize,
}

impl CarballoSplineLut {
    fn load() -> Self {
        let phase = bytes_to_f32(GRID_PHASE);
        let corr = bytes_to_f32(GRID_CORR);
        let nlooks = bytes_to_f32(GRID_NLOOKS);
        let p0 = bytes_to_f32(P0_BYTES);
        let p1 = bytes_to_f32(P1_BYTES);
        let n_corr = corr.len();
        let n_nlooks = nlooks.len();
        assert_eq!(p0.len(), phase.len() * n_corr * n_nlooks);
        assert_eq!(p1.len(), phase.len() * n_corr * n_nlooks);
        Self { phase, corr, nlooks, p0, p1, n_corr, n_nlooks }
    }

    /// Binary-search for bracketing index; returns (lo, frac) clamped to grid.
    fn bracket(grid: &[f32], x: f32) -> (usize, f32) {
        let n = grid.len();
        let x = x.clamp(grid[0], grid[n - 1]);
        let lo = grid.partition_point(|&g| g <= x).saturating_sub(1).min(n - 2);
        let t = ((x - grid[lo]) / (grid[lo + 1] - grid[lo])).clamp(0.0, 1.0);
        (lo, t)
    }

    #[inline(always)]
    fn val(&self, vals: &[f32], ia: usize, ic: usize, il: usize) -> f32 {
        vals[ia * self.n_corr * self.n_nlooks + ic * self.n_nlooks + il]
    }

    fn trilinear(&self, vals: &[f32], ia: usize, ta: f32, ic: usize, tc: f32, il: usize, tl: f32) -> f32 {
        let v000 = self.val(vals, ia,   ic,   il);
        let v001 = self.val(vals, ia,   ic,   il+1);
        let v010 = self.val(vals, ia,   ic+1, il);
        let v011 = self.val(vals, ia,   ic+1, il+1);
        let v100 = self.val(vals, ia+1, ic,   il);
        let v101 = self.val(vals, ia+1, ic,   il+1);
        let v110 = self.val(vals, ia+1, ic+1, il);
        let v111 = self.val(vals, ia+1, ic+1, il+1);

        let v00 = v000 + tl * (v001 - v000);
        let v01 = v010 + tl * (v011 - v010);
        let v10 = v100 + tl * (v101 - v100);
        let v11 = v110 + tl * (v111 - v110);
        let v0  = v00  + tc * (v01  - v00);
        let v1  = v10  + tc * (v11  - v10);
        v0 + ta * (v1 - v0)
    }

    /// Arc cost = `round(100 * max(-ln(p1/p0), 0))` as i32.
    ///
    /// `alpha` is the signed phase gradient for this arc direction.
    /// `gamma` is the per-edge coherence (minimum of the two endpoint pixels).
    pub fn cost(&self, alpha: f32, gamma: f32, nlooks: f32) -> i32 {
        let (ia, ta) = Self::bracket(&self.phase, alpha);
        let (ic, tc) = Self::bracket(&self.corr, gamma);
        let (il, tl) = Self::bracket(&self.nlooks, nlooks);

        let p0 = self.trilinear(&self.p0, ia, ta, ic, tc, il, tl).max(1e-30);
        let p1 = self.trilinear(&self.p1, ia, ta, ic, tc, il, tl).max(1e-30);

        let raw = -f32::ln(p1 / p0);
        (100.0 * raw.max(0.0)) as i32
    }
}

static LUT: OnceLock<CarballoSplineLut> = OnceLock::new();

pub fn get_or_load() -> &'static CarballoSplineLut {
    LUT.get_or_init(CarballoSplineLut::load)
}
