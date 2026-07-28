//! Trilinear interpolation of the Carballo/Touzi PDF cost tables.
//!
//! The embedded binary blobs are little-endian `f32` dumps of the `values`
//! arrays produced by `scripts/generate_carballo_tables.py` in its default
//! analytic mode. They encode the full cost model - Lee 1994 multilook phase
//! noise + Carballo slope marginalization + the Touzi 1999 true-coherence
//! marginalization.
//!
//! The model is documented and reproducible in `scripts/generate_carballo_tables.py`:
//! its default analytic mode computes the model from theory and regenerates
//! these blobs byte-for-byte (`--write-rust-bins`). The `--source-table-dir`
//! mode instead re-exports the historical ww-orig `.npz` / `.pkl` tables and is
//! retained only for provenance.
//! The tables store `p0(α, γ, L)` and `p1(α, γ, L)`:
//!   p0 = P(Δk = 0 | observation)
//!   p1 = P(Δk = +1 | observation)   [note: p0 + p1 ≠ 1 in general]
//! A −1 jump is the +1 jump of the opposite direction: the cost builder
//! prices each edge's two arcs as cost(α, …) and cost(−α, …).
//!
//! Grid dimensions (all in C-row-major order in the stored arrays):
//!   axis 0 - phase_diff α : 31 samples, uniformly spaced in [-π, π]
//!   axis 1 - coherence γ  : 21 samples, [0.0, 0.05, …, 1.0]
//!   axis 2 - nlooks L     : 21 samples, log-spaced [1.0, …, 300.0]
//!
//! Sizing rationale (measured off-node against a 241x51x21 reference, on arcs
//! cheap enough to carry flow):
//!
//!   * Store p0/p1, NOT the cost. They are bounded and smooth on [0, 1]; the
//!     cost has a log divergence and two clamps, and interpolating it directly
//!     is ~3x worse at the same grid.
//!   * α is nearly saturated at 31 nodes: refining to 121 buys only ~10%,
//!     because the stored quantities are *integrals* of the phase PDF and stay
//!     smooth in α even when the PDF itself is narrow.
//!   * γ and L are the binding axes (~30% each going 11 -> 21 when all nodes
//!     span [1, 80]). For the table actually embedded here, whose 21 L nodes
//!     cover the wider [1, 300] interval, an off-node check over the shared
//!     [1, 80] domain cuts mean interpolation error from 0.260 to 0.134 nats
//!     (~1.9x) and p99 from 3.1 to 1.7 nats. The earlier 0.090 / 2.9x result
//!     was for a different candidate with all 21 L nodes inside [1, 80].
//!   * The L axis runs to 300 because real GUNW coherence carries far more than
//!     80 looks (water-fit measurements give L ~ 143-276) and the arc costs
//!     keep changing that far out. 21 log-spaced nodes over [1, 300] is finer
//!     in log-L (ratio 1.33) than the old 11 over [1, 80] (ratio 1.55).

use std::sync::OnceLock;

static GRID_PHASE: &[u8] = include_bytes!("carballo_grid_phase.bin");
static GRID_CORR: &[u8] = include_bytes!("carballo_grid_corr.bin");
static GRID_NLOOKS: &[u8] = include_bytes!("carballo_grid_nlooks.bin");
static P0_BYTES: &[u8] = include_bytes!("carballo_p0.bin");
static P1_BYTES: &[u8] = include_bytes!("carballo_p1.bin");

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(
        bytes.len() % 4,
        0,
        "Carballo LUT byte blob length must be a multiple of 4"
    );
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub struct CarballoSplineLut {
    phase: Vec<f32>,  // length 31
    corr: Vec<f32>,   // length 21
    nlooks: Vec<f32>, // length 21
    p0: Vec<f32>,     // shape [31][21][21], row-major
    p1: Vec<f32>,
    n_corr: usize,
    n_nlooks: usize,
}

impl CarballoSplineLut {
    fn load_embedded() -> Self {
        let phase = bytes_to_f32(GRID_PHASE);
        let corr = bytes_to_f32(GRID_CORR);
        let nlooks = bytes_to_f32(GRID_NLOOKS);
        let p0 = bytes_to_f32(P0_BYTES);
        let p1 = bytes_to_f32(P1_BYTES);
        Self::from_parts(phase, corr, nlooks, p0, p1)
    }

    fn from_parts(
        phase: Vec<f32>,
        corr: Vec<f32>,
        nlooks: Vec<f32>,
        p0: Vec<f32>,
        p1: Vec<f32>,
    ) -> Self {
        let n_corr = corr.len();
        let n_nlooks = nlooks.len();
        assert!(
            phase.len() >= 2 && n_corr >= 2 && n_nlooks >= 2,
            "Carballo LUT grids must each contain at least two points"
        );
        assert_eq!(
            p0.len(),
            phase.len() * n_corr * n_nlooks,
            "Carballo p0 table shape must match phase*corr*nlooks grids"
        );
        assert_eq!(
            p1.len(),
            phase.len() * n_corr * n_nlooks,
            "Carballo p1 table shape must match phase*corr*nlooks grids"
        );
        Self {
            phase,
            corr,
            nlooks,
            p0,
            p1,
            n_corr,
            n_nlooks,
        }
    }

    /// Binary-search for bracketing index; returns (lo, frac) clamped to grid.
    fn bracket(grid: &[f32], x: f32) -> (usize, f32) {
        let n = grid.len();
        let x = x.clamp(grid[0], grid[n - 1]);
        let lo = grid
            .partition_point(|&g| g <= x)
            .saturating_sub(1)
            .min(n - 2);
        let t = ((x - grid[lo]) / (grid[lo + 1] - grid[lo])).clamp(0.0, 1.0);
        (lo, t)
    }

    #[inline(always)]
    fn val(&self, vals: &[f32], ia: usize, ic: usize, il: usize) -> f32 {
        vals[ia * self.n_corr * self.n_nlooks + ic * self.n_nlooks + il]
    }

    fn trilinear(
        &self,
        vals: &[f32],
        ia: usize,
        ta: f32,
        ic: usize,
        tc: f32,
        il: usize,
        tl: f32,
    ) -> f32 {
        let v000 = self.val(vals, ia, ic, il);
        let v001 = self.val(vals, ia, ic, il + 1);
        let v010 = self.val(vals, ia, ic + 1, il);
        let v011 = self.val(vals, ia, ic + 1, il + 1);
        let v100 = self.val(vals, ia + 1, ic, il);
        let v101 = self.val(vals, ia + 1, ic, il + 1);
        let v110 = self.val(vals, ia + 1, ic + 1, il);
        let v111 = self.val(vals, ia + 1, ic + 1, il + 1);

        let v00 = v000 + tl * (v001 - v000);
        let v01 = v010 + tl * (v011 - v010);
        let v10 = v100 + tl * (v101 - v100);
        let v11 = v110 + tl * (v111 - v110);
        let v0 = v00 + tc * (v01 - v00);
        let v1 = v10 + tc * (v11 - v10);
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
    LUT.get_or_init(CarballoSplineLut::load_embedded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_tables_load_and_cost_is_sane() {
        let lut = get_or_load();
        // Grid shapes match the documented embedded layout.
        assert_eq!(lut.phase.len(), 31);
        assert_eq!(lut.corr.len(), 21);
        assert_eq!(lut.nlooks.len(), 21);
        // The looks axis is the *effective* cap on the default unwrap path:
        // `cost` brackets into this grid, so nlooks above the top node clamps
        // here rather than at `lut::MAX_COST_MODEL_NLOOKS`.
        assert_eq!(lut.nlooks[lut.nlooks.len() - 1], 300.0);
        // A wrap line (alpha = pi) is free to cut; a smooth interior edge costs more.
        let wrap = lut.cost(std::f32::consts::PI, 0.5, 16.0);
        let smooth = lut.cost(0.0, 0.5, 16.0);
        assert!(wrap >= 0 && smooth >= wrap);
    }
}
