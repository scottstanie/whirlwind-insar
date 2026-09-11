//! Linear phase-ramp fit and removal (de-ramp) for wrapped interferograms.
//!
//! Large-scale phase ramps - ionospheric (the dominant one for NISAR) or
//! tropospheric - add many fringes across a frame. In the NISAR fixed-PRF mode
//! the valid mask splits a scene into subswath rectangles that must be bridged
//! back together; a steep ramp makes the per-region 2π offsets large and the
//! Goldstein / interpolation / bridge post-passes struggle to resolve them.
//! Fitting and subtracting a plane before unwrapping flattens the fringe rate,
//! and adding the same plane back afterward restores the true phase.
//!
//! **Congruence-preserving.** For ANY plane `r`, unwrapping `z · exp(-i·r)` and
//! adding `r` back yields a field congruent (mod 2π) to `arg(z)` at every pixel:
//!
//! ```text
//! unw = unwrap(z · e^{-i r}) + r
//!     ≡ arg(z · e^{-i r}) + r   (mod 2π)
//!     = (arg(z) - r)  + r       (mod 2π)   = arg(z)   (mod 2π)
//! ```
//!
//! So the de-ramp can only help or be neutral - it never changes which
//! integer-cycle solution is valid, and every per-pixel value the caller passed
//! in is preserved by the final K-transfer. The fitted plane only needs to be
//! *close* to the real ramp to reduce the fringe rate; it does not have to be
//! exact.
//!
//! **Estimator.** The plane is the mean wrapped phase gradient, recovered as the
//! complex covariance (circular mean) of adjacent-pixel phase differences. This
//! is single-pass, `O(1)` extra memory (two complex accumulators, no full-size
//! temporary and no FFT), sub-pixel accurate (not tied to FFT bins), and robust:
//! localized signals such as deformation or terrain, whose gradients do not share
//! a consistent direction, average out and leave the frame-spanning ramp.

use ndarray::{Array2, ArrayView2};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

/// Row/col slopes (radians per pixel) of a fitted linear phase ramp
/// `φ(i, j) = row_slope · i + col_slope · j`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RampSlopes {
    /// Phase increment per row step (down, axis 0), in radians.
    pub row_slope: f32,
    /// Phase increment per column step (right, axis 1), in radians.
    pub col_slope: f32,
}

/// True for a pixel that may enter the fit: finite, nonzero (`0+0j` is the nodata
/// convention), and - if a mask is given - valid.
#[inline]
fn is_valid(z: Complex32, valid: Option<bool>) -> bool {
    z.re.is_finite() && z.im.is_finite() && (z.re != 0.0 || z.im != 0.0) && valid.unwrap_or(true)
}

/// Unit phasor `z / |z|` in f64. Assumes `|z| > 0` (guaranteed by [`is_valid`]).
/// Normalising to unit magnitude makes every valid pixel vote equally, so bright
/// targets do not dominate the gradient (the same choice the Goldstein filter
/// makes).
#[inline]
fn unit(z: Complex32) -> Complex64 {
    let r = (z.re as f64).hypot(z.im as f64);
    Complex64::new(z.re as f64 / r, z.im as f64 / r)
}

/// Fit the dominant linear phase ramp of a wrapped interferogram.
///
/// Estimates the mean wrapped phase gradient in each axis via the circular mean
/// of adjacent-pixel phase differences:
///
/// ```text
/// row_slope = arg( Σ  û[i+1, j] · conj(û[i, j]) )
/// col_slope = arg( Σ  û[i, j+1] · conj(û[i, j]) )
/// ```
///
/// with `û = z / |z|` the unit phasor. Only pairs whose BOTH endpoints pass
/// [`is_valid`] contribute, so nodata (`0+0j`), non-finite pixels, and the gaps
/// between NISAR subswaths are skipped rather than dragging the estimate toward
/// zero. The sums use `Complex64` so the reduction stays precise over the tens of
/// millions of terms in a full frame.
///
/// The estimate is unambiguous up to `|slope| < π` (half a fringe per pixel, the
/// aliasing limit); a ramp steeper than that cannot be unwrapped regardless. An
/// empty or ramp-free scene returns `(0, 0)` - a no-op de-ramp.
///
/// Runs in a single rayon pass over rows with `O(1)` extra memory.
pub fn fit_ramp(igram: ArrayView2<Complex32>, mask: Option<ArrayView2<bool>>) -> RampSlopes {
    let (m, n) = igram.dim();
    if m == 0 || n == 0 {
        return RampSlopes {
            row_slope: 0.0,
            col_slope: 0.0,
        };
    }
    // One parallel pass over rows. Row i accumulates its within-row column
    // differences (col direction) and its differences to row i+1 (row
    // direction); the per-row partials reduce into two global complex sums.
    let zero = Complex64::new(0.0, 0.0);
    let (row_acc, col_acc) = (0..m)
        .into_par_iter()
        .map(|i| {
            let mut col_acc = zero;
            let mut row_acc = zero;
            for j in 0..n {
                let z = igram[(i, j)];
                if !is_valid(z, mask.map(|mm| mm[(i, j)])) {
                    continue;
                }
                let uz_conj = unit(z).conj();
                if j + 1 < n {
                    let zr = igram[(i, j + 1)];
                    if is_valid(zr, mask.map(|mm| mm[(i, j + 1)])) {
                        col_acc += unit(zr) * uz_conj;
                    }
                }
                if i + 1 < m {
                    let zd = igram[(i + 1, j)];
                    if is_valid(zd, mask.map(|mm| mm[(i + 1, j)])) {
                        row_acc += unit(zd) * uz_conj;
                    }
                }
            }
            (row_acc, col_acc)
        })
        .reduce(|| (zero, zero), |a, b| (a.0 + b.0, a.1 + b.1));
    RampSlopes {
        row_slope: row_acc.arg() as f32,
        col_slope: col_acc.arg() as f32,
    }
}

/// De-ramp a wrapped interferogram: return `igram · exp(-i·(row·i + col·j))`.
///
/// Removes the fitted plane from the phase while preserving magnitude. Nodata
/// (`0+0j`) and non-finite pixels stay `0+0j`, matching the solver's nodata
/// convention. The plane angle is evaluated in f64 so it stays accurate across
/// frames tens of thousands of pixels wide, where an f32 `sin`/`cos` argument
/// (which can reach thousands of radians) would lose precision.
///
/// Runs in a single rayon pass over rows; one output allocation, no temporaries.
pub fn deramp(igram: ArrayView2<Complex32>, slopes: RampSlopes) -> Array2<Complex32> {
    let (m, n) = igram.dim();
    let a = slopes.row_slope as f64;
    let b = slopes.col_slope as f64;
    let mut out = Array2::<Complex32>::zeros((m, n));
    if let Some(buf) = out.as_slice_mut() {
        buf.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let ai = a * i as f64;
            for (j, o) in row.iter_mut().enumerate() {
                let z = igram[(i, j)];
                if !z.re.is_finite() || !z.im.is_finite() || (z.re == 0.0 && z.im == 0.0) {
                    continue; // leave nodata as 0+0j
                }
                // Angle to REMOVE is (ai + b·j); multiply by exp(-i·angle).
                let (s, c) = (-(ai + b * j as f64)).sin_cos();
                *o = z * Complex32::new(c as f32, s as f32);
            }
        });
    }
    out
}

/// Add a linear plane `row·i + col·j` back onto an unwrapped-phase array.
///
/// The unwrapped-domain inverse of [`deramp`]: after unwrapping the de-ramped
/// phase, this restores the fitted ramp so the result is a valid unwrapping of
/// the original interferogram. Non-finite pixels (masked nodata the solver left
/// as `NaN`) pass through unchanged - `NaN + x` stays `NaN`. Evaluated in f64 for
/// accuracy at large `i`, `j`.
///
/// Runs in a single rayon pass over rows; one output allocation, no temporaries.
pub fn add_ramp(phase: ArrayView2<f32>, slopes: RampSlopes) -> Array2<f32> {
    let (m, n) = phase.dim();
    let a = slopes.row_slope as f64;
    let b = slopes.col_slope as f64;
    let mut out = Array2::<f32>::zeros((m, n));
    if let Some(buf) = out.as_slice_mut() {
        buf.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let ai = a * i as f64;
            for (j, o) in row.iter_mut().enumerate() {
                *o = (phase[(i, j)] as f64 + ai + b * j as f64) as f32;
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::f32::consts::PI;

    /// Build `exp(i·(a·i + b·j))` on an (m, n) grid.
    fn ramp_igram(m: usize, n: usize, a: f32, b: f32) -> Array2<Complex32> {
        Array2::from_shape_fn((m, n), |(i, j)| {
            Complex32::from_polar(1.0, a * i as f32 + b * j as f32)
        })
    }

    #[test]
    fn fit_recovers_known_slopes() {
        let (a, b) = (0.20_f32, -0.13_f32);
        let ig = ramp_igram(200, 180, a, b);
        let s = fit_ramp(ig.view(), None);
        assert!((s.row_slope - a).abs() < 1e-4, "row {} vs {a}", s.row_slope);
        assert!((s.col_slope - b).abs() < 1e-4, "col {} vs {b}", s.col_slope);
    }

    #[test]
    fn deramp_then_fit_is_flat() {
        let (a, b) = (0.31_f32, 0.07_f32);
        let ig = ramp_igram(150, 160, a, b);
        let s = fit_ramp(ig.view(), None);
        let flat = deramp(ig.view(), s);
        // The de-ramped phase has no dominant slope left.
        let s2 = fit_ramp(flat.view(), None);
        assert!(s2.row_slope.abs() < 1e-4, "residual row {}", s2.row_slope);
        assert!(s2.col_slope.abs() < 1e-4, "residual col {}", s2.col_slope);
    }

    #[test]
    fn deramp_preserves_nodata_and_magnitude() {
        let mut ig = ramp_igram(32, 40, 0.1, 0.2);
        // Vary amplitude and punch a nodata hole.
        ig[(5, 6)] = Complex32::new(3.0, 0.0) * ig[(5, 6)];
        ig[(10, 10)] = Complex32::new(0.0, 0.0);
        let s = fit_ramp(ig.view(), None);
        let out = deramp(ig.view(), s);
        assert_eq!(
            out[(10, 10)],
            Complex32::new(0.0, 0.0),
            "nodata not preserved"
        );
        // Magnitude preserved everywhere (de-ramp is a unit-modulus multiply).
        for i in 0..ig.nrows() {
            for j in 0..ig.ncols() {
                assert!((out[(i, j)].norm() - ig[(i, j)].norm()).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn add_ramp_is_deramp_inverse_on_phase() {
        // Round-trip: unwrapped-domain add_ramp undoes a de-ramp in phase.
        let (a, b) = (0.05_f32, 0.09_f32);
        let m = 64;
        let n = 48;
        // A smooth unwrapped truth = ramp + small bump.
        let truth = Array2::from_shape_fn((m, n), |(i, j)| {
            a * i as f32 + b * j as f32 + 0.5 * ((i as f32) / 10.0).sin()
        });
        let ig = truth.mapv(|p| Complex32::from_polar(1.0, p));
        let s = fit_ramp(ig.view(), None);
        let flat = deramp(ig.view(), s);
        // Wrapped de-ramped phase, then add the ramp back in the unwrapped domain.
        let flat_phase = flat.mapv(|z| z.arg());
        let restored = add_ramp(flat_phase.view(), s);
        // restored ≡ truth (mod 2π) at every pixel.
        for i in 0..m {
            for j in 0..n {
                let d = restored[(i, j)] - truth[(i, j)];
                let k = (d / (2.0 * PI)).round();
                assert!(
                    (d - 2.0 * PI * k).abs() < 1e-3,
                    "not congruent at ({i},{j}): d={d}"
                );
            }
        }
    }

    #[test]
    fn mask_excludes_gap_pixels() {
        // Two subswaths (columns) of a shared ramp separated by a nodata gap.
        // The gap pixels are 0+0j and masked; the fit must still recover the
        // shared slope from the two swaths' internal gradients.
        let (a, b) = (0.12_f32, 0.18_f32);
        let mut ig = ramp_igram(120, 120, a, b);
        let mut mask = Array2::from_elem((120, 120), true);
        for i in 0..120 {
            for j in 58..62 {
                ig[(i, j)] = Complex32::new(0.0, 0.0);
                mask[(i, j)] = false;
            }
        }
        let s = fit_ramp(ig.view(), Some(mask.view()));
        assert!((s.row_slope - a).abs() < 1e-3, "row {} vs {a}", s.row_slope);
        assert!((s.col_slope - b).abs() < 1e-3, "col {} vs {b}", s.col_slope);
    }

    #[test]
    fn empty_or_flat_scene_returns_zero() {
        // All-nodata scene.
        let ig = Array2::<Complex32>::zeros((16, 16));
        let s = fit_ramp(ig.view(), None);
        assert_eq!(s.row_slope, 0.0);
        assert_eq!(s.col_slope, 0.0);
        // A constant-phase scene has zero gradient.
        let flat = Array2::from_elem((16, 16), Complex32::from_polar(1.0, 0.7));
        let s2 = fit_ramp(flat.view(), None);
        assert!(s2.row_slope.abs() < 1e-6 && s2.col_slope.abs() < 1e-6);
    }
}
