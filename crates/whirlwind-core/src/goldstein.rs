//! Goldstein adaptive phase filter (Goldstein & Werner 1998).
//!
//! Overlapping-block 2D FFT filter on the wrapped interferogram. Each
//! `psize × psize` block is FFT-transformed, its magnitudes are shaped
//! by `|F|^alpha`, inverse-transformed, and overlap-added with a Hann
//! window into the output.
//!
//! Two ww-specific tweaks vs the classical Goldstein (see
//! `python/whirlwind/__init__.py::goldstein` for the matching Python
//! reference):
//!
//! 1. Input is normalised to unit magnitude before filtering. Without
//!    this, bright SLC pixels (urban) dominate the spectrum and
//!    `|F|^alpha` enhances amplitude structure rather than phase.
//! 2. Hann overlap-add window (smoother slope at the block centre than
//!    the dolphin/snaphu triangle window — fewer cross-block phase
//!    artifacts on the gradient estimate).
//!
//! Block iteration is parallelised over independent FFTs via rayon and
//! batched (write-back is serial) so peak memory stays small.

use ndarray::{Array2, ArrayView2, Axis, s};
use num_complex::Complex32;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Apply Goldstein adaptive filtering to a complex interferogram.
///
/// * `igram` — wrapped complex IG of shape `(m, n)`. Pixels that are
///   `0+0j` or non-finite are flagged as "empty" and preserved as
///   `0+0j` in the output.
/// * `alpha` — filter strength in `[0, 1]`. `0` = identity (returns a
///   unit-magnitude copy of the input).
/// * `psize` — square FFT patch size. Must be even and ≥ 4.
pub fn goldstein(igram: ArrayView2<Complex32>, alpha: f32, psize: usize) -> Array2<Complex32> {
    assert!(alpha >= 0.0, "alpha must be >= 0, got {alpha}");
    assert!(
        psize >= 4 && psize.is_multiple_of(2),
        "psize must be even and ≥ 4"
    );
    let (m, n) = igram.dim();
    let step = psize / 2;

    // Empty mask (zero magnitude or non-finite ⇒ keep as 0+0j in output).
    let empty = Array2::<bool>::from_shape_fn((m, n), |(i, j)| {
        let z = igram[(i, j)];
        !z.re.is_finite() || !z.im.is_finite() || (z.re == 0.0 && z.im == 0.0)
    });

    // Unit-magnitude normalisation (NaN-safe; empty pixels become 0+0j).
    let normed = Array2::<Complex32>::from_shape_fn((m, n), |(i, j)| {
        if empty[(i, j)] {
            Complex32::default()
        } else {
            let z = igram[(i, j)];
            let r = (z.re * z.re + z.im * z.im).sqrt();
            if r > 0.0 { z / r } else { Complex32::default() }
        }
    });

    // Reflect-pad so block FFTs fully cover the image (matches numpy
    // np.pad(mode="reflect")).
    let pad_top = step;
    let pad_left = step;
    let pad_bottom = step + (step - m % step) % step;
    let pad_right = step + (step - n % step) % step;
    let padded = reflect_pad(normed.view(), pad_top, pad_bottom, pad_left, pad_right);
    let (pm, pn) = padded.dim();

    // 2D Hann window (separable outer product).
    let weight = hann_2d(psize);

    // FFT planner — Arc<dyn Fft<f32>> is Send + Sync.
    let mut planner = FftPlanner::<f32>::new();
    let fft_fwd: Arc<dyn Fft<f32>> = planner.plan_fft_forward(psize);
    let fft_inv: Arc<dyn Fft<f32>> = planner.plan_fft_inverse(psize);
    let norm = 1.0 / (psize * psize) as f32; // match numpy.fft.ifft2

    // Enumerate every block start.
    let n_br = (pm - psize) / step + 1;
    let n_bc = (pn - psize) / step + 1;
    let positions: Vec<(usize, usize)> = (0..n_br)
        .flat_map(|br| (0..n_bc).map(move |bc| (br * step, bc * step)))
        .collect();

    let mut out = Array2::<Complex32>::zeros((pm, pn));
    let mut wsum = Array2::<f32>::zeros((pm, pn));

    // Process blocks in chunks so peak memory for the parallel collect
    // stays small (psize² · chunk_size · 8 bytes).
    let chunk_size = 1024;
    for chunk in positions.chunks(chunk_size) {
        let patches: Vec<(usize, usize, Vec<Complex32>)> = chunk
            .par_iter()
            .map_init(
                || vec![Complex32::default(); psize * psize],
                |scratch, &(r0, c0)| {
                    let mut buf = extract_patch(padded.view(), r0, c0, psize);
                    fft2_inplace(&mut buf, scratch, fft_fwd.as_ref(), psize);
                    for z in buf.iter_mut() {
                        let mag = (z.re * z.re + z.im * z.im).sqrt();
                        let scale = mag.powf(alpha);
                        *z *= scale;
                    }
                    fft2_inplace(&mut buf, scratch, fft_inv.as_ref(), psize);
                    for z in buf.iter_mut() {
                        *z *= norm;
                    }
                    (r0, c0, buf)
                },
            )
            .collect();

        for (r0, c0, patch) in patches {
            for i in 0..psize {
                for j in 0..psize {
                    let w = weight[i * psize + j];
                    out[(r0 + i, c0 + j)] += w * patch[i * psize + j];
                    wsum[(r0 + i, c0 + j)] += w;
                }
            }
        }
    }

    // Normalise by accumulated overlap-add weight.
    ndarray::Zip::from(&mut out).and(&wsum).for_each(|o, &w| {
        if w > 0.0 {
            *o /= w;
        }
    });

    // Crop back to original size; zero the empty mask.
    let cropped = out
        .slice(s![pad_top..pad_top + m, pad_left..pad_left + n])
        .to_owned();
    let mut result = cropped;
    ndarray::Zip::from(&mut result)
        .and(&empty)
        .for_each(|r, &is_empty| {
            if is_empty {
                *r = Complex32::default();
            }
        });
    result
}

fn extract_patch(src: ArrayView2<Complex32>, r0: usize, c0: usize, psize: usize) -> Vec<Complex32> {
    let mut buf = Vec::with_capacity(psize * psize);
    for i in 0..psize {
        for j in 0..psize {
            buf.push(src[(r0 + i, c0 + j)]);
        }
    }
    buf
}

/// In-place 2D FFT via separable 1D rows/cols (rustfft is 1D).
///
/// `scratch` must have length `p * p`; it's used for the column-pass
/// transpose and reused across calls so we don't reallocate per patch.
fn fft2_inplace(buf: &mut [Complex32], scratch: &mut [Complex32], fft: &dyn Fft<f32>, p: usize) {
    debug_assert_eq!(buf.len(), p * p);
    debug_assert_eq!(scratch.len(), p * p);
    // FFT each row.
    for r in 0..p {
        fft.process(&mut buf[r * p..(r + 1) * p]);
    }
    // Transpose into scratch.
    for r in 0..p {
        for c in 0..p {
            scratch[c * p + r] = buf[r * p + c];
        }
    }
    // FFT each "row" of the transpose (= original cols).
    for r in 0..p {
        fft.process(&mut scratch[r * p..(r + 1) * p]);
    }
    // Transpose back.
    for r in 0..p {
        for c in 0..p {
            buf[r * p + c] = scratch[c * p + r];
        }
    }
}

fn hann_2d(p: usize) -> Vec<f32> {
    let pi = std::f32::consts::PI;
    let w1: Vec<f32> = (0..p)
        .map(|i| 0.5 * (1.0 - (2.0 * pi * (i as f32) / ((p - 1) as f32)).cos()))
        .collect();
    let mut w2 = Vec::with_capacity(p * p);
    for i in 0..p {
        for j in 0..p {
            w2.push(w1[i] * w1[j]);
        }
    }
    w2
}

fn reflect_pad(
    src: ArrayView2<Complex32>,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) -> Array2<Complex32> {
    let (m, n) = src.dim();
    // numpy "reflect" needs at least 1 valid row/col beyond the pad on each
    // side: top uses src[1..=pad_top], bottom uses src[m-2..=m-1-pad_bottom];
    // both require pad ≤ m - 1. Same for columns.
    assert!(
        pad_top < m && pad_bottom < m,
        "reflect_pad: vertical pad ({pad_top}, {pad_bottom}) must be < image rows {m}"
    );
    assert!(
        pad_left < n && pad_right < n,
        "reflect_pad: horizontal pad ({pad_left}, {pad_right}) must be < image cols {n}"
    );
    let pm = m + pad_top + pad_bottom;
    let pn = n + pad_left + pad_right;
    let mut out = Array2::<Complex32>::zeros((pm, pn));
    // Copy interior.
    out.slice_mut(s![pad_top..pad_top + m, pad_left..pad_left + n])
        .assign(&src);
    // Reflect top / bottom (numpy "reflect" = abs index without endpoint
    // duplication: row i = src[abs(i)]).
    for r in 0..pad_top {
        let src_row = pad_top - r; // 1..=pad_top → src[1..=pad_top]
        let row = src.index_axis(Axis(0), src_row);
        let mut dst = out.index_axis_mut(Axis(0), r);
        for j in 0..n {
            dst[pad_left + j] = row[j];
        }
    }
    for r in 0..pad_bottom {
        let src_row = m - 2 - r; // src[m-2], src[m-3], ...
        let row = src.index_axis(Axis(0), src_row);
        let mut dst = out.index_axis_mut(Axis(0), pad_top + m + r);
        for j in 0..n {
            dst[pad_left + j] = row[j];
        }
    }
    // Reflect left / right on every output row (including the just-padded
    // top/bottom strips).
    for r in 0..pm {
        for c in 0..pad_left {
            let src_col = pad_left - c;
            out[(r, c)] = out[(r, pad_left + src_col)];
        }
        for c in 0..pad_right {
            let src_col = n - 2 - c;
            out[(r, pad_left + n + c)] = out[(r, pad_left + src_col)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn alpha_zero_returns_unit_magnitude_input() {
        // alpha=0 ⇒ |F|^0 = 1 ⇒ the FFT/IFFT round-trip is a no-op
        // (modulo overlap-add normalisation), so the output equals the
        // input wrapped phase with unit magnitude.
        let m = 128;
        let n = 96;
        let ig = Array2::<Complex32>::from_shape_fn((m, n), |(i, j)| {
            let phase = 0.05 * i as f32 + 0.03 * j as f32;
            // Vary amplitude so the unit-mag normalisation is exercised.
            let amp = 1.0 + 0.5 * (i as f32 / 10.0).sin();
            Complex32::from_polar(amp, phase)
        });
        let out = goldstein(ig.view(), 0.0, 32);
        for i in 0..m {
            for j in 0..n {
                let expected = Complex32::from_polar(1.0, ig[(i, j)].arg());
                let got = out[(i, j)];
                let diff = (got - expected).norm();
                assert!(
                    diff < 1e-3,
                    "diff at ({i},{j}) = {diff}, expected {expected:?}, got {got:?}",
                );
            }
        }
    }

    #[test]
    fn empty_pixels_preserved() {
        let m = 32;
        let n = 32;
        let mut ig = Array2::<Complex32>::from_shape_fn((m, n), |(i, j)| {
            Complex32::from_polar(1.0, 0.01 * i as f32 + 0.02 * j as f32)
        });
        ig[(10, 10)] = Complex32::default();
        ig[(15, 20)] = Complex32::default();
        let out = goldstein(ig.view(), 0.5, 16);
        assert_eq!(out[(10, 10)], Complex32::default());
        assert_eq!(out[(15, 20)], Complex32::default());
    }

    #[test]
    fn output_shape_matches_input() {
        for &(m, n) in &[(100usize, 100), (155, 229), (513, 769)] {
            let ig = Array2::<Complex32>::from_shape_fn((m, n), |(i, j)| {
                Complex32::from_polar(1.0, 0.001 * (i + j) as f32)
            });
            let out = goldstein(ig.view(), 0.7, 64);
            assert_eq!(out.dim(), (m, n));
        }
    }
}
