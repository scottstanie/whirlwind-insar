//! Carballo-style Bayesian edge costs.
//!
//! The cost of pushing one unit of flow on an arc encodes a log-likelihood
//! ratio for "this phase gradient should get a +1 cycle correction in the
//! direction of this arc" vs. "no correction". The PDF is Lee 1994 multilook
//! phase noise conditioned on coherence; the smoothed local phase gradient
//! enters as a shift parameter.
//!
//! Per arc direction we have four cost arrays (`cost_dn`, `cost_up`,
//! `cost_rt`, `cost_lt`) packed into a single `Vec<i32>` indexed by `arc_id`
//! using the layout in `crate::grid`.

pub mod hyp2f1;
pub mod lee_pdf;
pub mod lut;

use crate::grid::RectangularGridGraph;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView2, Axis};
use num_complex::Complex32;
use rayon::prelude::*;

/// Scale factor used when converting float Carballo costs to integers.
/// Integer costs enable Dial's bucket-queue Dijkstra; 100 keeps the
/// quantization error ≤ 0.005 per arc.
pub const COST_SCALE: f32 = 100.0;

/// Compute 7x7 box-filtered phase gradients (vertical & horizontal).
/// Mode = nearest (edge values replicate).
pub fn smooth_phase_gradients(igram: ArrayView2<Complex32>) -> (Array2<f32>, Array2<f32>) {
    smooth_phase_gradients_with_mask(igram, None)
}

/// Raw per-arc wrapped phase gradients (no smoothing).
///
/// Returns `(phase_dy, phase_dx)` with shapes `(m-1, n)` and `(m, n-1)`
/// respectively — the same shapes as [`smooth_phase_gradients`]. Each
/// entry is `arg(igram[h] * conj(igram[t]))` for the corresponding arc.
pub fn phase_gradients_raw(igram: ArrayView2<Complex32>) -> (Array2<f32>, Array2<f32>) {
    let (m, n) = igram.dim();
    let mut phase_dy = Array2::<f32>::zeros((m - 1, n));
    phase_dy
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n {
                let z = igram[(i + 1, j)] * igram[(i, j)].conj();
                row[j] = z.arg();
            }
        });
    let mut phase_dx = Array2::<f32>::zeros((m, n - 1));
    phase_dx
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n - 1 {
                let z = igram[(i, j + 1)] * igram[(i, j)].conj();
                row[j] = z.arg();
            }
        });
    (phase_dy, phase_dx)
}

/// Same as [`smooth_phase_gradients`] but mask-aware: at pixels whose
/// 7×7 window overlaps masked-out pixels, the average is taken over the
/// *valid* pixels only (rather than including masked zeros). This is
/// critical for real-data mask boundaries — without it, masked pixels
/// (set to `0+0j`) drag the smoothed gradient toward 0 within 3 pixels
/// of the boundary, biasing the cost field and inducing block-2π errors
/// in coherent regions near the boundary.
///
/// Implementation uses the algebraic identity `mean_valid = sum / count`
/// with two separable box-filter passes (one on `phase * valid`, one on
/// `valid` itself), then a per-pixel divide. Same O(k) complexity as the
/// unmasked path; ~2× the work.
pub fn smooth_phase_gradients_with_mask(
    igram: ArrayView2<Complex32>,
    pixel_mask: Option<ArrayView2<bool>>,
) -> (Array2<f32>, Array2<f32>) {
    let (m, n) = igram.dim();
    // Vertical gradient: angle(igram[i+1, j] * conj(igram[i, j])), shape (m-1, n).
    let mut phase_dy = Array2::<f32>::zeros((m - 1, n));
    phase_dy
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n {
                let z = igram[(i + 1, j)] * igram[(i, j)].conj();
                row[j] = z.arg();
            }
        });
    // Horizontal gradient: angle(igram[i, j+1] * conj(igram[i, j])), shape (m, n-1).
    let mut phase_dx = Array2::<f32>::zeros((m, n - 1));
    phase_dx
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n - 1 {
                let z = igram[(i, j + 1)] * igram[(i, j)].conj();
                row[j] = z.arg();
            }
        });

    let Some(mask) = pixel_mask else {
        let phase_dy_s = box_filter_2d(phase_dy.view(), 7);
        let phase_dx_s = box_filter_2d(phase_dx.view(), 7);
        return (phase_dy_s, phase_dx_s);
    };

    // Per-edge validity (1.0 where both endpoint pixels are valid).
    let mut valid_dy = Array2::<f32>::zeros((m - 1, n));
    valid_dy
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n {
                row[j] = if mask[(i, j)] && mask[(i + 1, j)] {
                    1.0
                } else {
                    0.0
                };
            }
        });
    let mut valid_dx = Array2::<f32>::zeros((m, n - 1));
    valid_dx
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n - 1 {
                row[j] = if mask[(i, j)] && mask[(i, j + 1)] {
                    1.0
                } else {
                    0.0
                };
            }
        });
    // Zero phase gradient where the edge is invalid so it contributes 0
    // to the sum (rather than leaking the wrapped-phase angle of 0+0j).
    ndarray::Zip::from(&mut phase_dy)
        .and(&valid_dy)
        .for_each(|p, &v| {
            if v == 0.0 {
                *p = 0.0;
            }
        });
    ndarray::Zip::from(&mut phase_dx)
        .and(&valid_dx)
        .for_each(|p, &v| {
            if v == 0.0 {
                *p = 0.0;
            }
        });

    let sum_dy = box_filter_2d(phase_dy.view(), 7);
    let cnt_dy = box_filter_2d(valid_dy.view(), 7);
    let sum_dx = box_filter_2d(phase_dx.view(), 7);
    let cnt_dx = box_filter_2d(valid_dx.view(), 7);

    // mean = sum / count; both passes already include a /(k^2) factor that
    // cancels, so the ratio is the unbiased mean over valid pixels.
    let mut out_dy = Array2::<f32>::zeros(sum_dy.dim());
    ndarray::Zip::from(&mut out_dy)
        .and(&sum_dy)
        .and(&cnt_dy)
        .for_each(|o, &s, &c| {
            *o = if c > 1e-6 { s / c } else { 0.0 };
        });
    let mut out_dx = Array2::<f32>::zeros(sum_dx.dim());
    ndarray::Zip::from(&mut out_dx)
        .and(&sum_dx)
        .and(&cnt_dx)
        .for_each(|o, &s, &c| {
            *o = if c > 1e-6 { s / c } else { 0.0 };
        });

    (out_dy, out_dx)
}

/// Separable box filter with size `k` (must be odd), nearest-edge replication.
/// O(k) per output pixel (no rolling-sum trick — kept simple; cost is ~1% of total).
pub fn box_filter_2d(a: ArrayView2<f32>, k: usize) -> Array2<f32> {
    assert!(k % 2 == 1);
    let half = (k / 2) as isize;
    let (m, n) = a.dim();
    let inv_k = 1.0 / (k as f32);

    // Horizontal pass: each output row depends only on input row i.
    let mut tmp = Array2::<f32>::zeros((m, n));
    tmp.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n {
                let mut s = 0.0;
                for dj in -half..=half {
                    let jj = ((j as isize + dj).clamp(0, n as isize - 1)) as usize;
                    s += a[(i, jj)];
                }
                row[j] = s * inv_k;
            }
        });

    // Vertical pass: each output column depends only on input column j.
    // We still write row-by-row to keep the output array layout-friendly.
    let tmp_view = tmp.view();
    let mut out = Array2::<f32>::zeros((m, n));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n {
                let mut s = 0.0;
                for di in -half..=half {
                    let ii = ((i as isize + di).clamp(0, m as isize - 1)) as usize;
                    s += tmp_view[(ii, j)];
                }
                row[j] = s * inv_k;
            }
        });
    out
}

/// Compute integer costs for every forward arc in the residual graph.
///
/// `igram`, `corr` have shape `(m_phase, n_phase)`. The residue grid (= node
/// grid) has shape `(m_phase + 1, n_phase + 1) = (m, n)`. The returned cost
/// array has length `num_forward`; reverse-arc costs are implicit
/// (`-forward`) and reconstructed by `Network` on demand.
pub fn compute_carballo_costs(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
) -> Vec<i32> {
    let (m_phase, n_phase) = igram.dim();
    let m = m_phase + 1;
    let n = n_phase + 1;
    let g = RectangularGridGraph::new(m, n);

    // Note on mask handling for smoothing: empirically the *biased*
    // smoothing (averaging in the 0+0j values from masked pixels) acts as
    // an implicit boundary penalty — it pulls the smoothed gradient
    // toward 0 near the boundary, which makes the Carballo cost
    // ~γ·π there (high), discouraging MCF from routing through the
    // boundary. Using mask-aware (unbiased) smoothing — see
    // `smooth_phase_gradients_with_mask` — removes that implicit fence
    // and worsens 2π block errors on real NISAR data. Kept here for
    // possible future use; not the default.
    let (phase_dy_s, phase_dx_s) = smooth_phase_gradients(igram);
    // phase_dy_s: (m_phase-1, n_phase) = (m-2, n-1)
    // phase_dx_s: (m_phase, n_phase-1) = (m-1, n-2)

    // `nlooks` is unused by the default Carballo cost (γ·max(0, π−α) has no
    // nlooks term); kept in the signature for API stability / parity with
    // `compute_crlb_costs`.
    let _ = nlooks;
    let corr_use = corr;

    // Per-edge coherence (minimum of the two endpoint pixels).
    let mut cor_dy = Array2::<f32>::zeros((m_phase - 1, n_phase)); // vertical edges in pixel space
    cor_dy
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n_phase {
                row[j] = corr_use[(i, j)].min(corr_use[(i + 1, j)]);
            }
        });
    let mut cor_dx = Array2::<f32>::zeros((m_phase, n_phase - 1)); // horizontal edges
    cor_dx
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n_phase - 1 {
                row[j] = corr_use[(i, j)].min(corr_use[(i, j + 1)]);
            }
        });
    let mask_dy = mask.map(|m_| {
        let mut out = Array2::<bool>::from_elem((m_phase - 1, n_phase), true);
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_phase {
                    row[j] = m_[(i, j)] && m_[(i + 1, j)];
                }
            });
        out
    });
    let mask_dx = mask.map(|m_| {
        let mut out = Array2::<bool>::from_elem((m_phase, n_phase - 1), true);
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_phase - 1 {
                    row[j] = m_[(i, j)] && m_[(i, j + 1)];
                }
            });
        out
    });

    // Per-direction topological cost (Carballo-style, sign-aware).
    //
    //   c_{+}(α, γ) = γ · max(0, π − α)
    //   c_{−}(α, γ) = γ · max(0, π + α) = c_{+}(−α, γ)
    //
    // `c_{+}` is the cost of pushing +2π through the arc; it is 0 when
    // α = +π (true wrap line in the + direction), γ·π when α = 0 (smooth
    // interior — never cut here), and γ·2π when α = −π (perversely add +2π
    // where the wrap is in the opposite direction — strongly avoid).
    // `c_{−}` is the mirror image. The asymmetry is essential: MCF must
    // see different costs for DOWN vs UP arcs at a single pixel edge,
    // otherwise the LP is degenerate and arbitrary tie-breaking flips the
    // unwrap topology under tiny input perturbations.
    //
    // The earlier symmetric `γ · (π − |α|)` form lost this distinction and
    // produced block-2π errors in coherent regions on real NISAR data; see
    // `paper/binary_vs_continuous.md` "Posterior reliability" section.
    //
    // Cost of pushing +2π through this arc, given a smoothed gradient α and
    // edge coherence γ: the direction-aware Carballo cost `γ · max(0, π − α)`.
    // The opposite direction is `cost_dir(-α, γ)`. (The SNAPHU-equivalent
    // statistical cost is the convex parabola in `compute_snaphu_smooth_costs`,
    // reached via `unwrap_convex` / `WHIRLWIND_TILE_CONVEX`.)
    let cost_dir = |alpha: f32, gamma: f32| -> f32 {
        let pi = std::f32::consts::PI;
        (gamma * (pi - alpha)).max(0.0)
    };

    // Forward-arc cost vector split into 4 direction slabs. Each slab is a
    // disjoint &mut [i32], so we fill them in parallel without aliasing.
    //   [0,            n_v)             DOWN
    //   [n_v,          2*n_v)           UP
    //   [2*n_v,        2*n_v + n_h)     RIGHT
    //   [2*n_v + n_h,  num_forward)     LEFT
    let mut cost = vec![0_i32; g.num_forward];
    let (down_slab, rest) = cost.split_at_mut(g.n_v);
    let (up_slab, rest) = rest.split_at_mut(g.n_v);
    let (right_slab, left_slab) = rest.split_at_mut(g.n_h);

    let phase_dy_s_v = phase_dy_s.view();
    let phase_dx_s_v = phase_dx_s.view();
    let cor_dy_v = cor_dy.view();
    let cor_dx_v = cor_dx.view();
    let mask_dy_ref = mask_dy.as_ref().map(|a| a.view());
    let mask_dx_ref = mask_dx.as_ref().map(|a| a.view());

    // RIGHT / LEFT slabs come from vertical pixel edges (alpha = phase_dy).
    //   right_arc(i+1, j)     → right_slab[(i+1)*stride_h + j]
    //   left_arc(i+1, j+1)    → left_slab [(i+1)*stride_h + j]
    // Skip residue row 0 so chunk index = pixel-edge row i; both slabs have
    // m grid rows × stride_h cells; we touch rows 1..m_phase (= rows 0..m_phase-1
    // of the body view).
    let stride_h = g.n - 1; // = n_phase
    let right_body = &mut right_slab[stride_h..];
    let left_body = &mut left_slab[stride_h..];
    right_body
        .par_chunks_mut(stride_h)
        .zip(left_body.par_chunks_mut(stride_h))
        .enumerate()
        .for_each(|(i, (right_row, left_row))| {
            if i >= m_phase - 1 {
                return; // residue rows past last pixel-edge row stay zero
            }
            for j in 0..n_phase {
                let alpha = phase_dy_s_v[(i, j)];
                let gamma = cor_dy_v[(i, j)];
                let masked = mask_dy_ref.as_ref().map(|mm| !mm[(i, j)]).unwrap_or(false);
                let (c_rt, c_lt) = if masked {
                    (0.0, 0.0)
                } else {
                    (cost_dir(-alpha, gamma), cost_dir(alpha, gamma))
                };
                right_row[j] = (c_rt * COST_SCALE).round() as i32;
                left_row[j] = (c_lt * COST_SCALE).round() as i32;
            }
        });

    // DOWN / UP slabs come from horizontal pixel edges (alpha = phase_dx).
    //   down_arc(i,   j+1)  → down_slab[i * stride_v + (j+1)]   for i ∈ [0, m_phase)
    //   up_arc  (i+1, j+1)  → up_slab  [i * stride_v + (j+1)]   for i ∈ [0, m_phase)
    // Both slabs have m_phase rows of width stride_v.
    let stride_v = g.n; // = n_phase + 1
    down_slab
        .par_chunks_mut(stride_v)
        .zip(up_slab.par_chunks_mut(stride_v))
        .enumerate()
        .for_each(|(i, (down_row, up_row))| {
            for j in 0..n_phase - 1 {
                let alpha = phase_dx_s_v[(i, j)];
                let gamma = cor_dx_v[(i, j)];
                let masked = mask_dx_ref.as_ref().map(|mm| !mm[(i, j)]).unwrap_or(false);
                let (c_dn, c_up) = if masked {
                    (0.0, 0.0)
                } else {
                    (cost_dir(alpha, gamma), cost_dir(-alpha, gamma))
                };
                let col = j + 1;
                down_row[col] = (c_dn * COST_SCALE).round() as i32;
                up_row[col] = (c_up * COST_SCALE).round() as i32;
            }
        });

    cost
}

// =========================================================================
// CRLB-weighted cost (for phase-linked inputs)
// =========================================================================
//
// Motivation: for inputs that are interferograms formed from phase-linked
// SLCs (Dolphin / EMI / EVD), the proper per-pixel noise model is *not* the
// sliding-window sample coherence. That's a downsampling-window estimator
// that biases low and ignores the off-diagonal structure of the coherence
// matrix. The right model is the Cramér-Rao Lower Bound on the per-acquisition
// phase variance σ²_t(x,y) that phase linking *also emits* as a byproduct,
// often as a `crlb_<DATE>.tif` raster.
//
// For an interferogram between acquisitions a and b, the per-pixel phase
// variance is (assuming independent linked-phase estimates, which is the
// dominant convention):
//     σ²_IG(p) = σ²_a(p) + σ²_b(p)
//
// And for an arc connecting two adjacent pixels p, q in the residue graph,
// the gradient noise variance is σ²_edge = σ²_IG(p) + σ²_IG(q). The arc
// cost is the same topological shape as the Carballo cost above, but with
// inverse-variance precision replacing coherence:
//     cost(α, σ²) = (1 / σ²_edge) · (π − |α|)   clipped to nonneg
//
// This is dimensional analysis on autopilot: 1/σ² is the Fisher information,
// (π − |α|) is the "wraparound budget" left on this gradient, the product is
// the log-likelihood ratio (up to additive constants) of "this gradient
// should get a +1 cycle correction".

/// Minimum CRLB variance accepted, in rad². Anything below this gets clamped
/// so the inverse-variance weight stays finite. 1e-3 corresponds to
/// γ_equiv ≈ 0.999 — essentially noiseless.
pub const CRLB_VARIANCE_FLOOR: f32 = 1e-3;

/// Variance assumed for missing CRLB (≤0 or non-finite). Phase linking
/// writes 0 for pixels it didn't pick (PS/DS thresholding) and for real
/// nodata at scene edges; those pixels are *unreliable*, not noiseless,
/// so they should get *low* inverse-variance weight (cheap to cut).
///
/// 50 rad² is roughly what a γ≈0.1 pixel would give from the Lee multilook
/// variance at L=5 looks. With it the per-edge weight is ~0.01, ~3 orders
/// of magnitude weaker than a typical PS pixel (σ²~0.2 ⇒ w~2.5). MCF still
/// has a residual cost gradient (so it prefers short routes through noise)
/// but routes flow *into* nodata rather than away from it.
pub const CRLB_VARIANCE_NODATA: f32 = 50.0;

/// Per-pixel variance after nodata / clamp policy. Negative, zero, NaN, or
/// non-finite inputs are treated as nodata (→ `CRLB_VARIANCE_NODATA`); all
/// other values are clamped to `[CRLB_VARIANCE_FLOOR, CRLB_VARIANCE_NODATA]`.
#[inline]
fn per_pixel_var(v: f32) -> f32 {
    if !v.is_finite() || v <= 0.0 {
        CRLB_VARIANCE_NODATA
    } else {
        v.clamp(CRLB_VARIANCE_FLOOR, CRLB_VARIANCE_NODATA)
    }
}

/// Compute integer costs from CRLB-derived per-IG phase variance.
///
/// * `igram`     — complex IG, shape (m_phase, n_phase).
/// * `variance`  — per-pixel phase variance for this IG (σ²_a + σ²_b),
///                 same shape, in rad². NoData = 0 (or NaN, or ≤0) is
///                 mapped to `CRLB_VARIANCE_NODATA` (cheap to cut through).
/// * `mask`      — optional valid-pixel mask.
pub fn compute_crlb_costs(
    igram: ArrayView2<Complex32>,
    variance: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
) -> Vec<i32> {
    let (m_phase, n_phase) = igram.dim();
    assert_eq!(
        variance.dim(),
        (m_phase, n_phase),
        "variance shape {:?} != igram shape {:?}",
        variance.dim(),
        (m_phase, n_phase)
    );
    let m = m_phase + 1;
    let n = n_phase + 1;
    let g = RectangularGridGraph::new(m, n);

    // See note in `compute_carballo_costs` re: biased vs mask-aware smoothing.
    let (phase_dy_s, phase_dx_s) = smooth_phase_gradients(igram);

    // Per-edge inverse variance. For vertical edges (between (i,j) and (i+1,j)),
    // the gradient variance is var(i,j) + var(i+1,j); the weight is 1 / that.
    // Use a small floor to avoid /0 on nodata pixels.
    let inv_var_dy = build_inv_var_dy(variance);
    let inv_var_dx = build_inv_var_dx(variance);

    let mask_dy = mask.map(|m_| {
        let mut out = Array2::<bool>::from_elem((m_phase - 1, n_phase), true);
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_phase {
                    row[j] = m_[(i, j)] && m_[(i + 1, j)];
                }
            });
        out
    });
    let mask_dx = mask.map(|m_| {
        let mut out = Array2::<bool>::from_elem((m_phase, n_phase - 1), true);
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_phase - 1 {
                    row[j] = m_[(i, j)] && m_[(i, j + 1)];
                }
            });
        out
    });

    // Direction-aware Carballo cost shape with inverse-variance weight:
    //   c_{+}(α, w) = w · max(0, π − α)
    //   c_{−}(α, w) = c_{+}(−α, w) = w · max(0, π + α)
    // See `compute_carballo_costs` for the topological motivation — the
    // symmetric `w · (π − |α|)` form was used here previously, but it makes
    // both directions equal at a pixel edge and recreates the degenerate
    // tie-breaking issue described in the Carballo path comment.
    let cost_dir = |alpha: f32, w: f32| -> f32 {
        let pi = std::f32::consts::PI;
        (w * (pi - alpha)).max(0.0)
    };

    let mut cost = vec![0_i32; g.num_forward];
    let (down_slab, rest) = cost.split_at_mut(g.n_v);
    let (up_slab, rest) = rest.split_at_mut(g.n_v);
    let (right_slab, left_slab) = rest.split_at_mut(g.n_h);

    let phase_dy_s_v = phase_dy_s.view();
    let phase_dx_s_v = phase_dx_s.view();
    let inv_var_dy_v = inv_var_dy.view();
    let inv_var_dx_v = inv_var_dx.view();
    let mask_dy_ref = mask_dy.as_ref().map(|a| a.view());
    let mask_dx_ref = mask_dx.as_ref().map(|a| a.view());

    let stride_h = g.n - 1;
    let right_body = &mut right_slab[stride_h..];
    let left_body = &mut left_slab[stride_h..];
    right_body
        .par_chunks_mut(stride_h)
        .zip(left_body.par_chunks_mut(stride_h))
        .enumerate()
        .for_each(|(i, (right_row, left_row))| {
            if i >= m_phase - 1 {
                return;
            }
            for j in 0..n_phase {
                let alpha = phase_dy_s_v[(i, j)];
                let w = inv_var_dy_v[(i, j)];
                let masked = mask_dy_ref.as_ref().map(|mm| !mm[(i, j)]).unwrap_or(false);
                let (c_rt, c_lt) = if masked {
                    (0.0, 0.0)
                } else {
                    (cost_dir(-alpha, w), cost_dir(alpha, w))
                };
                right_row[j] = (c_rt * COST_SCALE).round() as i32;
                left_row[j] = (c_lt * COST_SCALE).round() as i32;
            }
        });

    let stride_v = g.n;
    down_slab
        .par_chunks_mut(stride_v)
        .zip(up_slab.par_chunks_mut(stride_v))
        .enumerate()
        .for_each(|(i, (down_row, up_row))| {
            for j in 0..n_phase - 1 {
                let alpha = phase_dx_s_v[(i, j)];
                let w = inv_var_dx_v[(i, j)];
                let masked = mask_dx_ref.as_ref().map(|mm| !mm[(i, j)]).unwrap_or(false);
                let (c_dn, c_up) = if masked {
                    (0.0, 0.0)
                } else {
                    (cost_dir(alpha, w), cost_dir(-alpha, w))
                };
                let col = j + 1;
                down_row[col] = (c_dn * COST_SCALE).round() as i32;
                up_row[col] = (c_up * COST_SCALE).round() as i32;
            }
        });

    cost
}

/// `nshortcycle` constant for the SNAPHU-style convex cost. Scales the
/// continuous wrapped phase gradient α ∈ (-π, π] into integer offsets
/// in (-50, 50]; integer flow `k` lives in units of `nshortcycle`. Set
/// to 100 to match SNAPHU's `dr/cs/sct/smooth.c::DEF_NSHORTCYCLE`.
pub const NSHORTCYCLE: i32 = 100;

/// Just/Bamler 1994 small-angle approximation to multilook phase variance:
/// `σ² ≈ (1 − γ²) / (2 L γ²)` (radians²). Kept as a sanity-check fallback;
/// the convex-cost path uses [`lut::get_or_build_variance`] for the full
/// Lee 1994 numerical variance instead.
///
/// At γ=1 we floor to a small ε so the inverse weight stays finite.
#[inline]
#[allow(dead_code)]
fn just_bamler_variance(gamma: f32, nlooks: f32) -> f32 {
    let g = gamma.clamp(1e-3, 0.999);
    (1.0 - g * g) / (2.0 * nlooks * g * g)
}

/// Compute SNAPHU-style convex (quadratic) per-arc costs.
///
/// Each forward arc gets a parabolic cost `c_e(k) = w_e · (k · nshortcycle
/// − offset_e)²` where:
///   * `nshortcycle = 100` is the integer scale ([`NSHORTCYCLE`]).
///   * `offset_e = round(α_smooth · nshortcycle / 2π)`, the local
///     wrapped phase gradient mapped to integer flow units. Sign-aware
///     per-direction (DOWN/UP and RIGHT/LEFT get opposite signs at the
///     same pixel edge, mirroring the Carballo path's
///     `cost_dir(α)` / `cost_dir(−α)` split).
///   * `w_e = round(1 / σ²_e · COST_SCALE)`, the per-arc inverse noise
///     variance with σ² from the Just/Bamler approximation at the
///     min-coherence of the two endpoint pixels.
///
/// Returned vectors have length `g.num_forward`; `offsets[a]` and
/// `weights[a]` together define the cost on arc `a`. Reverse residual
/// arcs share the same `(offset, weight)` (the convex cost is symmetric
/// in flow sign relative to the offset; the *marginal* cost in each
/// direction is computed at use time by [`Network::marginal_cost`]).
///
/// Masked-out pixel edges get `(offset = 0, weight = 0)` — a flat zero
/// cost regardless of flow, equivalent to a free arc. Combined with
/// the pre-existing mask-arc forbidding in `Network::new_*_with_mask`,
/// these arcs are never traversed anyway, but zero weight keeps the
/// Dial bucket-count bounded.
///
/// The math:
///
///   `nshortcycle = 100`, integer flow `k`, offset `O ∈ (-50, 50]`:
///     cost(k=0)  = w · O²            (smooth region: small)
///     cost(k=1)  = w · (100 − O)²    (large unless O is near 50)
///     cost(k=-1) = w · (-100 − O)²   (large unless O is near -50)
///
///   The minimum integer is `argmin_k (k · 100 − O)²` which is `0` for
///   `O ∈ (-50, 50]` — so every arc *individually* prefers k=0, but the
///   strength of that preference varies. Near a wrap line (O ≈ ±50)
///   the cost difference between k=0 and k=±1 is small ("soft" arc,
///   easy routing channel); in a smooth interior (O ≈ 0) it's the
///   full `w · 100² = 10,000 w` ("stiff" arc).
///
///   *Marginal* cost of pushing one more unit on an arc currently
///   carrying `k` units: see [`Network::marginal_cost`].
pub fn compute_snaphu_smooth_costs(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
) -> (Vec<i32>, Vec<i32>) {
    use std::f32::consts::PI;
    let (m_phase, n_phase) = igram.dim();
    assert_eq!(
        corr.dim(),
        (m_phase, n_phase),
        "corr shape {:?} != igram shape {:?}",
        corr.dim(),
        (m_phase, n_phase)
    );
    let m = m_phase + 1;
    let n = n_phase + 1;
    let g = RectangularGridGraph::new(m, n);

    // SNAPHU's smooth-cost offset is the DEVIATION of the raw wrapped phase
    // gradient from its local box-mean: `offset = nshortcycle · (dpsi −
    // avgdpsi)` (snaphu_cost.c:1115-1116, with `dpsi` in cycles from
    // snaphu_util.c:149). The deviation spikes toward ±1 cycle at an isolated
    // wrap line (raw ≈ ±π while the box-mean ≈ 0) yet is ≈0 in smooth regions,
    // which is exactly the routing signal the convex cost needs. The earlier
    // implementation fed the *smoothed* gradient alone (`avgdpsi`), which the
    // 7×7 box washes to ≈0 both in smooth areas AND across wrap lines — leaving
    // |offset| ≲ 22 with no wrap-line information and the convex cost degenerate
    // to pure `w·k²` (paper/convex_cost_design.md Suspect 5; the doc's prose
    // mis-stated SNAPHU's offset as `avgdpsi` — the source uses the deviation).
    //
    // The difference is NOT re-wrapped: SNAPHU leaves `dpsi − avgdpsi` free to
    // exceed ½ cycle so the parabola minimum can sit at k = ±1. The absolute
    // (ramp-scale) flow is supplied separately by the coarse anchor / cascade,
    // mirroring SNAPHU's `unwrappedest` offset shift (snaphu_cost.c:1127-1132) —
    // so this cost belongs in the per-tile solve of the tiled+anchor pipeline,
    // not a standalone whole-image solve.
    let (raw_dy, raw_dx) = phase_gradients_raw(igram);
    let (phase_dy_s, phase_dx_s) = {
        let (sm_dy, sm_dx) = smooth_phase_gradients_with_mask(igram, mask);
        (&raw_dy - &sm_dy, &raw_dx - &sm_dx)
    };

    // Per-edge min-of-endpoints coherence (matches Carballo path).
    let mut cor_dy = Array2::<f32>::zeros((m_phase - 1, n_phase));
    cor_dy
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n_phase {
                row[j] = corr[(i, j)].min(corr[(i + 1, j)]);
            }
        });
    let mut cor_dx = Array2::<f32>::zeros((m_phase, n_phase - 1));
    cor_dx
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n_phase - 1 {
                row[j] = corr[(i, j)].min(corr[(i, j + 1)]);
            }
        });
    let mask_dy = mask.map(|m_| {
        let mut out = Array2::<bool>::from_elem((m_phase - 1, n_phase), true);
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_phase {
                    row[j] = m_[(i, j)] && m_[(i + 1, j)];
                }
            });
        out
    });
    let mask_dx = mask.map(|m_| {
        let mut out = Array2::<bool>::from_elem((m_phase, n_phase - 1), true);
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_phase - 1 {
                    row[j] = m_[(i, j)] && m_[(i, j + 1)];
                }
            });
        out
    });

    let mut offsets = vec![0_i32; g.num_forward];
    let mut weights = vec![0_i32; g.num_forward];
    let (down_off, rest) = offsets.split_at_mut(g.n_v);
    let (up_off, rest) = rest.split_at_mut(g.n_v);
    let (right_off, left_off) = rest.split_at_mut(g.n_h);
    let (down_w, rest) = weights.split_at_mut(g.n_v);
    let (up_w, rest) = rest.split_at_mut(g.n_v);
    let (right_w, left_w) = rest.split_at_mut(g.n_h);

    let phase_dy_s_v = phase_dy_s.view();
    let phase_dx_s_v = phase_dx_s.view();
    let cor_dy_v = cor_dy.view();
    let cor_dx_v = cor_dx.view();
    let mask_dy_ref = mask_dy.as_ref().map(|a| a.view());
    let mask_dx_ref = mask_dx.as_ref().map(|a| a.view());

    // Convert wrapped phase α ∈ (-π, π] to integer offset in (-50, 50].
    let alpha_to_offset =
        |alpha: f32| -> i32 { ((alpha / (2.0 * PI)) * (NSHORTCYCLE as f32)).round() as i32 };
    // Per-arc weight = inverse Lee 1994 wrapped-phase variance, scaled by
    // COST_SCALE so the convex parabolic cost lives in i32 range. We build
    // a γ → σ² LUT once per nlooks (`lut::get_or_build_variance`) from a
    // 1024-sample numerical integration of the full Lee 1994 PDF over
    // (-π, π], then read it per arc. Big upgrade over the Just/Bamler
    // small-angle approximation `(1 − γ²) / (2 L γ²)` which diverges from
    // the true variance at low γ and moderate L (the NISAR regime).
    //
    // At γ → 0 the variance saturates near π²/3 ≈ 3.29 (the wrapped phase
    // becomes uniform on (-π, π]), so weights stay bounded — no need for
    // a low-γ floor. At γ ≈ 0.999 the variance is small and the weight
    // can spike; clamp the resulting weight to a sane integer range to
    // protect downstream arithmetic.
    let var_lut = lut::get_or_build_variance(nlooks);
    let gamma_to_weight = |gamma: f32| -> i32 {
        let var = var_lut.eval(gamma).max(1e-4);
        let w = (1.0 / var) * COST_SCALE;
        // Clamp to avoid pathological i32 overflow at near-perfect γ.
        // Max realistic weight at γ=0.999 is ~5e4; 1e7 is a comfortable cap.
        w.min(1e7).round() as i32
    };

    // RIGHT / LEFT slabs from vertical pixel edges. Same convention as
    // the Carballo path: RIGHT uses -α, LEFT uses +α (sign-aware per
    // direction at one pixel edge).
    let stride_h = g.n - 1;
    let right_off_body = &mut right_off[stride_h..];
    let left_off_body = &mut left_off[stride_h..];
    let right_w_body = &mut right_w[stride_h..];
    let left_w_body = &mut left_w[stride_h..];
    right_off_body
        .par_chunks_mut(stride_h)
        .zip(left_off_body.par_chunks_mut(stride_h))
        .zip(right_w_body.par_chunks_mut(stride_h))
        .zip(left_w_body.par_chunks_mut(stride_h))
        .enumerate()
        .for_each(
            |(i, (((right_off_row, left_off_row), right_w_row), left_w_row))| {
                if i >= m_phase - 1 {
                    return;
                }
                for j in 0..n_phase {
                    let masked = mask_dy_ref.as_ref().map(|mm| !mm[(i, j)]).unwrap_or(false);
                    if masked {
                        right_off_row[j] = 0;
                        left_off_row[j] = 0;
                        right_w_row[j] = 0;
                        left_w_row[j] = 0;
                    } else {
                        let alpha = phase_dy_s_v[(i, j)];
                        let w = gamma_to_weight(cor_dy_v[(i, j)]);
                        right_off_row[j] = alpha_to_offset(-alpha);
                        left_off_row[j] = alpha_to_offset(alpha);
                        right_w_row[j] = w;
                        left_w_row[j] = w;
                    }
                }
            },
        );

    // DOWN / UP slabs from horizontal pixel edges. DOWN uses +α, UP uses -α.
    let stride_v = g.n;
    down_off
        .par_chunks_mut(stride_v)
        .zip(up_off.par_chunks_mut(stride_v))
        .zip(down_w.par_chunks_mut(stride_v))
        .zip(up_w.par_chunks_mut(stride_v))
        .enumerate()
        .for_each(
            |(i, (((down_off_row, up_off_row), down_w_row), up_w_row))| {
                for j in 0..n_phase - 1 {
                    let masked = mask_dx_ref.as_ref().map(|mm| !mm[(i, j)]).unwrap_or(false);
                    let col = j + 1;
                    if masked {
                        down_off_row[col] = 0;
                        up_off_row[col] = 0;
                        down_w_row[col] = 0;
                        up_w_row[col] = 0;
                    } else {
                        let alpha = phase_dx_s_v[(i, j)];
                        let w = gamma_to_weight(cor_dx_v[(i, j)]);
                        down_off_row[col] = alpha_to_offset(alpha);
                        up_off_row[col] = alpha_to_offset(-alpha);
                        down_w_row[col] = w;
                        up_w_row[col] = w;
                    }
                }
            },
        );

    (offsets, weights)
}

/// Per-vertical-edge inverse variance. For pixel-row i and pixel-col j the
/// vertical edge connects (i, j) ↔ (i+1, j); variance = var(i,j) + var(i+1,j).
fn build_inv_var_dy(variance: ArrayView2<f32>) -> Array2<f32> {
    let (m_phase, n_phase) = variance.dim();
    let mut out = Array2::<f32>::zeros((m_phase - 1, n_phase));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n_phase {
                let s = per_pixel_var(variance[(i, j)]) + per_pixel_var(variance[(i + 1, j)]);
                row[j] = 1.0 / s;
            }
        });
    out
}

/// Per-horizontal-edge inverse variance. (i, j) ↔ (i, j+1).
fn build_inv_var_dx(variance: ArrayView2<f32>) -> Array2<f32> {
    let (m_phase, n_phase) = variance.dim();
    let mut out = Array2::<f32>::zeros((m_phase, n_phase - 1));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n_phase - 1 {
                let s = per_pixel_var(variance[(i, j)]) + per_pixel_var(variance[(i, j + 1)]);
                row[j] = 1.0 / s;
            }
        });
    out
}

#[cfg(test)]
mod crlb_tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn crlb_cost_basic_shape() {
        let m = 16_usize;
        let n = 16_usize;
        // Synthetic smooth IG: phase ramp.
        let igram = Array2::from_shape_fn((m, n), |(i, j)| {
            let phase = 0.1 * (i as f32) + 0.05 * (j as f32);
            Complex32::from_polar(1.0, phase)
        });
        let variance = Array2::<f32>::from_elem((m, n), 0.1);
        let costs = compute_crlb_costs(igram.view(), variance.view(), None);
        let g = RectangularGridGraph::new(m + 1, n + 1);
        assert_eq!(costs.len(), g.num_forward);
    }

    #[test]
    fn low_variance_yields_higher_cost() {
        let m = 8;
        let n = 8;
        let igram = Array2::from_shape_fn((m, n), |(_, _)| Complex32::from_polar(1.0, 0.0));
        let var_low = Array2::<f32>::from_elem((m, n), 0.05);
        let var_high = Array2::<f32>::from_elem((m, n), 1.0);
        let costs_low = compute_crlb_costs(igram.view(), var_low.view(), None);
        let costs_high = compute_crlb_costs(igram.view(), var_high.view(), None);
        // Low variance → high precision → cost-to-tear should be higher.
        let sum_low: i64 = costs_low.iter().take(64).map(|&c| c as i64).sum();
        let sum_high: i64 = costs_high.iter().take(64).map(|&c| c as i64).sum();
        assert!(sum_low > sum_high * 5, "low-variance cost should dominate");
    }

    #[test]
    fn nodata_variance_yields_lower_cost_than_valid() {
        // Phase-linking writes 0 to CRLB rasters for pixels it didn't pick
        // (PS/DS thresholding) and for true nodata at scene edges. Those
        // pixels are *unreliable*, not noiseless, and must get LOW per-edge
        // cost (≈ free to route 2π discontinuities through them) — the
        // opposite of the pre-fix behavior, which clamped 0 → 1e-3 → highest
        // possible cost and caused MCF to route flow through actual PS
        // pixels (cheaper), corrupting the few good measurements.
        let m = 8;
        let n = 8;
        // Constant smooth IG so alpha=0 everywhere ⇒ max-magnitude cost.
        let igram = Array2::from_shape_fn((m, n), |_| Complex32::from_polar(1.0, 0.0));
        let var_nodata = Array2::<f32>::from_elem((m, n), 0.0); // CRLB nodata convention
        let var_valid = Array2::<f32>::from_elem((m, n), 0.2); // typical PS pixel
        let costs_nodata = compute_crlb_costs(igram.view(), var_nodata.view(), None);
        let costs_valid = compute_crlb_costs(igram.view(), var_valid.view(), None);
        let sum_nodata: i64 = costs_nodata.iter().take(64).map(|c| c.abs() as i64).sum();
        let sum_valid: i64 = costs_valid.iter().take(64).map(|c| c.abs() as i64).sum();
        assert!(
            sum_valid > sum_nodata * 50,
            "valid CRLB should cost ≫ nodata: valid={sum_valid}, nodata={sum_nodata}"
        );
    }

    #[test]
    fn nan_variance_treated_as_nodata() {
        let m = 8;
        let n = 8;
        let igram = Array2::from_shape_fn((m, n), |_| Complex32::from_polar(1.0, 0.0));
        let mut var = Array2::<f32>::from_elem((m, n), 0.2);
        var[(3, 3)] = f32::NAN;
        var[(3, 4)] = f32::NEG_INFINITY;
        // Should not panic, should not produce NaN/inf in cost.
        let costs = compute_crlb_costs(igram.view(), var.view(), None);
        for c in &costs {
            assert!(c.abs() < 1_000_000, "cost overflowed: {c}");
        }
    }
}

#[cfg(test)]
mod convex_tests {
    use super::*;
    use ndarray::Array2;

    /// Smooth-phase IG: every smoothed gradient ≈ 0, so every offset
    /// should round to 0. Coherence is constant ⇒ uniform weight.
    #[test]
    fn smooth_phase_gives_zero_offsets() {
        let m = 16;
        let n = 16;
        let igram = Array2::from_shape_fn((m, n), |_| Complex32::from_polar(1.0, 0.0));
        let corr = Array2::<f32>::from_elem((m, n), 0.7);
        let (offsets, weights) = compute_snaphu_smooth_costs(igram.view(), corr.view(), 4.0, None);

        // Every offset is zero on a constant-phase IG (smoothed gradient = 0).
        for &o in &offsets {
            assert_eq!(o, 0, "smooth IG should give zero offsets, got {o}");
        }
        // Weights are roughly uniform inside the unmasked interior. We don't
        // assert exact equality because border arcs reuse the frame-edge
        // computation (and inside-vs-outside split slabs differ in coverage).
        let interior_w: Vec<i32> = weights.iter().copied().filter(|&w| w > 0).collect();
        assert!(!interior_w.is_empty(), "expected some non-zero weights");
        let w_min = *interior_w.iter().min().unwrap();
        let w_max = *interior_w.iter().max().unwrap();
        assert_eq!(
            w_min, w_max,
            "uniform coherence should give uniform weights"
        );
    }

    /// SNAPHU's offset is the DEVIATION of the raw wrapped gradient from its
    /// local box-mean (`dpsi − avgdpsi`), not the smoothed gradient. So a
    /// *uniform* ramp (where every arc's gradient equals its neighborhood mean)
    /// gives ≈zero offsets — the absolute slope is the coarse anchor's job, not
    /// the per-arc cost's. The offset spikes only where the gradient deviates
    /// locally: a wrap line / discontinuity. This replaces the earlier test,
    /// which asserted nonzero offsets on a uniform ramp under the (wrong)
    /// smoothed-gradient offset model.
    #[test]
    fn deviation_offset_zero_on_ramp_nonzero_at_feature() {
        let (m, n) = (32, 32);
        let corr = Array2::<f32>::from_elem((m, n), 0.95);

        // (1) Uniform ramp, 1.0 rad/px along columns: deviation ≈ 0 (raw == mean
        // everywhere but the truncated-box border). Expect very few nonzeros.
        let ramp = Array2::from_shape_fn((m, n), |(_, j)| Complex32::from_polar(1.0, j as f32));
        let (off_ramp, _) = compute_snaphu_smooth_costs(ramp.view(), corr.view(), 10.0, None);
        let nz_ramp = off_ramp.iter().filter(|&&o| o != 0).count();
        assert!(
            nz_ramp <= off_ramp.len() / 5,
            "uniform ramp should give ~zero deviation offsets, got {nz_ramp} of {}",
            off_ramp.len()
        );

        // (2) Localized vertical wall (cols 15-16 raised toward π): the gradient
        // at the wall edges deviates sharply from the ~0 neighborhood mean, so
        // the deviation offset fires there.
        let wall = Array2::from_shape_fn((m, n), |(_, j)| {
            Complex32::from_polar(1.0, if (15..=16).contains(&j) { 3.0 } else { 0.0 })
        });
        let (off_wall, _) = compute_snaphu_smooth_costs(wall.view(), corr.view(), 10.0, None);
        let nz_wall = off_wall.iter().filter(|&&o| o != 0).count();
        assert!(
            nz_wall > 0,
            "localized wall should produce nonzero deviation offsets"
        );
        assert!(
            nz_wall > nz_ramp,
            "wall ({nz_wall}) should have more nonzero offsets than a uniform ramp ({nz_ramp})"
        );
    }

    /// Masked arcs get weight=0 and offset=0 (free arc; will be forbidden
    /// at Network construction anyway).
    #[test]
    fn masked_edges_get_zero_weight() {
        let m = 8;
        let n = 8;
        let igram = Array2::from_shape_fn((m, n), |_| Complex32::from_polar(1.0, 0.0));
        let corr = Array2::<f32>::from_elem((m, n), 0.7);
        let mut mask = Array2::<bool>::from_elem((m, n), true);
        // Mask out a 2x2 patch at the corner; the arcs that cross those
        // pixel-edges should get weight = 0.
        mask[(0, 0)] = false;
        mask[(0, 1)] = false;
        mask[(1, 0)] = false;
        mask[(1, 1)] = false;
        let (offsets, weights) =
            compute_snaphu_smooth_costs(igram.view(), corr.view(), 4.0, Some(mask.view()));
        // Some arcs must end up with zero weight (the ones spanning the masked corner).
        let n_zero = weights.iter().filter(|&&w| w == 0).count();
        assert!(
            n_zero > 0,
            "expected some zero-weight arcs near the masked corner"
        );
        // Wherever weight = 0, offset must also be 0 (free arc, no preference).
        for (o, &w) in offsets.iter().zip(weights.iter()) {
            if w == 0 {
                assert_eq!(
                    *o, 0,
                    "zero-weight arc must have zero offset, got offset={o}"
                );
            }
        }
    }
}
