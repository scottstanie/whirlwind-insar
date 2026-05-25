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
use ndarray::{Array2, ArrayView2, Axis};
use ndarray::parallel::prelude::*;
use num_complex::Complex32;
use rayon::prelude::*;
use std::f32::consts::TAU;
use std::sync::OnceLock;

/// Scale factor used when converting float Carballo costs to integers.
/// Integer costs enable Dial's bucket-queue Dijkstra; 100 keeps the
/// quantization error ≤ 0.005 per arc.
pub const COST_SCALE: f32 = 100.0;

/// Compute 7x7 box-filtered phase gradients (vertical & horizontal).
/// Mode = nearest (edge values replicate).
pub fn smooth_phase_gradients(
    igram: ArrayView2<Complex32>,
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
    let phase_dy_s = box_filter_2d(phase_dy.view(), 7);
    let phase_dx_s = box_filter_2d(phase_dx.view(), 7);
    (phase_dy_s, phase_dx_s)
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

/// Compute integer costs for every forward and reverse arc in the residual graph.
///
/// `igram`, `corr` have shape `(m_phase, n_phase)`. The residue grid (= node
/// grid) has shape `(m_phase + 1, n_phase + 1) = (m, n)`. The cost array has
/// length `2 * num_forward`, indexed by `arc_id` per `crate::grid` layout.
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

    let (phase_dy_s, phase_dx_s) = smooth_phase_gradients(igram);
    // phase_dy_s: (m_phase-1, n_phase) = (m-2, n-1)
    // phase_dx_s: (m_phase, n_phase-1) = (m-1, n-2)

    // Per-edge coherence (minimum of the two endpoint pixels).
    let mut cor_dy = Array2::<f32>::zeros((m_phase - 1, n_phase)); // vertical edges in pixel space
    cor_dy
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n_phase {
                row[j] = corr[(i, j)].min(corr[(i + 1, j)]);
            }
        });
    let mut cor_dx = Array2::<f32>::zeros((m_phase, n_phase - 1)); // horizontal edges
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

    // For now use a SNAPHU-style topological cost rather than the Carballo
    // LLR. The full Bayesian LLR can go negative for arcs at wrap lines, which
    // breaks Dijkstra; the cleanest fix is to either run a Bellman-Ford pass
    // for initial potentials, or pick a cost that's structurally non-negative.
    // This simple cost still encodes the right intuition: high cost in smooth
    // high-coherence regions (don't unwrap there); low cost near wrap lines.
    //
    //   cost = γ_edge · (π − |α_smooth|)
    //
    // which is 0 when |α|=π (free to unwrap at a wrap line), and γ·π when α=0
    // (avoid unwrapping in smooth interior). We still consume `lut` lazily so
    // the analytical Lee PDF code path stays exercised by tests; the LLR is
    // retained behind a `WHIRLWIND_LLR_COST=1` env knob for experiments.
    let use_llr = use_llr_cost();
    let lut = lut::get_or_build(nlooks);

    // Per-arc cost as a closure; reads `lut` (Sync) and a copied `use_llr` bool.
    let cost_dir = |alpha: f32, gamma: f32| -> f32 {
        if use_llr {
            let p0 = lut.eval(alpha, gamma);
            let p1 = lut.eval(alpha - TAU, gamma);
            -((p1.max(1e-30)).ln() - (p0.max(1e-30)).ln())
        } else {
            let pi = std::f32::consts::PI;
            (gamma * (pi - alpha.abs())).max(0.0)
        }
    };

    // Allocate the unified arc-cost array and split it into the 4 forward-direction
    // slabs + 1 reverse slab. Each slab is a disjoint &mut [i32], so we can fill
    // them in parallel without aliasing. Layout (see `crate::grid`):
    //   [0,        n_v)              DOWN
    //   [n_v,      2*n_v)            UP
    //   [2*n_v,    2*n_v + n_h)      RIGHT
    //   [2*n_v + n_h, num_forward)   LEFT
    //   [num_forward, 2*num_forward) reverse partners (cost = -forward)
    let mut cost = vec![0_i32; g.num_arcs()];
    let (forward, reverse) = cost.split_at_mut(g.num_forward);
    let (down_slab, rest) = forward.split_at_mut(g.n_v);
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
                let masked = mask_dy_ref
                    .as_ref()
                    .map(|mm| !mm[(i, j)])
                    .unwrap_or(false);
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
                let masked = mask_dx_ref
                    .as_ref()
                    .map(|mm| !mm[(i, j)])
                    .unwrap_or(false);
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

    // Residual reverse arcs: cost = -forward_cost (parallel copy).
    reverse
        .par_chunks_mut(8192)
        .zip(forward.par_chunks(8192))
        .for_each(|(rev_chunk, fwd_chunk)| {
            for (r, f) in rev_chunk.iter_mut().zip(fwd_chunk.iter()) {
                *r = -*f;
            }
        });

    cost
}

/// Cached env-var lookup for the LLR-cost toggle. Read once per process.
fn use_llr_cost() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("WHIRLWIND_LLR_COST").is_ok())
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

    let cost_dir = |alpha: f32, w: f32| -> f32 {
        let pi = std::f32::consts::PI;
        (w * (pi - alpha.abs())).max(0.0)
    };

    let mut cost = vec![0_i32; g.num_arcs()];
    let (forward, reverse) = cost.split_at_mut(g.num_forward);
    let (down_slab, rest) = forward.split_at_mut(g.n_v);
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
                let masked = mask_dy_ref
                    .as_ref()
                    .map(|mm| !mm[(i, j)])
                    .unwrap_or(false);
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
                let masked = mask_dx_ref
                    .as_ref()
                    .map(|mm| !mm[(i, j)])
                    .unwrap_or(false);
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

    reverse
        .par_chunks_mut(8192)
        .zip(forward.par_chunks(8192))
        .for_each(|(rev_chunk, fwd_chunk)| {
            for (r, f) in rev_chunk.iter_mut().zip(fwd_chunk.iter()) {
                *r = -*f;
            }
        });

    cost
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
                let s = per_pixel_var(variance[(i, j)])
                    + per_pixel_var(variance[(i + 1, j)]);
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
                let s = per_pixel_var(variance[(i, j)])
                    + per_pixel_var(variance[(i, j + 1)]);
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
        assert_eq!(costs.len(), g.num_arcs());
        // Reverse arcs are negatives of forward arcs.
        let (fwd, rev) = costs.split_at(g.num_forward);
        for (f, r) in fwd.iter().zip(rev.iter()) {
            assert_eq!(*r, -*f);
        }
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
