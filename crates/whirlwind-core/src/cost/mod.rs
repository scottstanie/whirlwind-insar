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
/// 100 matches the original Python convention. Integer costs enable Dial's
/// bucket-queue Dijkstra.
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
