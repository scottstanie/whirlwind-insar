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
use ndarray::{Array2, ArrayView2};
use num_complex::Complex32;
use std::f32::consts::TAU;

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
    for i in 0..m - 1 {
        for j in 0..n {
            let z = igram[(i + 1, j)] * igram[(i, j)].conj();
            phase_dy[(i, j)] = z.arg();
        }
    }
    // Horizontal gradient: angle(igram[i, j+1] * conj(igram[i, j])), shape (m, n-1).
    let mut phase_dx = Array2::<f32>::zeros((m, n - 1));
    for i in 0..m {
        for j in 0..n - 1 {
            let z = igram[(i, j + 1)] * igram[(i, j)].conj();
            phase_dx[(i, j)] = z.arg();
        }
    }
    let phase_dy_s = box_filter_2d(phase_dy.view(), 7);
    let phase_dx_s = box_filter_2d(phase_dx.view(), 7);
    (phase_dy_s, phase_dx_s)
}

/// Separable box filter with size `k` (must be odd), nearest-edge replication.
pub fn box_filter_2d(a: ArrayView2<f32>, k: usize) -> Array2<f32> {
    assert!(k % 2 == 1);
    let half = (k / 2) as isize;
    let (m, n) = a.dim();
    let mut tmp = Array2::<f32>::zeros((m, n));
    let inv_k = 1.0 / (k as f32);

    // Filter along columns (vertical pass).
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for dj in -half..=half {
                let jj = ((j as isize + dj).clamp(0, n as isize - 1)) as usize;
                s += a[(i, jj)];
            }
            tmp[(i, j)] = s * inv_k;
        }
    }
    // Filter along rows (horizontal pass).
    let mut out = Array2::<f32>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for di in -half..=half {
                let ii = ((i as isize + di).clamp(0, m as isize - 1)) as usize;
                s += tmp[(ii, j)];
            }
            out[(i, j)] = s * inv_k;
        }
    }
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
    for i in 0..m_phase - 1 {
        for j in 0..n_phase {
            cor_dy[(i, j)] = corr[(i, j)].min(corr[(i + 1, j)]);
        }
    }
    let mut cor_dx = Array2::<f32>::zeros((m_phase, n_phase - 1)); // horizontal edges
    for i in 0..m_phase {
        for j in 0..n_phase - 1 {
            cor_dx[(i, j)] = corr[(i, j)].min(corr[(i, j + 1)]);
        }
    }
    let mask_dy = mask.map(|m_| {
        let mut out = Array2::<bool>::from_elem((m_phase - 1, n_phase), true);
        for i in 0..m_phase - 1 {
            for j in 0..n_phase {
                out[(i, j)] = m_[(i, j)] && m_[(i + 1, j)];
            }
        }
        out
    });
    let mask_dx = mask.map(|m_| {
        let mut out = Array2::<bool>::from_elem((m_phase, n_phase - 1), true);
        for i in 0..m_phase {
            for j in 0..n_phase - 1 {
                out[(i, j)] = m_[(i, j)] && m_[(i, j + 1)];
            }
        }
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
    let use_llr = std::env::var("WHIRLWIND_LLR_COST").is_ok();
    let lut = lut::get_or_build(nlooks);
    let cost_dir = |alpha: f32, gamma: f32| -> f32 {
        if use_llr {
            let p0 = lut.eval(alpha, gamma);
            let p1 = lut.eval(alpha - TAU, gamma);
            -((p1.max(1e-30)).ln() - (p0.max(1e-30)).ln())
        } else {
            let _ = lut; // keep alive
            let pi = std::f32::consts::PI;
            (gamma * (pi - alpha.abs())).max(0.0)
        }
    };

    // Allocate cost array.
    let mut cost = vec![0_i32; g.num_arcs()];

    let store = |cost: &mut [i32], arc: Option<usize>, c: f32| {
        if let Some(a) = arc {
            cost[a] = (c * COST_SCALE).round() as i32;
        }
    };

    // Iterate over each (vertical) pixel edge (i in 0..m_phase-1, j in 0..n_phase).
    // The corresponding residue-grid arcs are RIGHT and LEFT between residues
    // (i+1, j) and (i+1, j+1). (A vertical pixel-edge → horizontal residue-arc.)
    for i in 0..m_phase - 1 {
        for j in 0..n_phase {
            let alpha = phase_dy_s[(i, j)];
            let gamma = cor_dy[(i, j)];
            let masked = mask_dy
                .as_ref()
                .map(|mm| !mm[(i, j)])
                .unwrap_or(false);
            let (c_rt, c_lt) = if masked {
                (0.0, 0.0)
            } else {
                (cost_dir(-alpha, gamma), cost_dir(alpha, gamma))
            };
            // Residue node coords: (i+1, j) → (i+1, j+1) is the RIGHT direction.
            let r = g.right_arc(i + 1, j);
            let l = g.left_arc(i + 1, j + 1);
            store(&mut cost, r, c_rt);
            store(&mut cost, l, c_lt);
        }
    }

    // Iterate over each horizontal pixel edge (i in 0..m_phase, j in 0..n_phase-1).
    // Corresponding residue arcs are DOWN and UP between residues (i, j+1) and (i+1, j+1).
    for i in 0..m_phase {
        for j in 0..n_phase - 1 {
            let alpha = phase_dx_s[(i, j)];
            let gamma = cor_dx[(i, j)];
            let masked = mask_dx
                .as_ref()
                .map(|mm| !mm[(i, j)])
                .unwrap_or(false);
            let (c_dn, c_up) = if masked {
                (0.0, 0.0)
            } else {
                (cost_dir(alpha, gamma), cost_dir(-alpha, gamma))
            };
            let d = g.down_arc(i, j + 1);
            let u = g.up_arc(i + 1, j + 1);
            store(&mut cost, d, c_dn);
            store(&mut cost, u, c_up);
        }
    }

    // Residual reverse arcs: cost = -forward_cost.
    for a in 0..g.num_forward {
        cost[a + g.num_forward] = -cost[a];
    }

    cost
}
