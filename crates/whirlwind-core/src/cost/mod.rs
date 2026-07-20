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
pub mod spline_lut;

use crate::grid::RectangularGridGraph;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView2, Axis};
use num_complex::Complex32;
use rayon::prelude::*;

/// Scale factor used when converting float Carballo costs to integers.
/// Integer costs enable Dial's bucket-queue Dijkstra; 100 keeps the
/// quantization error ≤ 0.005 per arc.
pub const COST_SCALE: f32 = 100.0;

/// Scale for the analytical Carballo LLR cost specifically. The raw LLR
/// is capped at `lut::MAX_CARBALLO_COST = 50.0`; multiplying by 6 gives
/// max integer cost = 300, matching the Dial's bucket-queue speed of the
/// earlier simplified formula while using the correct Lee 1994 shape.
pub const CARBALLO_COST_SCALE: f32 = 6.0;

/// Saturation ceiling for integer arc costs: `Network` stores forward costs
/// as `u16` (SNAPHU likewise uses `short` costs). Builders whose formula is
/// unbounded (CRLB / sparse inverse-variance weights at ultra-low variance)
/// clamp here - an arc this expensive (LLR ≈ 655 at `COST_SCALE = 100`) is
/// already "never cut here", so saturating the top end changes nothing
/// semantically. The bounded builders (parity spline ≤ 6,908, analytical
/// LUT ≤ 300) sit far below it.
pub const MAX_ARC_COST: f32 = 65_535.0;

/// Size of the sliding window used to average wrapped phase gradients into the
/// local mean (non-layover) slope that enters the cost model, expressed in the
/// two directions *relative to the examined phase difference* — exactly
/// SNAPHU's `KPARDPSI` / `KPERPDPSI` (and the `phase_grad_window` pair in
/// snaphu-py). A bigger window smooths the expected slope more (steadier in
/// high-fringe-rate deformation / topography, but blurs across wrap lines); a
/// smaller one is more local. The default `(7, 7)` matches SNAPHU.
///
/// The window is applied to each gradient array with its orientation swapped
/// so the *parallel* extent always runs along the gradient's own difference
/// direction: the vertical (azimuth) gradient is smoothed with `parallel`
/// rows × `perpendicular` cols, the horizontal (range) gradient with
/// `perpendicular` rows × `parallel` cols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseGradWindow {
    /// Extent (pixels) parallel to the examined phase difference (`KPARDPSI`).
    pub parallel: usize,
    /// Extent (pixels) perpendicular to the examined phase difference (`KPERPDPSI`).
    pub perpendicular: usize,
}

impl Default for PhaseGradWindow {
    fn default() -> Self {
        Self {
            parallel: 7,
            perpendicular: 7,
        }
    }
}

impl PhaseGradWindow {
    /// Build a window, panicking if either extent is 0 (SNAPHU's only
    /// requirement). The Python/CLI wrappers validate before reaching here, so
    /// this is a defensive backstop.
    pub fn new(parallel: usize, perpendicular: usize) -> Self {
        assert!(
            parallel >= 1 && perpendicular >= 1,
            "phase_grad_window extents must be >= 1, got ({parallel}, {perpendicular})"
        );
        Self {
            parallel,
            perpendicular,
        }
    }
}

/// Compute box-filtered phase gradients (vertical & horizontal) with the given
/// smoothing window. Mode = nearest (edge values replicate).
pub fn smooth_phase_gradients(
    igram: ArrayView2<Complex32>,
    window: PhaseGradWindow,
) -> (Array2<f32>, Array2<f32>) {
    smooth_phase_gradients_with_mask(igram, None, window)
}

/// How a triggered [`SlopeGuard`] rewrites the arc cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlopeGuardMode {
    /// Cost 0 in both directions - the edge is free to cut.
    ZeroCost,
    /// Re-evaluate the cost at zero expected slope, keeping the coherence
    /// weighting. Diagnostic arm: a coherent flat-terrain cut is *expensive*,
    /// so this makes aliased edges harder to cut rather than easier.
    ZeroSlope,
}

/// Validity guard for the Carballo cost on aliased phase gradients.
///
/// The arc cost is a log-likelihood ratio `-log(p1/p0)`: the evidence that this
/// edge carries a 2π cycle jump, given the locally expected slope. That
/// conditioning is only meaningful while the wrapped observation still
/// discriminates between hypotheses. Once the true fringe rate passes Nyquist -
/// a glacier shear margin, a rupture edge - one wrapped difference is
/// consistent with many true slopes, the likelihood ratio collapses toward 1,
/// and the honest cost is 0. Reporting the model's confident answer there is
/// what makes the solver refuse to cut along a real discontinuity and lay a
/// cheaper cut through the smooth interior instead (see
/// `docs/BUG_NISAR_CRYO_STACKED_CUTS.md`). This guard marks the model's domain
/// of validity rather than replacing it; isce3 PHASS hard-codes the same rule
/// as `|Δφ| >= 1 rad → cost 0`.
///
/// Disabled by default (threshold 0), so the ww-orig parity path is untouched.
#[derive(Debug, Clone, Copy)]
pub struct SlopeGuard {
    /// Raw per-edge wrapped `|Δφ|` (radians) at or above which the cost model
    /// is considered out of its validity domain. With `budget == 0` this is
    /// the threshold; with a budget it is the *floor*. `<= 0` and no budget
    /// disables the guard.
    pub threshold_rad: f32,
    /// EXPERIMENTAL. Maximum fraction of valid edges the guard may free. When
    /// `> 0` the threshold is chosen per frame as the matching quantile of the
    /// raw `|Δφ|` distribution (floored at `threshold_rad`), instead of being a
    /// fixed radian value.
    ///
    /// Motivation: a fixed threshold does not generalize across scenes. 1 rad
    /// fixes the cryo frame and `077_A_036` (2-3% of their edges aliased) but
    /// frees ~50% of the decorrelated `143_D_060` and visibly destabilizes it;
    /// 2 rad is safe there but stops fixing `077_A_036`. What separates the
    /// cases is *how much of the cost field the guard erases*, not coherence -
    /// aliased edges are low-coherence in every frame, fixed or broken (see
    /// `docs/BUG_NISAR_CRYO_STACKED_CUTS.md`). Budgeting that fraction turns a
    /// per-scene radian value into one scene-independent knob: it selects
    /// ~1 rad where few edges alias, and pushes toward π where most do, which
    /// disables the guard exactly where it does harm.
    pub budget: f32,
    pub mode: SlopeGuardMode,
}

/// A [`SlopeGuard`] with its per-frame threshold resolved (see
/// [`SlopeGuard::resolve`]). This is what the cost loops consult.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedSlopeGuard {
    threshold_rad: f32,
    mode: SlopeGuardMode,
}

/// Bin count for the budget quantile. 1024 bins over `[0, π]` puts the
/// threshold within ~0.003 rad, far finer than the effect being measured.
const GUARD_HIST_BINS: usize = 1024;

impl SlopeGuard {
    #[inline]
    pub fn enabled(&self) -> bool {
        self.threshold_rad > 0.0 || self.budget > 0.0
    }

    /// Resolve the per-frame threshold. Without a budget this just carries the
    /// configured radian value through; with one it picks the smallest cut
    /// that frees no more than `budget` of the valid edges, never going below
    /// `threshold_rad`.
    ///
    /// `edges` pairs each raw gradient array with its per-edge coherence, so
    /// mask-boundary edges (coherence 0, meaningless gradient) are excluded
    /// from the distribution exactly as they are excluded from firing.
    fn resolve(&self, edges: &[(ArrayView2<f32>, ArrayView2<f32>)]) -> ResolvedSlopeGuard {
        let mut threshold_rad = self.threshold_rad;
        if self.budget > 0.0 {
            let mut hist = vec![0_u64; GUARD_HIST_BINS];
            let mut total = 0_u64;
            for (raw, cor) in edges {
                ndarray::Zip::from(raw).and(cor).for_each(|&d, &g| {
                    if g > 1e-9 {
                        total += 1;
                        let frac = d.abs() / std::f32::consts::PI;
                        let b = (frac * GUARD_HIST_BINS as f32) as usize;
                        hist[b.min(GUARD_HIST_BINS - 1)] += 1;
                    }
                });
            }
            // Walk down from the steepest bin until spending one more would
            // exceed the budget; that bin's upper edge is the threshold.
            let allowed = (self.budget as f64 * total as f64) as u64;
            let mut acc = 0_u64;
            let mut chosen = std::f32::consts::PI;
            for b in (0..GUARD_HIST_BINS).rev() {
                if acc + hist[b] > allowed {
                    chosen = (b + 1) as f32 / GUARD_HIST_BINS as f32 * std::f32::consts::PI;
                    break;
                }
                acc += hist[b];
            }
            threshold_rad = threshold_rad.max(chosen);
        }
        ResolvedSlopeGuard {
            threshold_rad,
            mode: self.mode,
        }
    }
}

impl ResolvedSlopeGuard {
    /// A guard that never fires, for the disabled path.
    #[inline]
    fn off() -> Self {
        Self {
            threshold_rad: 0.0,
            mode: SlopeGuardMode::ZeroCost,
        }
    }

    #[inline]
    fn enabled(&self) -> bool {
        self.threshold_rad > 0.0
    }

    /// Whether the guard fires for the edge at `(i, j)`.
    ///
    /// `gamma > 0` excludes mask-boundary edges, whose raw gradient is
    /// meaningless because masked pixels enter the interferogram as `0+0j`.
    /// PHASS gates its rule the same way (`corr > small` on both endpoints).
    #[inline]
    fn fires(&self, raw: Option<&ArrayView2<f32>>, i: usize, j: usize, gamma: f32) -> bool {
        match raw {
            Some(r) => self.enabled() && gamma > 1e-9 && r[(i, j)].abs() >= self.threshold_rad,
            None => false,
        }
    }
}

/// Read the [`SlopeGuard`] configuration once from the environment:
/// `WHIRLWIND_SLOPE_GUARD_RAD` (radians; threshold, or floor when a budget is
/// set), `WHIRLWIND_SLOPE_GUARD_BUDGET` (max fraction of valid edges to free),
/// and `WHIRLWIND_SLOPE_GUARD_MODE` (`zerocost`, the default, or `zeroslope`).
/// With none of them set the guard is disabled and the cost field is untouched.
pub fn slope_guard() -> SlopeGuard {
    use std::sync::OnceLock;
    static G: OnceLock<SlopeGuard> = OnceLock::new();
    *G.get_or_init(|| SlopeGuard {
        threshold_rad: std::env::var("WHIRLWIND_SLOPE_GUARD_RAD")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0),
        budget: std::env::var("WHIRLWIND_SLOPE_GUARD_BUDGET")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0),
        mode: match std::env::var("WHIRLWIND_SLOPE_GUARD_MODE").as_deref() {
            Ok("zeroslope") => SlopeGuardMode::ZeroSlope,
            _ => SlopeGuardMode::ZeroCost,
        },
    })
}

/// Raw per-arc wrapped phase gradients (no smoothing).
///
/// Returns `(phase_dy, phase_dx)` with shapes `(m-1, n)` and `(m, n-1)`
/// respectively - the same shapes as [`smooth_phase_gradients`]. Each
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
/// 7x7 window overlaps masked-out pixels, the average is taken over the
/// *valid* pixels only (rather than including masked zeros). This is
/// critical for real-data mask boundaries - without it, masked pixels
/// (set to `0+0j`) drag the smoothed gradient toward 0 within 3 pixels
/// of the boundary, biasing the cost field and inducing block-2π errors
/// in coherent regions near the boundary.
///
/// Implementation uses the algebraic identity `mean_valid = sum / count`
/// with two separable box-filter passes (one on `phase * valid`, one on
/// `valid` itself), then a per-pixel divide. Same O(k) complexity as the
/// unmasked path; ~2x the work.
pub fn smooth_phase_gradients_with_mask(
    igram: ArrayView2<Complex32>,
    pixel_mask: Option<ArrayView2<bool>>,
    window: PhaseGradWindow,
) -> (Array2<f32>, Array2<f32>) {
    // Orientation per gradient: the `parallel` extent runs along the gradient's
    // own difference direction. Vertical (dy) differences run down rows, so
    // rows = parallel, cols = perpendicular; horizontal (dx) differences run
    // along cols, so the axes swap.
    let (dy_krow, dy_kcol) = (window.parallel, window.perpendicular);
    let (dx_krow, dx_kcol) = (window.perpendicular, window.parallel);
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
        let phase_dy_s = box_filter_2d(phase_dy.view(), dy_krow, dy_kcol);
        let phase_dx_s = box_filter_2d(phase_dx.view(), dx_krow, dx_kcol);
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

    let sum_dy = box_filter_2d(phase_dy.view(), dy_krow, dy_kcol);
    let cnt_dy = box_filter_2d(valid_dy.view(), dy_krow, dy_kcol);
    let sum_dx = box_filter_2d(phase_dx.view(), dx_krow, dx_kcol);
    let cnt_dx = box_filter_2d(valid_dx.view(), dx_krow, dx_kcol);

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

/// Separable box filter, `krow` taps down columns × `kcol` taps across rows,
/// nearest-edge replication. Both extents must be >= 1; odd sizes are centered,
/// even sizes lean one pixel toward higher indices (`lo = (k-1)/2`, `hi = k/2`).
/// O(krow + kcol) per output pixel (no rolling-sum trick - kept simple; cost is
/// ~1% of total).
pub fn box_filter_2d(a: ArrayView2<f32>, krow: usize, kcol: usize) -> Array2<f32> {
    assert!(krow >= 1 && kcol >= 1);
    let (lo_c, hi_c) = (((kcol - 1) / 2) as isize, (kcol / 2) as isize);
    let (lo_r, hi_r) = (((krow - 1) / 2) as isize, (krow / 2) as isize);
    let (m, n) = a.dim();
    let inv_kc = 1.0 / (kcol as f32);
    let inv_kr = 1.0 / (krow as f32);

    // Horizontal pass: each output row depends only on input row i.
    let mut tmp = Array2::<f32>::zeros((m, n));
    tmp.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n {
                let mut s = 0.0;
                for dj in -lo_c..=hi_c {
                    let jj = ((j as isize + dj).clamp(0, n as isize - 1)) as usize;
                    s += a[(i, jj)];
                }
                row[j] = s * inv_kc;
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
                for di in -lo_r..=hi_r {
                    let ii = ((i as isize + di).clamp(0, m as isize - 1)) as usize;
                    s += tmp_view[(ii, j)];
                }
                row[j] = s * inv_kr;
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
    window: PhaseGradWindow,
) -> Vec<i32> {
    let (m_phase, n_phase) = igram.dim();
    let m = m_phase + 1;
    let n = n_phase + 1;
    let g = RectangularGridGraph::new(m, n);

    // Note on mask handling for smoothing: empirically the *biased*
    // smoothing (averaging in the 0+0j values from masked pixels) acts as
    // an implicit boundary penalty - it pulls the smoothed gradient
    // toward 0 near the boundary, which makes the Carballo cost
    // ~γ·π there (high), discouraging MCF from routing through the
    // boundary. Using mask-aware (unbiased) smoothing - see
    // `smooth_phase_gradients_with_mask` - removes that implicit fence
    // and worsens 2π block errors on real NISAR data. Kept here for
    // possible future use; not the default.
    let (phase_dy_s, phase_dx_s) = smooth_phase_gradients(igram, window);
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

    // Analytical Carballo LLR cost from Lee 1994 multilook phase CDF.
    //
    //   cost(+, α, γ) = min(−log(CDF_Lee(α−π) / (1−CDF_Lee(α−π))), MAX)
    //                   for α > 0; MAX for α ≤ 0.
    //   cost(−, α, γ) = cost(+, −α, γ)    (opposite direction)
    //
    // At α = +π (wrap line): CDF(0) = 0.5 → cost = 0 (free to cross).
    // At α → 0  (smooth):    CDF(−π) → 0  → cost = MAX (never cut here).
    // The asymmetry between +/− directions is essential (see earlier
    // Carballo comment block); the LUT encodes it via the sign of α.
    let carb_lut = lut::get_or_build_carballo(nlooks);
    let cost_dir = |alpha: f32, gamma: f32| -> f32 { carb_lut.eval(alpha, gamma) };

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
    // m grid rows x stride_h cells; we touch rows 1..m_phase (= rows 0..m_phase-1
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
                right_row[j] = (c_rt * CARBALLO_COST_SCALE).round() as i32;
                left_row[j] = (c_lt * CARBALLO_COST_SCALE).round() as i32;
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
                down_row[col] = (c_dn * CARBALLO_COST_SCALE).round() as i32;
                up_row[col] = (c_up * CARBALLO_COST_SCALE).round() as i32;
            }
        });

    cost
}

/// Parity cost mode - matches Python `_cost.compute_carballo_costs` exactly:
///
/// * Scale = 100.0 (matching Python's `100 * -log(p1/p0)`)
/// * Cost zeroed only where **both** endpoint pixels are invalid
///   (Python passes `mask=~valid_mask`; zeros where `mask[a] && mask[b]` =
///   both-invalid; boundary arcs with one valid pixel retain a nonzero cost)
/// * p0/p1 probabilities loaded from the embedded ww-orig tables.
///
/// This is intentionally separate from `compute_carballo_costs`, which keeps
/// the faster analytical LUT used by the tiled/reuse production path.
pub fn compute_carballo_costs_parity(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    window: PhaseGradWindow,
) -> Vec<i32> {
    compute_carballo_costs_parity_impl(igram, corr, nlooks, mask, window, |c| c)
}

/// [`compute_carballo_costs_parity`] emitting the `u16` word `Network` stores
/// internally, so `unwrap_linear` can hand the vector to
/// [`crate::network::Network::new_linear_packed`] by move. Skipping the `i32`
/// intermediate removes the largest setup-phase transient (~4 arcs/pixel · 4
/// bytes) plus one full repack pass. Values are identical: the parity spline
/// cost is bounded (≤ ~6,908) and the conversion asserts the `u16` range the
/// same way `Network` construction does.
pub fn compute_carballo_costs_parity_packed(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    window: PhaseGradWindow,
) -> Vec<u16> {
    compute_carballo_costs_parity_impl(igram, corr, nlooks, mask, window, |c| {
        assert!(
            (0..=u16::MAX as i32).contains(&c),
            "arc cost {c} outside the u16 range [0, 65535] - the cost builder must clamp"
        );
        c as u16
    })
}

// The default `PhaseGradWindow` (7x7) reproduces the Python reference exactly;
// a non-default window is a deliberate opt-out of bit-for-bit parity.
fn compute_carballo_costs_parity_impl<T: Copy + Default + Send + Sync>(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    window: PhaseGradWindow,
    pack: impl Fn(i32) -> T + Sync,
) -> Vec<T> {
    let (m_phase, n_phase) = igram.dim();
    let m = m_phase + 1;
    let n = n_phase + 1;
    let g = RectangularGridGraph::new(m, n);

    // Biased (non-mask-aware) smoothing - matches Python's uniform_filter.
    let (phase_dy_s, phase_dx_s) = smooth_phase_gradients(igram, window);

    // Aliased-gradient validity guard (off by default; see `SlopeGuard`). It
    // keys off the RAW per-edge wrapped difference, not the smoothed slope:
    // a box average over a shear margin is diluted by its gentle neighbours,
    // which is exactly where the model stops discriminating. Nothing below is
    // computed when the guard is disabled, so the parity path is untouched.
    let guard_cfg = slope_guard();
    let raw_grads = guard_cfg.enabled().then(|| phase_gradients_raw(igram));
    let (raw_dy_v, raw_dx_v) = match &raw_grads {
        Some((dy, dx)) => (Some(dy.view()), Some(dx.view())),
        None => (None, None),
    };

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

    // "Both invalid" per-edge masks - True where NEITHER pixel is valid.
    // This matches Python's `mask_dy = logical_and(~valid[a], ~valid[b])`.
    let mask_dy_bi = mask.map(|m_| {
        let mut out = Array2::<bool>::from_elem((m_phase - 1, n_phase), false);
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_phase {
                    row[j] = !m_[(i, j)] && !m_[(i + 1, j)];
                }
            });
        out
    });
    let mask_dx_bi = mask.map(|m_| {
        let mut out = Array2::<bool>::from_elem((m_phase, n_phase - 1), false);
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_phase - 1 {
                    row[j] = !m_[(i, j)] && !m_[(i, j + 1)];
                }
            });
        out
    });

    // Resolve the guard's per-frame threshold now that the per-edge coherence
    // arrays exist (a budget picks the threshold from the |Δφ| distribution
    // over edges that could actually fire).
    let guard = match (raw_dy_v, raw_dx_v) {
        (Some(rdy), Some(rdx)) => guard_cfg.resolve(&[(rdy, cor_dy.view()), (rdx, cor_dx.view())]),
        _ => ResolvedSlopeGuard::off(),
    };

    // Use the embedded ww-orig spline tables for p0/p1.
    let sp_lut = spline_lut::get_or_load();

    // Masked "sea" arcs are cost-0, matching ww-orig (free sea).
    let sea = pack(0);

    let mut cost = vec![T::default(); g.num_forward];
    let (down_slab, rest) = cost.split_at_mut(g.n_v);
    let (up_slab, rest) = rest.split_at_mut(g.n_v);
    let (right_slab, left_slab) = rest.split_at_mut(g.n_h);

    let phase_dy_s_v = phase_dy_s.view();
    let phase_dx_s_v = phase_dx_s.view();
    let cor_dy_v = cor_dy.view();
    let cor_dx_v = cor_dx.view();
    let mask_dy_bi_ref = mask_dy_bi.as_ref().map(|a| a.view());
    let mask_dx_bi_ref = mask_dx_bi.as_ref().map(|a| a.view());

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
                let gamma = cor_dy_v[(i, j)];
                let both_invalid = mask_dy_bi_ref
                    .as_ref()
                    .map(|mm| mm[(i, j)])
                    .unwrap_or(false);
                let (c_rt, c_lt) = if both_invalid {
                    (sea, sea)
                } else if guard.fires(raw_dy_v.as_ref(), i, j, gamma) {
                    match guard.mode {
                        SlopeGuardMode::ZeroCost => (sea, sea),
                        SlopeGuardMode::ZeroSlope => {
                            let c = pack(sp_lut.cost(0.0, gamma, nlooks));
                            (c, c)
                        }
                    }
                } else {
                    (
                        pack(sp_lut.cost(-alpha, gamma, nlooks)),
                        pack(sp_lut.cost(alpha, gamma, nlooks)),
                    )
                };
                right_row[j] = c_rt;
                left_row[j] = c_lt;
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
                let gamma = cor_dx_v[(i, j)];
                let both_invalid = mask_dx_bi_ref
                    .as_ref()
                    .map(|mm| mm[(i, j)])
                    .unwrap_or(false);
                let (c_dn, c_up) = if both_invalid {
                    (sea, sea)
                } else if guard.fires(raw_dx_v.as_ref(), i, j, gamma) {
                    match guard.mode {
                        SlopeGuardMode::ZeroCost => (sea, sea),
                        SlopeGuardMode::ZeroSlope => {
                            let c = pack(sp_lut.cost(0.0, gamma, nlooks));
                            (c, c)
                        }
                    }
                } else {
                    (
                        pack(sp_lut.cost(alpha, gamma, nlooks)),
                        pack(sp_lut.cost(-alpha, gamma, nlooks)),
                    )
                };
                let col = j + 1;
                down_row[col] = c_dn;
                up_row[col] = c_up;
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
/// γ_equiv ≈ 0.999 - essentially noiseless.
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
/// * `igram`     - complex IG, shape (m_phase, n_phase).
/// * `variance`  - per-pixel phase variance for this IG (σ²_a + σ²_b),
///                 same shape, in rad². NoData = 0 (or NaN, or ≤0) is
///                 mapped to `CRLB_VARIANCE_NODATA` (cheap to cut through).
/// * `mask`      - optional valid-pixel mask.
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
    // CRLB path is experimental and always uses the default slope window.
    let (phase_dy_s, phase_dx_s) = smooth_phase_gradients(igram, PhaseGradWindow::default());

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
    // See `compute_carballo_costs` for the topological motivation - the
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
                right_row[j] = (c_rt * COST_SCALE).round().min(MAX_ARC_COST) as i32;
                left_row[j] = (c_lt * COST_SCALE).round().min(MAX_ARC_COST) as i32;
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
                down_row[col] = (c_dn * COST_SCALE).round().min(MAX_ARC_COST) as i32;
                up_row[col] = (c_up * COST_SCALE).round().min(MAX_ARC_COST) as i32;
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
/// Masked-out pixel edges get `(offset = 0, weight = 0)` - a flat zero
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
///   `O ∈ (-50, 50]` - so every arc *individually* prefers k=0, but the
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
    window: PhaseGradWindow,
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
    // 7x7 box washes to ≈0 both in smooth areas AND across wrap lines - leaving
    // |offset| ≲ 22 with no wrap-line information and the convex cost degenerate
    // to pure `w·k²`.
    //
    // The difference is NOT re-wrapped: SNAPHU leaves `dpsi − avgdpsi` free to
    // exceed ½ cycle so the parabola minimum can sit at k = ±1. The absolute
    // (ramp-scale) flow is supplied separately by the coarse anchor / cascade,
    // mirroring SNAPHU's `unwrappedest` offset shift (snaphu_cost.c:1127-1132) -
    // so this cost belongs in the per-tile solve of the tiled+anchor pipeline,
    // not a standalone whole-image solve.
    let (raw_dy, raw_dx) = phase_gradients_raw(igram);
    let (phase_dy_s, phase_dx_s) = {
        let (sm_dy, sm_dx) = smooth_phase_gradients_with_mask(igram, mask, window);
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
    // becomes uniform on (-π, π]), so weights stay bounded - no need for
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
mod slope_guard_tests {
    use super::*;
    use ndarray::arr2;

    /// The guard must be OFF unless explicitly configured: every parity claim
    /// (ww-orig equivalence, the byte-identical NISAR frame set) rests on the
    /// default cost field being untouched. Deliberately does not set the env
    /// var - `slope_guard()` caches in a `OnceLock`, so a test that enabled it
    /// would silently poison the rest of the suite.
    #[test]
    fn default_is_disabled() {
        assert!(
            !slope_guard().enabled(),
            "the slope guard must be opt-in; enabling it by default breaks ww-orig parity"
        );
    }

    /// A fixed-threshold guard, resolved (no budget => threshold passes through).
    fn fixed(threshold_rad: f32) -> ResolvedSlopeGuard {
        SlopeGuard {
            threshold_rad,
            budget: 0.0,
            mode: SlopeGuardMode::ZeroCost,
        }
        .resolve(&[])
    }

    #[test]
    fn fires_only_on_aliased_edges_with_valid_coherence() {
        let raw = arr2(&[[0.5_f32, 1.5], [-2.0, 0.99]]);
        let v = raw.view();
        let g = fixed(1.0);

        assert!(!g.fires(Some(&v), 0, 0, 0.8), "0.5 rad is below threshold");
        assert!(g.fires(Some(&v), 0, 1, 0.8), "1.5 rad is aliased");
        assert!(
            g.fires(Some(&v), 1, 0, 0.8),
            "guard keys off |dphi|, not sign"
        );
        assert!(
            !g.fires(Some(&v), 1, 1, 0.8),
            "0.99 rad is just below a 1.0 threshold"
        );
        // A mask-boundary edge has gamma == 0; its raw gradient is meaningless
        // (masked pixels enter as 0+0j), so the guard must not fire there.
        assert!(
            !g.fires(Some(&v), 0, 1, 0.0),
            "guard must not fire on a zero-coherence (mask boundary) edge"
        );
        // Disabled guard never fires, and no raw gradients means no guard.
        assert!(!fixed(0.0).fires(Some(&v), 0, 1, 0.8));
        assert!(!g.fires(None, 0, 1, 0.8));
    }

    /// The budget picks the threshold from the data so that only the requested
    /// fraction of valid edges is freed. This is what lets one scene-independent
    /// knob cover both a steep frame (few aliased edges -> a low threshold, the
    /// guard applies) and a decorrelated one (most edges aliased -> threshold
    /// pushed toward π, the guard backs off). Zero-coherence edges are excluded
    /// from the distribution, exactly as they are excluded from firing.
    #[test]
    fn budget_picks_a_threshold_that_frees_about_the_requested_fraction() {
        // 100 edges: 90 gentle (0.1 rad), 10 steep (3.0 rad).
        let mut raw = ndarray::Array2::<f32>::from_elem((10, 10), 0.1);
        for j in 0..10 {
            raw[(0, j)] = 3.0;
        }
        let cor = ndarray::Array2::<f32>::from_elem((10, 10), 0.8);

        // A 10% budget must let exactly the steep decile through.
        let g = SlopeGuard {
            threshold_rad: 0.0,
            budget: 0.10,
            mode: SlopeGuardMode::ZeroCost,
        }
        .resolve(&[(raw.view(), cor.view())]);
        let fired = (0..10)
            .flat_map(|i| (0..10).map(move |j| (i, j)))
            .filter(|&(i, j)| g.fires(Some(&raw.view()), i, j, 0.8))
            .count();
        assert_eq!(fired, 10, "a 10% budget must free the steep decile");

        // A budget far below the aliased fraction backs the guard off entirely
        // rather than freeing the whole steep population - the `143_D_060` case.
        let tight = SlopeGuard {
            threshold_rad: 0.0,
            budget: 0.01,
            mode: SlopeGuardMode::ZeroCost,
        }
        .resolve(&[(raw.view(), cor.view())]);
        let fired_tight = (0..10)
            .flat_map(|i| (0..10).map(move |j| (i, j)))
            .filter(|&(i, j)| tight.fires(Some(&raw.view()), i, j, 0.8))
            .count();
        assert_eq!(
            fired_tight, 0,
            "a budget below the aliased fraction must back off, not overspend"
        );

        // The configured radian value is a FLOOR under a budget: a generous
        // budget must not drag the threshold below it and free gentle edges.
        let floored = SlopeGuard {
            threshold_rad: 2.0,
            budget: 0.90,
            mode: SlopeGuardMode::ZeroCost,
        }
        .resolve(&[(raw.view(), cor.view())]);
        let fired_floored = (0..10)
            .flat_map(|i| (0..10).map(move |j| (i, j)))
            .filter(|&(i, j)| floored.fires(Some(&raw.view()), i, j, 0.8))
            .count();
        assert_eq!(fired_floored, 10, "the radian floor must bound the budget");
    }

    /// The `ZeroSlope` arm must stay *expensive* on a coherent edge - that is
    /// the whole reason it is only a diagnostic. If this ever inverts, the
    /// interpretation in `docs/BUG_NISAR_CRYO_STACKED_CUTS.md` is wrong.
    #[test]
    fn zero_slope_cost_is_not_free_on_coherent_edges() {
        let lut = spline_lut::get_or_load();
        assert!(
            lut.cost(0.0, 0.8, 16.0) > 0,
            "a coherent flat-slope cut must cost something"
        );
    }
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
        // cost (≈ free to route 2π discontinuities through them) - the
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
    use ndarray::{Array2, array};

    /// Rectangular box filter normalizes by its true tap count in each axis and
    /// centers odd windows. A constant field must come back unchanged.
    #[test]
    fn box_filter_rectangular_preserves_constant() {
        let a = Array2::<f32>::from_elem((9, 11), 3.5);
        for &(kr, kc) in &[(1usize, 1usize), (7, 3), (3, 7), (2, 4), (5, 5)] {
            let out = box_filter_2d(a.view(), kr, kc);
            assert_eq!(out.dim(), a.dim());
            for &v in out.iter() {
                assert!((v - 3.5).abs() < 1e-5, "kr={kr} kc={kc}: got {v}");
            }
        }
    }

    /// A 1x1 window is a no-op, so `smooth_phase_gradients` reduces to the raw
    /// gradients.
    #[test]
    fn unit_window_equals_raw_gradients() {
        let igram = array![
            [
                Complex32::from_polar(1.0, 0.0),
                Complex32::from_polar(1.0, 0.3)
            ],
            [
                Complex32::from_polar(1.0, 0.7),
                Complex32::from_polar(1.0, 1.1)
            ],
        ];
        let (raw_dy, raw_dx) = phase_gradients_raw(igram.view());
        let (s_dy, s_dx) = smooth_phase_gradients(igram.view(), PhaseGradWindow::new(1, 1));
        assert_eq!(raw_dy, s_dy);
        assert_eq!(raw_dx, s_dx);
    }

    /// The window orientation is swapped between the two gradients: for a purely
    /// row-varying phase (constant along columns), the horizontal gradient is
    /// zero and only the vertical gradient's `parallel` (row) extent smooths it,
    /// so `(par, perp)` and `(perp, par)` generally differ.
    #[test]
    fn orientation_swap_is_not_symmetric() {
        // Phase = f(row): step at row 8, zero horizontal gradient everywhere.
        let m = 24;
        let n = 12;
        let igram = Array2::from_shape_fn((m, n), |(i, _)| {
            let ph = if i < m / 2 { 0.0 } else { 0.9 };
            Complex32::from_polar(1.0, ph)
        });
        let (a_dy, _) = smooth_phase_gradients(igram.view(), PhaseGradWindow::new(9, 1));
        let (b_dy, _) = smooth_phase_gradients(igram.view(), PhaseGradWindow::new(1, 9));
        // (9,1) smooths dy over 9 rows; (1,9) smooths dy over 1 row (=raw). The
        // step region must differ between the two.
        let diff: f32 = (&a_dy - &b_dy).iter().map(|v| v.abs()).sum();
        assert!(
            diff > 1e-3,
            "orientation swap should change dy smoothing, diff={diff}"
        );
    }

    /// Smooth-phase IG: every smoothed gradient ≈ 0, so every offset
    /// should round to 0. Coherence is constant ⇒ uniform weight.
    #[test]
    fn smooth_phase_gives_zero_offsets() {
        let m = 16;
        let n = 16;
        let igram = Array2::from_shape_fn((m, n), |_| Complex32::from_polar(1.0, 0.0));
        let corr = Array2::<f32>::from_elem((m, n), 0.7);
        let (offsets, weights) = compute_snaphu_smooth_costs(
            igram.view(),
            corr.view(),
            4.0,
            None,
            PhaseGradWindow::default(),
        );

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
    /// gives ≈zero offsets - the absolute slope is the coarse anchor's job, not
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
        let (off_ramp, _) = compute_snaphu_smooth_costs(
            ramp.view(),
            corr.view(),
            10.0,
            None,
            PhaseGradWindow::default(),
        );
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
        let (off_wall, _) = compute_snaphu_smooth_costs(
            wall.view(),
            corr.view(),
            10.0,
            None,
            PhaseGradWindow::default(),
        );
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
        let (offsets, weights) = compute_snaphu_smooth_costs(
            igram.view(),
            corr.view(),
            4.0,
            Some(mask.view()),
            PhaseGradWindow::default(),
        );
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
