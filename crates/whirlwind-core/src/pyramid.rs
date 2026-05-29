//! Pyramidal (coarse-to-fine, multi-resolution) phase unwrap.
//!
//! Motivation — the aliasing trap in single-shot multilook-first.
//! [`crate::tile::unwrap_tiled`]'s `multilook` path suppresses noise by
//! coherently down-looking the complex igram by a big fixed factor `L`, then
//! unwraps the COARSE grid and block-replicates the result. That is a fine
//! noise filter, but it is a *destructive* one for steep signals: the coarse
//! grid's per-pixel gradient is `L×` the full-res gradient, so a full-res
//! fringe rate `g` (rad/pixel) becomes `L·g` on the coarse grid. As soon as
//! `L·g > π` the coarse grid is itself ALIASED — adjacent coarse pixels differ
//! by more than half a cycle, the coarse unwrap "locks on" to the wrong (too
//! few) integer cycle count, and block-replicating that wrong `K` field back to
//! full resolution can never recover the true surface. A dense-fringe signal
//! that full-res unwrapping gets right (a volcano eruption bowl, an earthquake
//! near-field) is silently destroyed by aggressive multilook-first.
//!
//! The fix here is classic multigrid / multi-resolution unwrapping. Instead of
//! one big jump, refine by powers of two (`base, base/2, …, 1`). Each finer
//! level does NOT re-unwrap the absolute phase; it unwraps only the *residual*
//! relative to the prediction inherited from the next-coarser level:
//!
//! 1. Coarsest level: ordinary whole-image unwrap of the `base×` down-looked
//!    igram. (This level must be unaliased at its own scale, i.e.
//!    `base·g < π` — choose `base` accordingly; the refinement below recovers
//!    the resolution a single big multilook would have thrown away.)
//! 2. Each finer level `f`: bilinearly upsample the previous level's unwrapped
//!    phase to this level's grid → `pred`. Rotate this level's complex igram by
//!    `exp(−i·pred)` so its phase becomes `wrap(angle − pred)` — the residual
//!    wrapped phase. Because `pred` already carries the large-scale gradient,
//!    the residual gradient is small (well under π), so a plain unwrap solves
//!    it without aliasing. The level's phase is `pred + unwrap(residual)`.
//!
//! `pred` is exactly the user's "previous solved K as a prior": per pixel,
//! `round((pred − angle)/2π)` is the integer cycle the coarse solve believes
//! this pixel sits in, and the residual unwrap only corrects deviations from
//! it. Refining all the way to `f = 1` always returns a full-resolution
//! surface — never the blocky block-replicated field of single-shot multilook.
//!
//! Base solver — why NOT the linear coherence cost. The default [`crate::unwrap`]
//! linear coherence cost mis-routes on smooth steep signals: a radial bowl whose
//! wrap-lines are concentric rings must drain those rings at the image boundary,
//! and the capacity-1 frame arcs can't carry the stacked flow (the same
//! pathology as the ignored `diagonal_ramp_512` regression). The symptom is
//! wrong integer cycles in the CORNERS (the steepest part of a bowl) even on a
//! perfectly clean input — `unwrap` scores only ~88 % on a clean `0.7π` bowl
//! while [`crate::unwrap_reuse`] and [`crate::unwrap_convex`] score 100 %. The
//! pyramid therefore defaults its per-level solve to [`BaseSolver::Reuse`]; see
//! `paper/pyramid_aliasing.md` for the sweep.
//!
//! What this does and does not buy you. It cannot recover a signal that is
//! genuinely aliased at full resolution (`g > π` — a hard Nyquist limit no
//! unwrapper escapes), nor escape the `base·g < π` wall at its coarsest level.
//! Its win is the noisy-but-steep middle: data noisy enough that a full-res
//! solve drowns in residues, yet steep enough that the big multilook a
//! single-shot filter would need aliases the signal. There the pyramid takes
//! its noise robustness from the coarse prediction and its resolution from the
//! residual refinement. See `scripts/dense_fringe_pyramid.py`.

use crate::UnwrapError;
use crate::tile::multilook_complex;
use ndarray::{Array2, ArrayView2, s};
use num_complex::Complex32;
use rayon::prelude::*;
use std::f32::consts::{PI, TAU};

/// Per-level base unwrap solver for the pyramid.
///
/// The residual at each finer level is a small-gradient field, and the
/// coarsest level is a heavily-looked (high-coherence) down-look — both regimes
/// where the convex / reuse solvers fix the linear cost's smooth-signal
/// boundary-stacking failure. [`Reuse`](BaseSolver::Reuse) is the default.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BaseSolver {
    /// Linear Carballo coherence cost ([`crate::unwrap`]). Fast, but
    /// mis-routes the corners of smooth steep signals (capacity-1 stacking).
    Linear,
    /// SNAPHU-style convex quadratic cost ([`crate::unwrap_convex`]).
    Convex,
    /// PHASS-style flow-reuse ([`crate::unwrap_reuse`]). Default.
    Reuse,
}

impl BaseSolver {
    /// Parse a solver name (`"linear"`, `"convex"`, `"reuse"`); case-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "linear" => Some(BaseSolver::Linear),
            "convex" => Some(BaseSolver::Convex),
            "reuse" => Some(BaseSolver::Reuse),
            _ => None,
        }
    }

    fn solve(
        self,
        igram: ArrayView2<Complex32>,
        corr: ArrayView2<f32>,
        nlooks: f32,
        mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f32>, UnwrapError> {
        match self {
            BaseSolver::Linear => crate::unwrap(igram, corr, nlooks, mask),
            BaseSolver::Convex => crate::unwrap_convex(igram, corr, nlooks, mask),
            BaseSolver::Reuse => crate::unwrap_reuse(igram, corr, nlooks, mask),
        }
    }
}

/// Solve the COARSEST level (absolute phase, no prediction yet). When the grid
/// exceeds `tile_size` it uses the full tiled path — whose global coarse anchor
/// is exactly right for absolute phase — to bound memory; otherwise a single
/// whole-image `solver` solve.
fn solve_coarsest(
    solver: BaseSolver,
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    tile_size: usize,
) -> Result<Array2<f32>, UnwrapError> {
    let (cm, cn) = igram.dim();
    if tile_size >= 4 && (cm > tile_size || cn > tile_size) {
        let ov = (tile_size / 4).max(2);
        crate::tile::unwrap_tiled(igram, corr, nlooks, mask, tile_size, ov, 1)
    } else {
        solver.solve(igram, corr, nlooks, mask)
    }
}

/// Solve a finer level's RESIDUAL field (`rig` = level igram rotated by
/// `exp(-i·pred)`), optionally tiling it to bound memory on a large frame.
///
/// Unlike the coarsest level, the residual is *relative* to a global
/// prediction, so the tiled path's absolute-phase anchor/cascade machinery is
/// not just unnecessary but actively harmful (it region-votes a near-flat field
/// into garbage). Instead we tile with a lightweight scheme that leans on the
/// prediction the residual is already measured against: solve each (overlapping)
/// tile independently — trivial, since the residual is small-gradient — then
/// gauge each tile to a common cycle by removing its rounded-2π median (the only
/// freedom a per-tile MCF solve has), and feather-composite the overlaps.
/// Because every tile's residual is referenced to the *same* prediction, this
/// needs no inter-tile 2π reconciliation as long as the residual stays within a
/// cycle — which a good prediction guarantees.
fn solve_residual(
    solver: BaseSolver,
    rig: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    rmask: ArrayView2<bool>,
    tile_size: usize,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = rig.dim();
    if !(tile_size >= 4 && (m > tile_size || n > tile_size)) {
        return solver.solve(rig, corr, nlooks, Some(rmask));
    }

    let ov = (tile_size / 4).max(2);
    let tiles = crate::tile::decompose(m, n, tile_size, ov);
    let solved: Vec<Array2<f32>> = tiles
        .par_iter()
        .map(|t| {
            let sig = rig.slice(s![t.r0..t.r1, t.c0..t.c1]).to_owned();
            let sco = corr.slice(s![t.r0..t.r1, t.c0..t.c1]).to_owned();
            let sm = rmask.slice(s![t.r0..t.r1, t.c0..t.c1]).to_owned();
            solver.solve(sig.view(), sco.view(), nlooks, Some(sm.view()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Feather-composite with a triangular taper (peaks at tile centre), after
    // gauging each tile by its rounded-2π median.
    let taper = |p: usize, len: usize| -> f32 { (p + 1).min(len - p) as f32 };
    let mut acc = Array2::<f32>::zeros((m, n));
    let mut wsum = Array2::<f32>::zeros((m, n));
    for (t, sub) in tiles.iter().zip(solved.iter()) {
        let mut vals: Vec<f32> = sub.iter().copied().filter(|v| v.is_finite()).collect();
        let off = if vals.is_empty() {
            0.0
        } else {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = vals[vals.len() / 2];
            (med / TAU).round() * TAU
        };
        let (tr, tc) = (t.rows(), t.cols());
        for ti in 0..tr {
            let gi = t.r0 + ti;
            let wr = taper(ti, tr);
            for tj in 0..tc {
                let v = sub[(ti, tj)];
                if v.is_finite() {
                    let w = wr * taper(tj, tc);
                    acc[(gi, t.c0 + tj)] += w * (v - off);
                    wsum[(gi, t.c0 + tj)] += w;
                }
            }
        }
    }
    let mut out = Array2::<f32>::from_elem((m, n), f32::NAN);
    for i in 0..m {
        for j in 0..n {
            if wsum[(i, j)] > 0.0 {
                out[(i, j)] = acc[(i, j)] / wsum[(i, j)];
            }
        }
    }
    Ok(out)
}

/// Bilinear upsample a coarse `(cm, cn)` field to `(m, n)`, skipping NaN
/// contributors. Coarse cell `(ci, cj)` is treated as the block-mean of the
/// `(m/cm)×(n/cn)` full-res block centred on it, so the source coordinate of
/// full-res pixel `i` is `(i + 0.5)·cm/m − 0.5` (center-aligned). Each output
/// is the weight-renormalised bilinear blend of the up-to-four surrounding
/// finite coarse cells; if all four are NaN the output stays NaN.
fn upsample_bilinear(coarse: &Array2<f32>, m: usize, n: usize) -> Array2<f32> {
    let (cm, cn) = coarse.dim();
    let mut out = Array2::<f32>::from_elem((m, n), f32::NAN);
    if cm == 0 || cn == 0 {
        return out;
    }
    let sy = cm as f32 / m as f32;
    let sx = cn as f32 / n as f32;
    for i in 0..m {
        let fy = ((i as f32 + 0.5) * sy - 0.5).clamp(0.0, (cm - 1) as f32);
        let i0 = fy.floor() as usize;
        let i1 = (i0 + 1).min(cm - 1);
        let wy = fy - i0 as f32;
        for j in 0..n {
            let fx = ((j as f32 + 0.5) * sx - 0.5).clamp(0.0, (cn - 1) as f32);
            let j0 = fx.floor() as usize;
            let j1 = (j0 + 1).min(cn - 1);
            let wx = fx - j0 as f32;

            let mut acc = 0.0_f32;
            let mut wsum = 0.0_f32;
            for (ii, wi) in [(i0, 1.0 - wy), (i1, wy)] {
                for (jj, wj) in [(j0, 1.0 - wx), (j1, wx)] {
                    let v = coarse[(ii, jj)];
                    let w = wi * wj;
                    if v.is_finite() && w > 0.0 {
                        acc += w * v;
                        wsum += w;
                    }
                }
            }
            if wsum > 0.0 {
                out[(i, j)] = acc / wsum;
            }
        }
    }
    out
}

/// Build the coarse→fine factor schedule `[base, base/2, …, 1]`, clamping
/// `base` down so every level grid is at least `2×2`.
fn factor_schedule(base_factor: usize, m: usize, n: usize) -> Vec<usize> {
    let mut base = base_factor.max(1);
    while base > 1 && (m / base < 2 || n / base < 2) {
        base /= 2;
    }
    let mut factors = Vec::new();
    let mut f = base;
    while f > 1 {
        factors.push(f);
        f /= 2;
    }
    factors.push(1);
    factors
}

/// Itoh-violation rate of the `f×` down-looked igram: the fraction of adjacent
/// coarse-pixel wrapped phase differences whose magnitude exceeds `0.6π`.
///
/// This directly measures the aliasing (Nyquist) condition. Phase unwrapping
/// assumes adjacent samples differ by less than half a cycle; a down-look that
/// pushes the per-pixel gradient past that produces a burst of large wrapped
/// jumps. Crucially this separates the two effects that confounded a plain
/// residue-density probe: phase *noise* contributes a roughly constant low rate
/// that does NOT grow with `f` (coherent averaging even shrinks it), whereas
/// *aliasing* makes the rate jump sharply once `f·g > π`. So a fixed absolute
/// threshold cleanly marks the aliasing onset (validated in
/// `scripts/dense_fringe_pyramid.py`).
fn itoh_violation_rate(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    f: usize,
) -> f32 {
    let (cig, _ccorr, cmask) = multilook_complex(igram, corr, mask, f);
    let (cm, cn) = cig.dim();
    if cm < 2 || cn < 2 {
        return 0.0;
    }
    let thr = 0.6 * PI;
    let wrap = |d: f32| -> f32 { d - TAU * (d / TAU).round() };
    let phase = |i: usize, j: usize| cig[(i, j)].arg();
    let valid = |i: usize, j: usize| cmask[(i, j)] && cig[(i, j)].norm() > 0.0;
    let mut viol = 0usize;
    let mut tot = 0usize;
    for i in 0..cm {
        for j in 0..cn {
            if !valid(i, j) {
                continue;
            }
            if i + 1 < cm && valid(i + 1, j) {
                tot += 1;
                if wrap(phase(i + 1, j) - phase(i, j)).abs() > thr {
                    viol += 1;
                }
            }
            if j + 1 < cn && valid(i, j + 1) {
                tot += 1;
                if wrap(phase(i, j + 1) - phase(i, j)).abs() > thr {
                    viol += 1;
                }
            }
        }
    }
    if tot == 0 {
        0.0
    } else {
        viol as f32 / tot as f32
    }
}

/// Choose the coarsest down-look factor that is still unaliased, via the
/// [`itoh_violation_rate`] probe. Walk `1, 2, 4, …, max_factor` and keep
/// doubling the down-look while the next level either sits below the benign
/// noise FLOOR or *meaningfully decreases* the rate (coherent averaging
/// suppressing noise on a still-unaliased grid). Stop the first time the rate
/// holds flat or rises — that is the aliasing onset — and return the factor
/// before it.
///
/// Why both conditions. An *absolute* threshold alone fails because phase noise
/// pushes the rate high (≈0.2–0.4 at 4 looks / γ≲0.3) with no aliasing — under
/// a pure threshold the probe would never downsample noisy data, defeating the
/// whole point. A *decrease* rule alone fails on clean gentle data (the rate is
/// already ≈0 and cannot decrease further). Together: keep going while the grid
/// is clean (`rate < FLOOR`) OR the down-look is still buying noise suppression
/// (`rate` dropped by ≥ `DECR`); a flat/rising rate is the aliasing fold.
///
/// LIMITATION — the constant-ramp blind spot. The probe (like any local
/// gradient/curl measure) detects aliasing through the wrapped jumps it creates,
/// which appear only where the gradient *varies* (any real localized signal:
/// bowls, point sources, faults). A perfectly constant-rate ramp aliases
/// coherently — adjacent aliased pixels' wrapped gradient folds back small — so
/// the probe cannot see it. Benign for *clean* ramps (it keeps base = 1 and the
/// reuse solver handles the unaliased full-res signal), but a *noisy* steep
/// near-constant ramp can fool it; pass an explicit `base_factor` there.
pub fn auto_base_factor(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    max_factor: usize,
) -> usize {
    // Benign noise/discretization floor, and the minimum rate *drop* that still
    // counts as "this down-look is suppressing noise (not yet aliasing)". Tuned
    // on the synthetic dense-fringe sweep; see `paper/pyramid_aliasing.md`.
    const FLOOR: f32 = 0.05;
    const DECR: f32 = 0.02;
    let (m, n) = igram.dim();
    let mut best = 1usize;
    let mut prev = itoh_violation_rate(igram, corr, mask, 1);
    let mut f = 2usize;
    while f <= max_factor && m / f >= 4 && n / f >= 4 {
        let r = itoh_violation_rate(igram, corr, mask, f);
        if r < FLOOR || r <= prev - DECR {
            best = f;
            prev = r;
            f *= 2;
        } else {
            break;
        }
    }
    best
}

/// Pyramidal coarse-to-fine phase unwrap (configurable base solver).
///
/// Inputs match [`crate::unwrap`] plus:
/// * `base_factor` — coarsest power-of-two down-look. `0` ⇒ choose it
///   automatically via [`auto_base_factor`] (capped at 16). The schedule is
///   `base_factor, base_factor/2, …, 1`; `1` degenerates to a single full-res
///   solve with `solver`.
/// * `solver` — per-level base unwrap ([`BaseSolver::Reuse`] recommended).
/// * `tile_size` — if ≥ 4, any level whose grid exceeds it is tiled (memory
///   bound for the finest levels of a large frame). `0` ⇒ never tile.
///
/// See the module docs for the algorithm, the `base·g < π` constraint, and why
/// the default solver is not the linear coherence cost.
pub fn unwrap_pyramid(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    mask: Option<ArrayView2<bool>>,
    base_factor: usize,
    solver: BaseSolver,
    tile_size: usize,
) -> Result<Array2<f32>, UnwrapError> {
    let (m, n) = igram.dim();
    if (m, n) != corr.dim() {
        return Err(UnwrapError::ShapeMismatch((m, n), corr.dim()));
    }
    if m < 2 || n < 2 {
        return Err(UnwrapError::TooSmall((m, n)));
    }

    let base = if base_factor == 0 {
        auto_base_factor(igram, corr, mask, 16)
    } else {
        base_factor
    };
    let factors = factor_schedule(base, m, n);

    // `prev` holds the previous (coarser) level's unwrapped phase on its own
    // grid; we upsample it to each finer level to form that level's prediction.
    let mut prev: Option<Array2<f32>> = None;
    for &f in &factors {
        let (cig, ccorr, cmask) = multilook_complex(igram, corr, mask, f);
        let (cm, cn) = cig.dim();
        let level_looks = nlooks * (f * f) as f32;

        let level_unw = match prev {
            // Coarsest level: absolute phase, no prior yet.
            None => solve_coarsest(
                solver,
                cig.view(),
                ccorr.view(),
                level_looks,
                Some(cmask.view()),
                tile_size,
            )?,
            // Finer level: unwrap only the residual against the upsampled prior.
            Some(ref p) => {
                let pred = upsample_bilinear(p, cm, cn);
                // Rotate the level igram by exp(-i·pred): the rotated phase is
                // wrap(angle - pred) (small gradient), magnitude (hence coherence)
                // is untouched. A pixel with no finite prediction is masked out
                // of the residual solve and stays NaN in the output.
                let mut rig = Array2::<Complex32>::zeros((cm, cn));
                let mut rmask = Array2::<bool>::from_elem((cm, cn), false);
                for ci in 0..cm {
                    for cj in 0..cn {
                        let pv = pred[(ci, cj)];
                        if cmask[(ci, cj)] && pv.is_finite() {
                            rig[(ci, cj)] = cig[(ci, cj)] * Complex32::from_polar(1.0, -pv);
                            rmask[(ci, cj)] = true;
                        }
                    }
                }
                let resid = solve_residual(
                    solver,
                    rig.view(),
                    ccorr.view(),
                    level_looks,
                    rmask.view(),
                    tile_size,
                )?;
                let mut lv = Array2::<f32>::from_elem((cm, cn), f32::NAN);
                for ci in 0..cm {
                    for cj in 0..cn {
                        let pv = pred[(ci, cj)];
                        let rv = resid[(ci, cj)];
                        if pv.is_finite() && rv.is_finite() {
                            lv[(ci, cj)] = pv + rv;
                        }
                    }
                }
                lv
            }
        };
        prev = Some(level_unw);
    }

    // The last level is f = 1, so `prev` is already full resolution.
    Ok(prev.expect("factor schedule always ends with f = 1"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn cone(m: usize, n: usize, grad: f32) -> Array2<f32> {
        let (ci, cj) = ((m as f32 - 1.0) * 0.5, (n as f32 - 1.0) * 0.5);
        Array2::from_shape_fn((m, n), |(i, j)| {
            let dy = i as f32 - ci;
            let dx = j as f32 - cj;
            grad * (dy * dy + dx * dx).sqrt()
        })
    }

    fn bowl(m: usize, n: usize, g_edge: f32) -> Array2<f32> {
        let (ci, cj) = ((m as f32 - 1.0) * 0.5, (n as f32 - 1.0) * 0.5);
        let r_max = (ci * ci + cj * cj).sqrt();
        let a = g_edge / (2.0 * r_max);
        Array2::from_shape_fn((m, n), |(i, j)| {
            let dy = i as f32 - ci;
            let dx = j as f32 - cj;
            a * (dy * dy + dx * dx)
        })
    }

    fn align_and_kfrac(unw: &Array2<f32>, truth: &Array2<f32>) -> f32 {
        let tau = std::f32::consts::TAU;
        let mean: f64 = unw
            .iter()
            .zip(truth.iter())
            .map(|(&a, &b)| (a - b) as f64)
            .sum::<f64>()
            / unw.len() as f64;
        let off = (mean / tau as f64).round() as f32 * tau;
        let correct = unw
            .iter()
            .zip(truth.iter())
            .filter(|&(&a, &b)| ((a - off - b) / tau).round() == 0.0)
            .count();
        correct as f32 / unw.len() as f32
    }

    fn synth(truth: &Array2<f32>, coh: f32) -> (Array2<Complex32>, Array2<f32>) {
        let wrapped = crate::simulate::wrap_phase(truth);
        let igram = wrapped.mapv(|p| Complex32::new(p.cos(), p.sin()));
        let corr = Array2::<f32>::from_elem(igram.dim(), coh);
        (igram, corr)
    }

    #[test]
    fn upsample_bilinear_monotone_on_ramp() {
        let coarse = Array2::from_shape_fn((8, 8), |(i, j)| (i + j) as f32);
        let up = upsample_bilinear(&coarse, 32, 32);
        assert!(up[(15, 17)].is_finite());
        assert!(up[(15, 18)] > up[(15, 16)]);
        assert!(up[(16, 17)] > up[(14, 17)]);
    }

    #[test]
    fn factor_schedule_halves_to_one() {
        assert_eq!(factor_schedule(8, 256, 256), vec![8, 4, 2, 1]);
        assert_eq!(factor_schedule(1, 256, 256), vec![1]);
        assert_eq!(factor_schedule(8, 6, 6), vec![2, 1]);
    }

    #[test]
    fn reuse_solver_fixes_clean_bowl_corners() {
        // The linear cost mis-routes the corners of a clean steep bowl; the
        // reuse solver does not. base=1 isolates the base-solver behaviour.
        let truth = bowl(192, 192, 0.6 * std::f32::consts::PI);
        let (igram, corr) = synth(&truth, 0.999);

        let lin = unwrap_pyramid(
            igram.view(),
            corr.view(),
            1.0,
            None,
            1,
            BaseSolver::Linear,
            0,
        )
        .unwrap();
        let reu = unwrap_pyramid(
            igram.view(),
            corr.view(),
            1.0,
            None,
            1,
            BaseSolver::Reuse,
            0,
        )
        .unwrap();
        let klin = align_and_kfrac(&lin, &truth);
        let kreu = align_and_kfrac(&reu, &truth);
        assert!(
            kreu > 0.99,
            "reuse base should solve the clean bowl, got {kreu}"
        );
        assert!(
            kreu > klin + 0.05,
            "reuse ({kreu}) should beat linear ({klin}) on corners"
        );
    }

    #[test]
    fn pyramid_recovers_clean_dense_cone() {
        // A steep but unaliased cone (g = 0.4π/pixel). A single ×8 multilook
        // would alias (8·0.4π = 3.2π); with a base whose coarsest level stays
        // unaliased (2·0.4π = 0.8π < π) the pyramid refines back to near-perfect.
        let truth = cone(192, 192, 0.4 * std::f32::consts::PI);
        let (igram, corr) = synth(&truth, 0.98);
        let unw = unwrap_pyramid(
            igram.view(),
            corr.view(),
            1.0,
            None,
            2,
            BaseSolver::Reuse,
            0,
        )
        .unwrap();
        assert!(align_and_kfrac(&unw, &truth) > 0.9);
    }

    #[test]
    fn pyramid_coarsest_must_be_unaliased() {
        // base = 8 on a 0.4π cone: coarsest gradient 8·0.4π = 3.2π — aliased.
        // Documents the hard base·g < π wall.
        let truth = cone(192, 192, 0.4 * std::f32::consts::PI);
        let (igram, corr) = synth(&truth, 0.98);
        let unw = unwrap_pyramid(
            igram.view(),
            corr.view(),
            1.0,
            None,
            8,
            BaseSolver::Reuse,
            0,
        )
        .unwrap();
        assert!(align_and_kfrac(&unw, &truth) < 0.6);
    }

    #[test]
    fn itoh_probe_separates_noise_from_aliasing() {
        // Gentle cone: violation rate stays ~0 even far down (probe → big base).
        let gentle = cone(256, 256, 0.05 * std::f32::consts::PI);
        let (gig, gcorr) = synth(&gentle, 0.999);
        assert!(auto_base_factor(gig.view(), gcorr.view(), None, 16) >= 4);

        // Steep bowl: 2·0.6π corner is borderline (refinement saves it) but
        // 4·0.6π aliases hard, so the probe must stop at 2 — exactly where the
        // residue-density probe failed (it sailed to 4 and destroyed the bowl).
        let steep = bowl(256, 256, 0.6 * std::f32::consts::PI);
        let (sig, scorr) = synth(&steep, 0.999);
        let b = auto_base_factor(sig.view(), scorr.view(), None, 16);
        assert!(b <= 2, "probe should stop before the bowl aliases, got {b}");
    }

    #[test]
    fn auto_base_recovers_clean_bowl() {
        // End-to-end: the auto base must recover a clean steep bowl (the case
        // the old residue-density probe got wrong by over-downsampling).
        let truth = bowl(256, 256, 0.6 * std::f32::consts::PI);
        let (igram, corr) = synth(&truth, 0.999);
        let unw = unwrap_pyramid(
            igram.view(),
            corr.view(),
            1.0,
            None,
            0,
            BaseSolver::Reuse,
            0,
        )
        .unwrap();
        assert!(align_and_kfrac(&unw, &truth) > 0.9);
    }

    #[test]
    fn tiled_finest_level_matches_untiled() {
        // Tiling a gentle scene's finest level must not change the answer
        // beyond seam-level noise.
        let truth = bowl(192, 192, 0.3 * std::f32::consts::PI);
        let (igram, corr) = synth(&truth, 0.999);
        let untiled = unwrap_pyramid(
            igram.view(),
            corr.view(),
            1.0,
            None,
            4,
            BaseSolver::Reuse,
            0,
        )
        .unwrap();
        let tiled = unwrap_pyramid(
            igram.view(),
            corr.view(),
            1.0,
            None,
            4,
            BaseSolver::Reuse,
            64,
        )
        .unwrap();
        let ku = align_and_kfrac(&untiled, &truth);
        let kt = align_and_kfrac(&tiled, &truth);
        assert!(
            kt > 0.9 && (kt - ku).abs() < 0.06,
            "tiled {kt} vs untiled {ku}"
        );
    }
}
