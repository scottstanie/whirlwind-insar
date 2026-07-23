//! Connected component growing for the legacy linear-cost path and the default
//! SNAPHU ambiguity-wiggle path.
//!
//! The legacy [`grow_components`] path labels each pixel-edge as a *cut* when at
//! least one of its underlying linear-cost arcs is unreliable:
//!
//! 1. The arc is forbidden (both directions saturated - masked-out pixel).
//! 2. The minimum raw forward cost across the two directions is below
//!    `cost_threshold` - the noise model is uninformative here, so any
//!    routing decision across this edge wasn't strongly supported by data.
//!
//! Then BFS the remaining pixels through non-cut edges to label connected
//! components. Small components are dropped; the top `max_ncomps` by size are
//! kept and renumbered 1..=max_ncomps.
//!
//! This is the gridded linear-cost adaptation of `GrowConnCompsMask` in SNAPHU's
//! `snaphu_tile.c`. SNAPHU works with convex piecewise costs and tests
//! `min(negcost, poscost)` - the local cost-function flatness around the
//! chosen flow. In our linear unit-capacity setting that collapses to the
//! raw arc cost (no curvature anywhere), and MCF flow placement is
//! deliberately *not* a cut signal: a high-cost branch cut means MCF paid
//! the right price to close a noise-induced residue pair, which is the
//! correct answer to encode, not an unreliable region. Low-cost edges where
//! MCF *does* place flow show up as cuts anyway by the raw-cost test.

use crate::cost;
use crate::grid::RectangularGridGraph;
use crate::network::Network;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex32;
use std::collections::VecDeque;
use std::f32::consts::TAU;

/// Parameters for the legacy linear [`grow_components`] path. The
/// `cost_threshold` is compared against the Carballo coherence cost grid
/// (`CARBALLO_COST_SCALE = 6`); the default of 50 corresponds to a per-edge
/// one-cycle probability of ~2.4e-4.
#[derive(Debug, Clone)]
pub struct ConnCompParams {
    /// Cut a pixel edge when min raw forward cost across the two underlying
    /// arcs is ≤ this. Higher → more boundaries and smaller components.
    pub cost_threshold: i32,
    /// Drop components smaller than this many pixels. This ABSOLUTE floor is the
    /// binding control: at 80 m it is 0.8 km/side, at 30 m 0.3 km - scene-size-
    /// and pixel-spacing-invariant (matches SNAPHU's `minregionsize`). A small
    /// coherent island stays a usable, self-consistent component the caller can
    /// re-reference into; only sub-floor speckle is dropped.
    pub min_size_px: usize,
    /// Vestigial fractional floor, kept only as an anti-pathology cap on huge
    /// frames (it can only RAISE `min_size_px`, never lower it). At the default
    /// 1e-4 it stays negligible (<100 px below ~1M valid). Do NOT raise toward
    /// 0.01 - on a NISAR frame 1% is a ~25 km minimum feature, which orphans
    /// every island a user might want to reference into. The effective floor is
    /// `max(min_size_px, ceil(min_size_frac * n_valid))`, so this fraction only
    /// bites when it exceeds the absolute `min_size_px` control.
    pub min_size_frac: f32,
    /// Keep at most this many components (largest by size). 0 → keep all. The
    /// `min_size_px` floor is the real speckle control; this is only a guard
    /// against a pathological scene emitting tens of thousands of labels, set
    /// generously so it never clips a genuine feature the floor admits.
    pub max_ncomps: u32,
}

impl Default for ConnCompParams {
    fn default() -> Self {
        Self {
            cost_threshold: 50,
            min_size_px: 100,
            min_size_frac: 0.0001,
            max_ncomps: 1024,
        }
    }
}

#[inline]
fn edge_is_cut(net: &Network, fwd1: usize, fwd2: usize, thresh: i32) -> bool {
    let nf = net.num_forward();
    let sat = |a: usize| net.is_arc_saturated(a);
    let forbidden1 = sat(fwd1) && sat(fwd1 + nf);
    let forbidden2 = sat(fwd2) && sat(fwd2 + nf);
    if forbidden1 || forbidden2 {
        return true;
    }
    (net.cost_fwd[fwd1].min(net.cost_fwd[fwd2]) as i32) <= thresh
}

/// Grow components on the pixel grid using a solved MCF network. Returns a
/// `(m_phase, n_phase)` `u32` label array; 0 = unassigned (cut off or smaller
/// than `max(min_size_px, ceil(min_size_frac * n_valid))`), renumbered by size.
pub fn grow_components(
    g: &RectangularGridGraph,
    net: &Network,
    pixel_mask: Option<ArrayView2<bool>>,
    params: &ConnCompParams,
) -> Array2<u32> {
    let m_phase = g.m - 1;
    let n_phase = g.n - 1;
    assert!(m_phase >= 1 && n_phase >= 1);
    if let Some(mm) = pixel_mask {
        assert_eq!(
            mm.dim(),
            (m_phase, n_phase),
            "pixel mask must be (m-1, n-1)"
        );
    }

    let valid = |i: usize, j: usize| pixel_mask.map(|m| m[(i, j)]).unwrap_or(true);
    let n_valid: usize = (0..m_phase)
        .flat_map(|i| (0..n_phase).map(move |j| (i, j)))
        .filter(|&(i, j)| valid(i, j))
        .count();
    // Absolute floor governs; the fraction only ever RAISES it (a generous cap
    // on huge frames), so a coherent island down to `min_size_px` is kept.
    let frac_floor = (params.min_size_frac as f64 * n_valid as f64).ceil() as usize;
    let min_size = params.min_size_px.max(frac_floor).max(1);

    let mut labels = Array2::<u32>::zeros((m_phase, n_phase));
    let mut next_label: u32 = 0;
    let mut sizes: Vec<usize> = vec![0]; // sizes[0] is a placeholder

    for si in 0..m_phase {
        for sj in 0..n_phase {
            if labels[(si, sj)] != 0 || !valid(si, sj) {
                continue;
            }
            next_label += 1;
            let label = next_label;
            let mut q: VecDeque<(usize, usize)> = VecDeque::new();
            q.push_back((si, sj));
            labels[(si, sj)] = label;
            let mut size = 0_usize;

            while let Some((i, j)) = q.pop_front() {
                size += 1;

                // Right: pixel edge (i, j)-(i, j+1) ↔ down(i,j+1)+up(i+1,j+1)
                if j + 1 < n_phase && labels[(i, j + 1)] == 0 && valid(i, j + 1) {
                    let fwd1 = g.down_arc(i, j + 1).unwrap();
                    let fwd2 = g.up_arc(i + 1, j + 1).unwrap();
                    if !edge_is_cut(net, fwd1, fwd2, params.cost_threshold) {
                        labels[(i, j + 1)] = label;
                        q.push_back((i, j + 1));
                    }
                }
                // Left: pixel edge (i, j-1)-(i, j) ↔ down(i,j)+up(i+1,j)
                if j >= 1 && labels[(i, j - 1)] == 0 && valid(i, j - 1) {
                    let fwd1 = g.down_arc(i, j).unwrap();
                    let fwd2 = g.up_arc(i + 1, j).unwrap();
                    if !edge_is_cut(net, fwd1, fwd2, params.cost_threshold) {
                        labels[(i, j - 1)] = label;
                        q.push_back((i, j - 1));
                    }
                }
                // Down: pixel edge (i, j)-(i+1, j) ↔ right(i+1,j)+left(i+1,j+1)
                if i + 1 < m_phase && labels[(i + 1, j)] == 0 && valid(i + 1, j) {
                    let fwd1 = g.right_arc(i + 1, j).unwrap();
                    let fwd2 = g.left_arc(i + 1, j + 1).unwrap();
                    if !edge_is_cut(net, fwd1, fwd2, params.cost_threshold) {
                        labels[(i + 1, j)] = label;
                        q.push_back((i + 1, j));
                    }
                }
                // Up: pixel edge (i-1, j)-(i, j) ↔ right(i,j)+left(i,j+1)
                if i >= 1 && labels[(i - 1, j)] == 0 && valid(i - 1, j) {
                    let fwd1 = g.right_arc(i, j).unwrap();
                    let fwd2 = g.left_arc(i, j + 1).unwrap();
                    if !edge_is_cut(net, fwd1, fwd2, params.cost_threshold) {
                        labels[(i - 1, j)] = label;
                        q.push_back((i - 1, j));
                    }
                }
            }
            sizes.push(size);
        }
    }

    finalize_labels(&mut labels, &sizes, next_label, min_size, params.max_ncomps);
    labels
}

/// Drop components smaller than `min_size`, keep at most `max_ncomps` (largest
/// by size), and renumber the survivors `1..=k` in descending-size order.
/// `sizes[l]` is the pixel count of label `l` (index 0 unused). Shared by both
/// component-growing entry points.
fn finalize_labels(
    labels: &mut Array2<u32>,
    sizes: &[usize],
    next_label: u32,
    min_size: usize,
    max_ncomps: u32,
) {
    let mut indices: Vec<u32> = (1..=next_label)
        .filter(|&l| sizes[l as usize] >= min_size)
        .collect();
    indices.sort_by(|&a, &b| sizes[b as usize].cmp(&sizes[a as usize]));
    if max_ncomps > 0 {
        indices.truncate(max_ncomps as usize);
    }
    let mut renumber = vec![0_u32; (next_label + 1) as usize];
    for (new_idx, &old) in indices.iter().enumerate() {
        renumber[old as usize] = (new_idx + 1) as u32;
    }
    labels.mapv_inplace(|l| renumber[l as usize]);
}

// =========================================================================
// SNAPHU-faithful connected components ("ambiguity wiggle")
// =========================================================================
//
// The legacy `grow_components` path above cuts an edge by its raw *linear* arc cost.
// That is the right collapse of SNAPHU's reliability test only because the
// default Carballo cost is linear (no curvature anywhere). SNAPHU itself works
// with a *convex* cost `c_e(k) = w_e · (k − k*_e)²` and, in `GrowConnCompsMask`
// (snaphu_tile.c), measures the reliability of each arc by perturbing the
// solved integer flow `k` by ±1 and watching how the cost responds:
//
//   poscost = c_e(k+1) − c_e(k)        negcost = c_e(k−1) − c_e(k)
//   reliability_e = min(poscost, negcost)
//
// A *stiff* edge (the solution sits deep in its parabola well) has a large
// reliability — bumping the ambiguity either way is expensive, so the routing
// decision there is confident. A *flat* edge (the solution sits near a
// half-cycle tie, or the achieved flow is on the wrong side of the minimum
// entirely) has a small or negative reliability — a ±1 cycle slip is nearly
// free, so the edge is unreliable and becomes a component boundary.
//
// This reproduces that test directly, "from correlation and output": it needs
// only the interferogram (wrapped phase + smoothed-gradient offsets), the
// coherence (inverse-variance weights), and the unwrapped phase (to recover
// the achieved per-edge ambiguity `k`). It does NOT need the solved MCF network
// or to know *how* the phase was unwrapped, so — like [`crate::components_only`]
// — it composes with the tiled / robust / linear phase paths.

/// Parameters for [`grow_components_snaphu`].
///
/// `reliability_threshold` is compared against `min(poscost, negcost)` in the
/// convex cost's native units (`weight · nshortcycle²`, where `weight =
/// COST_SCALE / σ²` and `nshortcycle = 100`). It scales with edge coherence,
/// so the *physical* default of `0` needs no per-scene calibration: an edge is
/// cut iff a ±1 ambiguity flip is no more expensive than the achieved flow —
/// i.e. the unwrap is at (or past) a cost tie across that edge. Raise it to cut
/// more aggressively (treat shallow wells as unreliable too); the cost-min
/// reliability of a clean edge at coherence γ is `≈ (COST_SCALE/σ²(γ)) · 10⁴`,
/// so a threshold near that value starts cutting low-coherence interiors.
#[derive(Debug, Clone)]
pub struct SnaphuConnCompParams {
    /// Cut a pixel edge when `min(poscost, negcost) <= this`. Higher → more
    /// boundaries (smaller components). See struct docs for units.
    pub reliability_threshold: i64,
    /// Drop components smaller than this many pixels (see [`ConnCompParams`]).
    pub min_size_px: usize,
    /// Vestigial fractional floor; only ever RAISES `min_size_px` (see
    /// [`ConnCompParams`]).
    pub min_size_frac: f32,
    /// Keep at most this many components (largest by size). 0 → keep all.
    pub max_ncomps: u32,
    /// Sliding window for the smoothed-gradient offsets that drive the
    /// ambiguity-wiggle reliability test (SNAPHU `KPARDPSI`/`KPERPDPSI`). Must
    /// match the window the phase was solved with for the reliability costs to
    /// be consistent with the achieved flow.
    pub phase_grad_window: cost::PhaseGradWindow,
    /// SNAPHU `ThickenCosts` behavior: convert each edge to a cut *strength*
    /// `max(0, threshold - reliability)`, smooth the strengths laterally
    /// (down-edges along the row direction, right-edges along the column
    /// direction, kernel `(2·self + neighbors) / n`), and cut wherever the
    /// smoothed strength is positive. A one-pixel reliable bridge through a
    /// wide unreliable band then no longer connects the two sides. Off by
    /// default; note the tie handling differs from the plain test (an edge
    /// with `reliability == threshold` passes here, matching SNAPHU's strict
    /// `< costthresh` cut).
    pub thicken_cuts: bool,
}

impl Default for SnaphuConnCompParams {
    fn default() -> Self {
        Self {
            reliability_threshold: 0,
            min_size_px: 100,
            min_size_frac: 0.0001,
            max_ncomps: 1024,
            phase_grad_window: cost::PhaseGradWindow::default(),
            thicken_cuts: false,
        }
    }
}

/// Smallest integer `n` with `wrap(x) = x − n·2π ∈ (−π, π]`, i.e. the integer
/// part of `x` in cycles. Companion of `integrate::wrap_n_cycle`.
#[inline]
fn cycles_of(x: f32) -> i32 {
    (x / TAU).round() as i32
}

/// Grow connected components by the SNAPHU convex-cost *ambiguity wiggle*
/// (see the module section above). Inputs are the interferogram, coherence,
/// `nlooks`, and the already-`unwrapped` phase — all `(m, n)` pixel-grid sized;
/// `mask` (if given) is `(m, n)` with `true` = valid. Returns an `(m, n)` `u32`
/// label array (0 = unassigned), renumbered by size, exactly like
/// [`grow_components`].
///
/// For each pixel edge the achieved integer ambiguity `k` is recovered purely
/// from the output:
///
/// ```text
/// 2π·k = (unw_b − unw_a) − wrap(ψ_b − ψ_a)
/// ```
///
/// (the unwrapped gradient minus the wrapped gradient, in cycles), and the
/// convex cost `c(k) = w·(k·100 − O)²` — with offset `O` and weight `w` from
/// [`cost::compute_snaphu_smooth_costs`] — is wiggled by ±1 around it.
pub fn grow_components_snaphu(
    igram: ArrayView2<Complex32>,
    corr: ArrayView2<f32>,
    nlooks: f32,
    unwrapped: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    params: &SnaphuConnCompParams,
) -> Array2<u32> {
    let (m, n) = igram.dim();
    assert_eq!(corr.dim(), (m, n), "corr must match igram shape");
    assert_eq!(unwrapped.dim(), (m, n), "unwrapped must match igram shape");
    assert!(m >= 2 && n >= 2);
    if let Some(mm) = mask {
        assert_eq!(mm.dim(), (m, n), "mask must be (m, n)");
    }

    // SNAPHU convex per-arc offsets (smoothed-gradient deviation, in
    // nshortcycle units) and weights (inverse Lee variance · COST_SCALE).
    let (offsets, weights) =
        cost::compute_snaphu_smooth_costs(igram, corr, nlooks, mask, params.phase_grad_window);
    let g = RectangularGridGraph::new(m + 1, n + 1);
    const NS: i64 = 100; // nshortcycle (matches Network::marginal_cost / cost::NSHORTCYCLE)

    // A pixel is valid when the mask allows it AND the output is finite (masked
    // pixels come back NaN from the integrator).
    let valid = |i: usize, j: usize| {
        mask.map(|mm| mm[(i, j)]).unwrap_or(true) && unwrapped[(i, j)].is_finite()
    };

    // Per-edge reliability `min(poscost, negcost)` for the convex cost on the
    // arc backing this pixel edge, given the achieved ambiguity `k`. `arc` is
    // the residue arc whose offset sign matches the forward (a→b) flow
    // convention used by the integrator, so `(k, O)` are paired consistently
    // and a wrong-direction slip reads as negative reliability.
    let reliability = |arc: usize, k: i64| -> i64 {
        let o = offsets[arc] as i64;
        let w = weights[arc] as i64;
        let u = k * NS - o;
        let poscost = w * (NS * NS + 2 * NS * u);
        let negcost = w * (NS * NS - 2 * NS * u);
        poscost.min(negcost)
    };

    // Recover the integer ambiguity across the edge a→b directly from the
    // output: `k = round(((unw_b − unw_a) − wrap(ψ_b − ψ_a)) / 2π)`.
    let ambiguity = |ai: usize, aj: usize, bi: usize, bj: usize| -> i64 {
        let psi_a = igram[(ai, aj)].arg();
        let psi_b = igram[(bi, bj)].arg();
        let wrapped_grad = psi_b - psi_a;
        let wrapped_grad = wrapped_grad - TAU * (wrapped_grad / TAU).round();
        let unw_grad = unwrapped[(bi, bj)] - unwrapped[(ai, aj)];
        cycles_of(unw_grad - wrapped_grad) as i64
    };

    let thresh = params.reliability_threshold;
    // cut_right[(i, j)]: horizontal edge (i, j)-(i, j+1). cut_down[(i, j)]:
    // vertical edge (i, j)-(i+1, j). Edges touching an invalid pixel are cut.
    let mut cut_right = Array2::<bool>::from_elem((m, n - 1), true);
    let mut cut_down = Array2::<bool>::from_elem((m - 1, n), true);
    if params.thicken_cuts {
        // SNAPHU `GrowConnCompsMask` + `ThickenCosts` semantics: per-edge cut
        // strength `max(0, thresh - reliability)`, laterally smoothed so a
        // thin reliable bridge through a wide unreliable band is still cut.
        // Invalid edges carry strength `thresh` (SNAPHU's masked arcs have
        // ~zero reliability, so they smear a soft barrier onto neighbors) and
        // are additionally cut unconditionally below.
        let invalid_strength = thresh.max(0);
        let mut s_right = Array2::<i64>::from_elem((m, n - 1), invalid_strength);
        let mut s_down = Array2::<i64>::from_elem((m - 1, n), invalid_strength);
        for i in 0..m {
            for j in 0..n - 1 {
                if valid(i, j) && valid(i, j + 1) {
                    let arc = g.down_arc(i, j + 1).unwrap();
                    let k = ambiguity(i, j, i, j + 1);
                    s_right[(i, j)] = (thresh - reliability(arc, k)).max(0);
                }
            }
        }
        for i in 0..m - 1 {
            for j in 0..n {
                if valid(i, j) && valid(i + 1, j) {
                    let arc = g.left_arc(i + 1, j + 1).unwrap();
                    let k = ambiguity(i, j, i + 1, j);
                    s_down[(i, j)] = (thresh - reliability(arc, k)).max(0);
                }
            }
        }
        // SNAPHU convolves each arc's strength with its lateral neighbors:
        // horizontal (right) edges across adjacent rows, vertical (down)
        // edges across adjacent columns, kernel (2·self + neighbors) / n.
        for i in 0..m {
            for j in 0..n - 1 {
                let mut acc = 2 * s_right[(i, j)];
                let mut cnt = 2.0_f64;
                if i >= 1 {
                    acc += s_right[(i - 1, j)];
                    cnt += 1.0;
                }
                if i + 1 < m {
                    acc += s_right[(i + 1, j)];
                    cnt += 1.0;
                }
                let smoothed = (acc as f64 / cnt).round() as i64;
                cut_right[(i, j)] = !(valid(i, j) && valid(i, j + 1)) || smoothed > 0;
            }
        }
        for i in 0..m - 1 {
            for j in 0..n {
                let mut acc = 2 * s_down[(i, j)];
                let mut cnt = 2.0_f64;
                if j >= 1 {
                    acc += s_down[(i, j - 1)];
                    cnt += 1.0;
                }
                if j + 1 < n {
                    acc += s_down[(i, j + 1)];
                    cnt += 1.0;
                }
                let smoothed = (acc as f64 / cnt).round() as i64;
                cut_down[(i, j)] = !(valid(i, j) && valid(i + 1, j)) || smoothed > 0;
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n - 1 {
                if valid(i, j) && valid(i, j + 1) {
                    // Horizontal edge ↔ down_arc(i, j+1) (offset uses +α, matching
                    // the integrator's forward j-increasing net flow).
                    let arc = g.down_arc(i, j + 1).unwrap();
                    let k = ambiguity(i, j, i, j + 1);
                    cut_right[(i, j)] = reliability(arc, k) <= thresh;
                }
            }
        }
        for i in 0..m - 1 {
            for j in 0..n {
                if valid(i, j) && valid(i + 1, j) {
                    // Vertical edge ↔ left_arc(i+1, j+1) (offset uses +α, matching
                    // the integrator's forward i-increasing net flow).
                    let arc = g.left_arc(i + 1, j + 1).unwrap();
                    let k = ambiguity(i, j, i + 1, j);
                    cut_down[(i, j)] = reliability(arc, k) <= thresh;
                }
            }
        }
    }

    // BFS the pixel grid through non-cut edges.
    let n_valid: usize = (0..m)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|&(i, j)| valid(i, j))
        .count();
    let frac_floor = (params.min_size_frac as f64 * n_valid as f64).ceil() as usize;
    let min_size = params.min_size_px.max(frac_floor).max(1);

    let mut labels = Array2::<u32>::zeros((m, n));
    let mut next_label: u32 = 0;
    let mut sizes: Vec<usize> = vec![0];
    let mut q: VecDeque<(usize, usize)> = VecDeque::new();

    for si in 0..m {
        for sj in 0..n {
            if labels[(si, sj)] != 0 || !valid(si, sj) {
                continue;
            }
            next_label += 1;
            let label = next_label;
            q.clear();
            q.push_back((si, sj));
            labels[(si, sj)] = label;
            let mut size = 0_usize;
            while let Some((i, j)) = q.pop_front() {
                size += 1;
                if j + 1 < n && labels[(i, j + 1)] == 0 && !cut_right[(i, j)] {
                    labels[(i, j + 1)] = label;
                    q.push_back((i, j + 1));
                }
                if j >= 1 && labels[(i, j - 1)] == 0 && !cut_right[(i, j - 1)] {
                    labels[(i, j - 1)] = label;
                    q.push_back((i, j - 1));
                }
                if i + 1 < m && labels[(i + 1, j)] == 0 && !cut_down[(i, j)] {
                    labels[(i + 1, j)] = label;
                    q.push_back((i + 1, j));
                }
                if i >= 1 && labels[(i - 1, j)] == 0 && !cut_down[(i - 1, j)] {
                    labels[(i - 1, j)] = label;
                    q.push_back((i - 1, j));
                }
            }
            sizes.push(size);
        }
    }

    finalize_labels(&mut labels, &sizes, next_label, min_size, params.max_ncomps);
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost;
    use crate::primal_dual;
    use crate::residue;
    use ndarray::Array2;
    use num_complex::Complex32;

    /// Smooth ramp + one strong residue boundary. Components should respect
    /// the masked stripe even though the wrapped phase is otherwise smooth.
    #[test]
    fn split_by_mask_stripe() {
        let m = 32;
        let n = 32;
        let mut truth = Array2::<f32>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                truth[(i, j)] = 0.3 * (i as f32 + j as f32);
            }
        }
        let igram: Array2<Complex32> =
            truth.mapv(|p| Complex32::from_polar(1.0, p.sin().atan2(p.cos())));
        let corr = Array2::<f32>::from_elem((m, n), 0.9);
        let mut mask = Array2::<bool>::from_elem((m, n), true);
        // Stripe of invalid pixels straight down the middle.
        for i in 0..m {
            mask[(i, n / 2)] = false;
        }
        let mask_view = mask.view();
        let costs = cost::compute_carballo_costs(
            igram.view(),
            corr.view(),
            5.0,
            Some(mask_view),
            cost::PhaseGradWindow::default(),
        );
        let graph = RectangularGridGraph::new(m + 1, n + 1);
        let residues = residue::compute_with_mask(igram.mapv(|z| z.arg()).view(), Some(mask_view));
        let mut net = Network::new_with_mask(&graph, residues.view(), &costs, Some(mask_view));
        primal_dual::run(&graph, &mut net, 50);
        let labels = grow_components(&graph, &net, Some(mask_view), &ConnCompParams::default());
        // Left and right halves should be in different components.
        let left = labels[(m / 2, n / 4)];
        let right = labels[(m / 2, 3 * n / 4)];
        assert!(left > 0, "left half should be labeled");
        assert!(right > 0, "right half should be labeled");
        assert_ne!(left, right, "mask stripe should separate components");
    }

    #[test]
    fn small_components_are_dropped() {
        let m = 64;
        let n = 64;
        let truth = Array2::<f32>::zeros((m, n));
        let igram: Array2<Complex32> = truth.mapv(|p| Complex32::from_polar(1.0, p));
        let corr = Array2::<f32>::from_elem((m, n), 0.9);
        let mut mask = Array2::<bool>::from_elem((m, n), true);
        // Two isolated islands separated from the main body by a masked moat:
        // a tiny 2x2 (4 px, below the 100-px floor) and a 12x12 (144 px, above).
        // A one-pixel masked ring around each isolates it.
        for i in 0..m {
            for j in 0..n {
                let small = i < 2 && j < 2;
                let big = (4..16).contains(&i) && (4..16).contains(&j);
                let moat = (i < 3 && j < 3) || ((3..17).contains(&i) && (3..17).contains(&j));
                if moat && !small && !big {
                    mask[(i, j)] = false;
                }
            }
        }
        let mask_view = mask.view();
        let costs = cost::compute_carballo_costs(
            igram.view(),
            corr.view(),
            5.0,
            Some(mask_view),
            cost::PhaseGradWindow::default(),
        );
        let graph = RectangularGridGraph::new(m + 1, n + 1);
        let residues = residue::compute_with_mask(igram.mapv(|z| z.arg()).view(), Some(mask_view));
        let mut net = Network::new_with_mask(&graph, residues.view(), &costs, Some(mask_view));
        primal_dual::run(&graph, &mut net, 50);
        // Default policy: absolute 100-px floor governs (frac is a negligible cap).
        let labels = grow_components(&graph, &net, Some(mask_view), &ConnCompParams::default());
        // The 2x2 island (4 px < 100) is dropped...
        assert_eq!(labels[(0, 0)], 0);
        // ...the 12x12 island (144 px >= 100) SURVIVES as its own component
        // (the whole point: a small coherent island is NOT dropped for being
        // disconnected - the caller can re-reference into it)...
        assert!(labels[(9, 9)] > 0);
        // ...and the main body is labeled.
        assert!(labels[(40, 40)] > 0);

        // The old 1% fraction would have orphaned the 12x12 island: assert the
        // absolute floor is what keeps it (raising min_size_px past 144 drops it).
        let strict = ConnCompParams {
            min_size_px: 300,
            ..Default::default()
        };
        let labels_strict = grow_components(&graph, &net, Some(mask_view), &strict);
        assert_eq!(labels_strict[(9, 9)], 0);
    }

    // ---------------------------------------------------------------------
    // SNAPHU-faithful "ambiguity wiggle" component growing.
    // ---------------------------------------------------------------------

    /// Gentle ramp (|gradient| < π so no wrap ambiguity), perfectly unwrapped,
    /// high coherence, no mask: the wiggle test finds no unreliable edge, so
    /// the whole frame is one component.
    #[test]
    fn snaphu_clean_ramp_is_one_component() {
        let (m, n) = (32, 32);
        let truth = Array2::from_shape_fn((m, n), |(i, j)| 0.2 * (i as f32 + j as f32));
        let igram: Array2<Complex32> = truth.mapv(|p| Complex32::from_polar(1.0, p));
        let corr = Array2::<f32>::from_elem((m, n), 0.9);
        // Perfect unwrap: congruent to wrapped input by construction.
        let labels = grow_components_snaphu(
            igram.view(),
            corr.view(),
            5.0,
            truth.view(),
            None,
            &SnaphuConnCompParams::default(),
        );
        let first = labels[(0, 0)];
        assert!(first > 0, "clean ramp should be labeled");
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    labels[(i, j)],
                    first,
                    "clean ramp must be a single component"
                );
            }
        }
    }

    /// A wide decorrelated horizontal band with a single high-coherence column
    /// bridging it. The plain per-edge test keeps the bridge (its own edges
    /// are deep wells) so the frame stays one component; SNAPHU's
    /// `ThickenCosts` smoothing smears the band's cut strength onto the
    /// bridge laterally and severs it. Uniformly coherent data must stay a
    /// single component under the same thickened threshold.
    #[test]
    fn snaphu_thicken_cuts_severs_thin_bridge() {
        let (m, n) = (32, 32);
        let truth = Array2::from_shape_fn((m, n), |(i, j)| 0.2 * (i as f32 + j as f32));
        let igram: Array2<Complex32> = truth.mapv(|p| Complex32::from_polar(1.0, p));
        // Coherence 0.05 in rows 12..21, except a coherent bridge at column 16.
        let corr = Array2::from_shape_fn((m, n), |(i, j)| {
            if (12..21).contains(&i) && j != 16 {
                0.05
            } else {
                0.95
            }
        });
        // Threshold between the band's shallow wells and the bridge's deep
        // ones: ~0.5 in the public 1/sigma2 knob units at 50 looks.
        let params = SnaphuConnCompParams {
            reliability_threshold: 500_000,
            min_size_px: 1,
            ..Default::default()
        };
        let plain =
            grow_components_snaphu(igram.view(), corr.view(), 50.0, truth.view(), None, &params);
        assert_eq!(
            plain[(2, 16)],
            plain[(30, 16)],
            "without thickening the coherent bridge connects the halves"
        );
        let thick_params = SnaphuConnCompParams {
            thicken_cuts: true,
            ..params.clone()
        };
        let thick = grow_components_snaphu(
            igram.view(),
            corr.view(),
            50.0,
            truth.view(),
            None,
            &thick_params,
        );
        let top = thick[(2, 16)];
        let bottom = thick[(30, 16)];
        assert!(top > 0 && bottom > 0, "both sides should stay labeled");
        assert_ne!(top, bottom, "thickened cuts must sever the 1-px bridge");

        // Uniform high coherence: thickening must not fragment anything.
        let corr_hi = Array2::<f32>::from_elem((m, n), 0.95);
        let uniform = grow_components_snaphu(
            igram.view(),
            corr_hi.view(),
            50.0,
            truth.view(),
            None,
            &thick_params,
        );
        let first = uniform[(0, 0)];
        assert!(first > 0);
        assert!(
            uniform.iter().all(|&l| l == first),
            "uniformly coherent frame must stay one component with thickening"
        );
    }

    /// A clean ramp whose *output* has been corrupted by a spurious 2π jump on
    /// the right half. The wrapped phase is smooth across the seam (the model
    /// expects no slip, offset ≈ 0), but the output shows a full-cycle slip
    /// there: the wiggle test reads that as negative reliability and cuts the
    /// seam, splitting the frame into two components.
    #[test]
    fn snaphu_output_slip_creates_a_cut() {
        let (m, n) = (32, 32);
        let truth = Array2::from_shape_fn((m, n), |(i, j)| 0.2 * (i as f32 + j as f32));
        let igram: Array2<Complex32> = truth.mapv(|p| Complex32::from_polar(1.0, p));
        let corr = Array2::<f32>::from_elem((m, n), 0.9);
        // Corrupt the output: add a full cycle to the right half.
        let mut unw = truth.clone();
        for i in 0..m {
            for j in n / 2..n {
                unw[(i, j)] += TAU;
            }
        }
        let labels = grow_components_snaphu(
            igram.view(),
            corr.view(),
            5.0,
            unw.view(),
            None,
            &SnaphuConnCompParams::default(),
        );
        let left = labels[(m / 2, n / 4)];
        let right = labels[(m / 2, 3 * n / 4)];
        assert!(left > 0 && right > 0, "both halves should be labeled");
        assert_ne!(
            left, right,
            "spurious 2π output slip must separate components"
        );
    }

    /// Mask stripe straight down the middle isolates the two halves: edges
    /// touching a masked (NaN-output) pixel are always cut.
    #[test]
    fn snaphu_split_by_mask_stripe() {
        let (m, n) = (32, 32);
        let truth = Array2::from_shape_fn((m, n), |(i, j)| 0.2 * (i as f32 + j as f32));
        let igram: Array2<Complex32> = truth.mapv(|p| Complex32::from_polar(1.0, p));
        let corr = Array2::<f32>::from_elem((m, n), 0.9);
        let mut mask = Array2::<bool>::from_elem((m, n), true);
        let mut unw = truth.clone();
        for i in 0..m {
            mask[(i, n / 2)] = false;
            unw[(i, n / 2)] = f32::NAN; // masked pixels come back NaN
        }
        let labels = grow_components_snaphu(
            igram.view(),
            corr.view(),
            5.0,
            unw.view(),
            Some(mask.view()),
            &SnaphuConnCompParams::default(),
        );
        let left = labels[(m / 2, n / 4)];
        let right = labels[(m / 2, 3 * n / 4)];
        assert!(left > 0 && right > 0, "both halves should be labeled");
        assert_ne!(left, right, "mask stripe should separate components");
        assert_eq!(labels[(0, n / 2)], 0, "masked pixels stay unassigned");
    }
}
