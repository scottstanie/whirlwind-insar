use ndarray::Array2;
use num_complex::Complex32;
use whirlwind_core::{simulate, unwrap_convex, unwrap_crlb_grounded, unwrap_reuse};

/// Constant 2π offset between unwrapped and truth is unobservable; subtract
/// the modal offset before comparing.
fn align_to_truth(unw: &Array2<f32>, truth: &Array2<f32>) -> Array2<f32> {
    let mean_diff: f64 = unw
        .iter()
        .zip(truth.iter())
        .map(|(&a, &b)| (a - b) as f64)
        .sum::<f64>()
        / (unw.len() as f64);
    let offset = (mean_diff / (2.0 * std::f64::consts::PI)).round() * 2.0 * std::f64::consts::PI;
    unw.mapv(|v| v - offset as f32)
}

fn max_abs_err(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// The same 6π smooth-ramp regression test, but with the SNAPHU-style
/// convex cost. Quadratic per-arc cost with nonzero preferred offset
/// should give the *correct* topology choice directly - no need for
/// the reuse hack, no need for a virtual ground. If this passes it's
/// the most principled fix to the original boundary-stacking failure.
#[test]
fn diagonal_ramp_512_convex() {
    let truth = simulate::diagonal_ramp((512, 512));
    let wrapped = simulate::wrap_phase(&truth);
    let igram = wrapped.mapv(|p| Complex32::new(p.cos(), p.sin()));
    let corr = Array2::<f32>::from_elem(igram.dim(), 0.999);
    let unw = unwrap_convex(igram.view(), corr.view(), 1.0, None).unwrap();
    let aligned = align_to_truth(&unw, &truth);
    let err = max_abs_err(&aligned, &truth);
    assert!(
        err < 1e-2,
        "max error {err} too large for convex-mode smooth ramp"
    );
}

/// The 6π diagonal ramp that the old unit-capacity solver (since removed)
/// could not unwrap PASSES under [`unwrap_reuse`] (PHASS-style flow-reuse):
/// the frame-along arcs each carry multiple units of flow at zero marginal
/// cost after the first push, so no spurious interior flow spills out. Plain
/// coherence cost, no virtual ground. This is the live corner-safe guard.
#[test]
fn diagonal_ramp_512_reuse() {
    let truth = simulate::diagonal_ramp((512, 512));
    let wrapped = simulate::wrap_phase(&truth);
    let igram = wrapped.mapv(|p| Complex32::new(p.cos(), p.sin()));
    let corr = Array2::<f32>::from_elem(igram.dim(), 0.999);
    let unw = unwrap_reuse(igram.view(), corr.view(), 1.0, None).unwrap();
    let aligned = align_to_truth(&unw, &truth);
    let err = max_abs_err(&aligned, &truth);
    assert!(
        err < 1e-2,
        "max error {err} too large for reuse-mode smooth ramp"
    );
}

/// The 6π diagonal ramp that the old unit-capacity solver could not unwrap
/// passes once a virtual ground node is enabled: each boundary residue
/// drains to ground independently, no interior arc gets spurious flow,
/// Itoh integration alone recovers the smooth ramp.
#[test]
fn diagonal_ramp_512_grounded() {
    let truth = simulate::diagonal_ramp((512, 512));
    let wrapped = simulate::wrap_phase(&truth);
    let igram = wrapped.mapv(|p| Complex32::new(p.cos(), p.sin()));
    // Synthetic clean ramp ⇒ low CRLB everywhere.
    let var = Array2::<f32>::from_elem(igram.dim(), 0.01);
    // ground_cost = 0: ground is free for the smooth-ramp case (no interior
    // residues to drag toward boundary).
    let unw = unwrap_crlb_grounded(igram.view(), var.view(), None, 0).unwrap();
    let aligned = align_to_truth(&unw, &truth);
    let err = max_abs_err(&aligned, &truth);
    assert!(
        err < 1e-2,
        "max error {err} too large for grounded smooth ramp"
    );
}

/// Single planted residue pair: a phase that loops once around an interior
/// point creates a +1/-1 pair. The unwrapper must route flow between them.
#[test]
fn vortex_pair() {
    let m = 32;
    let n = 32;
    let mut truth = ndarray::Array2::<f32>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let dy = i as f32 - (m as f32 / 2.0);
            let dx = j as f32 - (n as f32 / 2.0);
            truth[(i, j)] = dy.atan2(dx) * 3.0; // 3 turns
        }
    }
    let wrapped = whirlwind_core::simulate::wrap_phase(&truth);
    let igram = wrapped.mapv(|p| num_complex::Complex32::new(p.cos(), p.sin()));
    let corr = ndarray::Array2::<f32>::from_elem((m, n), 0.95);
    // This call must terminate (in finite time, no infinite augmentation loop).
    let _unw = whirlwind_core::unwrap_reuse(igram.view(), corr.view(), 5.0, None).unwrap();
}

#[test]
fn gaussian_bump_noisy() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let truth = simulate::gaussian_bump((64, 64), 8.0, 12.0);
    let gamma = Array2::<f32>::from_elem(truth.dim(), 0.85);
    let nlooks = 10;
    let (igram, corr) = simulate::simulate_ifg(&truth, &gamma, nlooks, &mut rng);

    let unw = unwrap_reuse(igram.view(), corr.view(), nlooks as f32, None).unwrap();
    let aligned = align_to_truth(&unw, &truth);
    let err = max_abs_err(&aligned, &truth);
    // With moderate noise and a smooth deformation, expect ≤ 2π anywhere.
    assert!(
        err < 6.5,
        "max error {err} > 2π - unwrapping diverged on noisy bump"
    );
}

/// Single-tile `unwrap_linear` uses `run_full_dijkstra`, which falls through to
/// the SINGLE-source SSP (`ssp::run_single_source`). This guards the correctness
/// bar for that path: on a steep, noisy ramp that leaves residue after the 8 PD
/// iterations (so the SSP fallback is actually exercised), the
/// `debug_assert!(rc >= 0)` inside single-source SSP must never fire - i.e. the
/// per-source capped potential update keeps reduced costs non-negative after
/// every early-exit Dijkstra, not just at SSP entry. `cargo test` is a debug
/// build, so the assertion is live; a regression panics here.
#[test]
fn single_source_ssp_keeps_nonnegative_reduced_costs() {
    use rand::SeedableRng;
    let (m, n) = (160usize, 160usize);
    let cycles = 4.0_f32;
    let truth = Array2::from_shape_fn((m, n), |(i, j)| {
        2.0 * std::f32::consts::PI * cycles * (i as f32 + j as f32) / (m as f32)
    });
    let gamma = Array2::<f32>::from_elem((m, n), 0.3);
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    let (igram, corr) = simulate::simulate_ifg(&truth, &gamma, 4, &mut rng);

    // run_full_dijkstra(8) → ssp::run_single_source; debug_assert is active.
    let unw = whirlwind_core::unwrap_linear(igram.view(), corr.view(), 4.0, None).unwrap();

    // Confirm the single-source SSP fallback was actually reached (else the
    // assertion above guards nothing on this input).
    let t = whirlwind_core::primal_dual::last_timings();
    assert!(
        t.ssp_iters > 0,
        "test did not exercise the SSP fallback (ssp_iters=0); make the ramp steeper/noisier"
    );
    assert!(
        unw.iter().all(|v| v.is_finite()),
        "unwrap_linear produced non-finite output"
    );
}

/// Regression for the masked-plane TEAR (the capacity-1 gutter-stacking
/// limitation; see `scripts/diag_tear_capacity_hypothesis.py`). A clean,
/// well-sampled diagonal plane masked to a full-width horizontal band
/// disconnects the top/bottom zero-fill seas, so each fringe's +/- boundary
/// charge pair needs one unit of flow ACROSS the band. The only zero-cost,
/// integration-invisible crossings are the two image-edge gutter columns -
/// capacity 1 each before the multi-unit gutter ring, so a 3-fringe ramp
/// forced one cut through the band interior (a 2pi tear over ~42% of the
/// band). With the multi-unit gutter ring all crossings ride the gutter for
/// free and the unwrap is EXACT.
#[test]
fn masked_band_plane_no_tear() {
    let n = 256usize;
    let cycles = 3.0_f32;
    let truth = Array2::from_shape_fn((n, n), |(i, j)| {
        let x = j as f32 / (n - 1) as f32 - 0.5;
        let y = i as f32 / (n - 1) as f32 - 0.5;
        2.0 * std::f32::consts::PI * cycles * (x + y)
    });
    let mask = Array2::from_shape_fn((n, n), |(i, _)| (64..n - 64).contains(&i));
    // Production nodata convention: zero-fill the masked sea.
    let igram = Array2::from_shape_fn((n, n), |(i, j)| {
        if mask[(i, j)] {
            Complex32::from_polar(1.0, truth[(i, j)])
        } else {
            Complex32::new(0.0, 0.0)
        }
    });
    let corr = Array2::from_shape_fn((n, n), |(i, j)| if mask[(i, j)] { 0.999 } else { 0.0 });

    let unw =
        whirlwind_core::unwrap_linear(igram.view(), corr.view(), 1.0, Some(mask.view())).unwrap();

    // Align on the band's mean offset, then require ZERO cycle error on every
    // valid pixel - the pre-fix solver left ~42% of the band one cycle low.
    let (vals, truths): (Vec<f32>, Vec<f32>) = unw
        .iter()
        .zip(truth.iter())
        .zip(mask.iter())
        .filter(|&(_, &keep)| keep)
        .map(|((&v, &t), _)| (v, t))
        .unzip();
    let mean_diff: f64 = vals
        .iter()
        .zip(truths.iter())
        .map(|(&a, &b)| (a - b) as f64)
        .sum::<f64>()
        / vals.len() as f64;
    let tau = 2.0 * std::f64::consts::PI;
    let offset = ((mean_diff / tau).round() * tau) as f32;
    let n_torn = vals
        .iter()
        .zip(truths.iter())
        .filter(|&(&v, &t)| ((v - offset - t) as f64 / tau).round() != 0.0)
        .count();
    assert_eq!(
        n_torn, 0,
        "masked band plane torn: {n_torn}/{} valid px off by >= 1 cycle",
        vals.len()
    );
}

/// Regression for the zero-cost masked-sea blowup. `unwrap_linear` does NOT
/// forbid masked arcs (cost 0), so a heavily-masked frame is a vast zero-cost
/// "sea"; the single-source SSP must cross it to pair residues. With a binary
/// heap this ballooned to millions of equal-distance entries (RSS climbed
/// without bound → hang/OOM-looking); Dial's bucket queue traverses the sea in
/// O(nodes). This test must simply COMPLETE with finite output on the valid
/// pixels - a regression to the heap would hang here.
#[test]
fn single_source_ssp_bounded_on_zero_cost_masked_sea() {
    use rand::SeedableRng;
    let (m, n) = (128usize, 128usize);
    let cycles = 5.0_f32;
    let truth = Array2::from_shape_fn((m, n), |(i, j)| {
        2.0 * std::f32::consts::PI * cycles * (i as f32 + j as f32) / (m as f32)
    });
    let gamma = Array2::<f32>::from_elem((m, n), 0.3);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let (igram, corr) = simulate::simulate_ifg(&truth, &gamma, 4, &mut rng);
    // Sparse, fragmented validity (~12 %): a wide masked sea between valid lines,
    // so leftover residues must route across cost-0 masked region in the SSP.
    let mask = Array2::from_shape_fn((m, n), |(i, j)| (i % 16 < 2) || (j % 16 < 2));

    let unw =
        whirlwind_core::unwrap_linear(igram.view(), corr.view(), 4.0, Some(mask.view())).unwrap();
    let t = whirlwind_core::primal_dual::last_timings();
    assert!(
        t.ssp_iters > 0,
        "test did not exercise the single-source SSP over the masked sea (ssp_iters=0)"
    );
    // Completion + finiteness on valid pixels is the guard (the old binary-heap
    // SSP would balloon/hang on the zero-cost sea instead of returning). Masked
    // pixels are intentionally NaN.
    assert!(
        unw.iter()
            .zip(mask.iter())
            .all(|(&v, &keep)| !keep || v.is_finite()),
        "unwrap_linear produced non-finite output on valid pixels"
    );
}
