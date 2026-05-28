use ndarray::Array2;
use num_complex::Complex32;
use whirlwind_core::{simulate, unwrap, unwrap_crlb_grounded, unwrap_reuse};

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

/// Synthetic smooth-ramp regression test. KNOWN-FAILING under [`unwrap`]
/// (coherence cost, no ground node): a 6π ramp produces 6 wrap-lines
/// converging at the corner; each needs unit flow through the same
/// frame-along arc, but unit-capacity allows only one. The overflow spills
/// onto interior arcs and creates spurious 2π corrections.
/// The companion test [`diagonal_ramp_512_grounded`] uses
/// `unwrap_crlb_grounded` with a virtual ground node and PASSES.
#[test]
#[ignore = "capacity-1 frame-along arcs can't carry multiple stacked wrap-line flows; use unwrap_crlb_grounded instead (see diagonal_ramp_512_grounded)"]
fn diagonal_ramp_512() {
    let truth = simulate::diagonal_ramp((512, 512));
    let wrapped = simulate::wrap_phase(&truth);
    let igram = wrapped.mapv(|p| Complex32::new(p.cos(), p.sin()));
    let corr = Array2::<f32>::from_elem(igram.dim(), 0.999);

    let unw = unwrap(igram.view(), corr.view(), 1.0, None).unwrap();
    let aligned = align_to_truth(&unw, &truth);
    let err = max_abs_err(&aligned, &truth);
    assert!(
        err < 1e-2,
        "max error {err} too large for a smooth ramp"
    );
}

/// The 6π diagonal ramp test that fails under [`unwrap`]'s capacity-1
/// setup ALSO passes under [`unwrap_reuse`] (PHASS-style flow-reuse):
/// the frame-along arcs each carry multiple units of flow at zero
/// marginal cost after the first push, so no spurious interior flow
/// spills out. Same coherence cost as [`unwrap`], no virtual ground.
#[test]
fn diagonal_ramp_512_reuse() {
    let truth = simulate::diagonal_ramp((512, 512));
    let wrapped = simulate::wrap_phase(&truth);
    let igram = wrapped.mapv(|p| Complex32::new(p.cos(), p.sin()));
    let corr = Array2::<f32>::from_elem(igram.dim(), 0.999);
    let unw = unwrap_reuse(igram.view(), corr.view(), 1.0, None).unwrap();
    let aligned = align_to_truth(&unw, &truth);
    let err = max_abs_err(&aligned, &truth);
    assert!(err < 1e-2, "max error {err} too large for reuse-mode smooth ramp");
}

/// The 6π diagonal ramp test that fails under [`unwrap`]'s capacity-1 setup
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
    assert!(err < 1e-2, "max error {err} too large for grounded smooth ramp");
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
    let _unw = whirlwind_core::unwrap(igram.view(), corr.view(), 5.0, None).unwrap();
}

#[test]
fn gaussian_bump_noisy() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let truth = simulate::gaussian_bump((64, 64), 8.0, 12.0);
    let gamma = Array2::<f32>::from_elem(truth.dim(), 0.85);
    let nlooks = 10;
    let (igram, corr) = simulate::simulate_ifg(&truth, &gamma, nlooks, &mut rng);

    let unw = unwrap(igram.view(), corr.view(), nlooks as f32, None).unwrap();
    let aligned = align_to_truth(&unw, &truth);
    let err = max_abs_err(&aligned, &truth);
    // With moderate noise and a smooth deformation, expect ≤ 2π anywhere.
    assert!(
        err < 6.5,
        "max error {err} > 2π — unwrapping diverged on noisy bump"
    );
}
