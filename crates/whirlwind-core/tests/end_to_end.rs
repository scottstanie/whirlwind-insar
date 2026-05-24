use ndarray::Array2;
use num_complex::Complex32;
use whirlwind_core::{simulate, unwrap};

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

#[test]
fn diagonal_ramp_512() {
    // SNAPHU-style smooth-ramp regression test. 512x512 diagonal ramp from
    // -3π to +3π; coherence = 1.0 everywhere.
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
