//! Scale benchmark: per-stage timings + peak RSS at multiple image sizes.
//!
//! Run with:
//!   cargo run --release --example bench_scale
//!
//! Outputs a markdown table to stdout suitable for pasting into the README.

use ndarray::Array2;
use num_complex::Complex32;
use rand::SeedableRng;
use std::time::Instant;
use whirlwind_core::{cost, grid, integrate, network, primal_dual, residue, simulate};

/// Peak resident set size in bytes, via getrusage. On macOS ru_maxrss is in
/// bytes; on Linux it's in kilobytes — we normalize.
fn peak_rss_bytes() -> u64 {
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe {
        libc::getrusage(libc::RUSAGE_SELF, &mut u as *mut libc::rusage);
    }
    let raw = u.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        raw // bytes
    } else {
        raw * 1024 // KiB -> bytes on Linux
    }
}

fn fmt_bytes(b: u64) -> String {
    const K: f64 = 1024.0;
    let f = b as f64;
    if f >= K * K * K {
        format!("{:.2} GiB", f / (K * K * K))
    } else if f >= K * K {
        format!("{:.1} MiB", f / (K * K))
    } else if f >= K {
        format!("{:.1} KiB", f / K)
    } else {
        format!("{} B", b)
    }
}

fn make_clean(m: usize, n: usize) -> (Array2<Complex32>, Array2<f32>) {
    let truth = simulate::diagonal_ramp((m, n));
    let wrapped = simulate::wrap_phase(&truth);
    let igram = wrapped.mapv(|p| Complex32::new(p.cos(), p.sin()));
    let cor = Array2::<f32>::from_elem((m, n), 0.99);
    (igram, cor)
}

fn make_noisy(m: usize, n: usize, gamma: f32, nlooks: usize) -> (Array2<Complex32>, Array2<f32>) {
    // Diagonal ramp (lots of fringes → wrap-line residues even at high γ).
    let truth = simulate::diagonal_ramp((m, n));
    let g = Array2::<f32>::from_elem((m, n), gamma);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
    simulate::simulate_ifg(&truth, &g, nlooks, &mut rng)
}

fn make_very_noisy(m: usize, n: usize) -> (Array2<Complex32>, Array2<f32>) {
    // Low coherence, real residue blizzard. Mirrors typical Sentinel-1 over land.
    let truth = simulate::diagonal_ramp((m, n));
    let g = Array2::<f32>::from_elem((m, n), 0.3);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
    simulate::simulate_ifg(&truth, &g, 4, &mut rng)
}

struct StageTimes {
    residue_ms: f64,
    cost_ms: f64,
    network_ms: f64,
    primal_dual_ms: f64,
    integrate_ms: f64,
    total_ms: f64,
    pd_dijkstra_ms: f64,
    pd_augment_ms: f64,
    pd_potential_ms: f64,
    pd_iters: u32,
    pd_ssp_iters: u32,
    pd_residues: u32,
}

fn run_one(igram: &Array2<Complex32>, cor: &Array2<f32>, nlooks: f32) -> StageTimes {
    let (m, n) = igram.dim();
    let t_total = Instant::now();

    let t = Instant::now();
    let wrapped_phase = igram.mapv(|z| z.arg());
    let residues = residue::compute(wrapped_phase.view());
    let residue_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let costs = cost::compute_carballo_costs(igram.view(), cor.view(), nlooks, None);
    let cost_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let g = grid::RectangularGridGraph::new(m + 1, n + 1);
    let mut net = network::Network::new(&g, residues.view(), &costs);
    let network_ms = t.elapsed().as_secs_f64() * 1000.0;

    let n_residues = residues.iter().filter(|&&r| r != 0).count() as u32;

    let max_iter: usize = std::env::var("WW_MAX_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let t = Instant::now();
    primal_dual::run(&g, &mut net, max_iter);
    let primal_dual_ms = t.elapsed().as_secs_f64() * 1000.0;
    let pdt = primal_dual::last_timings();

    let t = Instant::now();
    let _unw = integrate::integrate(wrapped_phase.view(), &g, &net);
    let integrate_ms = t.elapsed().as_secs_f64() * 1000.0;

    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    StageTimes {
        residue_ms,
        cost_ms,
        network_ms,
        primal_dual_ms,
        integrate_ms,
        total_ms,
        pd_dijkstra_ms: pdt.dijkstra_ms,
        pd_augment_ms: pdt.augment_ms,
        pd_potential_ms: pdt.potential_ms,
        pd_iters: pdt.iters,
        pd_ssp_iters: pdt.ssp_iters,
        pd_residues: n_residues,
    }
}

fn analytic_mem_bytes(m: usize, n: usize) -> u64 {
    // Working-set estimate (per-pixel bytes × m·n + small per-arc overhead).
    // Values reflect the actual data structures whirlwind-core allocates.
    let mn = (m as u64) * (n as u64);
    let res_grid = ((m as u64) + 1) * ((n as u64) + 1);
    // num_forward arcs in the (m+1, n+1) residual graph:
    //   2 * ((m+1-1)*(n+1) + (m+1)*(n+1-1))
    //   = 2 * (m*(n+1) + (m+1)*n)
    let m1 = m as u64;
    let n1 = n as u64;
    let num_forward = 2 * (m1 * (n1 + 1) + (m1 + 1) * n1);
    let num_arcs = 2 * num_forward;

    // Inputs (caller-provided, but we hold views): 2x f32 + 1x c64 ≈ ignore.
    // Per-pixel intermediates we own:
    let wrapped_phase = mn * 4; // f32
    let phase_dy_s = (m1 - 1) * n1 * 4;
    let phase_dx_s = m1 * (n1 - 1) * 4;
    let cor_dy = phase_dy_s;
    let cor_dx = phase_dx_s;
    let costs_temp = num_arcs * 4; // costs Vec<i32> built then dropped after Network::new
    // Network (persistent through primal-dual):
    let net_excess = res_grid * 4; // i32
    let net_potential = res_grid * 8; // i64
    let net_cost_fwd = num_forward * 4; // i32
    let net_saturated = num_arcs / 8 + 8; // bitvec, +slack
    // Dijkstra state, allocated each iteration:
    let sp_dist = res_grid * 8; // i64
    let sp_pred_arc = res_grid * 4; // i32
    let sp_pred_node = res_grid * 4;
    let sp_source = res_grid * 4;
    let sp_visited = res_grid; // Vec<bool>
    let heap_capacity = res_grid * 16; // pessimistic upper bound

    wrapped_phase
        + phase_dy_s
        + phase_dx_s
        + cor_dy
        + cor_dx
        + costs_temp
        + net_excess
        + net_potential
        + net_cost_fwd
        + net_saturated
        + sp_dist
        + sp_pred_arc
        + sp_pred_node
        + sp_source
        + sp_visited
        + heap_capacity
}

fn run_scene(
    label: &str,
    m: usize,
    n: usize,
    igram: Array2<Complex32>,
    cor: Array2<f32>,
    nlooks: f32,
) -> (StageTimes, u64, u64) {
    let rss_before = peak_rss_bytes();
    let st = run_one(&igram, &cor, nlooks);
    let rss_after = peak_rss_bytes();
    let analytic = analytic_mem_bytes(m, n);
    let mpx = (m * n) as f64 / 1e6;
    let throughput = mpx / (st.total_ms / 1000.0);
    let size = format!("{m}x{n}");
    println!(
        "| {:24} | {:9} | {:6.2} | {:7.1} | {:6.1} | {:6.1} | {:7.1} | {:6.1} | {:6.1} | {:6.2} | {:>12} | {:>9} |",
        label,
        size,
        mpx,
        st.residue_ms,
        st.cost_ms,
        st.network_ms,
        st.primal_dual_ms,
        st.integrate_ms,
        st.total_ms,
        throughput,
        fmt_bytes(analytic),
        fmt_bytes(rss_after.saturating_sub(rss_before)),
    );
    (st, analytic, rss_after)
}

fn main() {
    println!("# whirlwind-rs scale benchmark\n");
    println!(
        "Times in milliseconds. Peak RSS via `getrusage(RUSAGE_SELF)`; the RSS column\n\
              shows the **delta** to the prior call so each row gives a real per-image cost.\n"
    );

    println!("## Per-stage timing + memory\n");
    println!(
        "| Scene                    | size      |    Mpx | residue |   cost |    net |      pd | integ. |  total | Mpx/s  | analytic mem | ΔpeakRSS |"
    );
    println!(
        "|--------------------------|-----------|-------:|--------:|-------:|-------:|--------:|-------:|-------:|-------:|-------------:|---------:|"
    );

    let mut all = Vec::new();
    for &n in &[256usize, 512, 1024, 2048] {
        let (ig, co) = make_clean(n, n);
        let row = run_scene("clean diagonal ramp", n, n, ig, co, 1.0);
        all.push(("clean", n, row));
    }
    for &n in &[256usize, 512, 1024, 2048] {
        let (ig, co) = make_noisy(n, n, 0.7, 10);
        let row = run_scene("noisy ramp (γ=0.7, L=10)", n, n, ig, co, 10.0);
        all.push(("noisy", n, row));
    }
    // Skip 2048² very-noisy (~5 minutes) by default; pass --huge to include it.
    let include_huge = std::env::args().any(|a| a == "--huge");
    let sizes_vn: &[usize] = if include_huge {
        &[256, 512, 1024, 2048]
    } else {
        &[256, 512, 1024]
    };
    for &n in sizes_vn {
        let (ig, co) = make_very_noisy(n, n);
        let row = run_scene("very noisy ramp (γ=0.3, L=4)", n, n, ig, co, 4.0);
        all.push(("very-noisy", n, row));
    }

    println!("\n## Primal-dual internals\n");
    println!(
        "| Scene                    | size      | residues | pd iters | dijkstra |  augment | potential | ssp iters |"
    );
    println!(
        "|--------------------------|-----------|---------:|---------:|---------:|---------:|----------:|----------:|"
    );
    for (label, n, (st, _, _)) in &all {
        let scene = match *label {
            "clean" => "clean diagonal ramp",
            "noisy" => "noisy ramp (γ=0.7, L=10)",
            "very-noisy" => "very noisy ramp (γ=0.3, L=4)",
            _ => "?",
        };
        let size = format!("{n}x{n}");
        let pdt = primal_dual::PDTimings::default();
        let _ = pdt; // pull in trait to ensure types align
        let st_pdt = primal_dual::last_timings();
        let _ = st_pdt;
        println!(
            "| {:24} | {:9} | {:8} | {:8} | {:7.1} ms | {:6.1} ms | {:7.1} ms | {:9} |",
            scene,
            size,
            st.pd_residues,
            st.pd_iters,
            st.pd_dijkstra_ms,
            st.pd_augment_ms,
            st.pd_potential_ms,
            st.pd_ssp_iters,
        );
    }
}
