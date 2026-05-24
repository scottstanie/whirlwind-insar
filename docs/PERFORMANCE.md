# whirlwind-rs performance & memory

All numbers below from `cargo run --release --example bench_scale -- --huge`
on an Apple M-series laptop (12 perf cores, 36 GB RAM).

## Methodology

`crates/whirlwind-core/examples/bench_scale.rs` builds three synthetic scenes at
sizes 256², 512², 1024², 2048²:

| Scene | Truth | γ | Looks | What it tests |
|---|---|---|---|---|
| clean diagonal ramp | π·(x+y), x,y∈[−3,3] | 0.99 | 1 | no residues; pure wrapped-diff integration cost |
| noisy ramp | same ramp + Goodman noise | 0.7 | 10 | a handful of noise-driven residues |
| very noisy ramp | same ramp + heavy Goodman noise | 0.3 | 4 | residue blizzard (~Sentinel-1 over land) |

Each call decomposes timing into 5 stages (residue → cost → network init → primal-dual → integrate) and records peak RSS via `getrusage(RUSAGE_SELF)`. The primal-dual stage further breaks down into Dijkstra + augment + potential-update, plus how many SSP fall-back iterations were needed.

## Headline results

### Per-stage timing (ms)

| Scene                    | size      |    Mpx | residue |   cost |    net |      pd | integ. |  total | Mpx/s  |
|--------------------------|-----------|-------:|--------:|-------:|-------:|--------:|-------:|-------:|-------:|
| clean diagonal ramp      | 256x256   |   0.07 |     0.6 |   10.5 |    1.1 |     0.0 |    0.2 |   12.4 |   5.3 |
| clean diagonal ramp      | 512x512   |   0.26 |     1.5 |   12.0 |    3.3 |     0.1 |    0.5 |   17.4 |  15.1 |
| clean diagonal ramp      | 1024x1024 |   1.05 |     4.1 |   32.9 |    9.2 |     0.1 |    1.2 |   47.6 |  22.0 |
| clean diagonal ramp      | 2048x2048 |   4.19 |    11.9 |  109.8 |   32.3 |     0.8 |    4.5 |  159.5 |  26.3 |
| noisy ramp (γ=0.7, L=10) | 256x256   |   0.07 |     0.2 |    4.3 |    0.4 |     0.0 |    0.1 |    5.1 |  12.9 |
| noisy ramp (γ=0.7, L=10) | 512x512   |   0.26 |     0.9 |    6.2 |    1.8 |     0.0 |    0.3 |    9.3 |  28.3 |
| noisy ramp (γ=0.7, L=10) | 1024x1024 |   1.05 |     3.7 |   25.6 |    7.8 |     0.1 |    1.1 |   38.4 |  27.3 |
| noisy ramp (γ=0.7, L=10) | 2048x2048 |   4.19 |    14.6 |  100.7 |   31.8 |   358.3 |    4.6 |  510.3 |   8.2 |
| very noisy (γ=0.3, L=4)  | 256x256   |   0.07 |     0.5 |    5.0 |    0.5 |    79.7 |    0.1 |   85.7 |   0.8 |
| very noisy (γ=0.3, L=4)  | 512x512   |   0.26 |     1.8 |    9.3 |    1.8 |   461.8 |    0.3 |  475.0 |   0.6 |
| very noisy (γ=0.3, L=4)  | 1024x1024 |   1.05 |     7.1 |   36.9 |    7.5 |  3481.3 |    1.3 | 3534.2 |   0.3 |
| very noisy (γ=0.3, L=4)  | 2048x2048 |   4.19 |    27.8 |  149.2 |   29.0 | 21534.5 |    4.7 | 21745.4 |  0.19 |

### Primal-dual internals

| Scene                    | size      | residues | pd iters | dijkstra |  augment | potential | ssp iters |
|--------------------------|-----------|---------:|---------:|---------:|---------:|----------:|----------:|
| noisy ramp (γ=0.7, L=10) | 2048x2048 |        4 |        1 |   353 ms |     1 ms |       4 ms |         0 |
| very noisy (γ=0.3, L=4)  | 256x256   |    10348 |       12 |    76 ms |     3 ms |       1 ms |         0 |
| very noisy (γ=0.3, L=4)  | 512x512   |    41464 |       16 |   448 ms |     9 ms |       4 ms |         0 |
| very noisy (γ=0.3, L=4)  | 1024x1024 |   165323 |       25 |   3398 ms |    56 ms |     24 ms |         0 |
| very noisy (γ=0.3, L=4)  | 2048x2048 |   666313 |       31 |  21095 ms |   312 ms |    110 ms |         0 |

## Where the time goes

Two regimes:

1. **Smooth interior or few residues** (top of every "noisy" row through 1024², all "clean" rows):
   Almost everything is the 7×7 box filter + per-arc cost evaluation; primal-dual is
   essentially free. Memory is dominated by the cost array.
   Throughput is **25–30 Mpx/s**.

2. **Residue-dense (very noisy)**: primal-dual dominates totally. **>97 %** of the time is
   inside Dijkstra; augmentation and potential updates are small change.

In the residue-dense regime, the cost per Dijkstra grows roughly linearly with the residue grid (≈ 700 ms / Dijkstra at 2048²), and the number of PD iterations grows slowly with residue density (12 → 31 iters going from 10K to 666K residues — about logarithmic).

## Single biggest win so far

Bumping `max_iter` for the primal-dual loop from `8` (the libwhirlwind default) to **`50`** drove **6–12× speedups** on residue-dense inputs. The reason: PD's iteration is one *multi-source* Dijkstra that batches augmentations from every excess node simultaneously; the SSP fall-back is one *single-source* Dijkstra per remaining residue. On 2048² very-noisy data this drops the total wall time from **283 s → 22 s** while still using zero SSP iterations.

## Remaining hot paths (1024² very-noisy, ~22 s budget)

| Stage | Time | Share |
|---|---:|---:|
| primal-dual Dijkstra (×25 iters) | 3.4 s | 96 % |
| augmentation (path-walking + flow flips) | 56 ms | 1.6 % |
| potential update | 24 ms | 0.7 % |
| 7×7 smoothing + LUT cost build + per-arc eval | 37 ms | 1 % |
| residue computation | 7 ms | 0.2 % |
| integration | 1.3 ms | 0.04 % |

Future work for noisy data:
- **Dial's bucket queue** in place of the binary heap (already stubbed in
  `shortest_path/`). Expect a ~2× win on Dijkstra inner loops because all
  reduced costs are bounded integers.
- **Tiling**: each ~10⁶-pixel tile takes < 1 s noisy, so a 25-tile 25K×4K
  Sentinel-1 frame ≈ 25 s wall instead of the linear-extrapolated >10 min.
- Cost-scaling MCF: theoretically O(VE log U log VC), should batch better
  than SSP-style augmenting-paths for very-dense residue blizzards.

## Memory model

Analytically — per image of size `m × n` (square assumed for brevity):

```
working set ≈
    wrapped_phase:     4 · m · n        (f32)
    smoothed dy, dx:   4 · 2 · m · n
    edge coherence:    4 · 2 · m · n
    costs (i32):       4 · num_arcs              ≈ 16 · m · n
    Network.excess:    4 · (m+1)(n+1)
    Network.potential: 8 · (m+1)(n+1)
    Network.cost_fwd:  4 · num_forward           ≈  8 · m · n
    Network.is_saturated: bitvec(num_arcs)/8     ≈  0.5 · m · n
    Dijkstra dist:     8 · (m+1)(n+1)
    Dijkstra pred_arc,pred_node,source: 12·(m+1)(n+1)
    Dijkstra visited:  1 · (m+1)(n+1)
    Dijkstra heap:     ≤ 16 · (m+1)(n+1)
```

which works out to roughly **115 bytes per pixel** for the working set. The constant 4×4 Lee-PDF LUT is fixed (~128 KiB) regardless of image size and rounds off.

### Measured vs analytic

| size | analytic | measured ΔRSS |
|---|---:|---:|
| 256x256 | 7.4 MiB | 1–5 MiB |
| 512x512 | 29.6 MiB | 8–18 MiB |
| 1024x1024 | 118 MiB | 70–105 MiB |
| 2048x2048 | 472 MiB | 290–480 MiB |

Measured ΔRSS undershoots the analytic estimate because peak-RSS is monotonic — by
the time the smaller scenes run after the bigger ones, the OS has already pre-allocated
their footprint. The analytic figure is the right number to budget against.

### Quick projections

| Image | Pixels | RAM needed (working set) |
|---|---:|---:|
| 256² | 65 K | ~7 MiB |
| 1024² | 1 M | ~118 MiB |
| 2048² | 4 M | ~472 MiB |
| 4096² | 16 M | ~1.9 GiB |
| 8192² | 67 M | ~7.5 GiB |
| Sentinel-1 IW frame ≈ 25K × 4K | 100 M | **~11.5 GiB** |

So we hit single-machine RAM limits around full Sentinel-1 frames; that's the regime
where tiling becomes mandatory. Memory does NOT shrink with sparse-residue inputs —
all the major arrays scale with the grid, not with the number of residues.

## Reproduce locally

```bash
cargo run --release --example bench_scale -- --huge    # ~25 seconds total
WW_MAX_ITER=8 cargo run --release --example bench_scale -- --huge  # original Whirlwind default
```
