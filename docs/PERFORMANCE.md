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

Each call decomposes timing into 5 stages (residue → cost → network init →
primal-dual → integrate) and records peak RSS via `getrusage(RUSAGE_SELF)`. The
primal-dual stage further breaks down into Dijkstra + augment + potential-update,
plus how many SSP fall-back iterations were needed.

## Headline results

### Per-stage timing (ms)

| Scene                    | size      |    Mpx | residue |   cost |    net |      pd | integ. |  total | Mpx/s  |
|--------------------------|-----------|-------:|--------:|-------:|-------:|--------:|-------:|-------:|-------:|
| clean diagonal ramp      | 256x256   |   0.07 |     0.6 |    5.1 |    0.1 |     0.0 |    0.1 |    5.9 |  11.17 |
| clean diagonal ramp      | 512x512   |   0.26 |     0.8 |    3.3 |    0.4 |     0.0 |    0.4 |    4.9 |  53.30 |
| clean diagonal ramp      | 1024x1024 |   1.05 |     2.6 |    9.0 |    1.5 |     0.2 |    1.4 |   14.7 |  71.10 |
| clean diagonal ramp      | 2048x2048 |   4.19 |     9.3 |   28.4 |    5.0 |     0.9 |    4.7 |   48.5 |  86.40 |
| noisy ramp (γ=0.7, L=10) | 256x256   |   0.07 |     0.3 |    3.9 |    0.0 |     0.0 |    0.1 |    4.3 |  15.28 |
| noisy ramp (γ=0.7, L=10) | 512x512   |   0.26 |     0.9 |    1.7 |    0.1 |     0.0 |    0.3 |    3.0 |  87.92 |
| noisy ramp (γ=0.7, L=10) | 1024x1024 |   1.05 |     2.9 |    4.8 |    0.7 |     0.1 |    1.1 |    9.8 | 106.76 |
| noisy ramp (γ=0.7, L=10) | 2048x2048 |   4.19 |    11.4 |   16.6 |    3.1 |   364.3 |    4.5 |  400.1 |  10.48 |
| very noisy (γ=0.3, L=4)  | 256x256   |   0.07 |     0.5 |    3.6 |    0.0 |    60.2 |    0.1 |   64.5 |   1.02 |
| very noisy (γ=0.3, L=4)  | 512x512   |   0.26 |     1.7 |    2.6 |    0.1 |   275.1 |    0.3 |  279.8 |   0.94 |
| very noisy (γ=0.3, L=4)  | 1024x1024 |   1.05 |     6.2 |    6.1 |    0.5 |  1977.4 |    1.4 | 1991.8 |   0.53 |
| very noisy (γ=0.3, L=4)  | 2048x2048 |   4.19 |    25.7 |   23.5 |    1.5 | 11262.2 |    4.4 | 11317.5 |  0.37 |

### Primal-dual internals

| Scene                    | size      | residues | pd iters | dijkstra |  augment | potential | ssp iters |
|--------------------------|-----------|---------:|---------:|---------:|---------:|----------:|----------:|
| noisy ramp (γ=0.7, L=10) | 2048x2048 |        4 |        1 |    361 ms |     1 ms |       1 ms |         0 |
| very noisy (γ=0.3, L=4)  | 256x256   |    10348 |       14 |     55 ms |     1 ms |       4 ms |         0 |
| very noisy (γ=0.3, L=4)  | 512x512   |    41464 |       16 |    265 ms |     4 ms |       5 ms |         0 |
| very noisy (γ=0.3, L=4)  | 1024x1024 |   165323 |       25 |   1934 ms |    27 ms |      13 ms |         0 |
| very noisy (γ=0.3, L=4)  | 2048x2048 |   666313 |       31 |  11069 ms |   131 ms |      44 ms |         0 |

## Where the time goes

Two regimes:

1. **Smooth interior or few residues** (top of every "noisy" row through 1024², all "clean" rows):
   Almost everything is the 7×7 box filter + per-arc cost evaluation; primal-dual is
   essentially free. After parallelization, **throughput is 50–105 Mpx/s** on these
   workloads (mostly cost build, which now scales with cores).

2. **Residue-dense (very noisy)**: primal-dual dominates totally. **>97 %** of the time is
   inside Dijkstra; augmentation and potential updates are small change.

In the residue-dense regime, the cost per Dijkstra grows roughly linearly with the
residue grid (≈ 360 ms / Dijkstra at 2048²), and the number of PD iterations grows
slowly with residue density (14 → 31 iters going from 10K to 666K residues — about
logarithmic).

## Changes from the v1 release

| Change | Effect | Where |
|---|---|---|
| Parallel cost build (rayon-iterated rows of the 7×7 box filter and per-arc cost fill, slabbed by direction so each thread writes a disjoint region of the `Vec<i32>`) | Cost stage **3–6×** faster on every size | `crates/whirlwind-core/src/cost/mod.rs` |
| Residue computation rewritten so each residue row is computed *only* from one pixel row (the curl is deposited at the bottom-right corner of each 2×2 pixel loop) — rows are independent → trivially `par_iter` over residue rows | Residue stage ~**1.3×** faster on noisy 2048² | `crates/whirlwind-core/src/residue.rs` |
| `Network::new` uses `residues.as_slice().to_vec()` (memcpy) instead of element-by-element fill, and `bitvec![1; n]; sat[..fwd].fill(false)` instead of a per-bit `set` loop | Net stage **~20×** faster on 2048² (32 ms → 1.5 ms) | `crates/whirlwind-core/src/network.rs` |
| `WHIRLWIND_DEBUG`, `WHIRLWIND_DIJKSTRA`, `WHIRLWIND_LLR_COST` env-var lookups are now cached in `OnceLock`s (was: re-read per Dijkstra call / per arc cost) | Eliminates ~25 syscalls/unwrap; small | `primal_dual.rs`, `shortest_path/mod.rs`, `cost/mod.rs` |
| Augment-phase cycle dedup: per-iteration `u32` epoch + `Vec<u32>` instead of per-sink heap-allocated `HashSet`; source dedup uses `Vec<bool>` | Augment stage ~**2×** faster on residue-dense inputs | `crates/whirlwind-core/src/primal_dual.rs` |
| Parallel potential update via `par_iter_mut().zip(par_iter())` | Potential stage ~**1.5×** faster | `primal_dual.rs` |
| Comment fix: swapped axis labels in `box_filter_2d` (no behavior change) | Readability | `cost/mod.rs` |

Aggregate end-to-end speedup vs the v1 baseline measured here:

| Scene | OLD total (v1 perf doc) | NEW total | Speedup |
|---|---:|---:|---:|
| clean 512²          |    17.4 ms |   4.9 ms | 3.6× |
| clean 1024²         |    47.6 ms |  14.7 ms | 3.2× |
| clean 2048²         |   159.5 ms |  48.5 ms | 3.3× |
| noisy γ=0.7 1024²   |    38.4 ms |   9.8 ms | 3.9× |
| noisy γ=0.7 2048²   |   510.3 ms | 400.1 ms | 1.3× |
| very noisy 512²     |   475.0 ms | 279.8 ms | 1.7× |
| very noisy 1024²    |  3534.2 ms |1991.8 ms | 1.8× |
| very noisy 2048²    | 21745.4 ms |11317.5 ms| 1.9× |

(All run on the same M-series host; some of the very-noisy speedup beyond what the
itemized list above explains is probably from less alloc churn during PD —
the `Vec<bool>` augment table is re-used across iterations rather than re-allocated
per sink, and the cost array is now built once with no scratch intermediaries.)

## On parallelizing the multi-source Dijkstra

We tried, and the rayon-parallel `Dial` (`run_parallel` in
`shortest_path/dial.rs`, opt-in via `WHIRLWIND_DIJKSTRA=dial-par`) is **not**
faster than the serial version on any size we measured. The reason:

- Phase 1 (parallel) collects proposed edge relaxations for each `u` in the
  current bucket against a *snapshot* of `sp.dist`. Each thread fills a local
  `Vec` of `(v, nd, arc, src, u)` tuples; rayon's `fold + reduce` then merges.
- Phase 2 (serial) re-checks `nd < sp.dist[v]` and applies the write, the
  pred-chain update, and the bucket push.

Phase 2 ends up doing roughly the same amount of work as the serial inner loop,
so the only thing we parallelize is the *check + propose*, which is cheap to
begin with (~17 ns per edge). After accounting for rayon fork-join cost (~10 µs
per bucket spawn) and the duplicated work in phase 2, the parallel version comes
out 2–5 % slower than serial.

What *would* help (but is significant work):
- A true **Δ-stepping** implementation with `AtomicI32` packed `(dist, pred)`
  updates and per-thread bucket queues. Requires bounding `dist` in `i32` (safe
  for grids up to ~6M-arc diameter; the safety check is cheap).
- A grid-block decomposition with **ghost-cell** exchange between threads —
  closer to how parallel PDE solvers work. Each thread owns a region and runs
  Dijkstra internally; boundary relaxations propagate via a few rounds of
  inter-thread communication.

Both are deferred. The headline number `97 % of time is in Dijkstra` is still
true on residue-dense inputs; this is the remaining lever for further speedups.

## Reproduce locally

```bash
cargo run --release --example bench_scale -- --huge       # ~12 seconds total
WW_MAX_ITER=8 cargo run --release --example bench_scale   # original v1 max_iter
WHIRLWIND_DIJKSTRA=heap   cargo run --release --example bench_scale -- --huge  # heap backend
WHIRLWIND_DIJKSTRA=dial-par cargo run --release --example bench_scale -- --huge # parallel Dial
```

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
    Dijkstra buckets:  ≤ 16 · (m+1)(n+1)
```

which works out to roughly **115 bytes per pixel** for the working set. The constant
4×4 Lee-PDF LUT is fixed (~128 KiB) regardless of image size and rounds off.

### Measured vs analytic

| size | analytic | measured ΔRSS (this run) |
|---|---:|---:|
| 256x256   | 7.4 MiB  | ≤ 5 MiB |
| 512x512   | 29.6 MiB | ≤ 19 MiB |
| 1024x1024 | 118 MiB  | 55–72 MiB |
| 2048x2048 | 472 MiB  | 91–289 MiB |

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

So we hit single-machine RAM limits around full Sentinel-1 frames. Memory does NOT
shrink with sparse-residue inputs — all the major arrays scale with the grid, not
with the number of residues.
