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
| clean diagonal ramp      | 256x256   |   0.07 |     1.0 |    8.6 |    0.1 |     0.0 |    0.2 |   10.0 |   6.57 |
| clean diagonal ramp      | 512x512   |   0.26 |     1.2 |    3.9 |    0.5 |     0.1 |    0.6 |    6.3 |  41.73 |
| clean diagonal ramp      | 1024x1024 |   1.05 |     3.5 |   10.4 |    1.6 |     0.2 |    1.5 |   17.2 |  60.84 |
| clean diagonal ramp      | 2048x2048 |   4.19 |    10.3 |   28.1 |    4.6 |     0.6 |    4.7 |   48.5 |  86.48 |
| noisy ramp (γ=0.7, L=10) | 256x256   |   0.07 |     0.3 |    3.6 |    0.0 |     0.0 |    0.1 |    4.0 |  16.41 |
| noisy ramp (γ=0.7, L=10) | 512x512   |   0.26 |     0.9 |    1.7 |    0.1 |     0.1 |    0.3 |    3.1 |  85.86 |
| noisy ramp (γ=0.7, L=10) | 1024x1024 |   1.05 |     3.0 |    4.3 |    0.7 |     0.3 |    1.2 |    9.5 | 110.44 |
| noisy ramp (γ=0.7, L=10) | 2048x2048 |   4.19 |    11.6 |   16.3 |    2.8 |    20.5 |    4.6 |   56.1 |  74.70 |
| very noisy (γ=0.3, L=4)  | 256x256   |   0.07 |     0.6 |    3.9 |    0.0 |    30.4 |    0.1 |   35.0 |   1.87 |
| very noisy (γ=0.3, L=4)  | 512x512   |   0.26 |     1.9 |    2.2 |    0.1 |   163.0 |    0.3 |  167.5 |   1.57 |
| very noisy (γ=0.3, L=4)  | 1024x1024 |   1.05 |     6.7 |    6.4 |    0.4 |   982.5 |    1.2 |  997.3 |   1.05 |
| very noisy (γ=0.3, L=4)  | 2048x2048 |   4.19 |    26.5 |   22.3 |    1.5 |  5600.4 |    4.6 | 5655.4 |   0.74 |

### Primal-dual internals

| Scene                    | size      | residues | pd iters | dijkstra |  augment | potential | ssp iters |
|--------------------------|-----------|---------:|---------:|---------:|---------:|----------:|----------:|
| noisy ramp (γ=0.7, L=10) | 2048x2048 |        4 |        1 |     13 ms |     1 ms |       5 ms |         0 |
| very noisy (γ=0.3, L=4)  | 256x256   |    10348 |       14 |     25 ms |     1 ms |       4 ms |         0 |
| very noisy (γ=0.3, L=4)  | 512x512   |    41464 |       16 |    154 ms |     4 ms |       4 ms |         0 |
| very noisy (γ=0.3, L=4)  | 1024x1024 |   165323 |       25 |    944 ms |    25 ms |      10 ms |         0 |
| very noisy (γ=0.3, L=4)  | 2048x2048 |   666313 |       31 |   5431 ms |   125 ms |      26 ms |         0 |

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
| **Early-exit Dijkstra** — stop popping as soon as every deficit sink has been finalized. Cuts the wasted tail of late primal-dual iterations when only a few sinks remain. `ShortestPaths::popped[]` distinguishes finalized vs merely-relaxed nodes; the d_max cap in the potential update uses `popped` so the Ahuja-Magnanti-Orlin invariant still holds. | Dijkstra stage **2× on uniform-noisy 2048²** (11069→5431 ms); **7× on noisy γ=0.7 2048²** (361→13 ms); **3.3× on a realistic 4096² patchy scene** (28.9→8.7 s). | `crates/whirlwind-core/src/shortest_path/{dial,heap}.rs`, `primal_dual.rs`, `ssp.rs` |
| Parallel max-reduced-cost scan (sets the Dial bucket count) — was a serial O(E) loop per Dijkstra call; now `(0..num_arcs).into_par_iter().max()` | Few ms per Dijkstra call; small but free | `shortest_path/dial.rs` |
| Scratch buffers (`visited_epoch`, `source_used`, `path_info`, `deficits`) in `primal_dual::run` are now lifted out of the outer iteration loop and reused. Previously each ~67M-element `Vec<u32>` on 8192² was re-allocated every PD iter. | Cuts ~GiBs of alloc churn on multi-iter very-noisy runs; small wall-clock impact | `primal_dual.rs` |
| Comment fix: swapped axis labels in `box_filter_2d` (no behavior change) | Readability | `cost/mod.rs` |

Aggregate end-to-end speedup vs the v1 baseline measured here:

| Scene | OLD total (v1 perf doc) | NEW total | Speedup |
|---|---:|---:|---:|
| clean 512²          |    17.4 ms |   6.3 ms | 2.8× |
| clean 1024²         |    47.6 ms |  17.2 ms | 2.8× |
| clean 2048²         |   159.5 ms |  48.5 ms | 3.3× |
| noisy γ=0.7 1024²   |    38.4 ms |   9.5 ms | 4.0× |
| noisy γ=0.7 2048²   |   510.3 ms |  56.1 ms | **9.1×** |
| very noisy 512²     |   475.0 ms | 167.5 ms | 2.8× |
| very noisy 1024²    |  3534.2 ms | 997.3 ms | 3.5× |
| very noisy 2048²    | 21745.4 ms |5655.4 ms | 3.8× |

The biggest single win is **early-exit Dijkstra**, especially on inputs where the
sinks (deficit residues) cluster spatially. On the noisy γ=0.7 2048² scene, only
a handful of residues exist and they sit close to each other — every late PD
iteration is now O(local-frontier) instead of O(grid).

### Heavy-scene benchmark (added after the v1 perf pass)

`scripts/heavy_scene.py` builds large noisy ifgs and `scripts/bench_heavy.py`
times one configuration per subprocess (so the `WHIRLWIND_DIJKSTRA` OnceLock
isn't pinned). These two scenes are what the final speedup table above
extrapolates beyond:

| Scene (4096²+) | v1 serial Dial | new serial Dial | speedup |
|---|---:|---:|---:|
| 4096² γ=0.3 uniform (15.9 % residues, "very noisy") |  45.7 s | 23.1 s | 1.97× |
| 4096² patchy γ (mixed γ=0.30–0.90, 6.9 % residues, "realistic Sentinel-1") |  28.9 s |  8.7 s | 3.3× |
| 8192² γ=0.3 uniform (15.9 % residues, ~2.66M sources + 2.66M sinks) | 312.6 s | 166.5 s | 1.88× |

Run yourself with:

```bash
python scripts/heavy_scene.py --size 4096 --flavor noisy --low 0.30 \
    --out /tmp/heavy_4k_noisy.npz --summary
python scripts/bench_heavy.py --scene /tmp/heavy_4k_noisy.npz --no-snaphu
```

The 8192² case is the one that actually exercises the worst-case primal-dual
loop (5+ minutes on the v1 code) and is the easiest reproducible target for
measuring future optimization work.

## On parallelizing the multi-source Dijkstra

We tried, and the rayon-parallel `Dial` (`run_parallel` in
`shortest_path/dial.rs`, opt-in via `WHIRLWIND_DIJKSTRA=dial-par`) is **still
not** faster than the serial version on any size we measured — even after
early-exit was added. Measured on a 4096² uniform-noisy scene:

| backend | wall-time | notes |
|---|---:|---|
| serial `dial` (default) |  23.1 s | early-exit + parallel max-rc + alloc reuse |
| `dial-par` (rayon) |  39.7 s | phase 1 parallel, phase 2 serial |
| `heap` |  50.8 s | reference |

The reason:

- Phase 1 (parallel) collects proposed edge relaxations for each `u` in the
  current bucket against a *snapshot* of `sp.dist`. Each thread fills a local
  `Vec` of `(v, nd, arc, src, u)` tuples; rayon's `fold + reduce` then merges.
- Phase 2 (serial) re-checks `nd < sp.dist[v]` and applies the write, the
  pred-chain update, and the bucket push.

Phase 2 ends up doing roughly the same amount of work as the serial inner loop,
so the only thing we parallelize is the *check + propose*, which is cheap to
begin with (~17 ns per edge). The graph is also nearly memory-bound — each
relaxation touches `sp.dist[v]`, `sp.pred_*[v]`, two entries of `net.potential`,
and `net.is_saturated[arc]`. Adding cores doesn't add memory bandwidth
proportionally, so even the parallel phase doesn't scale linearly.

Where parallelism actually pays off in the primal-dual loop right now:

- **Max-reduced-cost scan** (sets the Dial bucket count) — parallel over arcs.
- **Potential update** — parallel `par_iter_mut().zip(par_iter())`.
- **Cost build, residue compute** — already parallel from the v1 perf pass.

What *would* help in Dijkstra proper (but is significant work):
- A true **Δ-stepping** implementation with `AtomicI32` packed `(dist, pred)`
  updates and per-thread bucket queues. Requires bounding `dist` in `i32` (safe
  for grids up to ~6M-arc diameter; the safety check is cheap).
- A grid-block decomposition with **ghost-cell** exchange between threads —
  closer to how parallel PDE solvers work. Each thread owns a region and runs
  Dijkstra internally; boundary relaxations propagate via a few rounds of
  inter-thread communication.

Both are deferred. On residue-dense inputs Dijkstra is still ≥ 95 % of total
time; that's the remaining lever for further speedups, and it now has half as
much fat on it after early-exit.

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
