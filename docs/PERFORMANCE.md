# whirlwind-rs performance & memory

All numbers below from `cargo run --release --example bench_scale -- --huge`
on an Apple M-series laptop (12 perf cores, 36 GB RAM).

> **What these numbers are (read first).** The tables below benchmark the
> **whole-image MCF** on synthetic scenes — a valid *baseline*, but **not** the
> production path. The shipped default is the **tiled** path (per-tile MCF +
> global coarse anchor + multi-scale cascade); whole-image MCF *runs away* on
> real noisy scenes (NISAR: 80 % K-match, 18 % multi-cycle) whereas tiled +
> anchor + cascade reaches **99.79 % K-match / 0 % multi-cycle in 3.9 s** vs
> SNAPHU 9×9's ~17 min, and stays memory-bounded to tile scale. Tiling is a
> **correctness necessity**, not just an optimization. Real-scene method,
> numbers, and figures: [`paper/report_anchor_cascade.md`](https://github.com/scottstanie/whirlwind-insar/blob/main/paper/report_anchor_cascade.md).
> The synthetic baselines and the memory/parallelism analysis below remain
> accurate for what they measure (the per-tile core).

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
| clean diagonal ramp      | 256x256   |   0.07 |     1.2 |    6.6 |    0.1 |     0.0 |    0.1 |    8.0 |   8.15 |
| clean diagonal ramp      | 512x512   |   0.26 |     1.1 |    4.0 |    0.5 |     0.1 |    0.5 |    6.1 |  43.16 |
| clean diagonal ramp      | 1024x1024 |   1.05 |     3.0 |   10.1 |    1.5 |     0.3 |    1.5 |   16.4 |  63.94 |
| clean diagonal ramp      | 2048x2048 |   4.19 |     9.9 |   30.1 |    5.0 |     0.8 |    4.7 |   50.8 |  82.63 |
| noisy ramp (γ=0.7, L=10) | 256x256   |   0.07 |     0.3 |    4.1 |    0.0 |     0.0 |    0.1 |    4.6 |  14.39 |
| noisy ramp (γ=0.7, L=10) | 512x512   |   0.26 |     1.0 |    2.2 |    0.1 |     0.1 |    0.3 |    3.6 |  72.22 |
| noisy ramp (γ=0.7, L=10) | 1024x1024 |   1.05 |     3.2 |    5.1 |    0.7 |     0.3 |    1.1 |   10.5 |  99.77 |
| noisy ramp (γ=0.7, L=10) | 2048x2048 |   4.19 |    11.8 |   17.2 |    3.8 |    21.8 |    4.7 |   59.7 |  70.29 |
| very noisy (γ=0.3, L=4)  | 256x256   |   0.07 |     0.6 |    3.7 |    0.0 |    27.1 |    0.1 |   31.5 |   2.08 |
| very noisy (γ=0.3, L=4)  | 512x512   |   0.26 |     1.7 |    2.5 |    0.1 |   158.4 |    0.3 |  163.1 |   1.61 |
| very noisy (γ=0.3, L=4)  | 1024x1024 |   1.05 |     6.5 |    6.3 |    0.6 |   898.5 |    1.4 |  913.3 |   1.15 |
| very noisy (γ=0.3, L=4)  | 2048x2048 |   4.19 |    25.4 |   23.5 |    1.6 |  4793.1 |    4.6 | 4848.5 |   0.87 |

### Primal-dual internals

| Scene                    | size      | residues | pd iters | dijkstra |  augment | potential | ssp iters |
|--------------------------|-----------|---------:|---------:|---------:|---------:|----------:|----------:|
| noisy ramp (γ=0.7, L=10) | 2048x2048 |        4 |        1 |     18 ms |     1 ms |       1 ms |         0 |
| very noisy (γ=0.3, L=4)  | 256x256   |    10348 |       14 |     22 ms |     1 ms |       4 ms |         0 |
| very noisy (γ=0.3, L=4)  | 512x512   |    41464 |       16 |    148 ms |     5 ms |       6 ms |         0 |
| very noisy (γ=0.3, L=4)  | 1024x1024 |   165323 |       25 |    856 ms |    25 ms |      13 ms |         0 |
| very noisy (γ=0.3, L=4)  | 2048x2048 |   666313 |       31 |   4618 ms |   127 ms |      30 ms |         0 |

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

## Heavy-scene benchmark

For changes that need finer-grained measurement than the small-scene battery
above, `scripts/heavy_scene.py` builds large noisy ifgs and
`scripts/bench_heavy.py` times one configuration per subprocess (so the
`WHIRLWIND_DIJKSTRA` OnceLock isn't pinned to a single backend across runs).

| Scene (4096²+) | wall time | notes |
|---|---:|---|
| 4096² γ=0.3 uniform | 23.1 s | 15.9 % residues, "very noisy" |
| 4096² patchy γ (mixed γ=0.30–0.90) | 8.7 s | 6.9 % residues, "realistic Sentinel-1" |
| 8192² γ=0.3 uniform | 166.5 s | 15.9 % residues, ~2.66 M sources + 2.66 M sinks |

Run yourself with:

```bash
python scripts/heavy_scene.py --size 4096 --flavor noisy --low 0.30 \
    --out /tmp/heavy_4k_noisy.npz --summary
python scripts/bench_heavy.py --scene /tmp/heavy_4k_noisy.npz --no-snaphu
```

The 8192² case exercises the worst-case primal-dual loop (multi-minute wall
time) and is the easiest reproducible target when small benches are too
noisy to measure a change.

### Mask acceleration (Sentinel-1 land/water)

If a pixel-grid `mask` is passed to `ww.unwrap(igram, corr, nlooks, mask)`,
`Network::new_with_mask` pre-saturates every arc that crosses an invalid
pixel-edge so Dijkstra skips that arc. `residue::compute_with_mask` also
zeros residues whose 2×2 pixel loop touches a masked pixel — otherwise the
arbitrary-valued masked region (typically `igram=0+0j` after `nan_to_num`)
produces a wall of spurious residues at the mask boundary that completely
dominate the MCF problem.

The combined effect is large for realistic scenes:

| Scene | no mask | with mask | speedup |
|---|---:|---:|---:|
| 4096² γ=0.7 land + 35 % blob-shaped water mask (`heavy_scene.py --flavor noisy --low 0.7 --nlooks 10 --mask-fraction 0.35 --mask-kind blobs`) | 75.0 s | **0.54 s** | **139×** |

(`/tmp/heavy_4k_realistic.npz` in the local repro.)

The 139× isn't really a "mask makes things faster" story — it's that **without
a mask, the unwrapper does an enormous amount of pointless work on invalid
pixels** whose phase is just `arctan2(0, 0) = 0`, producing residues at every
land/water boundary. The mask just tells the algorithm to skip them.

Build a `heavy_scene.py`-style synthetic mask: `--mask-kind blobs` (a few
large gaussian "land" areas, realistic for coastal scenes) or `--mask-kind
rects` (random rectangles; stress-test, pathologically fragmented). On
uniform-γ noisy scenes mask doesn't speed things up much — the valid land
region itself is still dense in real residues. The win is in the realistic
"clean land + noisy water" regime.

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
- **Cost build, residue compute** — already parallel.

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
WW_MAX_ITER=8 cargo run --release --example bench_scale   # cap PD iters lower
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
