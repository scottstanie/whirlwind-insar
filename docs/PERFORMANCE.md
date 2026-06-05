# Performance notes

This page summarizes runtime and memory behavior for the default 2D Python call:

```python
unw, conncomp = whirlwind.unwrap(igram, corr, nlooks, mask=mask)
```

For NISAR-scale comparisons against SNAPHU, PHASS, and ICU, see [NISAR 2D unwrapping comparison](NISAR_SUMMARY.md). The numbers below are mostly synthetic benchmarks intended to explain scaling and bottlenecks.

## Short version

- Smooth or lightly noisy scenes run at roughly 50-105 megapixels per second on the benchmark laptop.
- Very noisy residue-dense scenes are much slower, around 1 megapixel per second, because shortest-path work dominates.
- Peak memory is linear in pixel count. For planning, budget about 0.2 GB per megapixel for the current whole-image path.
- Passing a valid-pixel mask matters. On a 4096x4096 land/water synthetic scene, runtime drops from 75.0 s without a mask to 0.54 s with the mask.

All numbers on this page were measured with release builds on an Apple M-series laptop with 12 performance cores and 36 GB RAM.

## Why it is faster than SNAPHU

Whirlwind and SNAPHU follow the same broad structure: residues, statistical edge costs, minimum-cost flow, then integration. The speed difference comes from how the flow problem is represented and solved.

The main reason is the cost model. SNAPHU uses nonlinear, flow-dependent statistical costs, which make a heavier network to solve. Whirlwind uses a fixed linear Carballo (Lee 1994) cost with a capacity-1 flow, so the network is lighter while keeping the same residue-pairing structure. The shortest-path inner loop also uses a tuned Dial-bucket Dijkstra. The speedup is mostly serial work: switching from 1 thread to 12 is only about 1.2 to 1.3 times faster, and a single thread is still around 13 times faster than single-tile SNAPHU.

PHASS is faster than whirlwind (roughly 2 to 4 times) because it is a different class of algorithm: it grows regions with quality-guided cuts instead of solving a global residue-balanced flow. That is cheaper but lower quality, which is why PHASS agrees less with the production SNAPHU unwrap on several frames in the [NISAR comparison](NISAR_SUMMARY.md).

## Synthetic benchmark

Run:

```bash
cargo run --release --example bench_scale -- --huge
```

The benchmark builds synthetic wrapped interferograms at several sizes and records timing for residue computation, cost construction, network setup, minimum-cost flow, and integration.

| Scene | Size | Total time | Throughput | What it shows |
|---|---:|---:|---:|---|
| clean diagonal ramp | 2048x2048 | 50.8 ms | 82.6 Mpx/s | no residues; cost construction dominates |
| noisy ramp, gamma=0.7, L=10 | 2048x2048 | 59.7 ms | 70.3 Mpx/s | a few noise-driven residues |
| very noisy ramp, gamma=0.3, L=4 | 1024x1024 | 913.3 ms | 1.15 Mpx/s | residue-dense; shortest paths dominate |
| very noisy ramp, gamma=0.3, L=4 | 2048x2048 | 4.85 s | 0.87 Mpx/s | same bottleneck at larger size |

The same benchmark reports per-stage timings. In smooth scenes, cost construction is most of the work. In residue-dense scenes, the primal-dual shortest-path loop is more than 95 percent of runtime.

## Large noisy scenes

For larger stress tests, use `scripts/heavy_scene.py` and `scripts/bench_heavy.py`.

```bash
python scripts/heavy_scene.py --size 4096 --flavor noisy --low 0.30 --out /tmp/heavy_4k_noisy.npz --summary
python scripts/bench_heavy.py --scene /tmp/heavy_4k_noisy.npz --no-snaphu
```

| Scene | Runtime | Notes |
|---|---:|---|
| 4096x4096 uniform gamma=0.3 | 23.1 s | 15.9 percent residues |
| 4096x4096 patchy gamma=0.30-0.90 | 8.7 s | 6.9 percent residues |
| 8192x8192 uniform gamma=0.3 | 166.5 s | about 2.66 M sources and 2.66 M sinks |

## Mask behavior

Real scenes usually have water, shadow, layover, or nodata regions. Pass a boolean mask with `True` for valid pixels:

```python
unw, conncomp = whirlwind.unwrap(igram, corr, nlooks=10.0, mask=valid)
```

Without a mask, invalid pixels still have phase values, often zero after upstream fill. Those values create artificial residues along mask boundaries and can dominate the flow problem.

| Scene | No mask | With mask | Speedup |
|---|---:|---:|---:|
| 4096x4096 gamma=0.7 land plus 35 percent blob-shaped water mask | 75.0 s | 0.54 s | 139x |

This is not a general claim that masks always make computation faster. The large gain appears when invalid areas would otherwise create many artificial residues.

## Memory

Memory is linear in pixel count. The core arrays are per-pixel or per-edge arrays: wrapped phase, smoothed gradients, edge coherence, integer costs, network state, shortest-path state, and integration buffers.

The analytic working set is about 115 bytes per pixel. Observed process memory is higher because of allocator behavior, Python/Rust boundary overhead, thread-local buffers, and temporary arrays. For planning, use about 0.2 GB per megapixel.

| Image size | Pixels | Planning memory |
|---|---:|---:|
| 1024x1024 | 1.0 M | about 0.2 GB |
| 2048x2048 | 4.2 M | about 0.8 GB |
| 4096x4096 | 16.8 M | about 3.4 GB |
| 8192x8192 | 67.1 M | about 13.4 GB |
| 25000x4000 | 100 M | about 20 GB |

The planning rule is intentionally conservative. The measured NISAR sweep peaks around 3-4 GB for frames near 18-19 megapixels.

Measured examples:

| Scene size | Pixels | Peak RAM |
|---|---:|---:|
| NISAR GUNW frames near 4176x4257 | 18-19 M | 3-4 GB |
| 11500x11500 interferogram | 132.3 M | 17.5 GB |

## Reproduce

```bash
cargo run --release --example bench_scale -- --huge
python scripts/heavy_scene.py --size 4096 --flavor noisy --low 0.30 --out /tmp/heavy_4k_noisy.npz --summary
python scripts/bench_heavy.py --scene /tmp/heavy_4k_noisy.npz --no-snaphu
```

Use release builds for timing. Debug builds are much slower.
