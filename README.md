# whirlwind-rs

A Rust rewrite of the [Whirlwind](https://github.com/isce-framework/whirlwind) Bayesian
minimum-cost-flow phase unwrapper for InSAR. See [`ATBD-whirlwind.md`](ATBD-whirlwind.md)
for the algorithm theoretical basis.

## Layout

- `crates/whirlwind-core` — pure-Rust algorithms (residues, costs, min-cost flow,
  integration, synthetic ifg simulator).
- `crates/whirlwind-cli` — `whirlwind` CLI binary (`simulate` + `unwrap` subcommands).
- `crates/whirlwind-py` — `pyo3`/`maturin` Python bindings, importable as `whirlwind_rs`.

## Prerequisites

- Rust ≥ 1.85 (the workspace is on edition 2024). `rustup update stable`.
- Python ≥ 3.9 with `pip install maturin numpy` for the Python bindings.
- Optional, for the cross-library bench (`scripts/bench.py`):
  `pip install snaphu kamui rasterio matplotlib`.

## Quickstart

```bash
# Run all the Rust tests (unit + integration).
cargo test --workspace

# Build + install the Python module in editable mode.
(cd crates/whirlwind-py && maturin develop --release)

# Run the Python test battery.
python -m pytest python/tests

# Generate a synthetic interferogram and unwrap it via the CLI.
cargo run --release -p whirlwind-cli -- simulate --shape 256x256 --out /tmp/sim
cargo run --release -p whirlwind-cli -- unwrap \
    --igram-re /tmp/sim/igram_re.tif --igram-im /tmp/sim/igram_im.tif \
    --cor     /tmp/sim/cor.tif --nlooks 10 --out /tmp/sim/unw.tif
```

## Environment variables

All optional; defaults are sensible.

| Var | Default | Effect |
|---|---|---|
| `WHIRLWIND_DEBUG` | unset | If set, primal-dual prints per-iteration state to stderr. |
| `WHIRLWIND_LLR_COST` | unset | Use the Carballo log-likelihood cost (negative-cost-tolerant; currently needs Bellman-Ford preprocessing — not enabled). Default is the SNAPHU-style non-negative cost. |
| `WHIRLWIND_DIJKSTRA` | `dial` | Select the multi-source Dijkstra backend: `dial` (default, serial Dial's bucket queue), `heap` (binary heap, reference), `dial-par` (rayon-parallel Dial — see `docs/PERFORMANCE.md`; not faster than serial on workloads we measured). |
| `WW_MAX_ITER` | `50` | Primal-dual max iterations before SSP fall-back. Used by `examples/bench_scale.rs`. |

## Performance & memory

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for full per-stage timings,
hotspot analysis, and memory model. Headline (M-series, 8 perf cores, release build):

- **~75–110 Mpx/s** on clean / lightly-noisy data — cost-build-bound.
- **~1 Mpx/s** on uniform-noisy residue-dense scenes; **2× faster on the
  worst-case 8192² uniform-noisy bench** than the v1 release (312.6 s → 166.5 s)
  thanks to early-exit Dijkstra. **3.3×** on a realistic 4096² Sentinel-1-style
  mixed-coherence scene.
- **Dijkstra is still ≥ 95 %** of time on the residue-dense regime;
  it's the remaining lever for further speedups. Parallel Dial is implemented
  (`WHIRLWIND_DIJKSTRA=dial-par`) but isn't faster than serial in practice —
  see `docs/PERFORMANCE.md` for why.
- Memory **~115 bytes / pixel** working set:
  | image | pixels | RAM |
  |---|---:|---:|
  | 1024² | 1 Mpx | 118 MiB |
  | 2048² | 4 Mpx | 472 MiB |
  | 4096² | 16 Mpx | ~1.9 GiB |
  | 8192² | 67 Mpx | ~7.5 GiB |
  | Sentinel-1 IW (25K×4K) | 100 Mpx | ~11.5 GiB |

Reproduce per-stage timings: `cargo run --release --example bench_scale -- --huge`.
Reproduce the heavy 5-min-class scene: `python scripts/heavy_scene.py --size 8192 --flavor noisy && python scripts/bench_heavy.py --scene /tmp/heavy_scene.npz --no-snaphu --only "dial serial"`.

## Benchmarks vs SNAPHU and kamui (PUMA)

`python scripts/bench.py` is the canonical cross-library benchmark; it writes
[`scripts/out/BENCH_RESULTS.md`](scripts/out/BENCH_RESULTS.md) and `.json` and is
fully reproducible (single command, fixed seeds). For the long (5+ minute)
single-scene comparison, `scripts/heavy_scene.py` + `scripts/bench_heavy.py`
build and time a deliberately-hard scene — useful for measuring future
optimization work where ~50 ms benchmarks are too noisy.

Headline from a recent run on M-series (numbers are from before the early-exit
pass; re-run after that change shrinks the ww times another 1.5–7× depending
on regime — see `docs/PERFORMANCE.md`):

| Scene | size | whirlwind-rs | snaphu | kamui (PUMA) | ww vs snaphu |
|---|---|---:|---:|---:|---:|
| clean ramp | 2048x2048 | 0.063 s | 20.06 s  | 117.7 s | **320×** |
| noisy ramp γ=0.7 | 2048x2048 | 0.412 s | 12.42 s  | 122.9 s | 30×    |
| very noisy ramp γ=0.3 | 1024x1024 | 2.206 s | 14.54 s | 229.9 s | 6.6×    |

(`scripts/bench_vs_snaphu.py` is a simpler 2-library version kept for the
side-by-side PNG output it produces; `bench.py` is the one to use for tables.)

## Implementation notes

- **Carballo / Lee analytical cost is implemented** (`cost::lee_pdf`, `cost::hyp2f1`,
  `cost::lut`), with a lazy LUT cache per `nlooks`. By default we use a simpler
  SNAPHU-style topological cost (always non-negative — which keeps Dijkstra valid
  on initial iterations). The Carballo LLR cost can be re-enabled with
  `WHIRLWIND_LLR_COST=1` for experiments, but it can go negative and currently
  needs a Bellman-Ford preprocessing pass to set initial potentials.
- **Both original-Whirlwind bugs are fixed at the source:**
  - Boundary residues are zeroed inside `residue::compute` (rather than in a Python wrapper).
  - The reverse-arc `arc_flow` semantics: the residual-reverse of a forward arc
    reports the *forward* arc's saturation, which means net_flow = 0 on an arc
    with no real flow pushed (not 1, as the original C++ produced).
- **Extra fixes for noisy real data** (not in the original C++):
  - Multi-source Dijkstra source attribution: walk pred_node back to find the real
    source rather than relying on a stale `sp.source[]` that can drift after
    re-relaxation.
  - Potential update: cap unreached-node distances at `D_max` (the max reached
    distance), per Ahuja-Magnanti-Orlin §9. Without this, residual arcs that cross
    the reach/unreach boundary acquire negative reduced cost on the next iteration
    and Dijkstra produces cyclic predecessor chains.
- Debug tracing: set `WHIRLWIND_DEBUG=1` to print primal-dual / SSP iteration state.

## Status

- Workspace, all 3 crates, residue, cost (Lee + simple), min-cost flow, integration,
  simulator, pyo3 bindings, CLI, real-data verification, snaphu benchmark — done.
- Tiled parallel unwrap — not yet (stretch goal in the plan).
- The mask test xfails because integration seeds at (0, 0); if that pixel is masked
  out the seed value is wrong. Need to seed at the first valid pixel in the mask.
