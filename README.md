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
- Python ≥ 3.9.
- [uv](https://docs.astral.sh/uv/) is the recommended way to get the dev
  environment (test + bench dependencies, including snaphu/kamui for
  cross-library comparison). Alternatively, plain `pip install maturin numpy`
  works for just building the bindings.

## Quickstart

With uv (recommended):

```bash
uv sync                                   # create venv + install all dev deps
uv run maturin develop --release          # editable Rust build into the venv
uv run pytest python/tests                # python test battery
uv run python scripts/bench.py            # cross-library benchmark
```

Without uv:

```bash
pip install maturin numpy
(cd crates/whirlwind-py && maturin develop --release)
python -m pytest python/tests
```

Rust tests + CLI work standalone (no Python):

```bash
cargo test --workspace

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

- **~50–75 Mpx/s** on clean / lightly-noisy data — cost-build-bound.
- **~1 Mpx/s** on uniform-noisy residue-dense scenes; **1.9× faster on the
  worst-case 8192² uniform-noisy bench** than the v1 release (312.6 s → 166.5 s)
  thanks to early-exit Dijkstra. **3.3×** on a realistic 4096² Sentinel-1-style
  mixed-coherence scene.
- vs snaphu (same M-series): **177×** on noisy γ=0.7 2048², **406×** on clean 2048², **14.8×** on very-noisy γ=0.3 1024².
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

Headline from the most recent run on M-series (post-early-exit + inner-loop
micro-opts; see `docs/PERFORMANCE.md` for what changed):

| Scene | size | whirlwind-rs | snaphu | ww vs snaphu |
|---|---|---:|---:|---:|
| clean ramp | 2048x2048 | 0.057 s | 23.09 s | **406×** |
| noisy ramp γ=0.7 | 2048x2048 | 0.086 s | 15.20 s | **177×** |
| very noisy ramp γ=0.3 | 1024x1024 | 1.094 s | 16.21 s | **14.8×** |

Notes:

- The clean-ramp row taking *longer* than the noisy-ramp row for snaphu is
  not a measurement artifact — snaphu's smooth-cost initialization scales
  with both coherence and look count in non-obvious ways and is actually
  more expensive at γ=0.99 than at γ=0.7 with our default per-scene nlooks.
  Pin nlooks across scenes with `python scripts/bench.py --nlooks 4` if you
  want a fixed snaphu regime.
- kamui's PUMA was excluded from this run — it takes 4+ minutes on the
  1024² very-noisy case and was swap-thrashing on this machine.
  Previously-measured kamui numbers are in earlier `BENCH_RESULTS.json`.

For the **long-running** worst-case bench (5+ minutes on the v1 code; used
to measure Dijkstra-level changes where short benches are too noisy):

| Scene | size | residues | v1 serial Dial | new serial Dial | speedup |
|---|---|---:|---:|---:|---:|
| 8192² uniform γ=0.3 | 8192x8192 | 10.65 M | 312.6 s | **166.5 s** | **1.88×** |
| 4096² mixed-coherence (patchy γ map) | 4096x4096 | 1.16 M |  28.9 s |   **8.7 s** | **3.34×** |

Reproduce: `python scripts/heavy_scene.py --size 8192 --flavor noisy && python scripts/bench_heavy.py --scene /tmp/heavy_scene.npz --no-snaphu --only "dial serial"`.

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
- **Mask support is now end-to-end:** pass a `bool` mask (True = valid) to
  `ww.unwrap(igram, corr, nlooks, mask)`. Masked pixels' arcs are pre-saturated
  so Dijkstra skips them; integration BFS-walks the valid region from the
  first valid pixel and leaves masked pixels as NaN. On a synthetic
  4096² γ=0.7 scene with a 35 % water-mask: **0.54 s with mask vs 75 s
  without** — masked-region junk residues otherwise dominate the cost.
- Tiled parallel unwrap — not yet (stretch goal in the plan).
