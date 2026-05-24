# whirlwind-rs

A fast Bayesian phase unwrapper for InSAR. Given a complex interferogram and a
coherence raster, `whirlwind-rs` returns the unwrapped phase by formulating
unwrapping as a minimum-cost network flow problem on a rectangular grid: edge
costs come from Carballo-style Bayesian likelihoods that weight phase gradients
by their coherence, and the flow problem is solved with a primal-dual algorithm
backed by Dial's bucket-queue Dijkstra. See
[`ATBD-whirlwind.md`](ATBD-whirlwind.md) for the full algorithm description.

## Layout

- `crates/whirlwind-core` — pure-Rust algorithms (residues, costs, min-cost
  flow, integration, synthetic ifg simulator).
- `crates/whirlwind-cli` — `whirlwind` CLI binary (`simulate` + `unwrap`
  subcommands).
- `crates/whirlwind-py` — `pyo3` / `maturin` Python bindings, importable as
  `whirlwind_rs`.

## Prerequisites

- Rust ≥ 1.85 (the workspace is on edition 2024). `rustup update stable`.
- Python ≥ 3.9.
- [uv](https://docs.astral.sh/uv/) is the recommended way to set up the dev
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

The Python API is a single function:

```python
import numpy as np
import whirlwind_rs as ww

# igram: complex64 (m, n); corr: float32 (m, n) in [0, 1]; mask optional bool.
unw = ww.unwrap(igram, corr, nlooks=10.0, mask=mask)   # → float32 (m, n)
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
hotspot analysis, and memory model. Headline (M-series, 8 perf cores, release
build):

- **~50–105 Mpx/s** on clean / lightly-noisy data — cost-build-bound.
- **~1 Mpx/s** on uniform-noisy residue-dense scenes — Dijkstra-bound
  (Dijkstra is ≥ 95 % of total time in this regime; it's the lever for
  further speedups).
- vs snaphu (same M-series): **177×** on noisy γ=0.7 2048², **406×** on
  clean 2048², **14.8×** on very-noisy γ=0.3 1024².
- Memory **~115 bytes / pixel** working set:
  | image | pixels | RAM |
  |---|---:|---:|
  | 1024² | 1 Mpx | 118 MiB |
  | 2048² | 4 Mpx | 472 MiB |
  | 4096² | 16 Mpx | ~1.9 GiB |
  | 8192² | 67 Mpx | ~7.5 GiB |
  | Sentinel-1 IW (25K×4K) | 100 Mpx | ~11.5 GiB |

Reproduce per-stage timings: `cargo run --release --example bench_scale -- --huge`.

## Benchmarks vs SNAPHU and kamui (PUMA)

`python scripts/bench.py` is the canonical cross-library benchmark; it writes
[`scripts/out/BENCH_RESULTS.md`](scripts/out/BENCH_RESULTS.md) and `.json` and
is fully reproducible (single command, fixed seeds). For long (5+ minute)
single-scene comparisons, `scripts/heavy_scene.py` + `scripts/bench_heavy.py`
build and time a deliberately-hard scene — useful when ~50 ms benchmarks are
too noisy to measure a change.

Most recent headline run on M-series:

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
- kamui's PUMA is excluded from the most recent run — it takes 4+ minutes on
  the 1024² very-noisy case and was swap-thrashing on this machine. To
  include it, run `python scripts/bench.py` on a machine with ≥ 32 GB RAM.

(`scripts/bench_vs_snaphu.py` is a simpler 2-library version kept for the
side-by-side PNG output it produces; `bench.py` is the one to use for tables.)

## Mask support

Pass a `bool` mask (True = valid) to `ww.unwrap(igram, corr, nlooks, mask)`.
Arcs that cross an invalid pixel-edge are pre-saturated so Dijkstra skips
them, and residues whose 2×2 pixel loop touches a masked pixel are zeroed —
without this, the arbitrary phase values in masked regions (typically
`igram = 0+0j` from upstream `nan_to_num`) generate a wall of spurious
residues at every mask boundary that dominate the MCF problem. Integration
BFS-walks the valid region from the first valid pixel and leaves masked
pixels as NaN.

The win on realistic mixed land/water scenes is large: on a 4096² γ=0.7
land + 35 % blob-shaped water-mask scene, **0.54 s with mask vs 75 s without
(139×)**. See `docs/PERFORMANCE.md` for details.

## Implementation notes

- **Carballo / Lee analytical cost is implemented** (`cost::lee_pdf`,
  `cost::hyp2f1`, `cost::lut`), with a lazy LUT cache per `nlooks`. The
  default is a simpler SNAPHU-style topological cost, which is always
  non-negative — Dijkstra is only valid on non-negative-cost graphs on the
  initial iteration. The Carballo log-likelihood cost can be enabled with
  `WHIRLWIND_LLR_COST=1` for experiments, but it can go negative and
  currently needs a Bellman-Ford preprocessing pass to set initial
  potentials (not yet wired up).
- **Multi-source Dijkstra source attribution:** walk `pred_node` back to
  find the real source rather than reading a stored `sp.source[]`, which
  can drift after re-relaxation on noisy data.
- **Potential update:** cap unreached-node distances at `D_max` (the max
  reached distance), per Ahuja-Magnanti-Orlin §9. Without this cap,
  residual arcs that cross the reach / unreach boundary acquire negative
  reduced cost on the next iteration and Dijkstra produces cyclic
  predecessor chains.
- **Boundary residues** are zeroed in `residue::compute` — partial loops
  at the image edge aren't real phase singularities.
- Debug tracing: set `WHIRLWIND_DEBUG=1` to print primal-dual / SSP
  iteration state.

## Status

- Workspace, all 3 crates, residue, cost (Lee + simple), min-cost flow,
  integration, simulator, pyo3 bindings, CLI, real-data verification, snaphu
  benchmark — done.
- Mask support is end-to-end.
- Tiled parallel unwrap — not yet (designed in `docs/TILING_DESIGN.md`).
