# whirlwind-rs

A Rust rewrite of the [Whirlwind](https://github.com/isce-framework/whirlwind) Bayesian
minimum-cost-flow phase unwrapper for InSAR. See [`ATBD-whirlwind.md`](ATBD-whirlwind.md)
for the algorithm theoretical basis.

## Layout

- `crates/whirlwind-core` — pure-Rust algorithms (residues, costs, min-cost flow,
  integration, synthetic ifg simulator).
- `crates/whirlwind-cli` — `whirlwind` CLI binary (`simulate` + `unwrap` subcommands).
- `crates/whirlwind-py` — `pyo3`/`maturin` Python bindings, importable as `whirlwind_rs`.

## Quickstart

```bash
# Run all the Rust tests (unit + integration).
cargo test --workspace

# Build + install the Python module.
cd crates/whirlwind-py && maturin develop --release && cd ../..

# Run the Python test battery.
python -m pytest python/tests

# Generate a synthetic interferogram and unwrap it via the CLI.
cargo run --release -p whirlwind-cli -- simulate --shape 256x256 --out /tmp/sim
cargo run --release -p whirlwind-cli -- unwrap \
    --igram-re /tmp/sim/igram_re.tif --igram-im /tmp/sim/igram_im.tif \
    --cor     /tmp/sim/cor.tif --nlooks 10 --out /tmp/sim/unw.tif

# Run on real interferograms and save side-by-side PNGs.
python scripts/run_real_data.py           # Palos-Verdes (Sentinel-1)
python scripts/run_real_data.py --rosamond # + Capella Rosamond
```

## Performance & memory

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for full per-stage timings,
hotspot analysis, and memory model. Headline:

- **~25 Mpx/s** on clean / lightly-noisy data
- **0.2–0.8 Mpx/s** on residue-dense scenes; **97 %** of the time is Dijkstra in
  the primal-dual loop
- Memory **~115 bytes / pixel** working set:
  | image | pixels | RAM |
  |---|---:|---:|
  | 1024² | 1 Mpx | 118 MiB |
  | 2048² | 4 Mpx | 472 MiB |
  | Sentinel-1 IW (25K×4K) | 100 Mpx | ~11.5 GiB |
- **6–12× speed-up** on residue-dense inputs from a one-line change (`max_iter` 8 → 50);
  primal-dual now finishes natively with 0 SSP fall-back iterations on every scene we tested

Reproduce: `cargo run --release --example bench_scale -- --huge`.

## Benchmarks vs SNAPHU

`python scripts/bench_vs_snaphu.py` (mac, M-series, release build):

| Scene                          | whirlwind-rs | snaphu  | speedup |
| ------------------------------ | -----------: | ------: | ------: |
| diagonal ramp 512x512          |     0.013 s  | 0.355 s | 26.7×   |
| noisy bump 256x256             |     0.005 s  | 0.053 s | 10.2×   |
| Rosamond Capella 512x512 (low coh) |     0.267 s  | 1.337 s |  5.0×   |
| Palos-Verdes S1 (~155x229, masked) |     0.301 s  | 0.152 s |  0.5×   |

Whirlwind-rs is faster on synthetic and noisy scenes; SNAPHU wins on the small masked
Palos-Verdes case (most of our 0.3 s there is fixed setup, not flow solving).

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
