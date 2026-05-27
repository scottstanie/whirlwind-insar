# whirlwind-rs

A fast Bayesian phase unwrapper for InSAR — both individual interferograms
and full phase-linked time-series stacks. Written in Rust with Python
bindings.

Two entry points, sharing the same minimum-cost-flow core:

- **`whirlwind_rs.unwrap(igram, corr, nlooks, mask=None)`** — classical
  2D unwrap with the Carballo/SNAPHU-style coherence cost. Designed as a
  drop-in for boxcar interferograms.
- **`whirlwind_rs.unwrap_crlb(igram, variance, mask=None, tile_size=…)`** —
  CRLB-weighted unwrap for phase-linked interferograms (Dolphin / EVD /
  EMI). The per-pixel CRLB phase variance the phase-linker emits is a
  tighter noise model than sample coherence, and the unwrap reuses it
  end-to-end (cost weight → spanning-tree priority → reference-pixel
  selection → per-date posterior std).

A Python orchestrator (`scripts/unwrap_stack.py`) wraps `unwrap_crlb` over
a full Dolphin output dir to produce a closure-consistent, reference-anchored
unwrapped stack + per-pixel quality map + per-date posterior σ cube.

## Documents

- **[`ATBD-3d.md`](ATBD-3d.md)** — algorithm theoretical basis for the
  3D / time-series pipeline (CRLB cost, residue-boundary fix, tree-based
  closure, tiling, ground-node MCF). This is the current load-bearing
  doc.
- **[`ATBD-whirlwind.md`](ATBD-whirlwind.md)** — algorithm theoretical
  basis for the underlying 2D MCF unwrap (Carballo cost, residue grid,
  primal-dual SSP, integration). The 3D pipeline reuses this 2D core.
- **[`paper/whirlwind3d.pdf`](paper/whirlwind3d.pdf)** — IEEE GRSL letter
  draft (5 pp.) covering the publishable claims; build with
  `cd paper && latexmk -pdf whirlwind3d.tex`.
- **[`PERFORMANCE.md`](PERFORMANCE.md)** — per-stage timings,
  scaling, memory model, mask-acceleration numbers.
- **[`TILING_DESIGN.md`](TILING_DESIGN.md)** — design notes for
  the tiled solver (Stage 1 implemented; Stages 2–3 deferred).
- **[`ENV_VARS.md`](ENV_VARS.md)** — debug / research env vars.

The same docs are also published as a Material-themed mkdocs site (see
`mkdocs.yml`; build locally with `uv run mkdocs serve`). The Rust crate
API is documented inline via doc-comments and rendered by `cargo doc
--open` (or, once published, on
[docs.rs/whirlwind-core](https://docs.rs/whirlwind-core)).

## Layout

- `crates/whirlwind-core` — pure-Rust algorithms (residues, costs,
  min-cost flow, integration, tiled stitching, temporal-closure
  correction, synthetic-ifg simulator).
- `crates/whirlwind-cli` — `whirlwind` CLI binary (`simulate` + `unwrap`
  subcommands; 2D coherence-cost only for now).
- `crates/whirlwind-py` — `pyo3` / `maturin` Python bindings, importable
  as `whirlwind_rs`.
- `scripts/` — the Python orchestrator for 3D stack unwrap, the
  reproducer (`reproduce.sh`), and the cross-library benchmark harnesses.

## Prerequisites

- Rust ≥ 1.85 (workspace is on edition 2024). `rustup update stable`.
- Python ≥ 3.11.
- [uv](https://docs.astral.sh/uv/) is the recommended way to set up the
  dev environment (test + bench deps including snaphu/kamui for
  cross-library comparison). Alternatively, plain `pip install maturin
  numpy` works for just building the bindings.

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

## Using the Python API

### Single interferogram, coherence cost (boxcar / classical)

```python
import whirlwind_rs as ww

# igram: complex64 (m, n); corr: float32 (m, n) in [0, 1]; mask optional bool.
unw = ww.unwrap(igram, corr, nlooks=10.0, mask=mask)   # → float32 (m, n)
```

### Single interferogram, CRLB cost (phase-linked SLCs)

```python
# variance: float32 (m, n), σ²_IG = σ²_a + σ²_b read from
# crlb_<date>.tif files that the phase-linker emits. NoData = 0 is fine.
unw = ww.unwrap_crlb(igram, variance, mask=mask)

# Tiled variant, bounds per-IG MCF memory to tile-size scale:
unw = ww.unwrap_crlb(igram, variance, mask=mask,
                     tile_size=1024, tile_overlap=128)
```

### Full Dolphin stack → closure-consistent, anchored unwrapped stack

```bash
uv run python scripts/unwrap_stack.py \
    --dolphin /path/to/dolphin-output \
    --out     /tmp/whirlwind-stack \
    --window  1000 1000 2024 2024          # optional (i0 j0 i1 j1)
```

Emits per-IG `corrected/*.unw.tif`, a per-date phase cube
(`date_phases.tif`), per-pixel quality map (`quality.tif`,
max-|K|-over-triangles), per-date posterior σ (`date_phase_std.tif`), and
a JSON report with the temporal graph + reference pixel + run metadata.
See `ATBD-3d.md §8` for the full output spec.

## Validation against dolphin / SNAPHU

On a 52-acquisition / 150-IG / 4065 × 3802 Capella Palos Verdes stack
processed by Dolphin, `whirlwind-rs` agrees with Dolphin's SNAPHU output
**at 100 % of pixels modulo 2π on every IG**, with median absolute
per-IG RMS of 2.31 rad (anchored at Dolphin's own reference pixel). See
[`ATBD-3d.md §9`](ATBD-3d.md#9-comparison-with-existing-tools) for the
full table + figures and `scripts/compare_to_dolphin_unwrapped.py` for
the validator that produced the numbers.

The full pipeline is reproducible end-to-end on any Dolphin output
directory:

```bash
./scripts/reproduce.sh                  # 1024² tile, ~30 s, ~5 GB RAM
./scripts/reproduce.sh --full           # 4065 × 3802 single-piece, ~33 min, ~25 GB
./scripts/reproduce.sh --full --tile 1500   # full scene, tiled MCF
```

(The reproducer expects a Dolphin output dir at `$DOLPHIN_DIR`; the
Capella stack the paper uses is not redistributable.)

## 2D performance and benchmarks vs SNAPHU + kamui

`python scripts/bench.py` is the canonical cross-library 2D benchmark;
it writes `scripts/out/BENCH_RESULTS.md` and `.json` and is fully
reproducible (single command, fixed seeds). Headline numbers (M-series,
release build):

| Scene | size | whirlwind-rs | snaphu | speedup |
|---|---|---:|---:|---:|
| clean ramp | 2048² | 0.057 s | 23.09 s | **406×** |
| noisy ramp γ=0.7 | 2048² | 0.086 s | 15.20 s | **177×** |
| very noisy ramp γ=0.3 | 1024² | 1.094 s | 16.21 s | **14.8×** |

Single-IG throughput is **~50–105 Mpx/s** on clean / lightly-noisy data
(cost-build-bound) and **~1 Mpx/s** on uniform-noisy residue-dense
scenes (Dijkstra-bound). Memory ~115 bytes/pixel working set — a 100 Mpx
Sentinel-1 IW frame fits in ~11.5 GiB single-piece, or tile to cap. See
[`PERFORMANCE.md`](PERFORMANCE.md) for the full per-stage
timing breakdown and the discussion of why we don't ship the
rayon-parallel Dijkstra backend.

### Mask acceleration

Pass a `bool` mask (True = valid) to `unwrap` or `unwrap_crlb`. Arcs
crossing invalid pixel-edges are pre-saturated so Dijkstra skips them,
and residues whose 2×2 loop touches a masked pixel are zeroed — without
this, the arbitrary phase values in masked regions (typically
`igram = 0 + 0j` from upstream `nan_to_num`) generate a wall of spurious
residues at every mask boundary that dominate the MCF problem.

On a 4096² γ=0.7 land + 35 % blob-water-mask scene this is **0.54 s with
mask vs 75 s without (139×)**. See `PERFORMANCE.md` for details.

## Status

| Component | State |
|---|---|
| 2D MCF unwrap (coherence cost) | done |
| 2D MCF unwrap (CRLB cost) | done |
| Mask support, end-to-end | done |
| Tiled MCF + overlap-median stitch | done (`unwrap_crlb_tiled`) |
| Virtual ground-node MCF | done (`unwrap_crlb_grounded`) |
| Temporal closure correction (tree projection) | done, off by default — see [ATBD-3d §10.2](ATBD-3d.md#102-closure-correction-now-hurts-more-than-it-helps) |
| Per-pixel quality map from temporal triangles | done (`quality_triangles`) |
| `pyo3` Python bindings, CLI, real-data verification, snaphu benchmark | done |
| Per-region SNAPHU-style secondary MCF (TILING_DESIGN Stage 2) | not implemented |
| Spatial-coupling / LAMBDA-style integer LS for 3D | not implemented (future work, [ATBD-3d §10.5](ATBD-3d.md#105-what-would-actually-beat-the-current-default)) |

## License

MIT. See `Cargo.toml` for the workspace `license` field.
