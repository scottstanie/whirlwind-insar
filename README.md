# whirlwind-rs

A fast Bayesian phase unwrapper for InSAR — both individual interferograms
and phase-linked time-series stacks (2D per-IG is the validated, default path;
3D temporal closure is opt-in and currently regresses on tight unwraps — see
below). Written in Rust with Python bindings.

Two entry points, sharing the same minimum-cost-flow core:

- **`whirlwind.unwrap(igram, corr, nlooks, mask=None, *, multilook=1, tile_size=0, tile_overlap=0, goldstein_alpha=0.0) -> (unwrapped, conncomp)`**
  — classical 2D unwrap with the Carballo/SNAPHU-style coherence cost,
  returning the unwrapped phase **and** SNAPHU-style connected-component
  labels from a single solve. The default is the **verified single-tile
  linear MCF** path (`unwrap_linear`: ww-orig-parity Carballo Lee-1994
  coherence cost, capacity-1 min-cost flow, adaptive primal-dual → SSP
  fallback for masked frames) — it matches Python ww-orig on all 13 validated
  NISAR GUNW frames and beats PHASS on quality. It then runs two fast Python
  post-passes: SNAPHU-style connected components, and a default-on
  integration-component gauge "bridge" (`bridge=True`) that re-levels
  mask-disconnected regions to a coarse ×8 anchor (fixes the A_025 river from
  58% to 99.99% with zero regression; disable with `WHIRLWIND_NO_BRIDGE=1`).
  `tile_size=0` (default) does **not** auto-tile — it is single-tile linear on
  the whole image. The tiled pipeline is **opt-in and not validated** (fails
  on most scenes, ~65–89% vs single-tile ~99–100%) and is selected only by an
  explicit `tile_size>=4`, `multilook>1`, or
  `WHIRLWIND_UNWRAP_SOLVER=tiled`. For noisy / moderate-coherence scenes
  (e.g. Sentinel-1) pass **`multilook=8`**: a coherent down-look first
  suppresses the noise the linear cost can't route through. Goldstein
  pre-filtering is off by default (`goldstein_alpha=0`); enable via
  `goldstein_alpha>0` (the on-vs-off trade-off is under evaluation).
  The solver is selectable via `WHIRLWIND_UNWRAP_SOLVER=linear|tiled|reuse|convex`
  (default `linear`); `reuse` (PHASS-style whole-image), `convex`, `sparse`,
  and 3D-closure are experimental/research, not production.
- **`whirlwind.unwrap_crlb(igram, variance, mask=None, tile_size=…)`** —
  CRLB-weighted unwrap for phase-linked interferograms (Dolphin / EVD /
  EMI). The per-pixel CRLB phase variance the phase-linker emits is a
  tighter noise model than sample coherence, and the unwrap reuses it
  end-to-end (cost weight → spanning-tree priority → reference-pixel
  selection → per-date posterior std).

A Python orchestrator (`scripts/unwrap_stack.py`) wraps `unwrap_crlb` over
a full Dolphin output dir to produce a reference-anchored unwrapped stack +
per-pixel quality map + per-date posterior σ cube.

> **3D is not a closed-loop unwrapper by default.** `unwrap_stack.py` emits raw
> per-IG unwraps with reference-pixel anchoring (`--closure off`, the default and
> highest-quality option). The optional `--closure tree` enforces exact temporal
> consistency but currently *regresses*: median absolute RMS vs SNAPHU is 2.29 rad
> without closure vs 5.61 rad with it (see
> [`ATBD-3d.md §10.2`](ATBD-3d.md#102-closure-correction-now-hurts-more-than-it-helps)).
> The 2D per-IG path is what CI validates and what dolphin uses.

## Documents

- **[`ATBD-3d.md`](ATBD-3d.md)** — algorithm theoretical basis for the
  3D / time-series pipeline (CRLB cost, residue-boundary fix, tree-based
  closure, tiling, ground-node MCF). This is the current load-bearing
  doc.
- **[`ATBD-whirlwind.md`](ATBD-whirlwind.md)** — algorithm theoretical
  basis for the underlying 2D MCF unwrap (Carballo cost, residue grid,
  primal-dual SSP, integration). The 3D pipeline reuses this 2D core.
- **`paper/whirlwind3d.tex`** — IEEE GRSL letter draft (5 pp.) covering
  the publishable claims; build the PDF with
  `cd paper && latexmk -pdf whirlwind3d.tex`.
- **[`PERFORMANCE.md`](docs/PERFORMANCE.md)** — per-stage timings,
  scaling, memory model, mask-acceleration numbers.
- **[`TILING_DESIGN.md`](docs/TILING_DESIGN.md)** — pointer to the
  authoritative tiling account in `paper/tiling.md`.
- **[`ENV_VARS.md`](docs/ENV_VARS.md)** — debug / research env vars.

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
- `crates/whirlwind-py` — `pyo3` / `maturin` Python bindings (Rust source
  for the `_native` extension module). The `pyproject.toml` lives at the
  repo root; Python source at `python/whirlwind/`. Installs as the
  `whirlwind-insar` PyPI distribution, imports as `whirlwind`.
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
pip install .                          # builds the wheel via maturin and installs it
python -m pytest python/tests
```

For an editable install (rebuild on every Rust change):

```bash
pip install maturin numpy
maturin develop --release
python -m pytest python/tests
```

Rust tests + CLI work standalone (no Python):

```bash
cargo test --workspace

cargo run --release -p whirlwind-cli -- simulate --shape 256x256 --out /tmp/sim
cargo run --release -p whirlwind-cli -- unwrap \
    --phase /tmp/sim/wrapped.tif --cor /tmp/sim/cor.tif \
    --nlooks 10 --out /tmp/sim/unw.tif
```

## Using the Python API

### Single interferogram, coherence cost (boxcar / classical)

```python
import whirlwind as ww

# igram: complex64 (m, n); corr: float32 (m, n) in [0, 1]; mask optional bool.
# Returns (unwrapped_phase, conncomp). The default is the verified single-tile
# linear MCF path (ww-orig-parity Carballo Lee-1994 cost, capacity-1 min-cost
# flow, adaptive PD->SSP fallback) plus two fast post-passes: SNAPHU-style
# connected components and a default-on gauge "bridge" that re-levels
# mask-disconnected regions to a coarse x8 anchor (WHIRLWIND_NO_BRIDGE=1 off).
# tile_size=0 (default) does NOT auto-tile; tiling is opt-in and not validated.
unw, conncomp = ww.unwrap(igram, corr, nlooks=10.0, mask=mask)  # float32, uint32

# Noisy / moderate-coherence scene (e.g. Sentinel-1)? Multilook-first:
unw, conncomp = ww.unwrap(igram, corr, nlooks=50.0, mask=mask, multilook=8)

# Goldstein pre-filtering is opt-in (off by default; under evaluation):
unw, conncomp = ww.unwrap(igram, corr, nlooks=10.0, mask=mask, goldstein_alpha=0.7)
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

On the 13 validated NISAR GUNW frames the default single-tile linear MCF path
(`unwrap_linear`) matches Python ww-orig and beats PHASS on quality (PHASS is
faster but much more heuristic — hard-cut flow + flood-fill; see
[`PHASS_SPEED.md`](docs/PHASS_SPEED.md)). The default-on gauge bridge fixes the
A_025 river from 58% to 99.99% with zero regression. For the noisy Atlanta
Sentinel-1 OPERA frame, `multilook=8` is the lever (coherent averaging
suppresses the noise the linear cost can't route through). The older
tiled + anchor + cascade reports (e.g.
[`paper/report_anchor_cascade.md`](paper/report_anchor_cascade.md)) are
**historical**: those select-scene 99.xx% / 97.7% numbers were measured on the
opt-in, unvalidated tiled path and are **not** the shipped default.

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
scenes (Dijkstra-bound). Memory is linear in pixels — budget ~0.2 GB per
megapixel (see the Memory note below). See
[`PERFORMANCE.md`](docs/PERFORMANCE.md) for the full per-stage
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
mask vs 75 s without (139×)**. See [`PERFORMANCE.md`](docs/PERFORMANCE.md) for details.

### Memory

Peak RAM is **linear in pixel count** — the single-tile solve allocates a fixed
set of per-pixel / per-arc arrays and nothing that grows with scene content: the
residue-MCF network (per-node `excess`/`potential`, per-arc `cost`/`flow`/
saturation), the Dijkstra/Dial working state (`dist`, predecessors, bucket
queue), and the cost grids. So there is a fixed bytes-per-pixel budget.

**Practical rule: budget ≈ 0.2 GB of RAM per megapixel** (≈ 200 bytes/pixel peak
RSS). The validated NISAR bench peaks at ~3.5 GB on a 4176 × 4257 (17.8 Mpx)
frame; a 4400 × 4400 (19.4 Mpx) frame is ~3.8 GB. The core solver arrays alone
are ~88 bytes/pixel (an ~1.7 GB floor at 19.4 Mpx); observed RSS runs ~2× that
from allocator / rayon thread-local / `ndarray` capacity reserves, so plan
against the ~0.2 GB/Mpx figure. Memory is purely image-size-bound — **tiling is
the only lever that lowers peak RAM** (and is still experimental). See
[`PERFORMANCE.md`](docs/PERFORMANCE.md) for the per-array breakdown.

## Status

| Component | State |
|---|---|
| 2D MCF unwrap (coherence cost) | done |
| 2D MCF unwrap (CRLB cost) | **experimental** — tiled-only, never validated |
| Mask support, end-to-end | done |
| Tiled MCF + overlap-median stitch | **experimental, never validated** (`unwrap_crlb_tiled`; tiling never reached useful results for coherence *or* CRLB) |
| Single-tile linear MCF (coherence cost) + gauge bridge | **done, DEFAULT** (`unwrap(…)` / `unwrap_linear`; bridge default-on, `WHIRLWIND_NO_BRIDGE=1` off) |
| Tiled coherence path + global coarse anchor + multi-scale cascade | opt-in, **not validated** (~65–89% vs single-tile ~99–100%; `tile_size>=4` / `multilook>1` / `WHIRLWIND_UNWRAP_SOLVER=tiled`) |
| Feathered seam composite + `multilook=` for noisy scenes | experimental (part of the opt-in, unvalidated tiled path) |
| Virtual ground-node MCF | done (`unwrap_crlb_grounded`) |
| Temporal closure correction (tree projection) | done, off by default — see [ATBD-3d §10.2](ATBD-3d.md#102-closure-correction-now-hurts-more-than-it-helps) |
| Per-pixel quality map from temporal triangles | done (`quality_triangles`) |
| `pyo3` Python bindings, CLI, real-data verification, snaphu benchmark | done |
| Per-region SNAPHU-style secondary MCF (TILING_DESIGN Stage 2) | superseded — global coarse anchor + cascade reach SNAPHU quality without it |
| Spatial-coupling / LAMBDA-style integer LS for 3D | not implemented (future work, [ATBD-3d §10.5](ATBD-3d.md#105-what-would-actually-beat-the-current-default)) |

## License

MIT. See `Cargo.toml` for the workspace `license` field.
