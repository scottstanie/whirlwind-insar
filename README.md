# whirlwind

Fast Rust-backed 2D InSAR phase unwrapping with Python bindings.

Whirlwind unwraps a complex interferogram and returns both unwrapped phase and connected-component labels. The [NISAR comparison](docs/NISAR_SUMMARY.md) shows agreement with production SNAPHU on 2pi ambiguities, with lower runtime in the tested scenes.

> The package is `whirlwind-insar` on PyPI and GitHub; it imports as `whirlwind`.

## Quickstart

```bash
git clone https://github.com/scottstanie/whirlwind-insar.git
cd whirlwind-insar
pip install .
```

Source installs require Python 3.11+ and Rust.

## Usage

```python
import whirlwind as ww

unw, conncomp = ww.unwrap(igram, corr, nlooks=10.0, mask=mask)
```

`igram` is a complex wrapped interferogram, `corr` is coherence/correlation in `[0, 1]`, and `mask` is optional with `True` for valid pixels.

For noisy scenes, coarsen the solve with `downsample` (it unwraps a
coherently-averaged copy at the given factor to pick each block's 2π cycle, then
maps the cycles back onto the full-resolution phase). `nlooks` stays the effective looks of your
input `corr` — the down-look scaling is handled internally, so you do not raise
it yourself:

```python
unw, conncomp = ww.unwrap(igram, corr, nlooks=10.0, mask=mask, downsample=8)
```

## CLI

The `whirlwind` CLI is at feature parity with the Python `unwrap`: every knob the
Python API exposes (`downsample`, the `bridge` post-pass, persistent-scatterer
`interpolate`, Goldstein filtering, and the connected-component cost controls) is
available as a flag. Run `whirlwind unwrap --help` for the full list.

### Install

Pick whichever is simplest for you:

1. **Prebuilt binary (no Python or toolchain).** Download the archive for your
   platform from the [latest release][releases], unpack it, and run the
   `whirlwind` executable. A single self-contained binary - handy for MATLAB
   users driving it via `system('whirlwind unwrap ...')`.
2. **With the Python package.** The wheel ships a `whirlwind` console script
   (the same Rust CLI, entered in-process):

   ```bash
   uvx --from whirlwind-insar whirlwind --help   # zero-install try-out
   pip install whirlwind-insar                   # puts `whirlwind` on PATH
   ```

3. **From source with Cargo** (needs the Rust toolchain):

   ```bash
   cargo install --path crates/whirlwind-cli --locked
   ```

4. **Docker** (see below).

[releases]: https://github.com/scottstanie/whirlwind-insar/releases/latest

### Run

```bash
whirlwind unwrap \
    --phase wrapped_phase.tif \
    --cor coherence.tif \
    --mask valid_mask.tif \
    --nlooks 10 \
    --out unwrapped_phase.tif
```

`--phase` is the wrapped phase in radians: a float32 TIFF, or a flat binary
float32 file (see below). `--mask` is optional; nonzero means valid. When
`--mask` is omitted the CLI uses `coherence > 0` (and `igram != 0` with
`--ifg`) as the default valid mask, matching the Python API. The CLI writes a
connected-component label map by default next to `--out` (`foo.conncomp.tif` for
TIFF, `foo.unw.conncomp` for flat `.unw`); use `--conncomp PATH` to choose the
path or `--no-conncomp` to skip it.

### Flat-binary formats (snaphu / ROI_PAC / isce2 / GAMMA)

Headerless flat-binary rasters from the classic pipelines work directly - no
GDAL conversion needed:

```bash
# snaphu-style: complex64 .int + amp/cor .cc; width ("line length") on the CLI
whirlwind unwrap --ifg pair.int --cor pair.cc --cols 1024 --nlooks 10 --out pair.unw

# ROI_PAC / Stanford: geometry read from the <file>.rsc sidecar automatically
whirlwind unwrap --ifg 20150902_20150914.int --cor 20150902_20150914.cc \
    --nlooks 10 --out 20150902_20150914.unw

# isce2 stripmapStack / topsStack: the <file>.xml sidecars provide everything
whirlwind unwrap --ifg filt_fine.int --cor filt_fine.cor --nlooks 10 \
    --out filt_fine.unw

# GAMMA: big-endian; width from a .par/.off (or --cols + --big-endian)
whirlwind unwrap --ifg pair.diff --ifg-meta pair.off \
    --cor pair.cc --cor-meta pair.off --nlooks 10 --out-format float --out pair.unw
```

- `--ifg` is the raw complex64 interferogram (snaphu `COMPLEX_DATA`, i.e.
  `numpy.tofile()` of a complex64 array); `--phase` also accepts flat float32
  (snaphu `FLOAT_DATA`). Exactly one of the two is given.
- `--cor` may be single-band float32 (isce2 `.cor`, GAMMA `.cc`) or the
  two-band line-interleaved amplitude+correlation "rmg" layout (snaphu's
  default, ROI_PAC `.cc`): the band count is detected from the file size and
  the correlation is read from the second channel, exactly as snaphu does.
  `--cor-format alt-sample` covers snaphu's sample-interleaved variant.
- `--cols` (alias `--width`) is snaphu's "line length" / ROI_PAC `WIDTH`; the
  row count always comes from the file size. A `<file>.rsc` or `<file>.xml`
  next to each input supplies it automatically (and, for isce2, the dtype,
  band count, scheme, and byte order). Use `--ifg-meta`, `--phase-meta`, or
  `--cor-meta` when the sidecar is not next to that input.
- Output is chosen by extension (override with `--out-format`): `.tif` →
  TIFF; `.unw` → two-band amp+phase rmg (snaphu's default output layout);
  anything else → flat float32 phase. Conncomp follows the output style by
  default: u16 TIFF for TIFF outputs, or one-byte-per-pixel flat for flat
  outputs (the snaphu/isce2 convention). Flat outputs keep the input's byte
  order.
- `--mask` also accepts snaphu-style flat byte masks (nonzero = valid).

For noisy scenes, coarsen the solve with `--downsample` (as in the Python API);
the integration-component `--no-bridge`-able re-leveling pass runs by default:

```bash
whirlwind unwrap --phase wrapped_phase.tif --cor coherence.tif \
    --mask valid_mask.tif --nlooks 10 --downsample 8 \
    --out unwrapped_phase.tif
```

The CLI is pure Rust (no GDAL), so it also runs from a container. Pull the
prebuilt image (published to the GitHub Container Registry by CI), or build it
locally:

```bash
docker pull ghcr.io/scottstanie/whirlwind-insar:main   # prebuilt, or:
docker build -t ghcr.io/scottstanie/whirlwind-insar .  # build locally

docker run --rm -v "$PWD:/data" ghcr.io/scottstanie/whirlwind-insar unwrap \
    --phase /data/wrapped.tif --cor /data/cor.tif --nlooks 10 \
    --out /data/unw.tif
```

## Dolphin

[Dolphin](https://github.com/isce-framework/dolphin) can select Whirlwind as an unwrap method:

```bash
dolphin unwrap --unwrap-options.unwrap-method whirlwind ...
```

See the Dolphin docs for the rest of the Dolphin workflow.

## Development

```bash
uv sync
uv run maturin develop --release
uv run pytest python/tests
cargo test --workspace
```

## More

- [NISAR comparison](docs/NISAR_SUMMARY.md)
- [Why SNAPHU/PHASS differ](docs/SNAPHU_PHASS_SPEED.md)
- [Memory and scaling notes](docs/MEMORY_AND_SCALING.md)
- [Algorithm notes](docs/ALGORITHM.md)
- [Environment variables](docs/ENV_VARS.md)

## Repository Layout

- `python/whirlwind`: Python API.
- `crates/whirlwind-core`: Rust algorithms.
- `crates/whirlwind-py`: PyO3 bindings.
- `crates/whirlwind-cli`: CLI binary.
- `docs`: reference docs.
- `scripts`: benchmarks and development utilities.

## License

Licensed under either the BSD 3-Clause License or the Apache License, Version
2.0, at your option. See [LICENSE](LICENSE).
