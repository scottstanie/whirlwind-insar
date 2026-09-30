# whirlwind

Fast Rust-backed 2D InSAR phase unwrapping with Python bindings.

Whirlwind unwraps a complex interferogram and returns unwrapped phase and connected-component labels. It runs one to two orders of magnitude faster than single-tile SNAPHU, with less than half the peak memory.
For the algorithm and the full comparisons, see [our paper](https://arxiv.org/abs/2609.36267); to cite it, see [Reference](#reference).

The package is `whirlwind-insar` on PyPI and GitHub; it imports as `whirlwind`.

## Quickstart

`whirlwind` can be installed from PyPI,

```bash
pip install whirlwind-insar
```

or on Conda Forge:

```bash
conda install -c conda-forge whirlwind-insar
```

Source installs require Python 3.11+ and Rust.

## Usage

```python
import whirlwind as ww

unw, conncomp = ww.unwrap(igram, corr, nlooks=10.0, mask=mask)
```

`igram` is a complex wrapped interferogram, `corr` is coherence/correlation in `[0, 1]`, and `mask` is optional with `True` for valid pixels.

## CLI

A CLI is provided that mirrors the Python API. Run `whirlwind --help` for the full list.

```bash
➜ whirlwind --help
InSAR phase unwrapper.
...
```

### CLI installation

There are multiple ways to install whirlwind to use only the CLI:

1. **Prebuilt binary (no Python or toolchain).** Download the archive for your platform from the [latest release][releases], unpack it, and run the `whirlwind` executable. A single self-contained binary - handy for MATLAB users driving it via `system('whirlwind ...')`.
2. **With the Python package.** The wheel ships a `whirlwind` console script which can be run using [`uvx` with with the UV tool system](https://docs.astral.sh/uv/guides/tools/):
```bash
uvx --from whirlwind-insar whirlwind --help   # zero-install try-out
pip install whirlwind-insar                   # puts `whirlwind` on PATH
```

3. **Docker** via the [Github Container Registry](https://github.com/scottstanie/whirlwind-insar/pkgs/container/whirlwind-insar)

4. **From source with Cargo** using a local Git clone:

   ```bash
   cargo install --path crates/whirlwind-cli --locked
   ```

### Running the CLI

```bash
whirlwind \
    --phase wrapped_phase.tif \
    --cor coherence.tif \
    --mask valid_mask.tif \
    --nlooks 10 \
    --out unwrapped_phase.tif
```
where
- `--phase` is the wrapped phase in radians: a float32 TIFF, or a flat binary float32 file (see below). If you start from a complex-valued GeoTIFF, extract GDAL's PHASE derived subdataset first and pass that as `--phase`;
  - Alternative: `--ifg` is for flat complex64 rasters. The phase path reconstructs a unit-magnitude interferogram, so it does not preserve amplitude.
- `--cor coherent.tif` is the float32 interferometric sample coherence raster
- `--mask` is optional; nonzero means valid. When `--mask` is omitted the CLI uses `coherence > 0` (and `igram != 0` with `--ifg`) as the default valid mask, matching the Python API.
- `--nlooks` is the number of independent looks used to estimate the sample coherence raster

The CLI writes a SNAPHU-like connected-component label map by default next to `--out` (`foo.conncomp.tif` for TIFF, `foo.unw.conncomp` for flat `.unw`); use `--conncomp PATH` to choose the path or `--no-conncomp` to skip it.

### Flat-binary formats (snaphu / ROI_PAC / isce2 / GAMMA)

Whirlwind can also accept flat binary rasters:

```bash
# snaphu-style: complex64 .int + amp/cor .cc; width ("line length") on the CLI
whirlwind --ifg pair.int --cor pair.cc --cols 1024 --nlooks 10 --out pair.unw

# ROI_PAC / Stanford: geometry read from the <file>.rsc sidecar automatically
whirlwind --ifg 20150902_20150914.int --cor 20150902_20150914.cc \
    --nlooks 10 --out 20150902_20150914.unw

# isce2 stripmapStack / topsStack: the <file>.xml sidecars provide everything
whirlwind --ifg filt_fine.int --cor filt_fine.cor --nlooks 10 \
    --out filt_fine.unw

# GAMMA: big-endian; width from a .par/.off (or --cols + --big-endian)
whirlwind --ifg pair.diff --ifg-meta pair.off \
    --cor pair.cc --cor-meta pair.off --nlooks 10 --out-format float --out pair.unw
```

- `--ifg` is the raw flat complex64 interferogram (snaphu `COMPLEX_DATA`, i.e.  `numpy.tofile()` of a complex64 array). `--phase` accepts float32 wrapped phase as TIFF or flat binary (snaphu `FLOAT_DATA`) and reconstructs unit-magnitude complex values.  Exactly one of the two is given.
- `--cor` may be single-band float32 (isce2 `.cor`, GAMMA `.cc`) or the two-band line-interleaved amplitude+correlation "rmg" layout (snaphu's default, ROI_PAC `.cc`): the band count is detected from the file size and the correlation is read from the second channel, exactly as snaphu does.  `--cor-format alt-sample` covers snaphu's sample-interleaved variant.
- `--cols` (alias `--width`) is snaphu's "line length" / ROI_PAC `WIDTH`; the row count always comes from the file size. A `<file>.rsc` or `<file>.xml` next to each input supplies it automatically (and, for isce2, the dtype, band count, scheme, and byte order). Use `--ifg-meta`, `--phase-meta`, or `--cor-meta` when the sidecar is not next to that input.
- Output is chosen by extension (override with `--out-format`): `.tif` → TIFF; `.unw` → two-band amp+phase rmg (snaphu's default output layout); anything else → flat float32 phase. Conncomp follows the output style by default: u16 TIFF for TIFF outputs, or one-byte-per-pixel flat for flat outputs (the snaphu/isce2 convention). Flat outputs keep the input's byte order.
- `--mask` also accepts snaphu-style flat byte masks (nonzero = valid, zero = masked).

### Docker

```bash
docker pull ghcr.io/scottstanie/whirlwind-insar:main   # prebuilt, or:
docker build -t ghcr.io/scottstanie/whirlwind-insar .  # build locally

docker run --rm -v "$PWD:/data" ghcr.io/scottstanie/whirlwind-insar \
    --phase /data/wrapped.tif --cor /data/cor.tif --nlooks 10 \
    --out /data/unw.tif
```

## Dolphin

[Dolphin](https://github.com/isce-framework/dolphin) can select Whirlwind as an unwrap method:

```bash
dolphin unwrap --unwrap-options.unwrap-method WHIRLWIND ...
```

See the Dolphin docs for the rest of the Dolphin workflow.

## Development

```bash
git clone https://github.com/scottstanie/whirlwind-insar.git
cd whirlwind-insar
pip install .
```

```bash
uv sync
uv run maturin develop --release
uv run pytest python/tests
cargo test --workspace
```

## More

- [Algorithm](docs/ALGORITHM.md)
- [Options and recipes](docs/RECIPES.md)
- [Connected components](docs/CONNCOMP_TUNING.md)
- [Bridging disconnected regions](docs/BRIDGING.md)
- [Performance](docs/PERFORMANCE.md)
- [Environment variables](docs/ENV_VARS.md)

## Reference

If you use Whirlwind in your work, please cite the paper (submitted to IEEE TGRS; preprint on arXiv):

> Staniewicz, S., Gunter, G., Mirzaee, S., Govorcin, M., Oliver-Cabrera, T., & Fattahi, H. (2026). A Probabilistic-Cost Algorithm for Large-Scale 2D Phase Unwrapping. arXiv:2609.36267. https://arxiv.org/abs/2609.36267

```bibtex
@article{Staniewicz2026Whirlwind,
  title   = {A Probabilistic-Cost Algorithm for Large-Scale 2D Phase Unwrapping},
  author  = {Staniewicz, Scott and Gunter, Geoffrey and Mirzaee, Sara and Govorcin, Marin and Oliver-Cabrera, Talib and Fattahi, Heresh},
  journal = {arXiv preprint arXiv:2609.36267},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.36267},
  url     = {https://arxiv.org/abs/2609.36267}
}
```

The same citation is in [`CITATION.cff`](CITATION.cff), which GitHub shows under "Cite this repository".

## License

Licensed under either the BSD 3-Clause License or the Apache License, Version 2.0, at your option. See [LICENSE](LICENSE).
