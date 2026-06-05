# whirlwind-rs

Fast Rust-backed 2D InSAR phase unwrapping with Python bindings.

A minimum-cost-flow unwrapper that reaches SNAPHU-comparable quality several times faster, returning the unwrapped phase and connected-component labels from a single solve. See the [NISAR comparison](docs/NISAR_SUMMARY.md) for benchmarks.

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

For noisy scenes, down-look first:

```python
unw, conncomp = ww.unwrap(igram, corr, nlooks=50.0, mask=mask, multilook=8)
```

## CLI

Install:

```bash
cargo install --path crates/whirlwind-cli --locked
```

Run:

```bash
whirlwind unwrap \
    --phase wrapped_phase.tif \
    --cor coherence.tif \
    --mask valid_mask.tif \
    --nlooks 10 \
    --out unwrapped_phase.tif \
    --conncomp conncomp.tif
```

`--phase` is a float32 TIFF of wrapped phase in radians. `--mask` is optional; nonzero means valid.

The CLI is pure Rust (no GDAL), so it also runs from a container. Pull the
prebuilt image (published to the GitHub Container Registry by CI), or build it
locally:

```bash
docker pull ghcr.io/scottstanie/whirlwind-insar:main   # prebuilt, or:
docker build -t ghcr.io/scottstanie/whirlwind-insar .  # build locally

docker run --rm -v "$PWD:/data" ghcr.io/scottstanie/whirlwind-insar unwrap \
    --phase /data/wrapped.tif --cor /data/cor.tif --nlooks 10 \
    --out /data/unw.tif --conncomp /data/conncomp.tif
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
- [Performance notes](docs/PERFORMANCE.md)
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

MIT. See [LICENSE](LICENSE).
