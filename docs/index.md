# whirlwind-rs

Fast Rust-backed 2D InSAR phase unwrapping with Python bindings.

Start with the [project README](https://github.com/scottstanie/whirlwind-insar/blob/main/README.md) for installation, Python usage, CLI usage, and links to comparisons.

## Main pages

- [Algorithm notes](ALGORITHM.md): the 2D unwrapping pipeline and the main terms used by the rest of the docs.
- [NISAR comparison](NISAR_SUMMARY.md): quality, runtime, and memory comparison on NISAR GUNW scenes.
- [Performance notes](PERFORMANCE.md): synthetic timing, memory behavior, and mask behavior.
- [Environment variables](ENV_VARS.md): debug and benchmarking switches.

## Deeper references

- [Tiling design](TILING_DESIGN.md): historical design notes for the experimental tiled path. The shipped default is not the tiled path.
- [Full ATBD](https://github.com/scottstanie/whirlwind-insar/blob/main/ATBD-whirlwind.md): long-form derivation and implementation details.
