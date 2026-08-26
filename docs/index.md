# whirlwind

Fast Rust-backed 2D InSAR phase unwrapping with Python bindings. The package is
`whirlwind-insar` on PyPI and GitHub; it imports as `whirlwind`.

Start with the [project README](https://github.com/scottstanie/whirlwind-insar/blob/main/README.md) for installation, Python usage, CLI usage, and links to comparisons.

## Main pages

- [Algorithm notes](ALGORITHM.md): the 2D unwrapping pipeline and the main terms used by the rest of the docs.
- [Why it's Bayesian](BAYESIAN.md): the posterior probability the solver maximizes, and how the formulation differs from SNAPHU's MAP/ML costs.
- [NISAR comparison](NISAR_SUMMARY.md): quality, runtime, and memory comparison on NISAR GUNW scenes.
- [Why SNAPHU/PHASS differ](SNAPHU_PHASS_SPEED.md): qualitative explanation of the runtime differences in the NISAR comparison.
- [Memory and scaling notes](MEMORY_AND_SCALING.md): rough memory planning and qualitative scaling behavior.
- [Bridging disconnected regions](BRIDGING.md): how the relative 2π level between mask-split regions is set.
- [Tuning connected components](CONNCOMP_TUNING.md): the knobs that shape the conncomp label map.
- [Environment variables](ENV_VARS.md): debug and benchmarking switches.

## Deeper references

- [Full ATBD](ATBD-whirlwind.md): long-form derivation and implementation details.
