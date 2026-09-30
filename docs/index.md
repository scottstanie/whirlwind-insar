# whirlwind

Fast Rust-backed 2D InSAR phase unwrapping with Python bindings. The package is
`whirlwind-insar` on PyPI and GitHub; it imports as `whirlwind`.

Start with the [project README](https://github.com/scottstanie/whirlwind-insar/blob/main/README.md) for installation, Python usage, and CLI usage.

## Pages

- [Algorithm](ALGORITHM.md): the unwrapping pipeline in brief.
- [Options and recipes](RECIPES.md): the inputs to get right, and optional steps for noisy or fragmented scenes.
- [Connected components](CONNCOMP_TUNING.md): what the label map means and how to tune it.
- [Bridging disconnected regions](BRIDGING.md): how the 2π offset between regions split by the mask is set.
- [Performance](PERFORMANCE.md): runtime and memory to expect, and why Whirlwind differs from SNAPHU and PHASS.
- [Environment variables](ENV_VARS.md): thread count, plus debugging and benchmarking switches.

## Paper

The cost model, solver, and evaluation are described in Staniewicz et al. (2026), [*A Probabilistic-Cost Algorithm for Large-Scale 2D Phase Unwrapping*](https://arxiv.org/abs/2609.36267) (arXiv:2609.36267). See the [README](https://github.com/scottstanie/whirlwind-insar#reference) for a BibTeX entry.
