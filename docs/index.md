# whirlwind-rs

Fast Rust-backed 2D InSAR phase unwrapping with Python bindings.

This site collects reference documentation. Start with the project README for
installation and basic usage:

- **[Project README](https://github.com/scottstanie/whirlwind-insar/blob/main/README.md)**
  — install, Python usage, CLI usage, and links to comparisons.
- **[NISAR comparison](NISAR_SUMMARY.md)**
  — quality, runtime, and memory comparison on validated 2D scenes.
- **[ATBD — 2D MCF core](https://github.com/scottstanie/whirlwind-insar/blob/main/ATBD-whirlwind.md)**
  — Carballo cost, residue grid, primal-dual SSP, integration.

## Reference

- [Performance](PERFORMANCE.md) — per-stage timings, scaling, the memory model,
  and mask-acceleration numbers.
- [Tiling design](TILING_DESIGN.md) — historical design notes for the
  experimental, opt-in tiled-unwrap architecture. The shipped default is not the
  tiled path.
- [Environment variables](ENV_VARS.md) — debug / research environment variables.
