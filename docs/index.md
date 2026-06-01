# whirlwind-rs

A fast Bayesian minimum-cost-flow phase unwrapper for InSAR — both individual
interferograms and phase-linked time-series stacks. Written in Rust with Python
bindings.

This site collects the reference documentation. The project README, the
algorithm theoretical basis documents, and the research notes are maintained on
GitHub:

- **[Project README](https://github.com/scottstanie/whirlwind-insar/blob/main/README.md)**
  — install, quickstart, the Python / CLI API, and validation against
  SNAPHU / dolphin.
- **[ATBD — 3D / time series](https://github.com/scottstanie/whirlwind-insar/blob/main/ATBD-3d.md)**
  — CRLB cost, residue-boundary fix, tree-based closure, tiling, ground-node MCF.
- **[ATBD — 2D MCF core](https://github.com/scottstanie/whirlwind-insar/blob/main/ATBD-whirlwind.md)**
  — Carballo cost, residue grid, primal-dual SSP, integration.

## Reference

- [Performance](PERFORMANCE.md) — per-stage timings, scaling, the memory model,
  and mask-acceleration numbers.
- [Tiling design](TILING_DESIGN.md) — the shipped tiled-unwrap architecture
  (per-tile MCF + global coarse anchor + multi-scale cascade).
- [Environment variables](ENV_VARS.md) — debug / research environment variables.
