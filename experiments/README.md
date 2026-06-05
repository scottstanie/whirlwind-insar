# Experiments and historical notes

These documents are research notes and historical records. They are not part of
the shipped pipeline and are not linked from the main documentation. Behavior,
numbers, and recommendations here may be out of date.

For what whirlwind actually ships, start at the [project README](../README.md)
and the documentation under [`docs/`](../docs).

## Contents

- [TILING_DESIGN.md](TILING_DESIGN.md): notes from the tiled-unwrap attempt. The
  shipped default is the single-tile solver; the tiled path was never validated
  across the test set and is not exposed in the Python API.
- [PHASS_SPEED.md](PHASS_SPEED.md): a longer analysis of why whirlwind is faster
  than SNAPHU and slower than PHASS. The headline numbers live in the
  [NISAR comparison](../docs/NISAR_SUMMARY.md).
- [ATBD-3d.md](ATBD-3d.md): the 3D / time-series (CRLB) algorithm basis. The 3D
  path is experimental and not shipped.
- [proto_tile_linear.py](proto_tile_linear.py): a prototype of memory-bounded
  tiling that stitches independent single-tile solves with the bridge
  integer-gauge idea over tile overlaps. This is the approach a future tiling
  effort would likely start from. It is a prototype, not part of the package.

The deeper, dated research write-ups (cost-model experiments, pyramid aliasing,
the why-whole-image-runs-away analysis, and the like) live under
[`paper/`](../paper).
