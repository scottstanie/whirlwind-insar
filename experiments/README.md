# Experiments and historical notes

These documents are research notes and historical records. They are not part of
the shipped pipeline and are not linked from the main documentation. Behavior,
numbers, and recommendations here may be out of date.

For what whirlwind actually ships, start at the [project README](../README.md)
and the documentation under [`docs/`](../docs).

## Contents

- [SPEED_VS_ORIGINAL.md](SPEED_VS_ORIGINAL.md): historical comparison against the original Python prototype.
- [TILING_DESIGN.md](TILING_DESIGN.md): notes from the tiled-unwrap attempt. The shipped default is the single-tile solver; the tiled path was never validated across the test set and is not exposed in the Python API.
