# Tiling design

The tiled unwrap is the shipped default for large frames. **The authoritative,
full account lives in
[`paper/tiling.md`](https://github.com/scottstanie/whirlwind-insar/blob/main/paper/tiling.md)**
— why naive tiling fails, the stitch crux, the secondary-MCF and coarse-anchor
work, and the Stage-3 warm-start dead-ends. This page is a short summary; that
file is the source of truth.

## What ships

`whirlwind.unwrap(...)` with `tile_size=0` auto-tiles frames larger than 512 px:

1. **Per-tile MCF** — each overlapping tile is unwrapped independently with the
   corner-safe reuse solver, bounding peak memory to tile scale.
2. **Global coarse anchor** — the complex igram is multilooked ×8, solved whole
   (seam-free, runaway-free), upsampled, and each tile region's integer-2π level
   is snapped to it by coherence-weighted mode.
3. **Multi-scale cascade** (`coarse_refine` at f=16,8,4) + a **feathered seam
   composite**, then a **gated multi-shift re-solve + seam-repair** for
   fragmented scenes (a no-op on clean ones).

On a NISAR frame this matches SNAPHU 9×9 at 99.79 % K-match (0 % multi-cycle) in
~4 s. Noisy / moderate-coherence scenes (e.g. Sentinel-1) pass `multilook=L`.

The planned per-region secondary MCF ("Stage 2") and warm-started full-image
reoptimize ("Stage 3") were **superseded** by the coarse anchor + cascade — see
`paper/tiling.md` for why, including the Stage-3 warm-start approaches that were
prototyped and rejected.
