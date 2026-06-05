# Tiling design (HISTORICAL)

> **Status (2026-06-04): HISTORICAL record of a tiling attempt.** Tiling is
> **NOT** the default and is **NOT** validated. The shipped default of
> `whirlwind.unwrap(...)` is the single-tile linear MCF path (`tile_size=0` does
> **not** auto-tile — it runs single-tile linear on the whole image). The tiled
> pipeline described below is **opt-in and experimental** (selected only by
> `tile_size>=4`, `multilook>1`, or `WHIRLWIND_UNWRAP_SOLVER=tiled`); it fails on
> most scenes (~65–89 % vs single-tile ~99–100 %). Getting tiling to parity will
> probably require **exact overlap / connected-component reconciliation across
> tile seams**, which has not been built. The text below is preserved as a record
> of the approach only — do not treat any of it as current/shipped behavior.

**The fuller historical account lives in
[`paper/tiling.md`](https://github.com/scottstanie/whirlwind-insar/blob/main/paper/tiling.md)**
— why naive tiling fails, the stitch crux, the secondary-MCF and coarse-anchor
work, and the Stage-3 warm-start dead-ends.

## What was attempted (historical, not shipped)

The opt-in tiled path unwrapped frames as follows:

1. **Per-tile MCF** — each overlapping tile is unwrapped independently with the
   corner-safe reuse solver, bounding peak memory to tile scale.
2. **Global coarse anchor** — the complex igram is multilooked x8, solved whole
   (seam-free, runaway-free), upsampled, and each tile region's integer-2π level
   is snapped to it by coherence-weighted mode.
3. **Multi-scale cascade** (`coarse_refine` at f=16,8,4) + a **feathered seam
   composite**, then a **gated multi-shift re-solve + seam-repair** for
   fragmented scenes (a no-op on clean ones).

These numbers (e.g. a NISAR frame at 99.79 % K-match, 0 % multi-cycle, in ~4 s)
were **select-scene only** — the tiled path did not generalize across the
validated frame set, which is why it is not the default. Noisy /
moderate-coherence scenes (e.g. Sentinel-1) passed `multilook=L`.

Within this historical attempt, the planned per-region secondary MCF
("Stage 2") and warm-started full-image reoptimize ("Stage 3") were set aside in
favor of the coarse anchor + cascade — see `paper/tiling.md` for why, including
the Stage-3 warm-start approaches that were prototyped and rejected. None of this
is the current/shipped path; a future tiling effort would likely revisit exact
overlap / connected-component reconciliation rather than this cascade.
