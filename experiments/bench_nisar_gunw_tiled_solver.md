# Removed: `--solver tiled` path from `bench_nisar_gunw_whirlwind.py`

Historical record. The NISAR GUNW bench (`scripts/bench_nisar_gunw_whirlwind.py`)
used to expose a `--solver {linear,tiled}` switch plus `--tile-size` /
`--tile-overlap`. The `tiled` branch is removed from the bench to keep it focused
on the one validated path (single-tile `unwrap_linear`, the ww-orig parity path).

Why removed:

- **Stale call.** The tiled branch called
  `ww.unwrap(ig, coh, nlooks, mask, tile_size=..., tile_overlap=...)`, but the
  public `ww.unwrap()` no longer accepts `tile_size` / `tile_overlap` (tiling is
  reached internally / via `WHIRLWIND_UNWRAP_SOLVER`, not as kwargs), so the
  branch would raise `TypeError`.
- **Not validated at NISAR scale.** Tiling produced fast-but-wrong results on
  most NISAR frames; it was never the default. See the memory notes / the
  whirlwind tiling design doc.

For the external GUNW comparison that *does* want connected components, use
`aws-batch/compare_gunw.py`, which calls the plain public `ww.unwrap(...)`
(single-tile linear + SNAPHU conncomps) — the path an external user actually
gets.

The removed bench dispatch, for reference:

```python
run.add_argument(
    "--solver",
    choices=["linear", "tiled"],
    default="linear",
    help="...linear (default, RECOMMENDED) = single-tile whole-image "
    "unwrap_linear; tiled = ww.unwrap tiled path (EXPERIMENTAL/invalid at scale).",
)
run.add_argument("--tile-size", type=int, default=0, help="tile_size for --solver tiled...")
run.add_argument("--tile-overlap", type=int, default=0, help="tile_overlap...")

# ... in run_one_product():
if args.solver == "linear":
    ww_unw = ww._native.unwrap_linear(ig_complex, coh_solver, float(args.nlooks), mask)
    ww_cc = None
else:
    ww_unw, ww_cc = ww.unwrap(
        ig_complex, coh_solver, args.nlooks, mask,
        tile_size=args.tile_size, tile_overlap=args.tile_overlap,
    )
```
