# No-Goldstein unwrapping: global anchor + multi-scale cascade (NISAR), and multilook-first (Atlanta)

_2026-05-28. Builds on `tiling.md` (tiled MCF + secondary-net reconciliation + coarse region-refine)._

## TL;DR

| scene           | path                                   | match vs SNAPHU/OPERA mainland | \|dK\|≥2  | wall                | reference                                 |
| --------------- | -------------------------------------- | ------------------------------ | --------- | ------------------- | ----------------------------------------- |
| **NISAR**       | tiled + **anchor + cascade** (default) | **99.89%**                     | **0.00%** | 3.9 s               | SNAPHU 9x9 = 17 min                       |
| NISAR           | tiled + anchor (single f=8)            | 99.63%                         | 0.00%     | 4.1 s               |                                           |
| NISAR           | tiled, no anchor (prior)               | 99.21%                         | 0.21%     | 3.5 s               |                                           |
| **Atlanta S-1** | **multilook-8 + tiled**                | **97.66%**                     | 0.03%     | ~0.1 s coarse solve | OPERA/SNAPHU; snaphu(3x3) = 97.89% / 59 s |
| Atlanta S-1     | fine tiled (no multilook)              | 26.4%                          | 41.2%     | 8.7 s               | - fails (noise)                           |

Both scenes now reach SNAPHU/OPERA quality with no Goldstein workaround. **The lever is different per scene, and that difference is the whole story:** NISAR is high-coherence so the fine per-tile solve is trustworthy and only needs its integer cycle levels pinned (anchor + cascade); Atlanta is noisy moderate-coherence, where the fine solve is *garbage* and must be replaced by a multilooked solve that suppresses the noise first.

Figures: `plots/report_nisar.png`, `plots/report_atlanta_ml8.png`, `plots/nisar_variants.png`.

---

## NISAR - two new post-passes take 99.21% → 99.89% (artifacts gone)

The prior tiled result (99.21%) had a visible multi-cycle vertical streak and a couple of rectangular blocks. Both were *wrong integer cycle levels* of sub-regions that the per-tile MCF + secondary-net reconciliation could not reach (a coherent wrong island sharing no high-confidence seam with the mainland is invisible to the relative largest-region vote).

**1. Global coarse anchor** (`compute_coarse_anchor` in `tile.rs`). Multilook the complex igram x8 (coherent down-look - never average wrapped phase), unwrap that tiny image in ONE whole-image solve (no tiles ⇒ no seams ⇒ one self-consistent surface; x64 fewer pixels and x64 effective looks ⇒ no runaway), upsample, and snap each no-jump region's integer 2π level to it via a coherence-weighted **mode over the whole region** (robust to local anchor error). This pins absolute cycle levels rather than relative ones → reaches the no-seam wrong islands. → 99.21% → **99.63%**, |dK|≥2 0.21% → **0.00%** (the streak is gone).

**2. Multi-scale cascade** `coarse_refine` at f = 16 → 8 → 4 (each re-anchored to the same global field). A block fragmented at one scale is caught whole at a coarser one; region boundaries resolve below the single-pass 8-px granularity. → 99.63% → **99.89%**.

Both are gated behind the existing tiled path; `WHIRLWIND_NO_ANCHOR=1` reverts to the old single-f=8 anchorless vote for before/after. The CRLB path and all prior tests are byte-identical (anchor defaults to `None`).

**Important metric note.** The 99.xx% is measured on SNAPHU's `cc==1` mainland only (14.5 M of 25 M valid px). The remaining ~10.5 M px are genuinely-decorrelated low-coherence pixels where SNAPHU itself is uncertain; whirlwind and SNAPHU disagree pixel-by-pixel there (full-frame "match" ~70%) but that is *irreducible noise, not coherent artifacts* - it is not visible as blocks. So the mainland number is the right success metric, and the visual (phase diff RMS = 0.23 rad on the mainland) confirms it.

**Residual thin vertical lines** (the ones still faintly visible): they sit at columns spaced exactly 448 px = `tile_size − overlap` (the tile step), i.e. **tile seams** in moderate-coherence columns, where the correct cross-seam offset varies along the seam and a single per-tile integer can't satisfy it. A handful of others (e.g. col 959, coh 0.13) are fully-decorrelated columns where there is no signal - SNAPHU fills them with mottle, whirlwind with blocks; neither is "correct." 21 of whirlwind's 30 worst vertical-line columns also tear in SNAPHU. Candidate fix (not yet done): a feathered overlap composite or a seam-local heal pass.

## Atlanta S-1 OPERA - fine solve fails on noise; multilook-first recovers it

whirlwind's fine tiled unwrap produces **vertical stripes everywhere** on Atlanta (26% match, |dK|≥2 = 41%, 38 k fragmented components). This is **not** a tiling/scale problem and **not** bad input: on *identical* 5x-subsampled input, snaphu unwraps cleanly (97.89%) while whirlwind whole-image gets 11% and tiled gets 41%.

The decisive experiment: **decimation** (subsample, no averaging) by 5 → whirlwind 11%; **multilooking** (coherent block-average) by 8 → whirlwind whole-image **83%**, and **multilook-8 + tiled** → **97.66%** (matching snaphu's 97.89%). So:

- whirlwind's linear Carballo / unit-capacity cost cannot route correctly through Atlanta's **noisy moderate-coherence** phase (snaphu's statistical cost can);
- **multilooking suppresses that noise** (decimation does not), after which the same tiled+anchor+cascade pipeline reaches SNAPHU quality;
- the whole-image coarse solve (83%) still has residual runaway; **tiling the coarse** removes it (97.66%).

This validates the user's "multilook to constrain the large-scale features" intuition. L=8 is the sweet spot (L=16 over-aliases the ramp → 59%; L=4 keeps too much noise → 35%).

**Two real options for an Atlanta-class (noisy) product path:**
1. **Multilook-first mode** - multilook xL, run tiled+anchor+cascade, upsample. Already validated at 97.7% / sub-second. Cheapest; aliases sub-L fringes (fine for low-gradient scenes). Natural next step: expose as a `multilook=` kwarg on `unwrap`.
2. **Statistical / convex cost** (`convex_cost_diagnosis`) so the *fine* solve survives noise like snaphu's does - the general fix, larger effort.

## Reproduce

```
# build (maturin develop is broken on this env)
python -m maturin build --release
pip install --force-reinstall --no-deps target/wheels/whirlwind_insar-0.1.0-cp311-abi3-macosx_11_0_arm64.whl

# NISAR before/after + metrics + arrays
python scripts/phass_experiments/run/run_nisar_anchor.py        # anchor vs no-anchor
python scripts/phass_experiments/run/run_nisar_cascade.py       # + cascade
python scripts/phass_experiments/report/make_report_figures.py nisar

# Atlanta
python scripts/phass_experiments/run/run_atlanta_ml8.py         # multilook-8 whole vs tiled
python scripts/phass_experiments/run/run_atlanta_sub5.py        # 5x: snaphu vs ww whole/tiled
python scripts/phass_experiments/report/make_atlanta_report.py
```
