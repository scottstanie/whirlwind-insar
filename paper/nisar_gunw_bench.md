# NISAR GUNW benchmark — whirlwind vs production (2026-06)

Tool: [`scripts/bench_nisar_gunw_whirlwind.py`](../scripts/bench_nisar_gunw_whirlwind.py).
Downloads NISAR L2 GUNW beta products via `earthaccess` (`NISAR_L2_GUNW_BETA_V1`),
re-wraps the production 80 m `unwrappedPhase`, runs `whirlwind.unwrap(ig, coh,
nlooks, mask)`, and compares to the production unwrap. Records runtime, RSS,
ambiguity-match (global + per-connected-component), unwrap recall, conncomp
counts, and a 6-panel plot per frame. Data + plots:
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_expand/`.

## Methodology notes

- **Water masking (default `--mask-policy water_only`).** The GUNW `mask` layer is
  a 3-digit code `[water][subswath_ref][subswath_sec]`; we drop water (it's not a
  surface to unwrap) and keep everything else. Masking the *subswath-invalid*
  pixels too (`nisar_land`) carves vertical seams that trip the multi-shift gate
  (see #61), so the default keeps them. NISAR itself *fills* these gaps via
  `preprocess_wrapped_phase` before unwrapping.
- **Per-component match.** The absolute integer-2π level of a region isolated by
  water/decorrelation is unobservable, so we align ww to production *within each
  production connected component* before scoring accuracy. This separates
  "right shape within a region" from "guessed the same arbitrary inter-region
  offset" (the latter needs bridging, #63).
- **Recall.** Of pixels that have data, what fraction does each unwrapper label
  (`conncomp > 0`)? The precision/recall lens that ruled out ICU (rarely wrong,
  gave up easily).
- `--nlooks 16` is the azimuth looks; NISAR's true effective looks are higher
  (crossmul 5x6 / phase_unwrap 13x16). See `nisar_gunw_snaphu_settings` memo.

## Results (13 frames, cycle 003, full frame, water-masked)

Median per-component match **97.2%**; **ww recall 94.5% vs production 92.0%**
(whirlwind gives up *less* than NISAR's snaphu).

| frame | runtime | per-comp match | global match | recall ww / prod | ww_cc / prod_cc |
| ----- | ------: | -------------: | -----------: | ---------------: | --------------: |
| A_013 |   1.0 s |         100.0% |       100.0% |      98.6 / 99.5 |           1 / 1 |
| A_020 |   3.9 s |          99.8% |        99.6% |      99.4 / 98.4 |           1 / 1 |
| A_022 |   5.8 s |          99.8% |        99.5% |      99.0 / 98.1 |           1 / 1 |
| A_016 |   1.7 s |          99.7% |        96.9% |      99.1 / 98.9 |           8 / 3 |
| A_025 |   2.4 s |          99.0% |        98.8% |      91.1 / 92.0 |          12 / 2 |
| A_018 |   2.1 s |          98.3% |        95.7% |      86.0 / 81.4 |          13 / 1 |
| A_028 |   4.1 s |          97.2% |        92.6% |      88.1 / 89.1 |          31 / 1 |
| D_074 |   0.9 s |          96.1% |        96.0% |      93.2 / 96.9 |           9 / 1 |
| A_030 |   2.0 s |          92.3% |        90.5% |      77.3 / 81.8 |          71 / 3 |
| A_035 |   2.1 s |          68.9% |        72.3% |      94.5 / 67.7 |           5 / 2 |
| D_075 |  12.7 s |          54.3% |        50.1% |      90.2 / 64.4 |           3 / 1 |
| D_078 |  21.2 s |          50.1% |        49.6% |      97.6 / 93.6 |           1 / 1 |
| D_077 |  12.8 s |          48.1% |        47.1% |      95.8 / 82.8 |           1 / 2 |

## Worst offenders → tracked bugs

- **D_074/D_075/D_077/D_078 (descending steep ramps): #61.** The gated multi-shift
  re-solve mis-winds clean steep ramps when the valid mask has holes. On D_078,
  even 1.6% water holes push `coherent_cut_rate` 2.09e-3 over the 1.5e-3 floor →
  multi-shift fires (3 re-solves) → vertical banding, 0.99→0.50 match, 2.5→21 s.
  No holes (`not_127`) → gate off → 0.99, 2.5 s. The same gate correctly rescues
  fragmented A_016, so the fix is making it non-regressing / scene-aware.
- **A_035: #62.** Vertical single-pixel streak artifacts (recurring); the bounded
  `heal_thin_slivers` isn't catching them.
- **A_030 / A_028 / A_018: #64.** Connected-component over-segmentation vs NISAR
  (ww `min_size_px=100` vs snaphu `min_region_size=300`).
- **Inter-island offsets (A_016 residual): #63.** Port phase bridging to reconcile
  water/decorrelation-isolated components (adapt the tile-offset reconciliation).
