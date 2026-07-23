# The SSP-tail runtime investigation (2026-07-23)

Context: the 3005-frame provisional GUNW campaign (median per-component
ambiguity match 0.9999) showed a long runtime tail. This log records what the
tail is, which optimizations were tested against it, and what remains. All
timings are single-run `WHIRLWIND_TIMING=1` numbers on the same machine and
frame unless noted.

## What the tail is

- **77 MHz (`_7700_`) products are the tail**: 8.7% of campaign frames but
  21.1% of the 51.2 h total unwrap runtime (median 88 s vs ~50 s for
  20/40 MHz; p90 300 s; all top-15 slowest frames are 7700). They are not
  garbled data: quicklooks show real, near-aliased glacier/mountain fringes,
  and their match scores are still ≥0.98.
- **Campaign wall times are ~5.6x contention-inflated** (6 concurrent
  workers): frame 023_155 recorded 936 s in the campaign but solves in
  ~164 s alone. Treat `runtime_s` in campaign CSVs as throughput
  accounting, not single-frame benchmarks.
- **Anatomy of a slow solve** (023_155_D_053, 24 Mpx, valid 20%): the 8
  primal-dual passes take ~8 s and drain excess 5000→507; the serial
  single-source SSP tail then drains the remaining 507 sources one unit at a
  time in ~150 s — ~95% of the solve. Each expensive SSP search floods
  ~1/3 of the grid before popping its nearest deficit.

## Experiments (chronological)

| Change | Result on 023_155 solve | Verdict |
|---|---|---|
| `outgoing` reserve + spare-capacity writes (committed 9febc04) | 82.0→75.5 s on a 198-source frame A/B, byte-identical | **Kept** |
| `WHIRLWIND_PD_MAX_ITER`/`WHIRLWIND_PD_MIN_DRAIN`: extend PD past 8 passes (committed 3999c19, default-off) | 167→158 s (~5%) | Knobs kept, not a lever: per-pass augments collapse 33→19→…→2; PD only picks off the cheap sources |
| `open_water` mask (erode water 2 km, keep shores/rivers; committed 655843c) | 023_088: 223→245 s, per-comp match 0.984→0.992 | Quality option, **runtime-negative**: masked nodes remain zero-cost flood corridors — every pass still pops the full 24.3M grid at valid=0.109 |
| `visit_outgoing` closure visitor (trait + converted SSP/dial loops) | 164→216 s (and 227 s for a stack-array variant) | **Reverted**: LLVM does not dissolve the relaxation closure's captured environment; loop state falls out of registers |
| `#[inline(always)]` on `outgoing` | 161.5→160.0 s | **Reverted (no-op)**: the ~25% of `sample` hits inside `outgoing` are the intrinsic neighbor-generation work, not call overhead — inlining only re-labels them |

All variants were byte-identical (NaN-aware) on `ww_unw`/`ww_cc` and reached
the identical MCF objective; only speed differed.

## Conclusion

The SSP inner loop is now memory-bound (scattered `pd[v]` gathers + bucket
ops over a 24 Mpx graph); micro-optimization is exhausted. The remaining
levers are algorithmic:

1. **Masked-plateau compression.** Masked regions are zero-reduced-cost
   corridors that every flood traverses node-by-node; on ocean frames the
   flood visits ~10x more nodes than the valid domain. Contracting zero-cost
   masked components into supernodes (or restricting floods to the valid
   domain + gutter) attacks that multiple directly.
2. Bidirectional / deficit-side search to shrink each SSP flood.
3. The still-open Dial-SSP trajectory mismatch vs ww-orig
   (`docs/BUG_RIDGECREST_STRANDED_RESIDUES.md` lineage): ww-orig's SSP
   completes after 8 PD passes where ours needs hundreds of searches.

## Repro

```bash
# Slow-frame diagnostic (hardest campaign frames, local):
cd /Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_provisional
WHIRLWIND_TIMING=1 WHIRLWIND_DEBUG=1 python ~/repos/whirlwind-insar/aws-batch/compare_gunw.py \
  nisar_data_hardest/NISAR_L2_PR_GUNW_023_155_D_053_...h5 --out-dir out

# Eyeball whether a slow frame is garbled or genuinely hard:
python scripts/quicklook_gunw_hardest.py FILE.h5 --out-dir quicklooks
```

The glacier benchmark set (20 products + delivered `.rc.yaml` runconfigs, for
a later SNAPHU-with-production-parameters comparison) lives in
`.../nisar_provisional/nisar_glacier/`; whirlwind baselines accumulate in its
`ww_baseline/` as downloads complete.
