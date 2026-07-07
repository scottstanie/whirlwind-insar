# Ridgecrest block-tear bug: stranded residues → stacked 2π integration tears

**Status (2026-07-07): ROOT CAUSE FOUND AND FIXED.**
`run_single_source`'s Dial bucket-advance exit tested **cumulative** empty
advances over the whole search instead of **consecutive** ones (the three
backends in `dial.rs` all reset the counter on every non-empty bucket; the SSP
copy did not), so any search whose distance range spanned more than
`k = max_rc + 1` total empty skips self-terminated with live entries still
queued — stranding exactly the long-range leftover residues. One-line fix
(`bucket_advances = 0;` after the empty-scan, matching `dial.rs`) plus a
regression test (`ssp::single_source_tests::pairs_a_distant_sink_beyond_one_bucket_cycle`,
verified to fail on the unfixed code). Results with the fix alone (BFS guard
disabled): crop 64.0% → **97.4%**, zero stranding, zero remaining excess; full
frame 90.3% → **99.43%** in 164 s (block eliminated, conncomps 20 → 1,
vs SNAPHU-5×5's 99.81% self-recovery in 1301 s). Runtime rose 98 → 164 s
because the formerly-stranded searches now complete — the cost of doing the
work rather than silently skipping it.

A `-C debug-assertions=yes` build of the repro never fired the
`rc >= 0` assert, so reduced costs stay non-negative throughout: the earlier
"invalid potentials from the belt PD" hypothesis was **wrong** and has been
corrected in the code comments. The cost-ignoring residual-BFS balance guard
(`ssp::drain_residual_bfs`, first commit of this PR) is retained as a safety
net — the network must never be integrated unbalanced — but is now a no-op on
this scene. Disable it with `WHIRLWIND_NO_BFS_DRAIN=1`.

**Earlier guard-only status:** BFS guard alone gave full frame 90.3% → 97.75%,
crop 64.0% → 93.9% (fewest-hops pairing, small local wedge). Superseded by the
root-cause fix above, which lets the cost-aware SSP pair everything optimally.

**Original diagnosis (2026-07-07).** Found while benchmarking the
paper's Sentinel-1 rewrap-recovery test. Whirlwind offsets a 3.4-Mpixel
far-field block by −19 cycles on OPERA DISP-S1 F16941 (74 Mpixel, 2019
Ridgecrest coseismic pair). This is **not** cost-model behavior, aliasing
ambiguity, or bridging — it is incomplete residue pairing in the solver.

## Symptom

Rewrap the production displacement of
`OPERA_L3_DISP-S1_IW_F16941_VV_20190529T015026Z_20190710T014947Z` and unwrap
(corr = `estimated_phase_quality`, L=18, water+finite mask). Whirlwind returns
90.3% per-component agreement; the disagreement is one far-field block north
of the M7.1 rupture offset by exactly −19 cycles, bounded by thin horizontal
stripe bands (which `components_snaphu` flags as strip components).

## Evidence chain

1. **Objective accounting** (reconstructed parity costs from the embedded LUT,
   validated by `unwrap_linear_ext_costs` reproducing the native output at
   98.7% of pixels): whirlwind's returned solution costs **~2× (full frame) to
   6–8× (crop)** the *production solution charged under whirlwind's own cost
   model*. No zero-cost valid arcs exist anywhere (min-direction cost on
   differing arcs: median 300–400). So the returned flow is far from optimal —
   the tear is not a cheap-corridor artifact and not a degenerate tie.
2. **`WHIRLWIND_DEBUG=1` on a 2400×6509 crop** (rows 800:3200, cols 3000:9509
   of the cached inputs; 18 s):
   - PD(8) drains 16,526 → 116 residues.
   - Single-source SSP: **49 of 116 sources STRANDED** — each floods a
     100–400K-node residual pocket containing **zero** of the ~109 deficits.
     (The `ssp.rs` comment itself: in a balanced network this is "a
     REACHABILITY/SSP BUG, not expected control flow." Flow decomposition
     against any feasible completion guarantees an augmenting path exists, so
     residual-arc traversal is incomplete somewhere.)
   - Adaptive PD resume recovers some, then stalls:
     `ADAPTIVE FINAL remaining_excess=40` (dial) / `56` (heap).
3. **The unbalanced network is then integrated anyway.** Per the
   `unwrap_linear` comment, each unpaired residue becomes a full-width 2π tear
   in the row-major integration. The block offset equals the unpaired-pair
   count exactly:
   | backend | unpaired pairs | block offset |
   |---|---|---|
   | dial, full frame | 19 | −19 cycles |
   | dial, crop       | 20 | −20 cycles |
   | heap, crop       | 28 | −28 cycles |

## Why here and not on the NISAR frames

The coseismic near-fault belt is residue-dense with high-cost walls on all
sides; after PD saturates the cheap crossings, the leftover excess nodes sit
in residual pockets the SSP search cannot escape. The 13 NISAR frames leave
zero (or near-zero) stranded residues, so the defect is invisible there.
(D_075's 88.2% should be re-checked for `STRANDED`/`remaining_excess` after
the fix.)

## Where to look / fix sketch

- Root cause: residual reachability. Audit reverse-arc availability in
  `network.rs` / `residual_graph.rs` for every arc class (grid forward/reverse,
  boundary gutter ring, ground arcs) under the saturation bookkeeping;
  `debug_assert!(rc >= 0)` in `ssp.rs` is compiled out in release, so run a
  debug build on the repro to see if the invariant fires.
- Independent guard regardless of root cause: **never integrate an unbalanced
  network silently.** If `remaining_excess > 0` after the adaptive resume,
  pair the leftovers by any complete method (direct residual BFS ignoring
  cost, or cost-capped matching) — a suboptimal pairing costs a local error;
  an unpaired residue costs a full-width tear.
  **This is what shipped** (`ssp::drain_residual_bfs`): a plain FIFO BFS over
  the residual graph gated only on `is_arc_saturated`, augmenting the
  fewest-hops path per leftover unit. It cannot strand on a connected balanced
  graph (which `unwrap_linear` always builds — kept frame deposits ⇒ residue
  sum 0, no forbidden arcs), so it guarantees `remaining_excess == 0`. Runs
  only when the cost-aware passes leave excess, so it is a no-op on frames that
  already drain (NISAR parity untouched). Follow-up to shrink the residual
  near-fault wedge: cost-aware completion (cycle-cancel on the now-balanced
  flow) or the real root-cause fix in the Dial SSP.

## Repro

```bash
# inputs cache (already built): .../s1-testing/ridgecrest/rewrap_bench/inputs_full.npz
V=.venv/bin/python
B=/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/s1-testing/ridgecrest/rewrap_bench
WHIRLWIND_DEBUG=1 $V $B/repro_crop.py 2> debug.log      # dial, 18 s, match=64%, −20 block
WHIRLWIND_DIJKSTRA=heap $V $B/repro_crop.py             # −28 block
grep -E 'STRANDED|ADAPTIVE FINAL' debug.log
# full-frame benchmark + figure: whirlwind-paper/scripts/s1_ridgecrest_rewrap.py
# objective accounting: $B/diag_cost_accounting.py
```
