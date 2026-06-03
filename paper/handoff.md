# Handoff: where things stand on the no-Goldstein unwrapping work

Last updated: 2026-06-03 (single-tile Carballo parity + SSP-runtime saga).

This is the entry point for anyone picking up the no-Goldstein
unwrapping investigation. It's intended to (a) keep someone from
re-running experiments we already did and (b) make it easy to see
which leads we *haven't* fully exhausted, so the search continues
forward rather than in circles.

The two longer-form companion writeups are:
* `paper/phass_experiments.md` — PHASS-style cost-shape experiments,
  the reuse prototype, hard-cut tests, and the path-dependence ruling-out
  experiment that pointed us at convex cost.
* `paper/convex_cost_design.md` — the SNAPHU-style convex prototype
  design plan and first-run results.
* **`paper/tiling.md`** — the **tiled coherence unwrap** (now the
  fast/low-memory no-Goldstein path, NISAR 96.6 % in 3.5 s), plus
  source-verified *corrections* to the convex/reuse/unit-capacity
  conclusions below, and the corrected Atlanta status. **Read this first
  for the current best path.**

Read those for context. Use this doc to find the *current* status
of each line of investigation.

## 2026-06-03 update: single-tile Carballo parity + the SSP-runtime saga

Canonical, code-verified status + benchmark table now lives in
**`ATBD-whirlwind.md` §9.6**. This section is the *don't-repeat* log for the
several circles we burned getting there.

**What now works (verified).** Single-tile `unwrap_linear` matches Python
`ww-orig` at **99.49 %** on full D_077, via embedded ww-orig Carballo spline
tables (trilinear, `cost/spline_lut.rs`, `WHIRLWIND_CARBALLO_LUT_DIR` override)
+ full-completion Dijkstra. On D_077 it beats single-tile SNAPHU on *both* axes:
**158 s / 99.49 %** vs SNAPHU **588 s / 99.30 %** (PHASS 19.6 s / 94.7 %). Peak
RSS **6.4 GB**. Merged: PR #66 (parity), #67 (LUT override + buffer reuse).

**Red herrings — do NOT re-run / re-conclude these:**

1. **"Single-tile is slow because it swaps / OOMs." FALSE.** `/usr/bin/time -l`
   on the 6.4 GB D_077 run reported **0 swaps, 565 page faults** — it does not
   swap. A 1472 s wall-time was misread as swap twice; it was (a) CPU contention
   from running `cargo`/`maturin` *concurrently* with the measurement (starved
   rayon → ~1.4× parallelism instead of ~13×), and (b) the SSP algorithm (below).
   *Lesson:* measure peak RSS with `/usr/bin/time -l` (default in benches), use
   `mprof` only to chase spikes, and never run other heavy jobs during a timing
   run.

2. **"unwrap_linear's runtime is fine / 158 s on `main`." FALSE as committed.**
   The 158 s used the *single-source* SSP that was later reverted; `main`'s
   *multi-source* `ssp::run` makes single-tile D_077 ≈**1472 s** (slower than
   SNAPHU). See item 4.

3. **"`run_full_dijkstra_no_ssp` = Python parity (8 PD then integrate)." FALSE.**
   No-SSP scores **11.19 %** on D_077: the 8 PD iterations alone route only ~11 %
   of the flow; the **SSP fallback does the bulk**. `unwrap_linear` *requires*
   SSP. (Our PD under-routes vs Python's per-iteration; SSP makes up the rest.)
   A `run_full_dijkstra_no_ssp` variant was added and then reverted as useless.

4. **"The single-source SSP rewrite was a perf regression — revert it globally."
   WRONG (this was the biggest circle).** The two SSP algorithms have *opposite*
   tradeoffs by scenario:
   * old **multi-source** `ssp::run`: fast on tiled many-small-leftover
     (A_016 1.7 s) but **catastrophic on whole-image** (D_077 1472 s — a
     near-whole-image Dijkstra *per single unit*);
   * new **single-source** SSP: fast on whole-image (D_077 158 s) but slow on
     tiled (A_016 145 s).
   Since the **product is single-tile**, reverting the single-source SSP was a 9×
   regression there. Don't pick one globally.

**Open / next: dual-SSP.** Keep multi-source `ssp::run` for the early-exit
(`run`, tiled/default) path; add `ssp::run_single_source` used only by
`run_full_dijkstra` (single-tile). Correctness caveat (the trap behind the
earlier clamp): the single-source potential update must keep reduced costs
non-negative *after every early-exit Dijkstra*, not just at SSP entry — popped
nodes get exact distance, unpopped are implicitly ≥ the sink distance by Dijkstra
pop order. Acceptance bar: D_077 **158 s / 99.49 %**, convex + tiled tests
unchanged, and `debug_assert!(rc ≥ 0)` **on — no clamp** — must never fire during
single-source SSP. (Validate the assertion via a focused *debug* test on a
moderate noisy ramp through `run_full_dijkstra`; a debug full-frame D_077 is too
slow.)

**Also note:** `ssp.rs` module doc-comment still says "single-source" but the
code is multi-source-seeded (`dijkstra_multi_source_into`) + one augmentation —
fix that comment as part of the dual-SSP work.

## tl;dr — what works today, α=0 (no Goldstein)

| function | mechanism | PV K-match | NISAR K-match | wall (NISAR) |
|---|---|---:|---:|---:|
| `unwrap`                  | linear coherence, unit-capacity MCF      | 90.67 % | 80.01 % |  75 s |
| **`unwrap` tiled ts=512** | per-tile MCF + consensus 2π stitch       | — | **96.6 %** |  **3.5 s** |
| **`unwrap_reuse`**        | linear coh + PHASS-style flow reuse      | **99.75 %** | **92.70 %** |  93 s |
| `unwrap_convex`           | SNAPHU-style quadratic cost (BUGGY — see `tiling.md`) | 99.80 % | 68.80 % | 409 s |

**Update (session 2): the tiled path (`unwrap(..., tile_size=512,
tile_overlap=64)`) is the new fast/low-memory no-Goldstein winner and
*beats* whole-image. The `unwrap_convex` 68.8 % is an unsound-solver
artifact, not a verdict on convex cost; the convex `offset` is also the
wrong quantity. See `paper/tiling.md` for the full corrected picture.**
| `unwrap_grounded`         | linear coh + virtual-ground node         | (bad on real data — specialized) | — | — |
| `unwrap_crlb_*`           | CRLB-variance cost variants              | (designed for phase-linked stacks) | — | — |

References (same NISAR scene):
* SNAPHU 9×9 tiled: K-match definition, ~17 min wall.
* `dolphin --unwrap-method PHASS`: 97.93 % K-match, ~60 s, no Goldstein.
* whirlwind `unwrap` + `--goldstein-alpha 0.7`: **99.90 %**, **38 s**.

**The production answer remains Goldstein α=0.7** — empirically the
fastest path to SNAPHU-quality unwraps. The work documented below is
the research effort to get there *without* Goldstein.

## Has any of this slowed down the production path?

**No.** The reuse / convex / grounded prototypes are all new
top-level entry points (`unwrap_reuse`, `unwrap_convex`,
`unwrap_grounded`); the original `unwrap` / `unwrap_with_components` /
`unwrap_crlb_*` functions are byte-identical in their call paths
because all the conditional logic for the prototypes goes through
`reuse_mode` / `convex_mode` flags on `Network` that default to
false and short-circuit out in the existing default paths.

The original production path (Carballo cost + Goldstein α=0.7 + Dial
bucket-queue Dijkstra + unit-capacity MCF) still clocks **38 s on
NISAR for 99.90 % K-match** — unchanged from PR #19.

Per-prototype wall times (NISAR 6811×6912, 25M valid px):

| function | wall | × baseline | notes |
|---|---:|---:|---|
| `unwrap`         |  75 s | 1.0 × | linear coh cost, unit-cap MCF, Dial |
| `unwrap`+Goldstein α=0.7 | 38 s | 0.5 × | *faster* than no Goldstein (filter shrinks the residue field) |
| `unwrap_reuse`   |  93 s | 1.2 × | linear coh + multi-unit flow + reduced-cost-0 on used arcs |
| `unwrap_convex`  | 402 s | 5.4 × | quadratic cost, heap backend (Dial bucket vec would be GB-scale on convex weights) |

The convex prototype is the only one with notable slowdown. Two
contributing factors:
* Convex cost lives on much wider integer range (max marginal ≈
  10⁷ vs ~300 for Carballo). The Dial bucket-queue Dijkstra needs
  `K+1` buckets where K = max reduced cost — physically impossible
  at convex magnitudes. The `shortest_path::dijkstra_multi_source`
  dispatcher therefore forces convex networks to the binary-heap
  backend, which is O(E log V) vs Dial's O(V + E + K) and slower
  per-edge in practice.
* The Lee 1994 variance LUT replacement for Just/Bamler (committed
  earlier as the σ² fix) gave a side-effect 90× speedup on the
  synthetic diagonal-ramp test (91 s → 0.97 s) because the weight
  magnitudes are now more sensible. So the convex runtime is
  *much* better than the first-cut implementation.

The reuse prototype runtime tax (1.2× on NISAR, 5× on PV) is the
honest cost of multi-unit flow without saturation. If we ever ship
reuse as a non-prototype path, profiling its per-relax overhead is
the obvious cleanup.

## What has been ruled out (don't re-run these)

### Cost-shape tweaks confined to the linear unit-capacity SSP

All these were tried on top of `unwrap` (linear coh cost, unit-capacity)
in 2026-05 PHASS experiments. None close the no-Goldstein gap on NISAR:

* **Hard cuts** (`WHIRLWIND_HARD_CUT_THRESH` = 1.0 and 2.0 rad). 1.0
  is pathological in our Dial bucket queue (PV: 472 s). 2.0 hurts NISAR
  (-6.5 pp). Mechanism documented in `paper/phass_experiments.md`.
* **PHASS γ² (flat-clamp)** (`WHIRLWIND_PHASS_COST`). Symmetric in
  arc direction → degenerate tie-breaking in our LP. NISAR 67.45 %.
* **Faithful PHASS γ²·100 + 255-cliff** (`WHIRLWIND_PHASS_FAITHFUL_GOOD_CORR`).
  Pathological — PV killed at >13 min vs 0.7 s baseline. Even with reuse
  the 255-cliff overwhelms Dial / heap. *Confirmed not solver-recoverable
  in our architecture.*

### Path-order / tie-breaking as cause of NISAR residual error

After the reuse prototype landed at 92.7 % on NISAR, the suspicion was
that the residual 5 pp gap to dolphin PHASS was a *path-dependence*
artifact — reuse locks in a wrong "highway." Tested three Dijkstra
backends with materially different tie-breaking (serial Dial / parallel
Dial / binary heap). Results identical to 0.005 %. **Not path-order.**
The reuse 92.7 % solution is the genuine cost-optimum of the linear
coherence-cost model.

### Convex-cost knobs (suspects 1, 2, 5 from `convex_cost_design.md`)

The convex prototype regresses on NISAR (92.7 % → 68.8 %). Three
plausible knobs tested, none move it:

* **σ² calibration**: Just/Bamler vs full Lee 1994 numerical variance.
  NISAR 68.55 → 68.80 %. Ruled out.
* **Offset polarity** (`WHIRLWIND_CONVEX_OFFSET_FLIP`). Identical
  result to default. Ruled out.
* **Raw vs smoothed gradient for offset input**
  (`WHIRLWIND_CONVEX_OFFSET_RAW`). NISAR 68.80 → 68.98 %. Same
  ~1M-pixel +4-cycle blob in the upper-right. Ruled out.

The diagnostic also showed NISAR's max offset is 22 (vs ±50 saturation
point) even with raw gradients, because the 7×7 box smoothing or the
nshortcycle=100 scaling doesn't capture wrap-line geometry on this scene.
But the *answer* doesn't depend on this — so it's not the cause of the
regression.

## What's actually still open

### Convex prototype solver correctness (suspect 4)

The remaining hypothesis: **standard SSP potential update doesn't
maintain non-negative reduced costs across iterations in convex mode**.
The Phase-4 analysis in `convex_cost_design.md` claimed it does; the
diagonal_ramp_512_convex test passes in DEBUG mode with the
`debug_assert(rc >= 0)` enabled, so it's not violated on simple
synthetic. But the four-thousand-times-larger NISAR scene with realistic
weight ranges may silently trip negative reduced costs that release
builds drop.

How to test:
1. Compile the e2e harness in DEBUG mode (`cargo test --test end_to_end
   ...`) and run a small but realistic real scene (PV is small enough,
   ~8 s in release). If the assert fires, suspect 4 is real.
2. Or: add a release-build counter for `rc < 0` events in the relax
   sites and run NISAR convex once with `WHIRLWIND_DEBUG=1`.

If suspect 4 is confirmed, the next prototype direction is a
**Bellman-Ford / SPFA pre-pass after every primal-dual iteration**
to re-establish valid potentials. Standard convex MCF requires this
in the general case.

### Atlanta S-1 scene (in progress)

A third real scene was added late on 2026-05-28: OPERA Atlanta
Sentinel-1 frame at `/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar/opera.{int.phs,int.cor,displacement,conncomp}.tif`.
~78 M px (bigger than NISAR), ~39 % NaN, median coh 0.57, displacement
range ±0.09 m. Provides:
* a third unwrap reference (the displacement file is the SNAPHU-derived
  OPERA solution),
* a *with-NaN* test that exercises the mask-aware code,
* a different SAR system (S-1 C-band, λ ≈ 5.6 cm) for cross-checking
  that conclusions generalize beyond NISAR.

Runner: `scripts/phass_experiments/run/run_atlanta.py <mode>`.

Results pending; see `outputs/atlanta_*.npz` once runs complete.

### Other unexhausted lanes (high-cost prototypes)

Not yet started:

* **Parallel-arc decomposition for convex MCF**. Replace each
  parabolic-cost arc with N parallel unit-capacity linear arcs of
  increasing marginal cost. Sidesteps the in-place-convex-cost
  potential issues entirely. Inflates arc count by N (typically 3-5)
  but is the textbook convex-MCF reduction. If suspect 4 is real and
  the Bellman-Ford pre-pass is too painful, this is plan B.
* **PHASS-style branch-cut detection separate from cost**. Use the raw
  gradient field to identify wrap-line *arcs* (long aligned runs of
  `|wrap(Δφ_raw)| ≈ π`), and zero only those (a smarter hard cut than
  per-arc thresholding). Lighter than convex but bespoke.
* **Iterative re-cost SSP** (Klein cycle-cancellation). Standard
  technique for convex MCF that handles negative residual cycles.
  More invasive than Bellman-Ford pre-pass.

## Where the code lives

* `crates/whirlwind-core/src/lib.rs`: top-level `unwrap_*` functions.
* `crates/whirlwind-core/src/cost/mod.rs`: `compute_carballo_costs`,
  `compute_snaphu_smooth_costs`, env knobs:
  `WHIRLWIND_LLR_COST`, `WHIRLWIND_DEVIATION_COST`,
  `WHIRLWIND_HARD_CUT_THRESH`, `WHIRLWIND_PHASS_COST`,
  `WHIRLWIND_PHASS_FAITHFUL_GOOD_CORR`,
  `WHIRLWIND_CONVEX_OFFSET_FLIP`, `WHIRLWIND_CONVEX_OFFSET_RAW`,
  `WHIRLWIND_DIJKSTRA`.
* `crates/whirlwind-core/src/cost/lut.rs`: Lee PDF & variance LUTs.
* `crates/whirlwind-core/src/network.rs`: `reuse_mode`, `convex_mode`,
  `flow_count`, `is_used`, `marginal_cost`, `new_reuse_with_mask`,
  `new_convex_with_mask`.
* `crates/whirlwind-core/src/shortest_path/{dial.rs,heap.rs}`: relax
  sites have `is_used` (reuse) and `convex_mode` (marginal_cost)
  branches. Convex networks always route to `heap` regardless of
  `WHIRLWIND_DIJKSTRA`.
* `scripts/phass_experiments/`: one `run_<variant>.py` per cost knob,
  `analyze.py` for K-quality tables, `plot_*.py` for K-field panels.

## Plot files (verify visually)

Comparison panels saved under
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots/`:
* `nisar_reuse_panel.png` — SNAPHU / baseline / reuse K-fields + Δ K.
* `pv_reuse_panel.png` — same for PV.
* `nisar_convex_panel.png` — adds the convex variant to the NISAR comparison.
* (Atlanta plot to be added once runs complete.)

## Decision rule for the next round of investigation

If your goal is to *close the no-Goldstein NISAR gap*: start with
suspect 4 (Bellman-Ford pre-pass for convex). If that doesn't fix it,
the parallel-arc decomposition is the next step.

If your goal is *production unwrapping quality today*: the Goldstein
α=0.7 default is already at 99.90 % vs SNAPHU 9×9 on NISAR in 38 s.
There is no current production reason to pursue the convex track.

If the scientific framing for a paper is the priority: the reuse-vs-
unit-cap improvement (80 % → 92.7 % on NISAR, 90.67 % → 99.75 % on PV,
and `diagonal_ramp_512` going from fail to perfect) is already a
defensible result by itself.
