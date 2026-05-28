# Handoff: where things stand on the no-Goldstein unwrapping work

Last updated: 2026-05-28 evening.

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

Read those for context. Use this doc to find the *current* status
of each line of investigation.

## tl;dr — what works today, α=0 (no Goldstein)

| function | mechanism | PV K-match | NISAR K-match | wall (NISAR) |
|---|---|---:|---:|---:|
| `unwrap`                  | linear coherence, unit-capacity MCF      | 90.67 % | 80.01 % |  75 s |
| **`unwrap_reuse`**        | linear coh + PHASS-style flow reuse      | **99.75 %** | **92.70 %** |  93 s |
| `unwrap_convex`           | SNAPHU-style quadratic cost              | 99.80 % | 68.80 % | 409 s |
| `unwrap_grounded`         | linear coh + virtual-ground node         | (bad on real data — specialized) | — | — |
| `unwrap_crlb_*`           | CRLB-variance cost variants              | (designed for phase-linked stacks) | — | — |

References (same NISAR scene):
* SNAPHU 9×9 tiled: K-match definition, ~17 min wall.
* `dolphin --unwrap-method PHASS`: 97.93 % K-match, ~60 s, no Goldstein.
* whirlwind `unwrap` + `--goldstein-alpha 0.7`: **99.90 %**, **38 s**.

**The production answer remains Goldstein α=0.7** — empirically the
fastest path to SNAPHU-quality unwraps. The work documented below is
the research effort to get there *without* Goldstein.

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

Runner: `scripts/phass_experiments/run_atlanta.py <mode>`.

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
