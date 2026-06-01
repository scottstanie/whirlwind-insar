# PHASS-style cost variants in whirlwind, tested

Companion to `paper/different_vs_snaphu_costs.md`. That doc explains *why*
whirlwind's linear-cost SSP can't reproduce SNAPHU's smooth-cost
unwrap. This doc tests whether a different *cost shape* — borrowed
from PHASS — closes the gap without changing the solver.

**Bottom line — two answers, depending on what you ask:**

* **Our PHASS-flavoured *cost* in our solver: no.** On both NISAR (47 M
  px, snaphu_9×9 reference) and Palos Verdes (Capella C13_SP, 750 k
  px), none of the PHASS-style cost variants at α=0 (no Goldstein)
  beats our Carballo cost by more than 1 pp in K-agreement, and most
  are noticeably worse.
* **PHASS as an algorithm (run via dolphin's binding): yes.** Dolphin's
  `--unwrap-method PHASS` at α=0 hits **97.93 % K-match with SNAPHU 9×9
  on NISAR in ~1 minute** (29 % conncomp coverage, 53 components).
  That's within 2 pp of SNAPHU at zero filtering, using the actual ISCE3
  PHASS C++ code we read in `~/repos/isce3/cxx/isce3/unwrap/phass`.

So the gap at α=0 is **not** in the PHASS *idea* — it's specifically in
*our reimplementation* of pieces of that idea on top of a different
solver. Goldstein remains the working lever for whirlwind-α=0 today,
but PHASS-as-shipped already shows that a cost-side-only fix is
achievable; the open question is reproducing it inside whirlwind's SSP
without inheriting the degeneracies we hit below.

## What we tested

Whirlwind α=0 with four cost variants (toggled via env vars in
`crates/whirlwind-core/src/cost/mod.rs`):

| mode         | env                                                | what it does |
|--------------|----------------------------------------------------|---|
| `baseline`   | (none)                                             | Default Carballo `γ·(π − α_smooth)` |
| `hard_cut`   | `WHIRLWIND_HARD_CUT_THRESH=2.0`                    | Plus: any arc with `|wrap(Δphase_raw)| ≥ 2.0` is forced to cost = 0 (PHASS-style cut, but at 2.0 rad — see below). |
| `phass_cost` | `WHIRLWIND_PHASS_COST=0.5`                         | Replace cost with `γ² · π` saturated at `0.5² · π` (PHASS coh-only, no α term). Conncomp threshold lowered 4× to match the lower magnitudes. |
| `phass_full` | both env vars set                                  | PHASS cost + cuts. |

The conncomp `cost_threshold` is scaled by 0.25 for `phass_cost` /
`phass_full` because PHASS γ² magnitudes (≈ 78 at γ_sat) are about 3-4×
smaller than Carballo (≈ 250 in a coherent ramp); without rescaling,
every component is rejected as "below threshold" and coverage is 0%.

### Why the hard-cut threshold is 2.0, not PHASS's 1.0

PHASS uses `phase_diff_th = 1.0 rad` (`PhassUnwrapper.cc:71`). Tried that
first on PV: **runtime exploded from 0.7 s → 472 s, K-agreement collapsed
from 90.7 % → 47.3 %.** Hard cuts at 1.0 rad fire on too many noisy arcs
in coherent regions, creating a vast zero-cost subgraph that the SSP
solver has to chew through and that re-routes wraps through random noise
locations. 2.0 rad fires only on near-wrap-line gradients (~60 % of π)
and is closer to a meaningful "this arc looks like a real wrap" signal.

Kept the 1.0-rad PV result on disk (`pv_hard_cut_lo.npz`) as a documented
slow/bad case.

## Results

### Palos Verdes (Capella C13_SP `20251129_20251205`, 871×864)

K-agreement is on SNAPHU's `cc=1` mainland (191 478 px, 25 % of frame).
SNAPHU smooth single-tile is the reference, run in 12.3 s.

| mode | wall | K=match % | `|dK|`=1 % | `|dK|`≥2 % |
|---|---:|---:|---:|---:|
| default Carballo            |   0.7 s | **90.67** | 1.09 | 8.25 |
| Carballo + cut @2.0 rad     |   0.6 s | **91.64** | 0.53 | 7.83 |
| Carballo + cut @1.0 rad     | 472.2 s | 47.34 | 12.17 | 40.49 |
| PHASS γ²                    |   0.9 s | 88.39 | 4.27 | 7.34 |
| PHASS γ² + cut @2.0         |  86.6 s | 76.24 | 2.55 | 21.21 |

### NISAR (HH 50 m posting `20251224_20260117`, 6811×6912)

K-agreement vs the saved SNAPHU 9×9 reference (`cc=1` mainland: 14.5 M
px, 31 % of frame; SNAPHU wall: 17 min, 9×9 tiling).

| mode | wall | K=match % | `|dK|`=1 % | `|dK|`≥2 % |
|---|---:|---:|---:|---:|
| default Carballo            |  75.0 s | **80.01** | 1.71 | 18.28 |
| Carballo + cut @2.0 rad     |  69.8 s | 73.55 | 8.26 | 18.18 |
| PHASS γ²                    |  92.5 s | 67.45 | 2.30 | 30.26 |
| PHASS γ² + cut @2.0 (skipped) | — | — | — | — |
| **dolphin `--unwrap-method PHASS`** | ~60 s | **97.93** | 2.07 | 0.00 |
| Goldstein α=0.7 (PR #19)    |  38 s |  **99.90** | — | — |

The last row is from earlier work — same NISAR scene with the production
default of `--goldstein-alpha 0.7`. **38 s and 99.9 % match.** With
Goldstein on, whirlwind beats SNAPHU's 17 min wall by 27× while
agreeing with it pixel-for-pixel on the cc=1 mainland.

### Visual evidence

`/Volumes/.../phass_experiments/plots/{pv,nisar}_k_panel.png` shows the
K-fields side by side per scene. SNAPHU's leftmost panel is the smooth
clean reference; the α=0 whirlwind variants either have tiny conncomp
coverage (default Carballo, hard_cut) or large-but-wrong K patches
(PHASS γ² gets 20 % NISAR conncomp coverage but blocky misroute
patterns instead of the smooth SNAPHU ramp).

### dolphin PHASS observation

We ran dolphin's PHASS binding on the same NISAR scene as a sanity check
on the PHASS *algorithm*, separate from our whirlwind reimplementation:

```bash
dolphin unwrap --ifg-filenames .../20251224_20260117.int.looked.tif \
               --cor-filenames .../20251224_20260117.int.coh.looked.cleaned.tif \
               --output-path .../phass_experiments/dolphin_phass \
               --nlooks 100 \
               --unwrap-options.unwrap-method PHASS
```

Completed in roughly 60 s on this laptop, no Goldstein, and produced
**29.41 % conncomp coverage** (53 components, vs SNAPHU's 1 component
at 30.84 %) with **97.93 % K-match against SNAPHU 9×9** on the cc=1
mainland (after a uniform −5 cycle global offset — expected, since
PHASS doesn't pin to any particular K=0 anchor). `|dK|=1: 2.07 %`,
`|dK|≥2: 0.00 %` — no multi-cycle misroutes anywhere.

Plot: `plots/nisar_dolphin_phass_vs_snaphu.png`. The mountain region
shape and the surrounding ramp match SNAPHU's pattern; the conncomp
count is much higher because PHASS does region-grow seeding rather than
single-component recovery, which is the *point* of PHASS (lakes for
SWOT) and a separate concern from K-agreement.

## Why none of *our* variants help

The PHASS γ² (coh-only, no α term) cost is **direction-symmetric**:
`cost_dir(α, γ) = cost_dir(−α, γ) = γ² · π`. The whirlwind cost docs
already warn about this: a symmetric per-arc cost is degenerate under
our LP (`crates/whirlwind-core/src/cost/mod.rs:296`), and "arbitrary
tie-breaking flips the unwrap topology under tiny input perturbations."
PHASS gets away with it because amplitude-edge cuts and phase-gradient
cuts break the ties geometrically; we don't have an amplitude-edge
detector and the phase-gradient cuts alone aren't enough.

Hard cuts (`γ_edge=0` at high `|dpsi|`) add per-arc cheap channels, but
they fire on isolated noise arcs too — and an isolated cheap arc in an
otherwise coherent region just gives MCF a free place to dump residue
pairs that have no business going there. We saw this in the
[[different-vs-snaphu-costs]] *deviation* experiment too; same failure
mode, same explanation. PHASS doesn't suffer from this as badly because
its coh-only cost is *saturated* — every coherent arc has the same flat
high cost, so a single arc dropping to 0 is uniquely cheap. In our
Carballo cost the surrounding γ·(π − α) values vary continuously, so
a cut arc isn't structurally distinguishable.

## So, is Goldstein necessary?

**For whirlwind today: practically, yes.** Cost-shape tweaks confined to
our linear unit-capacity SSP do not close the no-Goldstein gap on either
scene we tested.

**For phase unwrapping in general: clearly no** — both SNAPHU and PHASS
match (each other and themselves) at α=0 without any Goldstein
prefilter. The dolphin-PHASS row above demonstrates this directly: at
α=0, PHASS 97.93 % vs SNAPHU within 2 pp. The Goldstein gain in
whirlwind is *implementation-specific*, not algorithmically fundamental.

What we now know matters:

1. The structural cost-shape gap discussed in
   [[different-vs-snaphu-costs]] (linear vs convex per-arc cost) is the
   load-bearing piece of why magnitude-matched ports fail.
2. Direction-symmetric coh-only costs are degenerate in our solver
   without geometric tie-breakers (amplitude edges, asymmetric per-arc
   gradient signal).
3. Hard cuts at small thresholds (PHASS's 1.0 rad default) blow up our
   SSP runtime because they create huge zero-cost subgraphs.

Closing the gap inside whirlwind would mean either:

* Adding asymmetric tie-breaking to a coh-only base cost — most likely
  a Canny-on-amplitude branch-cut prior, the way PHASS actually does
  it. We don't have an amplitude path today.
* Promoting the solver to convex per-arc cost (iterative-recost SSP or
  Goldberg parallel-arc reduction). Neither is prototyped here.
* Keeping Goldstein and shipping the PR #19 default. **38 s, 99.9 %
  K-match with SNAPHU 9×9** — that's the working answer today, and the
  only PHASS-class number we beat empirically.

If a user truly needs no-Goldstein at scale, the practical
recommendation right now is `dolphin unwrap --unwrap-method PHASS`
rather than whirlwind α=0; that's a directly-validated path on the same
NISAR scene.

## 2026-05-28 follow-up: the reviewer's three suggestions

A second pair of eyes (transcripted into [[recent_phase_unwrapping_skeptical_review]])
spotted three concrete things that this writeup hadn't fully ruled out.
Each was tested.

### 1. The whirlwind PHASS cost port was *unfaithful*. Faithful port: pathological.

ISCE3 PHASS (`PhassUnwrapper.cc:119-141`) uses `γ²·100` for low-coh edges
and **jumps to 255** for `γ² > good_corr²` — a step function, not a
flat clamp. Whirlwind's port (`cost/mod.rs:cost_dir`) instead capped the
high-coh tail at `good_corr²·π` (flat ceiling). The reviewer expected
restoring the 255-cliff to close most of the gap.

A faithful port was implemented (γ²·100 base; cost=2.55 above good_corr,
which times COST_SCALE=100 reproduces the 255 PHASS emits) and tested:

| scene | mode | wall | result |
|---|---|---:|---|
| PV (750k px, baseline 0.7 s)   | γ²+255-cliff, no cuts | **>14 min, killed** | pathological |
| NISAR (47M px, baseline 75 s)   | γ²+255-cliff, no cuts | **>17 min, killed**  | pathological |

The 255-cliff is fundamentally incompatible with our Dial bucket-queue
linear-cost SSP. Reason (after re-reading `ASSP.cc:2034`): PHASS's own
Dijkstra zeros the *reduced* cost on any arc that already carries flow,
so once a wrap line is laid down it becomes a free reusable highway for
subsequent demands. The 255-cliff is fine in PHASS because expensive
arcs only get paid for once. In whirlwind, unit-capacity + linear cost
means every wrap line pays the cliff in full and the SSP has to
re-discover routes for each demand — the cliff produces many near-tied
candidate paths in the bucket queue and the solver chokes.

The faithful cost recipe has been reverted; the existing
`WHIRLWIND_PHASS_COST` env var still maps to the flat-clamp version
(documented negative result).

### 2. PHASS is not unit-capacity; whirlwind is.

Confirmed in source: `ASSP.h:44` declares `flow_limit_per_arc = 4`, but
every actual capacity check is commented out (`ASSP.cc:2033-2110`) and
the reused-arc reduced-cost line at `ASSP.cc:2034` is what enforces the
"free highway" behavior described above. Whirlwind's `BitVec`
saturation hard-limits each arc to one unit.

This is the root architectural difference and the most plausible single
explanation for the cost+SSP-only gap. It's also a far more invasive
change than a cost knob — multi-capacity needs new flow accounting in
`primal_dual::run`, `network::Network`, and `integrate_*`. Not
prototyped here.

### 3. Amplitude/Canny is *not* what gave dolphin its 97.93 % match.

`Phass.cpp:23` explicitly sets `_usePower = false` in the no-amplitude
overload. Our dolphin invocation passed phase + coherence only, so the
97.93 % came purely from cost + flow-reuse, not amplitude edges. The
earlier "we don't have an amplitude-edge detector" framing in the
discussion above was a red herring for this particular benchmark.

### 4. Coherence-cost + virtual-ground node.

Reviewer also suggested testing `unwrap_grounded` (parallel to
`unwrap_crlb_grounded`) on real data, since the smooth-ramp regression
test proves ground fixes stacked-boundary failures. Added the function
and tested both scenes (ground_cost values 0, 50, 100, 200):

| scene | mode | wall | K=match |
|---|---|---:|---:|
| PV    | baseline           | 0.7 s | 90.67 % |
| PV    | grounded gc=0      | 0.7 s | 18.70 % |
| PV    | grounded gc=50     | 0.7 s | 22.63 % |
| PV    | grounded gc=100    | 0.7 s | 22.63 % |
| PV    | grounded gc=200    | 0.7 s | 22.07 % |
| NISAR | baseline           | 75 s  | 80.01 % |
| NISAR | grounded gc=100    | 49 s  | 41.91 % |

Strictly worse on real data, at every ground cost tested. Ground node
drains interior residues to the boundary along non-physical paths
because real data has dense interior residue pairs that *want* to pair
internally. The grounded variant is right for the
`diagonal_ramp_512`-style stacked boundary regression and wrong as a
default for noisy real scenes. Kept the new `unwrap_grounded` API
(mirrors `unwrap_crlb_grounded`) for callers who know they're in the
boundary-stacking regime.

### Net take

The PHASS-class K-agreement gap at α=0 is not closable via cost-shape
tweaks confined to our SSP — the limiting abstraction is **linear,
unit-capacity, per-arc-cost SSP itself**, not any specific scalar tuning.
That diagnosis is now consistent across both the cost-shape experiments
(this doc, May) and the second-pass reviewer notes from after the
faithful-PHASS redo. SNAPHU works because it has nonlinear/convex flow
costs with curvature and offsets; PHASS works because used cut edges
become cheap/reusable. Whirlwind has one unit per directed arc and pays
the local cost independently every time, so it cannot express either.

Where that leaves the scientific story:

* **What works empirically today, with Goldstein α=0.7.** 38 s, 99.9 %
  K-match with SNAPHU 9×9 on the 6811×6912 NISAR scene; SNAPHU's own
  tiled wall on the same scene is 17 min. That's the only "we beat
  the reference" data point we currently have, and it is the publishable
  claim *if* the K-transfer back to original wrapped phase counts as
  "no smoothing of the delivered output" (it does — Goldstein only
  stabilises the integer ambiguity decision, the emitted phase is
  congruent to the unfiltered input modulo 2π by construction).
* **What does not work.** Trying to be a PHASS/SNAPHU peer *without*
  Goldstein, with the current solver. The simple-NISAR-unfiltered
  benchmark is not closable by parameter sweeps. We measured.
* **What it would take to drop the Goldstein requirement.** Either a
  PHASS-like branch-cut mode (integer flow counts, reused cuts cheap)
  or SNAPHU-like convex per-arc costs (nonzero preferred offsets,
  negative marginal costs handled via Bellman–Ford / Klein
  cycle-cancellation). Both are real prototyping work, not scalar
  tweaks. Neither is started here.

The "Carballo papers looked better than Chen's" instinct that this
codebase started from isn't refuted by these experiments — what we
have is a fast Carballo-style ambiguity solver that happens to need a
preconditioner to match the no-filter behavior of the reference
algorithms. Whether that's the right scientific framing for a paper, or
whether the right next step is a solver rewrite, is the open question.

## 2026-05-28 part 2: PHASS-style flow-reuse prototype — diagnosis confirmed

Implemented `Network::reuse_mode` + an override in
`shortest_path::dial` that forces reduced cost = 0 on any arc with
`flow_count != 0`. Same Carballo cost as the baseline; same primal-dual
Dial driver; same Python entry shape. Exposed as `unwrap_reuse` /
`whirlwind.unwrap_reuse`. No new solver code: it's a multi-unit
relaxation of the existing one.

Result on the same two scenes, α=0 (no Goldstein), no other changes:

| scene | mode | wall | K=match | `|dK|`=1 | `|dK|`≥2 |
|---|---|---:|---:|---:|---:|
| PV    | baseline (unit-cap)  |   0.7 s | 90.67 % |  1.09 % | 8.25 % |
| PV    | **reuse**            |   3.7 s | **99.75 %** | 0.25 % | **0.00 %** |
| NISAR | baseline (unit-cap)  |  75 s   | 80.01 % |  1.71 % | 18.28 % |
| NISAR | **reuse**            |  93 s   | **92.70 %** | 0.24 % | **7.06 %** |
| NISAR | dolphin PHASS (ref)  | ~60 s   | 97.93 % | 2.07 % | 0.00 % |
| NISAR | Goldstein α=0.7      |  38 s   | 99.90 % |   —    |   —    |

The diagnosis holds. Flow-reuse alone (with the existing cost shape,
no amplitude edges, no curvature) closes essentially all of the PV gap
and ~2/3 of the NISAR gap to dolphin PHASS. The runtime tax is 1.2-5×
baseline, well inside acceptable. And — possibly the cleanest signal —
the ignored `diagonal_ramp_512` regression test (6π smooth ramp,
boundary stacking failure under unit-capacity MCF) **passes** under
reuse with max error 0.0 rad. That's a stronger pass than even the
`unwrap_crlb_grounded` workaround, and it's the same cost as the
failing baseline; only the flow model changed. Test added as
`diagonal_ramp_512_reuse`.

What's left in the NISAR gap to dolphin PHASS (92.7 % → 97.9 %, the
remaining 5 pp): one or more of —
* **Hard cuts** at `phase_diff_th = 1.0 rad` (PHASS adds these on top
  of cost-only routing). Now testable cleanly on top of reuse — the
  earlier "hard cuts blow up runtime" failure was an artifact of the
  unit-capacity SSP, not the cuts themselves.
* **Cost shape**: PHASS γ²·100 with the 255-cliff. Earlier diagnosed
  as pathological in unit-cap SSP; under reuse the cliff should be
  digestible since paths can carry multi-unit flow.
* **Solver tuning**: bucket-queue size for the wider cost range,
  augmentation strategy.

None of those require core-algorithm work — they're knobs on top of
the now-working reuse path. The hard question (linear unit-capacity
SSP as a fundamental limit) is **answered**: it was the limit, and
relaxing the unit-capacity piece alone closes most of the gap.

### Hard-cut follow-up (negative result)

Tested reuse + `WHIRLWIND_HARD_CUT_THRESH=1.0` (PHASS's actual
threshold) and `=2.0` (the practical pre-reuse setting):

| mode | wall | K=match | `|dK|`=1 | `|dK|`≥2 |
|---|---:|---:|---:|---:|
| reuse alone           |  93 s   | **92.70 %** | 0.24 %  |  7.06 % |
| reuse + hard_cut 1.0  | killed at >8 min  | — | — | — |
| reuse + hard_cut 2.0  | 125 s   | 91.30 % | 1.73 % | 6.98 % |

`hard_cut=1.0` is still pathological even with reuse — the zero-cost
subgraph creates an unbounded bucket-0 in Dial. `hard_cut=2.0` runs
cleanly but **hurts** K-agreement (-1.4 pp; `|dK|=1` rises from
0.24 → 1.73 %). Mechanism: hard cuts pre-bake zero-cost arcs *before*
any flow is pushed. With reuse, the first augmenting paths get locked
into those pre-baked cuts. At threshold 2.0 the cuts fire on
within-coherent-region noise as well as true wrap-line gradients —
false positives become spurious "highways" that the routing then
reinforces via reuse. PHASS escapes this because its auction-based
augmentation handles tied costs differently than our Dial bucket
queue does; the cut threshold is calibrated for *that* solver, not
ours.

Net: the residual ~5 pp NISAR gap is **not** closable by the cost-knob
side. The two remaining options are:
1. **Convex SNAPHU-style cost** (per-arc curvature, nonzero preferred
   offsets) — bigger prototype, the other lane from the 2026-05-28
   diagnosis.
2. **Smarter cut placement** — limit zero-cost arcs to clusters that
   look like actual wrap-line topology (long aligned runs of high
   `|wrap(Δphase_raw)|`), rather than per-arc thresholding. Effectively
   PHASS's amplitude/Canny detector, but driven from phase instead.
   Lighter than convex but more bespoke.

This is the first time whirlwind has had a competitive no-Goldstein
data point on a real NISAR scene. The PR-#19 Goldstein α=0.7 default
is still the fastest path, but for the scientific story, reuse-mode
unwrap is now the cleaner positioning.

## Reproduction

```bash
PY=/Users/staniewi/miniforge3/envs/mapping-312/bin/python

# Rebuild the editable Rust extension (once, after any cost/mod.rs edit):
cd /Users/staniewi/repos/whirlwind-insar
maturin develop --release

# Sequential — never run more than one heavy unwrap at a time on this laptop:
$PY scripts/phass_experiments/run/run_snaphu_pv.py
for scene in pv nisar; do
  for mode in baseline hard_cut phass_cost; do
    $PY scripts/phass_experiments/run/run_one.py "$scene" "$mode"
  done
done
$PY scripts/phass_experiments/analyze/analyze.py
```

Outputs land in
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/`:

* `outputs/<scene>_<mode>.npz` — `unw`, `cc`, `k`, `elapsed`
* `outputs/pv_snaphu.npz`      — SNAPHU smooth reference (single tile)
* `outputs/results.md`         — the tables above
* `plots/<scene>_k_panel.png`  — side-by-side K-field comparison

For NISAR the SNAPHU 9×9 reference is the `.snaphu_9x9.{unw,cc}.tif`
TIFFs next to the input data, generated in earlier work.

`phass_full` mode is parameterised but expected to take ~hours on
NISAR (86 s on PV scales superlinearly); not run.
