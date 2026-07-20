# NISAR GUNW hard frames: what we found, fixed, and ruled out

**Handoff note, 2026-07-20, branch `nisar-gunw-local-bench`.** Written so
someone picking this up cold knows what is settled, what is opt-in, and which
hypotheses are already dead so they are not re-investigated.

Context: a local (no-AWS) campaign unwrapped **1,382 NISAR GUNW frames** and
compared each against the production SNAPHU unwrap. Median per-component
agreement was 0.9998, but ~15 frames sat between 8% and 63%. Those frames turned
out to be **two distinct bugs**, both now fixed, plus a third open item.

Note on the metric: "agreement" is against the production unwrap, which is not
ground truth. It is the right target for this work (we want to match what the
NISAR team ships) but score movements must always be sanity-checked against the
imagery - that is how the original defect was spotted in the first place, as a
visible vertical/horizontal rip. Also check `prod_unwrapped_recall` before
trusting a frame's score: production declines to unwrap some frames almost
entirely, and a score computed over 5% of an image means very little.

---

## 1. Bridging across masked gaps - FIXED (GPT Sol, commit `79619fe`)

**Symptom.** Frames with a strong ionospheric gradient plus a river or similar
masked feature. Whirlwind correctly identified the regions as separate
components but assigned them wrong relative 2π offsets, so the frame scored
8-58% despite looking locally correct.

**Cause.** The bridge post-pass used a 500-pixel window to estimate the offset
between regions. That averages across multiple ionospheric fringes and invents
integer offsets. It could also let a tiny island anchor a million-pixel region.

**Fix.** 32-pixel window plus a size-monotone tree, in
`crates/whirlwind-core/src/bridge.rs`.

| frame | before | after |
| ----- | -----: | ----: |
| 008_055_D_073 | 8.13% | 99.98% |
| 009_055_D_071 | 52.38% | 99.99% |
| 003_127_D_069 | 52.45% | 99.96% |
| 004_033_A_019 | 52.69% | 99.96% |
| 008_049_A_035 | 54.62% | 99.92% |
| 003_106_A_036 | 58.34% | 99.83% |
| 003_148_A_019 | 58.53% | 99.96% |
| 004_015_D_054 | 58.89% | 99.51% |

Also established in that pass, and **not worth retrying**: interpolation only
fills valid low-coherence pixels (masked water is re-zeroed, so it cannot
connect river-separated regions); neither interpolation, Goldstein filtering,
nor downsampling fixed the genuine failures. The same work exposed focused A/B
controls on `aws-batch/compare_gunw.py` (`--interpolate`, `--downsample`,
`--goldstein-alpha`, `--phase-grad-window`, `--no-bridge`), which everything
since has been built on.

---

## 2. The stacked-cut / block-tear bug - FIXED (commit `11c2ae8`)

This one took three rounds and killed several plausible hypotheses. Full
detail in [BUG_NISAR_CRYO_STACKED_CUTS.md](BUG_NISAR_CRYO_STACKED_CUTS.md);
summary here.

**Symptom.** The campaign's single worst frame after the bridge fix - a
cryosphere scene, `009_074_A_137` - had ~47% of its pixels offset by exactly
−3 cycles from production, split by three stacked one-cycle cut lines. One
connected mask region, so bridging was structurally a no-op. 50.57%
per-component. isce3 PHASS unwrapped the same frame cleanly (99.19%), which was
the key independent signal that the block was an artifact rather than real
signal.

### Ruled out: arc capacity (Fable, commit `26eb85b`)

The standing theory was the capacity-1 network: unit-capacity arcs cannot carry
two units of correction, so parallel corrections get sprayed onto neighbouring
arcs as stacked cuts. Tested directly by implementing a **true Costantini
uncapacitated linear MCF** - `Network::multi_mode` / `unwrap_linear_multi`,
reachable via `WHIRLWIND_UNWRAP_SOLVER=multi`. Every arc multi-unit, every unit
charged the full arc cost (unlike `reuse`'s free-after-first rule), riding the
unchanged parity solver.

Clean negative result: it finds a genuinely cheaper optimum (total cost
7,373,481 vs 7,589,085, −2.8%) and the stacked cuts do collapse into shared
crossings - but agreement is **unchanged at 50.6%**. Capacity was never the
problem. The solver is kept (opt-in, cheap to maintain) as the standing answer
to "isn't this just capacity-1 stacking?".

### Located: the cost surface (Fable, `scripts/phass_cost_ablation.py`)

Grafting isce3 PHASS's arc costs onto whirlwind's own unchanged capacity-1
solver (via `unwrap_linear_ext_costs`) scored **98.97%**. So the defect lived in
the cost surface, not the solver. Ingredient ablation: squared coherence alone
reached 87.9%; PHASS's "zero cost where the wrapped gradient ≥ 1 rad" rule
supplied the rest; its high-coherence clamp was inert (alters 1.2% of arcs,
changes no output).

### Root cause and the actual fix: bound the Carballo cost's validity domain

Swapping in PHASS's whole surface is not the right fix - it also swaps in
squared coherence and regresses other frames. The useful question was *which
part* of the Carballo cost is wrong, and the answer is its **domain**, not its
shape.

The Carballo arc cost is a log-likelihood ratio `−log(p1/p0)`: the evidence that
an edge carries a 2π cycle jump, given the locally expected slope. That
conditioning only means something while the wrapped observation still
discriminates between hypotheses. Once the true fringe rate passes Nyquist - a
glacier shear margin, a rupture edge - one wrapped difference is consistent with
many true slopes, the likelihood ratio collapses toward 1, and the honest cost
is 0. Whirlwind instead reported the model's confident answer, which made
cutting *along* the real discontinuity expensive, so the solver laid a cheaper
cut straight through the smooth interior. That is the block.

`cost::SlopeGuard` declares that domain: where the **raw** per-edge wrapped
`|Δφ|` reaches a threshold, the cost is 0. Raw rather than smoothed, because a
box average over a shear margin is diluted by its gentle neighbours - exactly
where the model stops discriminating. Gated on `γ > 0` so mask-boundary edges
(masked pixels enter as `0+0j`) never trigger.

Cryo frame, same solver and pipeline in every arm:

| arm | per-comp |
| --- | -------: |
| baseline (guard off) | 50.57% |
| 1.0 rad (PHASS's threshold) | 99.14% |
| 2.0 rad | 99.73% |
| *(grafting PHASS's entire cost surface)* | *98.97%* |

Bounding the domain **beats replacing the cost model**. `077_A_036` turned out
to be the same bug: previously recorded as "a real within-region solve issue"
that only 4x downsampling helped (54.83% → 85.33%), it reaches 99.26% with no
downsampling.

### Two more hypotheses killed by measurement

Both were plausible, both are **dead** - do not re-investigate without new
evidence:

1. **The slope estimator / circular mean.** Whirlwind box-averages raw wrapped
   angles arithmetically (`cost/mod.rs`), which is genuinely wrong near the ±π
   branch cut. But it understates steep slopes by only **11%** on this frame,
   and only 2.5% of edges are steep - it cannot explain a 48-point gap. Proven
   directly by the `zeroslope` guard arm, which re-evaluates the cost at zero
   slope while keeping coherence weighting: **50.55% vs 50.57% baseline, i.e.
   nothing.** (The estimator is still worth fixing on its own merits - a
   complex-domain circular mean is wrap-safe and coherence-weighted - but it is
   not this bug.)
2. **A coherence gate** (fire only on coherent edges, so a steep gradient means
   discontinuity rather than noise). Aliased edges are low-coherence in *every*
   frame, including both frames the guard fixes: cryo 0.241 vs 0.381 overall,
   `077_A_036` 0.209 vs 0.416. A gate at γ>0.4 would fire on 0.6% of the cryo
   frame instead of 2.5%, discarding most of the fix while barely changing the
   frame it was meant to protect. **Coherence does not separate the good cases
   from the bad one.**

### What does discriminate: the aliased FRACTION

No single radian threshold generalizes. 1 rad fixes `077_A_036` but frees ~50%
of the decorrelated `143_D_060` and visibly destabilizes it (blotchy field, ±6
cycle errors - real damage, confirmed in the imagery, not a metric artifact).
2 rad is safe there but stops fixing `077_A_036`.

The operative quantity is **how much of the cost field the guard erases** -
2-3% is a fix, 50% is destruction. Hence `WHIRLWIND_SLOPE_GUARD_BUDGET`: choose
the threshold per frame as a quantile of the raw `|Δφ|` distribution, floored in
radians, so at most a set fraction of edges is freed.

| frame | baseline | **budget 0.03** | best *fixed* threshold |
| ----- | -------: | --------------: | ---------------------: |
| 074_A_137 (cryo) | 50.57% | **99.38%** | 99.73% (2.0) |
| 077_A_036 | 54.83% | **99.26%** | 99.26% (1.0) |
| 035_D_123 | 59.59% | **94.43%** | 62.98% (1.0) |
| 143_D_060 | 62.26% | **62.32%** | 25.73% (1.0) ← wrecked |
| 106_A_036 | 99.83% | 99.65% | 99.44% |
| 127_D_069 | 99.96% | 99.72% | 99.72% |

**Worst regression −0.25 pp, mean +21.28 pp.** `035_D_123` is fixed *only* by
the budget rule - no fixed threshold got it above 63% - because what it needed
was a threshold selective enough to free 3% rather than the 12.7% that a flat
1 rad frees there. Stable across budgets 0.03 and 0.05, so not knife-edge
tuned. The radian floor is load-bearing in the other direction: on frames with
few aliased edges it stops the budget overspending.

**Three of the campaign's worst frames are one bug.**

---

## 3. Status and how to use it

Everything is **opt-in and off by default**. With the guard disabled the raw
gradients are never even allocated, so the parity path is untouched by
construction; the 13-frame parity set is unchanged at mean **0.9891**
per-component (`A_016`/`A_025` 1.0000, `D_075` 0.8807, matching the historical
record), and the guard-off arm reproduces the cryo frame's known 50.5736%.

```bash
# recommended setting
WHIRLWIND_SLOPE_GUARD_BUDGET=0.03 WHIRLWIND_SLOPE_GUARD_RAD=1.0 \
  PYTHONPATH=python python aws-batch/compare_gunw.py "$H5" --out-dir out --force

# uncapacitated linear solver (diagnostic)
WHIRLWIND_UNWRAP_SOLVER=multi ...
```

| env var | meaning |
| ------- | ------- |
| `WHIRLWIND_SLOPE_GUARD_RAD` | threshold in radians, or the floor when a budget is set |
| `WHIRLWIND_SLOPE_GUARD_BUDGET` | max fraction of valid edges the guard may free |
| `WHIRLWIND_SLOPE_GUARD_MODE` | `zerocost` (default) or `zeroslope` (diagnostic arm) |
| `WHIRLWIND_UNWRAP_SOLVER=multi` | uncapacitated linear MCF |

Tooling (all reuse `compare_gunw.py`'s cached `full_arrays.npz`, so inputs and
metric are byte-for-byte the benchmark's; each arm is a separate process because
`slope_guard()` caches in a `OnceLock`):

- `scripts/phass_cost_ablation.py` - one frame, one cost variant
  (`carballo` / `phass` / `phass-nogradzero` / `phass-noclamp`), runs the real
  bridge + conncomp tail and writes the 8-panel comparison figure.
- `scripts/run_slope_guard_sweep.sh` - threshold sweep on one frame.
- `scripts/slope_guard_frame_sweep.py` - many frames × arms → `sweep.md`
  with a regression summary.

Data: 15 hard frames at
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw_hardest/`,
13-frame parity set at `.../nisar_gunw/`.

## 4. Open items

1. **Make the guard the default?** The 13-frame parity gate **passed**:
   5 frames improved, 8 unchanged, **0 regressed**, worst change −0.01 pp,
   mean per-comp 98.91% → 98.95%. Every 100% frame stays at 100% and the
   weakest (`D_075`) improves. On that set the guard is free, not a tradeoff.
   With the hard frames (worst −0.25 pp, mean +21.28 pp) that makes
   `budget=0.03` / floor 1.0 defensible as a default. The one thing not yet
   done is a full 1,382-frame campaign re-run, which is the only way to see
   the tail - recommended before flipping the default.
3. **The circular-mean estimator fix** - a real defect, small effect, worth
   doing on its own merits and cleanly separable from the guard.
4. **`143_D_060` is a poor test frame** (production unwraps only 4.7% of it).
   Judge it on imagery, not score.
5. **Paper angle.** This is a clean ablation showing the cost surface, not the
   solver class, dominates on steep scenes - and it reinforces that whirlwind's
   contribution is the cost model and evaluation rather than the solver.
