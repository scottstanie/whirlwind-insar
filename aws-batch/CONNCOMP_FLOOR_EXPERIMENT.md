# The conncomp coherence floor: a knob to test at campaign scale

**Status: not changed.** `--conncomp-min-coherence` stays at `auto`. This note
records why it is worth testing, what the single-frame evidence looks like, and
what would have to hold across a campaign before the default moves.

## Where the gap comes from

Switching `--mask-policy` to `subswath` restricts the solve to paired RSLC
observations while keeping water. The delivered NISAR workflow makes the same
validity classification, although it uses it for phase preprocessing rather
than as SNAPHU's hard mask. On the tested river frames, this stopped severing
the integration domain into hundreds of regions and removed the resulting
component re-leveling failures.

It leaves a *label* gap. Solving through water means whirlwind also labels
water. On 023_169_A_016 (a river frame), 918k pixels — 10.2% of the valid
domain — get `ww_cc > 0` where production reports `cc = 0`. Spatially it is
exactly the lake and river network.

Production also solves phase through water and applies separate connected-
component rules, but those rules are not equivalent to our coherence floor.
For this single-tile runconfig, `regrow_conncomps` has no effect and
`min_region_size` is a tile-mode setting; SNAPHU's cost-based labeling and
`min_conncomp_frac: 0.01` are the relevant comparison. The coherence floor is
one Whirlwind-side approximation to investigate, not a reproduction of them.

## Single-frame evidence

The two coherence populations separate cleanly on that frame:

| | median coherence | fraction < 0.2 |
| --- | --- | --- |
| production keeps (`cc > 0`) | 0.58 | 1.5% |
| production drops (`cc = 0`) | 0.14 | 71% |

Sweeping the floor:

| floor | of the disagreement, still labeled | of the whole valid domain, dropped |
| --- | --- | --- |
| 0.08 | 87.5% | 1.4% |
| 0.15 | 45.5% | 6.1% |
| 0.20 | 28.9% | 8.7% |
| 0.25 | 20.3% | 11.1% |
| 0.30 | 14.9% | 14.1% |

Reproduce with `scripts/plot_conncomp_water_floor.py <crop>_arrays.npz`.

The floor removes disagreement several times faster than it shrinks the valid
labeling domain on this frame,
and it changes **labels only, never phase** — the 0.9996 ambiguity match on that
frame is unaffected by any value in the table.

## Why not just change it

1. **One frame is not a campaign.** The separation above is clean because this
   scene's water is genuinely decorrelated. A frame whose *land* sits at
   coherence 0.2 — cryo, dense vegetation, long temporal baselines — would pay
   the "valid domain dropped" column without collecting the benefit. The A_140
   table in `README.md` shows a scene where 0.15 already drops 65% of labels.

2. **`auto` scales the wrong way for this purpose.** `auto` is
   `0.32/sqrt(nlooks)` — **0.045** at 50 looks. It gets *gentler* as looks rise,
   so on a well-multilooked product like GUNW it closes almost none of the gap.
   That is not a bug: `auto` answers "is this coherence estimate distinguishable
   from zero", which legitimately falls as `1/sqrt(L)`. Production's rules answer
   "is this phase useful enough to label". Different questions, and no single
   formula serves both.

3. **It interacts with `nlooks`.** `conncomp_reliability_from_coherence` uses
   `sigma2 = (1 - g^2) / (2 * nlooks * g^2)` with the **raw** value —
   `MAX_COST_MODEL_NLOOKS` (80) caps the cost LUT only. On 004_077_A_036 at a
   fixed floor of 0.2, component count went 15 / 8 / 9 / 20 across nlooks
   16 / 50 / 80 / 144. Any floor sweep has to hold `nlooks` fixed, or it is
   measuring two things at once.

4. **The metric that matters is not yet defined.** `ambiguity_match_frac` is
   insensitive to this knob by construction. What we actually want is a
   labeling agreement score against production's conncomps — see below.

## The experiment

Run at campaign scale (a few hundred frames, spread across the land table so
cryo / vegetated / arid / river scenes are all represented), holding a recorded
fixed `--nlooks 50` and `--mask-policy subswath` for this controlled sweep:

```bash
for FLOOR in auto 0.10 0.15 0.20 0.25; do
  python run_local.py \
    --manifest manifest.txt \
    --root /data/ww-bench-floor-$FLOOR \
    --workers 8 --delete-after \
    --compare-arg=--conncomp-min-coherence --compare-arg=$FLOOR
done
```

Use a separate `--root` per floor: a completed granule is recorded in
`runs.jsonl` and skipped on rerun, so reusing one root would silently return the
first floor's results.

Report per floor, per frame:

- `ww_nonzero_cc_frac` vs `prod_nonzero_cc_frac` — do we converge on
  production's coverage, or overshoot past it?
- **Label agreement, both directions.** Production-labeled pixels we drop are a
  real loss; water we stop labeling is the win. Today's `prod_unwrapped_recall`
  only measures one side.
- `ww_num_cc` — a floor that fixes coverage by shattering the frame into
  components is not a fix. Watch for scenes where this climbs sharply.
- `ambiguity_match_frac_percomp` — expected flat. If it moves, something other
  than labeling changed and the run is not clean.

**Decision rule.** Move the default only if some fixed floor beats `auto` on
label agreement across the *whole* distribution, not on the mean — specifically
if it does not make any frame's coverage or component count materially worse.
If the winner is scene-dependent (likely), the outcome is a documented
recommendation per scene type, or a smarter `auto` that keys off the observed
coherence distribution rather than `nlooks` alone — not a new constant.
