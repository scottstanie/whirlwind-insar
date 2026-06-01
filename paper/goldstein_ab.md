# Goldstein on-vs-off A/B (NISAR mainland, 2026-06-01)

Decides whether Goldstein pre-filtering should be the default for
`whirlwind.unwrap`. Run via `scripts/report_goldstein_ab.py`: two sequential
`unwrap` calls (α=0 and α=0.7) on the 40 MHz NISAR HH scene (6811×6912, 47 Mpx),
K-match vs SNAPHU 9×9 on the cc=1 mainland (14.5M px), modal cycle offset removed.

| Goldstein α | K-match | \|dK\|=1 | \|dK\|≥2 | coverage | #cc | runtime |
|-------------|---------|----------|----------|----------|-----|---------|
| 0.0 (off)   | 99.974% | 0.024%   | 0.003%   | 30.82%   | 6   | 1096 s  |
| 0.7 (on)    | 99.989% | 0.011%   | 0.000%   | 30.90%   | 6   | 1057 s  |

**Verdict: keep Goldstein OFF by default.** On this clean, high-quality NISAR
scene the two are essentially a wash — both ≥ 99.97% K-match, near-identical
coverage and runtime. Goldstein-on is marginally better (+0.015 pp, and it
removes the handful of |dK|≥2 pixels) but not enough to justify defaulting an
FFT pre-filter on. Goldstein stays available opt-in (`goldstein_alpha > 0`) for
noisier / lower-coherence scenes where it historically helps more; that regime
is not characterized by this single-scene A/B and is left for future evaluation.

**Runtime caveat (independent of Goldstein):** both runs took ~18 min on the
47 Mpx frame — over the < 5 min target. Goldstein is not the cause (α=0 and 0.7
are within ~4%). Tracked in **#52** — most likely the gated multi-shift re-solve
firing (≈4× the base tiled unwrap).

Plot: `<WD>/Documents/Learning/goldstein_ab/goldstein_ab_nisar.png`
(SNAPHU K-field + per-variant dK-vs-SNAPHU).
