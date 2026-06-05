# Goldstein on-vs-off A/B (NISAR mainland, 2026-06-01)

Decides whether Goldstein pre-filtering should be the default for
`whirlwind.unwrap`. Run via `scripts/report_goldstein_ab.py`: two sequential
`unwrap` calls (α=0 and α=0.7) on the 40 MHz NISAR HH scene (6811x6912, 47 Mpx),
K-match vs SNAPHU 9x9 on the cc=1 mainland (14.5M px), modal cycle offset removed.

| Goldstein α | K-match | \|dK\|=1 | \|dK\|≥2 | coverage | #cc | runtime |
| ----------- | ------- | -------- | -------- | -------- | --- | ------- |
| 0.0 (off)   | 99.974% | 0.024%   | 0.003%   | 30.82%   | 6   | 1096 s  |
| 0.7 (on)    | 99.989% | 0.011%   | 0.000%   | 30.90%   | 6   | 1057 s  |

**Verdict: keep Goldstein OFF by default.** On this clean, high-quality NISAR
scene the two are essentially a wash — both ≥ 99.97% K-match, near-identical
coverage and runtime. Goldstein-on is marginally better (+0.015 pp, and it
removes the handful of |dK|≥2 pixels) but not enough to justify defaulting an
FFT pre-filter on. Goldstein stays available opt-in (`goldstein_alpha > 0`) for
noisier / lower-coherence scenes where it historically helps more; that regime
is not characterized by this single-scene A/B and is left for future evaluation.

**Runtime note:** the ~18 min in the table above was a **debug build** artifact
(the A/B ran the `maturin develop` debug extension). Profiled in **release**
(`scripts/profile_nisar_runtime.py`, `WHIRLWIND_TIMING=1`) the same 47 Mpx frame
unwraps in **~31 s** — comfortably under the < 5 min target. Breakdown: per-tile
MCF solve ~27 s (86%), heal ~1.7 s, the global conncomp build only ~1.2 s, the
multi-shift gate correctly **did not fire** (coherent-cut rate 7.6e-6 ≪ 1.5e-3).
So K-match here is build-independent and the runtime is fine; just build
`--release` for real-data runs. See #52.

Plot: `<WD>/Documents/Learning/goldstein_ab/goldstein_ab_nisar.png`
(SNAPHU K-field + per-variant dK-vs-SNAPHU).
