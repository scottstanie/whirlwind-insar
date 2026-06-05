# NISAR 2D unwrapping comparison

This page compares Whirlwind with the unwrapping used in NISAR GUNW products and with a few other 2D unwrappers. The metric is agreement with the production NISAR unwrapped phase after re-wrapping it to create the input wrapped phase.

The comparison uses 13 HH NISAR GUNW frames with `nlooks=16`. Runtimes and memory are from one Apple M-series laptop, so treat them as relative numbers rather than universal benchmarks.

## Summary

- Whirlwind agrees with the production SNAPHU unwrap at 99-100 percent of pixels on 12 of 13 frames.
- The remaining frame, D_075, is difficult for every method in this sweep; Whirlwind agrees with production SNAPHU on 88.2 percent of pixels there, while PHASS agrees on 48.4 percent.
- Runtime is 14-41 seconds per frame for Whirlwind, compared with 465-1242 seconds for single-tile SNAPHU and 90-110 seconds for SNAPHU 9x9 tiled plus reoptimization.
- Peak memory is about 3-4 GB per NISAR frame, compared with about 7-8 GB for single-tile SNAPHU and about 3-3.8 GB for SNAPHU 9x9.

## Metric

The quality number is per-connected-component 2pi ambiguity agreement with the production GUNW unwrap, after median alignment within each component. This checks whether the integer cycle field agrees with the production result while avoiding a single global reference-pixel offset dominating the score.

## Results

| Frame | Whirlwind vs production SNAPHU | PHASS vs production SNAPHU | Note |
|---|---:|---:|---|
| A_013 | 100.0 | 99.3 | |
| A_016 | 100.0 | 99.6 | |
| A_018 | 100.0 | 85.7 | |
| A_020 | 99.8 | 99.4 | |
| A_022 | 100.0 | 99.4 | |
| A_025 | 100.0 | 67.0 | low-coherence river |
| A_028 | 100.0 | 92.9 | |
| A_030 | 100.0 | 75.4 | |
| D_074 | 98.8 | 91.2 | |
| D_075 | 88.2 | 48.4 | hard frame for all methods in the sweep |
| D_077 | 99.5 | 94.7 | |
| D_078 | 99.8 | 96.9 | |
| A_035 | 100.0 | 94.6 | |

![13-frame NISAR GUNW comparison](figures/nisar_summary.png)

The full per-frame table with runtime and memory is in [nisar_4way_results.csv](nisar_4way_results.csv).

## Runtime and memory

| Engine | Runtime | Peak memory | Notes |
|---|---:|---:|---|
| Whirlwind | 14-41 s | 3-4 GB | Rust-backed 2D MCF path |
| SNAPHU, single tile | 465-1242 s | 7-8 GB | quality reference, slowest configuration |
| SNAPHU, 9x9 tiled + reoptimize | 90-110 s | 3-3.8 GB | production-style tiled configuration |
| PHASS | 5.5-23 s | 1.7-2.4 GB | faster, lower agreement on several frames |
| isce2 ICU | 109-204 s | 1.5-2.8 GB | leaves some low-coherence areas disconnected |

## A_025 river case

A low-coherence river splits A_025 into disconnected land regions. The MCF solve can unwrap each region internally but does not observe the relative 2pi offset between disconnected valid regions. Whirlwind applies a bridge post-pass that uses a coarse connected view of the scene to set those relative offsets when the integer shift is clear. On A_025 that changes the agreement from 58 percent to 99.99 percent without changing the other 12 frames in this sweep.

![A_025 bridge before/after](figures/A_025_bridge.png)

## Algorithm in brief

1. Compute residues from the wrapped phase.
2. Build Carballo/Lee coherence-based edge costs.
3. Solve a minimum-cost-flow problem to pair residues through low-cost paths.
4. Integrate the corrected gradients through the valid mask.
5. Return the unwrapped phase and SNAPHU-style connected-component labels.

See [Algorithm notes](ALGORITHM.md) for the main algorithm description and [Performance notes](PERFORMANCE.md) for synthetic timing and memory details.

## Reproduce

- 4-way sweep: `scripts/sweep_all_unwrappers.sh`
- Bridge before/after sweep: `scripts/bench_bridge_all.py`
- A_025 bridge diagnostics: `scripts/proto_bridge_a025.py`, `scripts/diag_bridge_partition.py`
