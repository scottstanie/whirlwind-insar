# Memory and scaling notes

This page gives rough planning guidance for the default 2D Python call:

```python
unw, conncomp = whirlwind.unwrap(igram, corr, nlooks, mask=mask)
```

Measured NISAR runtimes, memory, and quality numbers live in [NISAR 2D unwrapping comparison](NISAR_SUMMARY.md) and the raw [`nisar_4way_results.csv`](nisar_4way_results.csv). This page intentionally avoids duplicating benchmark tables.

## Runtime characteristics

The computational runtime of Whirlwin depends mostly on residue density and graph connectivity.

- Smooth or lightly noisy scenes are dominated by O(mn) setup work: wrapped phase, residues, costs, integration, and connected-component labeling. Since these scenes are fast for most unwrappers, this is not of much concern.
- Residue-heavy scenes spend most of their time in primal-dual (PD) and Successive Shortest Paths (SSP) work.
- A valid-pixel mask matters. Scenes with large areas of non-zero wrapped phase which are not labelled as invalid by the `mask` can create much larger runtimes.

For the SNAPHU and PHASS comparisons, see
[Whirlwind, SNAPHU, and PHASS performance notes](SNAPHU_PHASS_SPEED.md).

## Memory

Memory is linear in pixel count. The core arrays are per-pixel or per-edge arrays: wrapped phase, smoothed gradients, edge coherence, integer costs, network state, shortest-path state, integration buffers, and connected components.

These are planning estimates, not benchmark results:

| Image size | Pixels | Planning memory |
| ---------- | -----: | --------------: |
| 1024x1024  |  1.0 M |    about 0.2 GB |
| 2048x2048  |  4.2 M |    about 0.8 GB |
| 4096x4096  | 16.8 M |    about 3.4 GB |
| 8192x8192  | 67.1 M |   about 13.4 GB |
| 25000x4000 |  100 M |     about 20 GB |

For measured examples, use the NISAR comparison page rather than copying values here.
