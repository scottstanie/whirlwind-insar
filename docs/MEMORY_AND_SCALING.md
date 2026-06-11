# Memory and scaling notes

This page gives rough planning guidance for the default 2D Python call:

```python
unw, conncomp = whirlwind.unwrap(igram, corr, nlooks, mask=mask)
```

Measured NISAR runtimes, memory, and quality numbers live in
[NISAR 2D unwrapping comparison](NISAR_SUMMARY.md) and the raw
[`nisar_4way_results.csv`](nisar_4way_results.csv). This page intentionally
avoids duplicating benchmark tables.

## Runtime shape

Runtime depends mostly on residue density and graph connectivity.

- Smooth or lightly noisy scenes are dominated by O(mn) setup work: wrapped
  phase, residues, costs, integration, and connected-component labeling.
- Residue-heavy scenes spend most of their time in primal-dual and SSP
  shortest-path work.
- A valid-pixel mask matters. Filled invalid areas can create artificial
  residues along mask boundaries; passing the mask keeps those areas from
  dominating the solve.
- More CPU cores help the O(mn) setup stages, but the current whole-frame
  PD/SSP solve is mostly serial.

For the SNAPHU and PHASS comparison, see
[Whirlwind, SNAPHU, and PHASS performance notes](SNAPHU_PHASS_SPEED.md).

## Memory

Memory is linear in pixel count. The core arrays are per-pixel or per-edge
arrays: wrapped phase, smoothed gradients, edge coherence, integer costs,
network state, shortest-path state, integration buffers, and connected
components.

The analytic working set is about 115 bytes per pixel. Observed process memory
is higher because of allocator behavior, Python/Rust boundary overhead,
thread-local buffers, and temporary arrays. For planning, use about 0.2 GB per
megapixel for the current whole-image path.

These are planning estimates, not benchmark results:

| Image size | Pixels | Planning memory |
| ---------- | -----: | --------------: |
| 1024x1024  |  1.0 M |    about 0.2 GB |
| 2048x2048  |  4.2 M |    about 0.8 GB |
| 4096x4096  | 16.8 M |    about 3.4 GB |
| 8192x8192  | 67.1 M |   about 13.4 GB |
| 25000x4000 |  100 M |     about 20 GB |

The planning rule is intentionally conservative. For measured examples, use the
NISAR comparison page rather than copying values here.

## Whole-frame vs tiled paths

The default public path is a whole-frame solve. A tiled or warm-started approach
can reduce runtime, but if it finishes with a whole-frame reoptimization then
the final solve still needs whole-frame memory.

The current tiled path is experimental and is not the validated default.
