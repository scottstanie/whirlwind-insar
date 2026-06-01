# Connected components

How whirlwind labels connected-component regions of self-consistent unwrapped
phase, and why. Implemented in `crates/whirlwind-core/src/conncomp.rs`
(`grow_components`), exposed via `unwrap_with_conncomp` /
`unwrap_crlb_with_conncomp`. Mirrors SNAPHU's component logic.

## Why components at all

A 2-D unwrapping is only defined up to an additive integer number of cycles per
**disconnected** region: wherever the result is *torn* (a 2π branch cut through
decorrelated pixels), the two sides have no enforced relative offset. A connected
component is a maximal region with **no internal tear** — inside it, every pixel
is integer-consistent with every other, so it has a single well-defined relative
phase. The absolute offset *between* components (or of an isolated island) is the
user's reference choice. Labelling components tells a downstream consumer "trust
phase differences within a label; do not trust them across labels."

## The algorithm (one MCF solve → BFS labelling)

Components are grown directly from the **already-solved** min-cost-flow network —
no second solve. The unwrap and the components come from the same MCF state.

1. **Pixel grid graph.** For an `(m, n)` phase image the MCF runs on the
   `(m+1, n+1)` grid of pixel *corners* (`RectangularGridGraph`). Each edge
   between two neighbouring **pixels** corresponds to two arcs of that corner
   grid (e.g. the pixel edge `(i,j)–(i,j+1)` maps to `down(i,j+1)` + `up(i+1,j+1)`).

2. **Cut rule** (`edge_is_cut`). A pixel edge is a **cut** (the two pixels are *not*
   joined) when either underlying arc is:
   - **mask-forbidden** — an endpoint pixel is invalid (both arc directions
     saturated), or
   - **low-cost** — `min(raw forward cost of the two arcs) ≤ cost_threshold`.
     The per-arc cost is the Carballo/CRLB cost, which is small exactly where
     coherence is low; those low-coherence edges are where the MCF places its
     branch cuts. So a cut = "masked, or a tear through decorrelated phase."
   `cost_threshold` (default 50, on the `COST_SCALE = 100` scale where the
   Carballo cost spans ~0–314) sets how aggressively low-coherence edges tear:
   higher → more/larger cuts → smaller components.

3. **Flood fill.** BFS over non-cut edges (4-connectivity) assigns every reachable
   valid pixel one label. Each unvisited valid pixel starts a new label. This is a
   plain connected-components labelling; the only subtlety is that adjacency is
   gated by the cut rule above.

4. **Size filter + cap** (the policy knobs):
   - `min_size_px` (default **100**): drop any component smaller than this many
     pixels — an **absolute** floor (≈0.8 km at 80 m, 0.3 km at 30 m), scene-size-
     and pixel-spacing-invariant, matching SNAPHU's `minregionsize`. This is the
     real speckle control; see [below](#why-an-absolute-floor).
   - `min_size_frac` (default **1e-4**): a vestigial fractional cap that only ever
     *raises* the floor on very large frames (`min_size = max(min_size_px,
     ceil(min_size_frac · n_valid))`); it can never drop it to kilometre scale.
   - `max_ncomps` (default **1024**): keep at most this many components (largest by
     size); `0` = keep all. A generous anti-pathology guard — the floor, not this
     count, is meant to do the real filtering.
   Surviving components are sorted by descending size and renumbered `1..=K`;
   `0` is background (cut off, masked, or filtered out).

The output is a `u32` label image, same `(m, n)` shape as the phase.

## Why an absolute floor

The size floor was originally `min_size_frac = 0.01` (1%), copied from SNAPHU's
`minconncompfrac`. On a NISAR/OPERA frame that is the wrong unit: 1% of ~7 M valid
pixels ≈ 70 000 px ≈ a **21–25 km** minimum feature at 80 m (worse at 30 m) — it
orphaned every coherent island. A *fraction* scales the minimum with **scene
area**, which is meaningless; an absolute **pixel** count scales the physical
minimum with **resolution**, which is what you want (finer data → keep finer
features), and needs no resolution input. The floor moved to an absolute 100 px;
the fraction was demoted to a safety cap.

The honest caveat: size is a *blunt proxy*. The true noise/island discriminator is
**coherence** (decorrelated speckle is low-coherence; real islands are high-
coherence). `cost_threshold` already cuts low-coherence edges, so the surviving
fragments are mostly real; the size floor then removes the residual sub-100-px
speckle. To keep coherent islands *below* 100 px without re-admitting speckle, the
right tool is a per-component mean-coherence gate (future work), not a smaller
size floor. All three knobs are caller-exposed.

## Not to be confused with the report heuristic

`scripts/phass_experiments/report/make_report_figures.py::ww_conncomp` computes a
*different*, cost-free approximation for plotting: connected components over edges
with `|Δunw| < π` (no 2π tear), on a downsampled grid. It does not use the MCF
costs or the `cost_threshold` rule, so it over-merges low-coherence speckle into
one big component and is only a visualisation aid — not the same as
`grow_components`.
