# Bridging disconnected regions

When a low-coherence feature (a river, a water body, a subswath seam) splits the
valid area into pieces, a minimum-cost-flow unwrapper cannot observe the relative
2π level between those pieces. The integrator seeds each disconnected region at
an arbitrary cycle, so each region is internally correct but may sit a whole
number of cycles above or below its neighbours. **Bridging** estimates and
removes those inter-region offsets. In whirlwind it is the default-on `bridge`
post-pass of [`unwrap`](API.md), and is also available standalone as
`whirlwind.bridge_components`.

## Why it is easy to miss

The natural agreement metric — per-connected-component cycle match — **cannot
see a bridging error**. It aligns each production component independently before
scoring, which deliberately removes any constant per-region offset. A frame can
therefore read 100 % per-component agreement while two regions sit at the wrong
relative level.

To measure bridging you need an *absolute* metric: remove a single global offset
(the median cycle of the largest region) and then count the fraction of valid
pixels on the same integer cycle as the reference unwrap. The diagnostic
`scripts/diag_bridge_isce3_compare.py` reports both, and the difference between
them is exactly the bridging error.

## The model

Let the valid mask split into integration regions $R_1, \dots, R_K$ (the
4-connected components of the mask — the partition the integrator seeds
independently). Within a region the relative 2π level is already pinned by the
MCF flow; *between* regions it is a free gauge. Bridging chooses one integer
shift $s_i$ per region,

$$ u'(p) = u(p) + 2\pi\, s_i \quad \text{for } p \in R_i, $$

to make the regions mutually consistent, fixing the largest region as the
reference ($s_\text{ref} = 0$).

## Algorithm

whirlwind uses a pure-numpy port of the algorithm in isce3's NISAR GUNW workflow
(`isce3.unwrap.bridge_phase.bridge_unwrapped_phase`):

1. **Label** the integration regions (the native `label_components`, a 4-connected
   BFS — no scipy). Keep regions of at least `min_px` pixels.
2. **Adjacency.** For every pair of regions, find the closest pair of boundary
   pixels (the natural place to bridge — where the true phase gap across the
   void is smallest). Boundary-pixel sets are strided to at most `max_boundary`
   points for the nearest-pair search.
3. **Spanning tree.** Build a minimum spanning tree over those closest-pair
   distances, rooted at the largest region. Each region is thus referenced
   through its nearest neighbour rather than directly to one global anchor — the
   shifts compose along the tree.
4. **Offsets.** Walking the tree outward from the root, for each edge take the
   median unwrapped phase in a local box (half-width `radius`, clamped to a
   scene-relative size) around each of the two bridge endpoints, round the
   parent-to-child difference to an integer number of cycles, and add that shift
   to the child region. Because the tree is walked from the root, a region's
   parent is already corrected when the region itself is processed, so offsets
   propagate transitively.

A single-region (or coherently connected) frame produces no bridges and is
returned byte-identical.

The key choices are using the phase **locally at the region boundaries** (where
the true cross-void phase difference is sub-cycle) and propagating along a
**spanning tree** (so far regions chain through near neighbours). An earlier
whirlwind version compared whole-region medians against a coarse 8×-downlooked
anchor; that was less robust — on A_016 it left the two largest regions three
cycles off, scoring identically to no bridging.

## Results

Absolute inter-region agreement with the production NISAR GUNW unwrap on the
13-frame set (`scripts/diag_bridge_isce3_compare.py all`):

| Frame | no bridge | whirlwind bridge | isce3 bridge |
|---|---:|---:|---:|
| A_016 | 93.5 | **99.9** | 99.9 |
| A_018 | 99.5 | **99.9** | 99.9 |
| A_025 | 46.2 | **99.9** | 70.3 |
| A_030 | 98.3 | **99.9** | 98.3 |

The other nine frames are single-region (bridging is a structural no-op) or
already consistent. whirlwind matches isce3 on A_016 / A_018 and is more robust
on A_025 (a low-coherence river) and A_030, where isce3's settings leave large
regions mis-levelled. whirlwind needs no scipy and no coherence input for the
post-pass — only the unwrapped phase and the mask.

![A_016 bridging](figures/bridge_compare_A_016.png)

The bottom row paints each region by its integer cycle error versus production
(0 = correct). Without bridging the two large regions are −3 cycles off (deep
red); the whirlwind and isce3 bridges both flatten them to zero.

## Reproduce

```bash
# absolute-metric comparison vs isce3, one or all frames
python scripts/diag_bridge_isce3_compare.py A_016
python scripts/diag_bridge_isce3_compare.py all
# figure
python scripts/plot_bridge_compare.py A_016
```
