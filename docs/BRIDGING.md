# Bridging disconnected regions

When a river, water body, subswath seam, or other low-coherence gap splits the valid mask, the MCF solve unwraps each side independently. The wrapped phase does not contain the integer 2π offset between disconnected regions. After integration, each region is internally consistent, but may be one or more cycles above or below another.

Bridging estimates one integer-cycle shift per region and applies it after unwrap. In Whirlwind it is enabled by default in [`unwrap`](ALGORITHM.md), and can also be run directly with `whirlwind.bridge_components`.

## Why component scores miss it


## The model

Let the valid mask split into integration regions $R_1, \dots, R_K$ (the 4-connected components of the mask, which the integrator seeds independently).  Within a region, the MCF flow fixes the relative 2π level. Between regions, the integer level is unobserved. Bridging chooses one integer shift $s_i$ per region,

$$ u'(p) = u(p) + 2\pi\, s_i \quad \text{for } p \in R_i, $$

and fixes the largest region as the reference ($s_\text{ref} = 0$).

## Algorithm

Whirlwind's Rust bridge is based on the algorithm in isce3's NISAR GUNW workflow (`isce3.unwrap.bridge_phase.bridge_unwrapped_phase`), with a smaller local window and size-monotone tree for strong-gradient scenes:

1. Label the integration regions (the native `label_components`, a 4-connected BFS — no scipy). Keep regions of at least `min_px` pixels.
2. For every pair of regions, find the closest pair of boundary pixels.  Boundary-pixel sets are strided to at most `max_boundary` points for the nearest-pair search.
3. Visit regions from largest to smallest. Attach each region to its nearest already-visited region, which is therefore at least as large. This size-monotone tree prevents a tiny river island from becoming the phase reference for a much larger landmass.
4. For each edge, take the median unwrapped phase in a local 32-pixel-half-width box around each bridge endpoint, round the parent-to-child difference to integer cycles, and add that shift to the child region. The parent is already corrected when the child is processed, so corrections propagate through the tree.

A single-region (or coherently connected) frame produces no bridges and is returned byte-identical.

The method reads phase locally at the region boundaries, where the cross-gap phase difference is smallest, and propagates offsets from larger anchors into smaller regions. Whole-region medians—or even 500-pixel "local" windows on strong ionospheric ramps—can span several fringes and round the real gradient into a false integer offset.

## Results

On the July 2026 1,382-frame NISAR campaign, eight of the nine non-cryo frames in the top-ten failure list were bridge-estimation errors rather than solver errors. Re-running the identical wrapped inputs with the 32-pixel, size-monotone bridge changed their per-production-component ambiguity agreement as follows:

| Frame | old bridge | new bridge |
| ----- | ---------: | ---------: |
| 008_055_D_073 | 8.13% | 99.98% |
| 009_055_D_071 | 52.38% | 99.99% |
| 003_127_D_069 | 52.45% | 99.96% |
| 004_033_A_019 | 52.69% | 99.96% |
| 008_049_A_035 | 54.62% | 99.92% |
| 003_106_A_036 | 58.34% | 99.83% |
| 003_148_A_019 | 58.53% | 99.96% |
| 004_015_D_054 | 58.89% | 99.51% |

`004_077_A_036` is a genuine within-region solve difference (54.83% with bridging disabled and with the new bridge), not another bridge regression. The cryosphere frame `009_074_A_137` also has one connected valid-mask region, so bridging is structurally a no-op there; its stacked-cut diagnosis is recorded in [NISAR cryosphere stacked-cut artifact](BUG_NISAR_CRYO_STACKED_CUTS.md).

Low-coherence interpolation was disabled in the campaign. Enabling it does not fill masked water—the interpolator only replaces valid low-coherence pixels, and masked pixels are re-zeroed before the solve. Focused A/B tests on the two genuine solver cases did not improve either one. Goldstein filtering likewise did not help; 4× downsampling improved `004_077_A_036` to 85.33% but degraded the cryosphere frame.

Absolute inter-region agreement with the production NISAR GUNW unwrap on the 13-frame set (`scripts/diag_bridge_isce3_compare.py all`):

| Frame | no bridge | whirlwind bridge | isce3 bridge |
| ----- | --------: | ---------------: | -----------: |
| 005_A_016 |      93.5 |             99.9 |         99.9 |
| 005_A_018 |      99.5 |             99.9 |         99.9 |
| 005_A_025 |      46.2 |             99.9 |         70.3 |
| 005_A_030 |      98.3 |             99.9 |         98.3 |

The other nine frames are single-region (bridging is a no-op) or already consistent. Whirlwind matches isce3 on 005_A_016 / 005_A_018. On 005_A_025 (a low-coherence river) and 005_A_030, Whirlwind corrects regions that remain mis-levelled with the isce3 settings used here. The post-pass needs only the unwrapped phase and mask; it does not need scipy or a coherence raster.

The script removes one global offset, taken as the median cycle of the largest region, then counts valid pixels on the same integer cycle as the reference unwrap.

![005_A_016 bridging](figures/bridge_compare_A_016.png)

The bottom row colors each region by integer cycle error versus production (0 = correct). Without bridging, the two large regions are −3 cycles off; the Whirlwind and isce3 bridges both bring them to zero.

## Reproduce

```bash
# absolute-metric comparison vs isce3, one or all frames
python scripts/diag_bridge_isce3_compare.py 005_A_016
python scripts/diag_bridge_isce3_compare.py all
# figure
python scripts/plot_bridge_compare.py 005_A_016
```
