"""Frozen numpy reference for ``whirlwind.bridge_components``.

A pure-numpy original of the integration-component gauge bridging post-pass,
from the algorithm isce3's NISAR GUNW workflow.
(``isce3.unwrap.bridge_phase.bridge_unwrapped_phase``).
The canonical implementation now lives in Rust (``crates/whirlwind-core/src/bridge.rs``,
bound as ``whirlwind.bridge_components``); this copy exists only so
``test_bridge_parity.py`` can assert the native version reproduces it exactly.
Do not edit the algorithm here - behavior changes belong in the Rust
implementation, and this oracle should only ever change in lockstep with an
intentional, test-acknowledged behavior change there.

The algorithm:

  1. Label the integration regions (connected components of the valid mask).
  2. For every pair of regions, find the closest boundary-pixel pair (the
     natural place to bridge - where the true phase gap is smallest).
  3. Build a minimum spanning tree of those distances, rooted at the largest
     region, so each region is referenced through its nearest neighbor rather
     than directly to one global anchor.
  4. Walking the tree outward from the root, compare the median unwrapped phase
     in a local box around the two bridge endpoints, round the difference to an
     integer number of cycles, and shift the child region (and, transitively,
     its descendants - the parent is already corrected when its child is
     processed).

Reading the level locally at the boundary (rather than a whole-region median,
which a residual within-region ramp would bias) and chaining through nearest
neighbors along the MST are what keep the integer rounding correct. A
single-region (or coherently connected) frame yields no bridges and is
byte-identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from whirlwind import label_components

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _boundary_coords(
    region: "NDArray[np.integer]",
    labels: "list[int]",
    max_boundary: int,
) -> "dict[int, NDArray[np.float64]]":
    """Strided boundary-pixel (y, x) coordinates for each region label.

    A boundary pixel is a valid pixel with a 4-neighbor of a different label;
    these line the gaps between regions, so the nearest boundary pair between two
    regions is the natural bridge endpoint. Each set is strided down to at most
    ``max_boundary`` points - the closest pair then lands within a few pixels of
    the true contact, which the radius-wide endpoint median absorbs.
    """
    is_diff = np.zeros(region.shape, dtype=bool)
    is_diff[:-1, :] |= region[:-1, :] != region[1:, :]
    is_diff[1:, :] |= region[1:, :] != region[:-1, :]
    is_diff[:, :-1] |= region[:, :-1] != region[:, 1:]
    is_diff[:, 1:] |= region[:, 1:] != region[:, :-1]
    is_boundary = is_diff & (region > 0)

    coords: dict[int, NDArray[np.float64]] = {}
    for lab in labels:
        ys, xs = np.nonzero(is_boundary & (region == lab))
        if ys.size > max_boundary:
            step = int(np.ceil(ys.size / max_boundary))
            ys, xs = ys[::step], xs[::step]
        coords[lab] = np.stack([ys, xs], axis=1).astype(np.float64)
    return coords


def bridge_components_reference(
    unw: "NDArray[np.float32]",
    mask: "NDArray[np.bool_] | None" = None,
    *,
    radius: int = 500,
    min_px: int = 500,
    max_boundary: int = 2000,
) -> "NDArray[np.float32]":
    """Re-level the disconnected regions of an unwrapped phase image.

    See the module docstring for the algorithm. The offset is read straight from
    the unwrapped phase at the region boundaries, so this needs only the
    unwrapped phase and a valid mask - no coherence or interferogram.

    Parameters
    ----------
    unw : ndarray of float32
        Unwrapped phase, masked/nodata pixels left at 0 (or NaN).
    mask : ndarray of bool, optional
        Valid-pixel mask defining the integration regions. Defaults to the
        finite, nonzero pixels of ``unw``.
    radius : int, default 500
        Half-width of the box around each bridge endpoint over which the
        region's local phase level is taken (clamped to a scene-relative size).
    min_px : int, default 500
        Ignore integration regions smaller than this many pixels.
    max_boundary : int, default 2000
        Cap on boundary pixels sampled per region for the nearest-pair search.

    Returns
    -------
    ndarray of float32
        ``unw`` with each region shifted by an integer number of cycles to
        agree with the reference (largest) region. A single-region frame is
        returned unchanged.
    """
    tau = 2.0 * np.pi
    m, n = unw.shape
    if mask is None:
        mask = np.isfinite(unw) & (unw != 0)  # masked/nodata left at 0 or NaN
    mask = np.ascontiguousarray(mask, dtype=bool)

    # Integration components = 4-connected components of the valid mask (native
    # BFS labeller, matching integrate_with_mask's partition; no scipy needed).
    region, n_region = label_components(mask)
    if n_region <= 1:
        return unw  # single integration component -> structural no-op

    sizes = np.bincount(region.ravel(), minlength=n_region + 1)
    big = [lab for lab in range(1, n_region + 1) if sizes[lab] >= min_px]
    if len(big) <= 1:
        return unw  # nothing sizeable to bridge

    ref = max(big, key=lambda lab: sizes[lab])  # largest region = MST root
    bcoords = _boundary_coords(region, big, max_boundary)

    # Complete graph of closest-boundary distances; remember the endpoint pair
    # (parent-side first) for each edge. K is small (sizeable regions only).
    K = len(big)
    dist = np.full((K, K), np.inf)
    endpts: dict[tuple[int, int], tuple[NDArray, NDArray]] = {}
    for a in range(K):
        bi = bcoords[big[a]]
        for b in range(a + 1, K):
            bj = bcoords[big[b]]
            d2 = (bi[:, None, 0] - bj[None, :, 0]) ** 2 + (
                bi[:, None, 1] - bj[None, :, 1]
            ) ** 2
            fi, fj = np.unravel_index(int(np.argmin(d2)), d2.shape)
            dist[a, b] = dist[b, a] = float(np.sqrt(d2[fi, fj]))
            endpts[(a, b)] = (bi[fi], bj[fj])  # (coord in big[a], coord in big[b])

    # Prim's MST rooted at the reference; record edges in growth order so a
    # parent is always already corrected when its child is processed.
    in_tree = [False] * K
    in_tree[big.index(ref)] = True
    edges: list[tuple[int, int]] = []  # (parent_idx, child_idx)
    for _ in range(K - 1):
        best = None
        for u in range(K):
            if not in_tree[u]:
                continue
            for v in range(K):
                if in_tree[v] or not np.isfinite(dist[u, v]):
                    continue
                if best is None or dist[u, v] < best[0]:
                    best = (dist[u, v], u, v)
        if best is None:
            break  # graph not fully connected (shouldn't happen for a clique)
        _, u, v = best
        in_tree[v] = True
        edges.append((u, v))

    out = unw.copy()
    # The endpoint median must stay LOCAL relative to the scene: a box that grows
    # to a large fraction of the frame reintroduces the within-region ramp, which
    # then mis-rounds as a gauge jump. Cap to a scene-relative size so the window
    # is ~500 px on a NISAR-sized frame but shrinks on small frames.
    r = int(min(radius, max(16, min(m, n) // 8)))
    for u_idx, v_idx in edges:
        # Recover (parent endpoint, child endpoint) from the a<b storage order.
        if u_idx < v_idx:
            yx_par, yx_chi = endpts[(u_idx, v_idx)]
        else:
            yx_chi, yx_par = endpts[(v_idx, u_idx)]
        par_lab, chi_lab = big[u_idx], big[v_idx]

        val_par = _endpoint_median(out, region, par_lab, yx_par, r)
        val_chi = _endpoint_median(out, region, chi_lab, yx_chi, r)
        if not (np.isfinite(val_par) and np.isfinite(val_chi)):
            continue
        s = -int(np.rint((val_chi - val_par) / tau))  # cycles to add to the child
        if s != 0:
            out[region == chi_lab] += tau * s
    return out


def _endpoint_median(
    unw: "NDArray[np.float32]",
    region: "NDArray[np.integer]",
    lab: int,
    yx: "NDArray[np.float64]",
    radius: int,
) -> float:
    """Median unwrapped phase of ``lab``'s pixels in a square box of half-width
    ``radius`` around the endpoint ``yx`` (the region's local level at the gap)."""
    m, n = unw.shape
    y, x = int(yx[0]), int(yx[1])
    y0, y1 = max(0, y - radius), min(m, y + radius + 1)
    x0, x1 = max(0, x - radius), min(n, x + radius + 1)
    sub_unw = unw[y0:y1, x0:x1]
    sub_reg = region[y0:y1, x0:x1]
    vals = sub_unw[sub_reg == lab]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))
