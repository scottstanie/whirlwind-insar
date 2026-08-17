"""Connect regions split apart by nodata, so the solver can level them itself.

Why this exists
---------------
A NISAR fixed-PRF frame arrives cut into subswath stripes separated by ~140 px
(~11 km) of nodata. Nothing in the interferogram connects one stripe to the
next, so an MCF integrator seeds each stripe independently and every stripe ends
up on an arbitrary 2*pi level. Something then has to choose those integers.

Choosing them *after* the solve (see :func:`whirlwind.bridge_components`) means
guessing the phase change across a gap you have no data for; the minimum-jump
rule assumes it is zero, which is wrong by a cycle or more whenever a bulk
ionospheric or tropospheric signal crosses the gap.

Connecting the regions first makes the frame a single component, so the solve
resolves the levels as part of the ordinary cost minimisation, using the real
data on both sides. The synthesised pixels are handed to the solver at a low
coherence: cheap enough that the MCF will route the required cycles through
them, distrusted enough that they never outvote real data.

Why not the interpolator
------------------------
:func:`whirlwind.interpolate` also writes phase from surrounding pixels, so it
is worth being precise about why it is not the same tool.

It computes ``angle(sum_i w_i * exp(i*phi_i))`` over nearby high-coherence
pixels -- a weighted circular mean. A circular mean is the short-arc answer by
construction: average two edges four fringes apart and you get the phase halfway
around the *short* way, not four fringes of continuation. That is exactly the
quantity a gap crossing has to preserve, so the interpolator computes the wrong
value for this job even where it runs.

Two practical things stop it running at all on nodata, both incidental rather
than fundamental: it skips pixels whose phasor is exactly zero, and its output
is scaled by the pixel's own magnitude (``amp * exp(i*ang)``), so a 0+0j pixel
would stay zero even with that skip removed. Its ``interp_max_radius`` (101 px
by default) is also shorter than a NISAR subswath gap, so the middle of a gap
has no neighbours to average.

The fill continues the fringe rate
----------------------------------
So instead of averaging, each gap is crossed by extrapolating the local fringe
rate from both sides, picking the integer that makes the two extrapolations
agree at the gap centre, and blending between them. On simulated scenes with
known truth this stays exact out to at least 1.4 cycles of true phase change per
gap, where an interpolating fill breaks at 0.5.

Note that this per-crossing integer is a *proposal*, not the final answer: it
fixes the phase along one synthesised line, and the solver still chooses each
region's 2*pi level globally, weighing every crossing against its cost model. A
few crossings whose extrapolation is wrong do not decide the region.
"""

from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi

__all__ = ["connect_gaps"]


def _row_runs(row_valid: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` spans of consecutive valid pixels in a line."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], row_valid.view(np.int8), [0]))))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _edge_level_and_slope(phase: np.ndarray) -> tuple[float, float]:
    """Level and per-pixel fringe rate at the far end of a 1-D phase segment.

    Both are medians (of the unwrapped segment, and of its first differences),
    so a handful of decorrelated pixels in the window cannot swing the estimate.
    """
    unwrapped = np.unwrap(phase)
    slope = float(np.median(np.diff(unwrapped)))
    level = float(np.median(unwrapped[-8:]))
    return level, slope


def _scan_lines(
    wrapped: np.ndarray, mask: np.ndarray, max_gap: int, edge: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cross every interior gap along axis 1, returning (phase, filled) maps.

    Operates line by line along rows; callers pass a transposed view to get the
    same treatment down columns.
    """
    phase_out = np.zeros(mask.shape, dtype=np.float64)
    filled = np.zeros(mask.shape, dtype=bool)

    for i in range(mask.shape[0]):
        runs = _row_runs(mask[i])
        for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
            gap = a2 - b1
            if not (1 <= gap <= max_gap):
                continue
            left = wrapped[i, max(a1, b1 - edge) : b1]
            right = wrapped[i, a2 : min(b2, a2 + edge)]
            if left.size < 8 or right.size < 8:
                continue
            lvl_l, slope_l = _edge_level_and_slope(left)
            # Mirror the right window so its "far end" is also the gap side.
            lvl_r, slope_r = _edge_level_and_slope(right[::-1])
            slope_r = -slope_r

            # Extrapolate both sides to the gap centre and pick the integer that
            # reconciles them: this is what carries the fringe count across.
            centre = (gap + 1) / 2.0
            k = round(((lvl_l + slope_l * centre) - (lvl_r - slope_r * centre)) / TWOPI)

            x = np.arange(1, gap + 1, dtype=np.float64)
            w = x / (gap + 1)
            phase_out[i, b1:a2] = (1.0 - w) * (lvl_l + slope_l * x) + w * (
                lvl_r + TWOPI * k - slope_r * (gap + 1 - x)
            )
            filled[i, b1:a2] = True

    return phase_out, filled


def connect_gaps(
    igram: np.ndarray,
    mask: np.ndarray,
    max_gap: int = 300,
    edge: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise phase across interior nodata gaps to join separated regions.

    Each gap is crossed by continuing the local fringe rate in from both sides,
    so the number of fringes spanning the gap is carried across rather than
    minimised. See the module docstring for why that distinction matters.

    Which pixels this touches
    -------------------------
    Every run of invalid pixels lying **between two valid runs along a row or
    down a column**, up to ``max_gap`` wide. Both axes are scanned, so a frame
    cut into vertical stripes (NISAR's cross-track subswath gaps) and one cut
    into horizontal bands (an along-track gap) are handled the same way. Where a
    hole is crossable both ways, the row crossing is the one kept.

    It is a geometric rule, not a NISAR-specific one: interior holes from water,
    layover or shadow masking are crossed too if they are narrow enough. If you
    do not want a particular kind of hole crossed, either keep it wider than
    ``max_gap`` or leave those pixels valid in ``mask``.

    Gaps touching the frame edge are never crossed -- they have only one side to
    extrapolate from -- and neither are gaps with fewer than 8 valid pixels
    available on the near side.

    Parameters
    ----------
    igram : complex64 array
        Wrapped interferogram. Pixels outside ``mask`` are ignored.
    mask : bool array
        Valid-pixel mask. ``False`` runs enclosed by valid data on both sides
        are the gaps this crosses.
    max_gap : int
        Widest gap to cross, in pixels. Wider holes are left alone -- past some
        width the fringe rate on either side stops predicting the far side, and
        an untouched gap that gets bridged afterwards is better than a
        confidently wrong crossing.
    edge : int
        Pixels either side used to estimate the local level and fringe rate.

    Returns
    -------
    connected : complex64 array
        ``igram`` with the gap pixels populated (magnitude 1).
    added : bool array
        Which pixels were synthesised. Add these to the mask before solving, and
        drop them from the result afterward -- they are not measurements.

    Examples
    --------
    >>> connected, added = connect_gaps(igram, mask)
    >>> unw, cc = unwrap(connected, np.where(added, 0.1, corr), nlooks,
    ...                  mask | added)
    >>> unw[added] = 0  # synthesised pixels carry no measurement

    Passing ``connect_gaps=True`` to :func:`whirlwind.unwrap` does all of that
    for you.
    """
    igram = np.ascontiguousarray(igram, dtype=np.complex64)
    mask = np.ascontiguousarray(mask, dtype=bool)
    if igram.shape != mask.shape:
        raise ValueError(f"igram {igram.shape} and mask {mask.shape} must match")
    if max_gap < 1:
        raise ValueError(f"max_gap must be >= 1, got {max_gap}")
    if edge < 8:
        raise ValueError(f"edge must be >= 8, got {edge}")

    wrapped = np.angle(igram).astype(np.float64)

    # Scan rows, then columns via a transposed view. Both run against the
    # ORIGINAL mask so a column crossing never extrapolates from phase the row
    # pass invented; on overlap the row result wins.
    phase_r, added_r = _scan_lines(wrapped, mask, max_gap, edge)
    phase_c, added_c = _scan_lines(
        np.ascontiguousarray(wrapped.T), np.ascontiguousarray(mask.T), max_gap, edge
    )
    phase_c, added_c = phase_c.T, added_c.T

    out = igram.copy()
    only_c = added_c & ~added_r
    out[only_c] = np.exp(1j * phase_c[only_c]).astype(np.complex64)
    out[added_r] = np.exp(1j * phase_r[added_r]).astype(np.complex64)
    return out, added_r | added_c
