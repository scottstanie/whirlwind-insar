"""Create phase paths across bounded nodata runs before unwrapping.

The path extrapolates local unwrapped phase slope from both sides of a gap and
reconciles the two estimates by an integer number of cycles. This differs from
phasor interpolation, whose circular mean contains no spatial winding number.
The generated pixels are solver inputs rather than measurements and must be
removed from returned phase and connected-component products.
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

    Both values are robust estimates from the locally unwrapped segment.
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

    Rows and columns are scanned independently; the row estimate is retained
    where both scans reach the same pixel. Only bounded invalid runs no wider
    than ``max_gap`` and with at least eight valid samples on each side are
    crossed. The selection is geometric and does not distinguish subswath gaps
    from water, layover, shadow, or other invalid holes.

    Parameters
    ----------
    igram : complex64 array
        Wrapped interferogram. Pixels outside ``mask`` are ignored.
    mask : bool array
        Valid-pixel mask. ``False`` runs enclosed by valid data on both sides
        are the gaps this crosses.
    max_gap : int
        Widest bounded invalid run to cross, in pixels.
    edge : int
        Pixels either side used to estimate the local level and fringe rate.

    Returns
    -------
    connected : complex64 array
        ``igram`` with the gap pixels populated (magnitude 1).
    added : bool array
        Pixels synthesised by this operation.
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

    # Both scans use the input mask, so one axis never extrapolates from phase
    # synthesised by the other. The row estimate wins on overlap.
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
