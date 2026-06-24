"""Reference helper for the isce3 GUNW `algorithm: whirlwind` branch.

This is a self-contained version of the unwrapping call used by the isce3 RUNW
workflow (`nisar/workflows/unwrap.py`). It is provided so a team maintaining
their own isce3 fork can drop it into `nisar/unwrap/` and call it from the
workflow, instead of inlining the logic. The in-tree `unwrap.py` branch inlines
the same call, so this file is optional.

It deliberately has no isce3 imports: it takes plain numpy arrays and returns
plain numpy arrays, so it is trivial to unit-test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def run_whirlwind(
    igram: "NDArray[np.complexfloating]",
    coherence: "NDArray[np.floating]",
    nlooks: float,
    mask: "NDArray[np.bool_] | None" = None,
    *,
    bridge: bool = True,
    downsample: int = 1,
    conncomp_reliability: float = 0.0,
    goldstein_alpha: float = 0.0,
) -> "tuple[NDArray[np.float32], NDArray[np.uint32]]":
    """Unwrap a wrapped interferogram with whirlwind.

    Parameters
    ----------
    igram : complex array
        Wrapped interferogram (any complex dtype; cast to complex64).
    coherence : float array
        Coherence in [0, 1], same shape as ``igram``.
    nlooks : float
        Effective number of looks of ``coherence`` (>= 1).
    mask : bool array, optional
        Valid-pixel mask, ``True`` = valid. If your mask file follows the
        SNAPHU/isce3 convention (nonzero = valid), pass ``raster != 0``. If you
        have an *invalid*-pixel mask (``True`` = invalid), pass ``~invalid``.
    bridge, downsample, conncomp_reliability, goldstein_alpha
        See ``whirlwind.unwrap``. ``bridge`` re-levels regions the mask splits
        apart; skip the generic isce3 bridge post-pass when using this.

    Returns
    -------
    (unwrapped_phase, connected_components)
        ``float32`` phase and ``uint32`` component labels, same shape as input.
    """
    import whirlwind as ww

    unw, conncomp = ww.unwrap(
        np.ascontiguousarray(igram, dtype=np.complex64),
        np.ascontiguousarray(coherence, dtype=np.float32),
        float(nlooks),
        mask,
        bridge=bridge,
        downsample=downsample,
        conncomp_reliability=conncomp_reliability,
        goldstein_alpha=goldstein_alpha,
    )
    return np.asarray(unw, dtype=np.float32), np.asarray(conncomp, dtype=np.uint32)
