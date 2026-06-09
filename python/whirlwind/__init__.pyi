"""Type stubs for the `whirlwind` package.

``unwrap`` is the Python wrapper defined in ``__init__.py``; everything below is
re-exported from the ``_native`` extension. The experimental CRLB / ``unwrap_reuse``
solvers, the temporal-closure stack functions, the quality maps, and the synthetic
scene generators are importable but intentionally kept off the public API (not in
``__all__``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__version__: str

from ._native import (
    compute_residues as compute_residues,
    goldstein as goldstein,
    interpolate as interpolate,
    label_components as label_components,
    num_threads as num_threads,
    set_num_threads as set_num_threads,
    unwrap_sparse as unwrap_sparse,
    wrap_phase as wrap_phase,
)

def unwrap(
    igram: NDArray[np.complex64],
    corr: NDArray[np.float32],
    nlooks: float,
    mask: NDArray[np.bool_] | None = ...,
    *,
    bridge: bool = ...,
    downsample: int = ...,
    interpolate: bool = ...,
    interp_cutoff: float = ...,
    interp_num_neighbors: int = ...,
    interp_max_radius: int = ...,
    interp_min_radius: int = ...,
    interp_alpha: float = ...,
    cost_threshold: int = ...,
    conncomp_cycle_prob: float | None = ...,
    conncomp_sigma: float | None = ...,
    min_size_px: int = ...,
    max_ncomps: int = ...,
    goldstein_alpha: float = ...,
    goldstein_psize: int = ...,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """Unwrap an interferogram, returning ``(unwrapped_phase, conncomp)``.

    The main entry point: an exact MCF solver (SNAPHU-comparable quality, faster)
    plus a default-on ``bridge`` post-pass that re-levels mask-disconnected
    regions. See the full docstring in ``whirlwind/__init__.py`` for the
    connected-component knobs (``cost_threshold`` / ``conncomp_sigma`` /
    ``conncomp_cycle_prob``) and other parameters.
    """

def bridge_components(
    unw: NDArray[np.float32],
    mask: NDArray[np.bool_] | None = ...,
    *,
    radius: int = ...,
    min_px: int = ...,
    max_boundary: int = ...,
) -> NDArray[np.float32]:
    """Re-level the disconnected regions of an unwrapped phase image.

    The MST gauge-bridging post-pass `unwrap(bridge=True)` applies by default,
    exposed standalone. See ``whirlwind/_bridge.py`` for the algorithm.
    """
