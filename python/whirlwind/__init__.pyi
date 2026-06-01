"""Type stubs for the `whirlwind` package.

``unwrap`` and ``unwrap_crlb_stack`` are Python wrappers defined in
``__init__.py``; everything else is re-exported from the ``_native`` extension.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._native import (
    closure_correct as closure_correct,
    closure_refine_mcf as closure_refine_mcf,
    compute_residues as compute_residues,
    diagonal_ramp as diagonal_ramp,
    goldstein as goldstein,
    num_threads as num_threads,
    quality_map as quality_map,
    quality_triangles as quality_triangles,
    set_num_threads as set_num_threads,
    simulate_ifg as simulate_ifg,
    unwrap_convex as unwrap_convex,
    unwrap_crlb as unwrap_crlb,
    unwrap_crlb_grounded as unwrap_crlb_grounded,
    unwrap_grounded as unwrap_grounded,
    unwrap_pyramid as unwrap_pyramid,
    unwrap_reuse as unwrap_reuse,
    unwrap_sparse as unwrap_sparse,
    wrap_phase as wrap_phase,
)

def unwrap(
    igram: NDArray[np.complex64],
    corr: NDArray[np.float32],
    nlooks: float,
    mask: NDArray[np.bool_] | None = ...,
    *,
    multilook: int = ...,
    tile_size: int = ...,
    tile_overlap: int = ...,
    cost_threshold: int = ...,
    min_size_px: int = ...,
    max_ncomps: int = ...,
    goldstein_alpha: float = ...,
    goldstein_psize: int = ...,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """MCF unwrap returning ``(phase, conn_components)``.

    Robust tiled pipeline + global connected components + (by default)
    Goldstein pre-filtering. The main entry point. See the docstring in
    ``whirlwind/__init__.py`` for parameter detail.
    """

def unwrap_crlb_stack(
    igram_cube: NDArray[np.complex64],
    variance_cube: NDArray[np.float32],
    mask: NDArray[np.bool_] | None = ...,
    cost_threshold: int = ...,
    min_size_px: int = ...,
    max_ncomps: int = ...,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """Per-IG CRLB unwrap + connected components over a 3D stack."""
