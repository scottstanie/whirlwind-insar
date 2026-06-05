"""Type stubs for the `whirlwind` package.

``unwrap`` is the Python wrapper defined in ``__init__.py``; everything below is
re-exported from the ``_native`` extension. The experimental CRLB unwrappers and
the ``unwrap_reuse`` solver are importable but intentionally kept off the public
API (not in ``__all__``) until validated.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__version__: str

from ._native import (
    closure_correct as closure_correct,
    closure_refine_mcf as closure_refine_mcf,
    compute_residues as compute_residues,
    diagonal_ramp as diagonal_ramp,
    goldstein as goldstein,
    label_components as label_components,
    num_threads as num_threads,
    quality_map as quality_map,
    quality_triangles as quality_triangles,
    set_num_threads as set_num_threads,
    simulate_ifg as simulate_ifg,
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
    multilook: int = ...,
    cost_threshold: int = ...,
    conncomp_cycle_prob: float | None = ...,
    conncomp_sigma: float | None = ...,
    conncomp_coh_floor: float | None = ...,
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
    ``conncomp_cycle_prob`` / ``conncomp_coh_floor``) and other parameters.
    """
