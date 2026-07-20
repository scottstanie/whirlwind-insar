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
    bridge_components as bridge_components,
    compute_residues as compute_residues,
    goldstein as goldstein,
    interpolate as interpolate,
    label_components as label_components,
    num_threads as num_threads,
    set_num_threads as set_num_threads,
    unwrap_sparse as unwrap_sparse,
    wrap_phase as wrap_phase,
)

def cost_threshold_from_cycle_prob(cycle_prob: float) -> int: ...
def conncomp_min_coherence_auto(nlooks: float) -> float:
    """Looks-aware default conncomp coherence floor (0.32/sqrt(nlooks), clipped
    to [0.02, 0.30]); 0.08 at nlooks=16."""
    ...

def conncomp_reliability_from_coherence(coherence: float, nlooks: float) -> float:
    """``conncomp_reliability`` value (in 1/sigma2 units) that cuts conncomp edges
    below a target coherence (``1 / sigma2(coherence)``). A guessable way to set
    the knob; see the full docstring in ``whirlwind/__init__.py``."""

def unwrap(
    igram: NDArray[np.complex64],
    corr: NDArray[np.float32],
    nlooks: float,
    mask: NDArray[np.bool_] | None = ...,
    *,
    bridge: bool = ...,
    downsample: int = ...,
    interpolate: bool = ...,
    interp_across_mask: bool = ...,
    interp_cutoff: float = ...,
    interp_num_neighbors: int = ...,
    interp_max_radius: int = ...,
    interp_min_radius: int = ...,
    interp_alpha: float = ...,
    conncomp_algorithm: str = ...,
    conncomp_min_coherence: float | str | None = ...,
    conncomp_reliability: float = ...,
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
    regions. Connected components default to the SNAPHU-faithful
    ``conncomp_algorithm="snaphu"`` grow, tuned by ``conncomp_reliability``
    (raise to label fewer, lower-coherence pixels). See the full docstring in
    ``whirlwind/__init__.py`` for all parameters.
    """
