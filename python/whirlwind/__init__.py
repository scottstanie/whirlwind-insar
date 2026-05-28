"""whirlwind-rs: Rust-backed InSAR phase unwrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._native import (
    closure_correct,
    closure_refine_mcf,
    compute_residues,
    diagonal_ramp,
    goldstein,
    num_threads,
    quality_map,
    quality_triangles,
    set_num_threads,
    simulate_ifg,
    unwrap,
    unwrap_crlb,
    unwrap_crlb_grounded,
    unwrap_crlb_with_conncomp,
    unwrap_grounded,
    unwrap_reuse,
    unwrap_sparse,
    wrap_phase,
)
from ._native import unwrap_with_conncomp as _unwrap_with_conncomp_native

if TYPE_CHECKING:
    from numpy.typing import NDArray


def unwrap_crlb_stack(
    igram_cube: "NDArray[np.complex64]",
    variance_cube: "NDArray[np.float32]",
    mask: "NDArray[np.bool_] | None" = None,
    cost_threshold: int = 50,
    min_size_frac: float = 0.01,
    max_ncomps: int = 64,
) -> "tuple[NDArray[np.float32], NDArray[np.uint32]]":
    """Per-IG CRLB unwrap + conncomp over a 3D stack.

    Loops over the leading axis calling :func:`unwrap_crlb_with_conncomp`
    per IG. Each per-IG MCF solve is independent, so this is just a
    convenient Python wrapper — there is no shared state between IGs. For
    parallel execution use ``multiprocessing`` or ``concurrent.futures``.

    Parameters
    ----------
    igram_cube : complex64, shape ``(E, m, n)``
        Stack of complex interferograms.
    variance_cube : float32, shape ``(E, m, n)``
        Per-pixel CRLB-derived phase variance σ²_IG = σ²_a + σ²_b in
        rad² (typically ``crlb_<date_a>.tif + crlb_<date_b>.tif``).
    mask : bool, optional
        Either ``(m, n)`` (one mask for the whole stack) or
        ``(E, m, n)`` (per-IG mask). ``False`` ⇒ excluded pixel.
    cost_threshold, min_size_frac, max_ncomps
        Forwarded to :func:`unwrap_crlb_with_conncomp`. See that function
        for the meaning of each.

    Returns
    -------
    unwrapped : float32, shape ``(E, m, n)``
    conncomps : uint32, shape ``(E, m, n)``
        Per-IG component labels; 0 = background / dropped.
    """
    if igram_cube.ndim != 3:
        raise ValueError(f"igram_cube must be 3D, got shape {igram_cube.shape}")
    if variance_cube.shape != igram_cube.shape:
        raise ValueError(
            f"variance_cube shape {variance_cube.shape} != igram_cube shape "
            f"{igram_cube.shape}"
        )
    n_edges, m, n = igram_cube.shape

    if mask is not None:
        if mask.ndim == 2:
            assert mask.shape == (m, n), (
                f"2D mask shape {mask.shape} != ({m}, {n})"
            )
            per_ig_mask = None
        elif mask.ndim == 3:
            assert mask.shape == igram_cube.shape, (
                f"3D mask shape {mask.shape} != igram cube shape {igram_cube.shape}"
            )
            per_ig_mask = mask
        else:
            raise ValueError(f"mask must be 2D or 3D, got {mask.ndim}D")
    else:
        per_ig_mask = None

    unw_out = np.empty((n_edges, m, n), dtype=np.float32)
    cc_out = np.empty((n_edges, m, n), dtype=np.uint32)

    for e in range(n_edges):
        ig = np.ascontiguousarray(igram_cube[e], dtype=np.complex64)
        var = np.ascontiguousarray(variance_cube[e], dtype=np.float32)
        m_e = per_ig_mask[e] if per_ig_mask is not None else mask
        unw_out[e], cc_out[e] = unwrap_crlb_with_conncomp(
            ig,
            var,
            mask=m_e,
            cost_threshold=cost_threshold,
            min_size_frac=min_size_frac,
            max_ncomps=max_ncomps,
        )
    return unw_out, cc_out


def unwrap_with_conncomp(
    igram: "NDArray[np.complex64]",
    corr: "NDArray[np.float32]",
    nlooks: float,
    mask: "NDArray[np.bool_] | None" = None,
    cost_threshold: int = 50,
    min_size_frac: float = 0.01,
    max_ncomps: int = 64,
    goldstein_alpha: float = 0.7,
    goldstein_psize: int = 64,
) -> "tuple[NDArray[np.float32], NDArray[np.uint32]]":
    """MCF unwrap + SNAPHU-style connected components (Carballo cost).

    Pre-filters the wrapped phase with the Goldstein adaptive filter
    (Goldstein & Werner 1998) to inform the MCF, then *applies the
    resulting 2π·k integer-cycle field to the original wrapped phase*.
    The output therefore preserves every per-pixel phase value the
    caller passed in; Goldstein only decides which 2π cycle each pixel
    sits on. Set ``goldstein_alpha=0`` to skip the filter entirely.

    Validated on a 40 MHz NISAR HH GSLC: agreement with SNAPHU
    80% → 99.5% within ±π/2 vs the no-filter baseline, while keeping
    the original wrapped phase under the integration.

    Parameters
    ----------
    igram : complex64
        Wrapped interferogram. Mask out-of-data pixels to ``0+0j``.
    corr : float32
        Sample coherence in ``[0, 1]``.
    nlooks : float
        Effective number of looks (≥ 1).
    mask : bool, optional
        Valid-pixel mask (True = valid).
    cost_threshold, min_size_frac, max_ncomps :
        Connected-component growing parameters (see
        :class:`whirlwind_core::conncomp::ConnCompParams`).
    goldstein_alpha : float, default 0.7
        Goldstein filter strength in ``[0, 1]``. 0 disables filtering.
    goldstein_psize : int, default 64
        Goldstein FFT patch size.
    """
    if goldstein_alpha <= 0:
        return _unwrap_with_conncomp_native(
            igram, corr, nlooks,
            mask=mask, cost_threshold=cost_threshold,
            min_size_frac=min_size_frac, max_ncomps=max_ncomps,
        )

    ig_filt = goldstein(igram, alpha=goldstein_alpha, psize=goldstein_psize)
    if mask is not None:
        ig_filt = ig_filt.copy()
        ig_filt[~mask] = 0
    unw_filt, cc = _unwrap_with_conncomp_native(
        ig_filt, corr, nlooks,
        mask=mask, cost_threshold=cost_threshold,
        min_size_frac=min_size_frac, max_ncomps=max_ncomps,
    )
    # Transfer the integer 2π·k field from the filtered unwrap onto the
    # *original* wrapped phase, rounding against the original (not the
    # filtered) phase to avoid the dolphin-#364 artefact:
    # any pixel where Goldstein moved phase across the ±π discontinuity
    # would otherwise pick up a spurious ±2π cycle, producing visible
    # outlines along fringe boundaries.
    tau = np.float32(2 * np.pi)
    phase_orig = np.angle(igram).astype(np.float32)
    k = np.round((unw_filt - phase_orig) / tau).astype(np.float32)
    unw = (phase_orig + tau * k).astype(np.float32)
    if mask is not None:
        unw[~mask] = 0.0
    return unw, cc


# goldstein() is the Rust-backed native binding re-exported from
# ``._native``. See crates/whirlwind-core/src/goldstein.rs for the
# implementation and the ww-specific choices (unit-magnitude
# normalisation, Hann window) that move agreement-with-SNAPHU from
# 87% → 99.5% within ±π/2 on the NISAR HH test scene.


__all__ = [
    "closure_correct",
    "closure_refine_mcf",
    "compute_residues",
    "diagonal_ramp",
    "goldstein",
    "num_threads",
    "quality_map",
    "quality_triangles",
    "set_num_threads",
    "simulate_ifg",
    "unwrap",
    "unwrap_crlb",
    "unwrap_crlb_grounded",
    "unwrap_crlb_stack",
    "unwrap_crlb_with_conncomp",
    "unwrap_grounded",
    "unwrap_reuse",
    "unwrap_sparse",
    "unwrap_with_conncomp",
    "wrap_phase",
]
