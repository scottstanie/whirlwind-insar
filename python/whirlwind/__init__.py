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
    unwrap_crlb,
    unwrap_crlb_grounded,
    unwrap_convex,
    unwrap_grounded,
    unwrap_reuse,
    unwrap_sparse,
    wrap_phase,
)
from ._native import _unwrap_native

if TYPE_CHECKING:
    from numpy.typing import NDArray


def unwrap_crlb_stack(
    igram_cube: "NDArray[np.complex64]",
    variance_cube: "NDArray[np.float32]",
    mask: "NDArray[np.bool_] | None" = None,
    cost_threshold: int = 50,
    min_size_px: int = 100,
    max_ncomps: int = 1024,
) -> "tuple[NDArray[np.float32], NDArray[np.uint32]]":
    """Per-IG CRLB unwrap + conncomp over a 3D stack.

    Loops over the leading axis calling :func:`unwrap_crlb`
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
    cost_threshold, min_size_px, max_ncomps
        Forwarded to :func:`unwrap_crlb`. See that function
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
        unw_out[e], cc_out[e] = unwrap_crlb(
            ig,
            var,
            mask=m_e,
            cost_threshold=cost_threshold,
            min_size_px=min_size_px,
            max_ncomps=max_ncomps,
        )
    return unw_out, cc_out


def unwrap(
    igram: "NDArray[np.complex64]",
    corr: "NDArray[np.float32]",
    nlooks: float,
    mask: "NDArray[np.bool_] | None" = None,
    *,
    multilook: int = 1,
    tile_size: int = 0,
    tile_overlap: int = 0,
    cost_threshold: int = 50,
    min_size_px: int = 100,
    max_ncomps: int = 1024,
    goldstein_alpha: float = 0.0,
    goldstein_psize: int = 64,
) -> "tuple[NDArray[np.float32], NDArray[np.uint32]]":
    """MCF unwrap returning ``(unwrapped_phase, conn_components)``.

    The main entry point. Runs the robust tiled pipeline for the phase
    (auto-tile large frames at 512, gated multi-shift re-solve, global
    coarse anchor, multi-scale cascade, seam-repair) and grows SNAPHU-style
    connected components globally from the same Carballo coherence cost.

    Optionally pre-filters with the Goldstein adaptive filter (off by
    default; set ``goldstein_alpha > 0`` to enable). When enabled, Goldstein
    (Goldstein & Werner 1998) informs the MCF, then the resulting 2π·k
    integer-cycle field is *applied to the original wrapped phase* — so the
    output preserves every per-pixel phase value the caller passed in;
    Goldstein only decides which 2π cycle each pixel sits on. (The
    Goldstein-on vs -off trade-off is currently under evaluation.)

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
    multilook : int, default 1
        > 1 coherently down-looks first (noisy / moderate-coherence scenes),
        unwraps the coarse frame, then upsamples.
    tile_size, tile_overlap : int
        ``tile_size=0`` (default) auto-tiles frames larger than 512 px at
        512 / overlap-64. Set ``tile_size`` ≥ 4 (with ``tile_overlap`` ≥ 2)
        to force a tile size.
    cost_threshold, min_size_px, max_ncomps :
        Connected-component growing parameters (see
        :class:`whirlwind_core::conncomp::ConnCompParams`).
    goldstein_alpha : float, default 0.0
        Goldstein filter strength in ``[0, 1]``. 0 (default) disables
        filtering; a typical "on" value is 0.7.
    goldstein_psize : int, default 64
        Goldstein FFT patch size (only used when ``goldstein_alpha > 0``).

    Returns
    -------
    unwrapped : float32, shape ``(m, n)``
    conncomp : uint32, shape ``(m, n)``
        Component labels; 0 = background / dropped.
    """
    if goldstein_alpha <= 0:
        return _unwrap_native(
            igram, corr, nlooks,
            mask=mask, tile_size=tile_size, tile_overlap=tile_overlap, multilook=multilook,
            cost_threshold=cost_threshold, min_size_px=min_size_px, max_ncomps=max_ncomps,
        )

    ig_filt = goldstein(igram, alpha=goldstein_alpha, psize=goldstein_psize)
    if mask is not None:
        ig_filt = ig_filt.copy()
        ig_filt[~mask] = 0
    unw_filt, cc = _unwrap_native(
        ig_filt, corr, nlooks,
        mask=mask, tile_size=tile_size, tile_overlap=tile_overlap, multilook=multilook,
        cost_threshold=cost_threshold, min_size_px=min_size_px, max_ncomps=max_ncomps,
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
    "unwrap_convex",
    "unwrap_grounded",
    "unwrap_reuse",
    "unwrap_sparse",
    "wrap_phase",
]
