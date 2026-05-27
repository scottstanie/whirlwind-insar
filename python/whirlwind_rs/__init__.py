"""whirlwind-rs: Rust-backed InSAR phase unwrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._native import (
    closure_correct,
    closure_refine_mcf,
    compute_residues,
    diagonal_ramp,
    quality_map,
    quality_triangles,
    simulate_ifg,
    unwrap,
    unwrap_crlb,
    unwrap_crlb_grounded,
    unwrap_crlb_with_conncomp,
    unwrap_with_conncomp,
    wrap_phase,
)

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


__all__ = [
    "closure_correct",
    "closure_refine_mcf",
    "compute_residues",
    "diagonal_ramp",
    "quality_map",
    "quality_triangles",
    "simulate_ifg",
    "unwrap",
    "unwrap_crlb",
    "unwrap_crlb_grounded",
    "unwrap_crlb_stack",
    "unwrap_crlb_with_conncomp",
    "unwrap_with_conncomp",
    "wrap_phase",
]
