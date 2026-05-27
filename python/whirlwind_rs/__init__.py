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


def goldstein_prefilter(
    igram: "NDArray[np.complex64]",
    alpha: float = 0.7,
    win: int = 64,
    step: int | None = None,
) -> "NDArray[np.complex64]":
    """Adaptive (Goldstein) phase pre-filter for the wrapped interferogram.

    Multiplies the FFT magnitude of each overlapping block by ``|F|^alpha``
    and inverse-transforms. The result is a phase-only image (magnitudes
    discarded — the filter operates on unit-amplitude phasors so SLC
    amplitude does not bias toward a low-pass filter on coherent areas).

    For noisy real-data interferograms (NISAR, multilooked Sentinel-1, etc.)
    pre-filtering with the defaults before calling
    :func:`unwrap_with_conncomp` brings ww into close agreement with SNAPHU
    on coherent regions. Without it, high-frequency phase noise leaks into
    the 7×7 smoothed-gradient estimate that feeds the Carballo cost, and
    the MCF picks a topologically different "cut" than SNAPHU's
    variance-aware ``smooth`` cost would. Validated on a 40 MHz NISAR HH
    GSLC: 80% → 99.5% within ±π/2 vs SNAPHU.

    Parameters
    ----------
    igram : complex64
        Wrapped interferogram. Mask out-of-data pixels to ``0+0j``
        beforehand (consistent with the rest of the ww API).
    alpha : float, default 0.7
        FFT-magnitude exponent. 0 = identity; 1 = strongest filter.
    win : int, default 64
        Square block size for the windowed FFT.
    step : int, optional
        Stride between block top-lefts. Defaults to ``win // 2``
        (50% overlap, recommended).
    """
    h, w = igram.shape
    if step is None:
        step = win // 2
    mag = np.abs(igram)
    z = np.where(mag > 0, igram / np.maximum(mag, 1e-30), 0).astype(np.complex64)
    w1 = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win) / (win - 1)))
    w2 = (w1[:, None] * w1[None, :]).astype(np.float32)
    out = np.zeros_like(z)
    wsum = np.zeros((h, w), dtype=np.float32)
    for i0 in range(0, h - win + 1, step):
        for j0 in range(0, w - win + 1, step):
            block = z[i0 : i0 + win, j0 : j0 + win]
            f = np.fft.fft2(block)
            f_filt = f * (np.abs(f) ** alpha)
            block_filt = np.fft.ifft2(f_filt).astype(np.complex64)
            out[i0 : i0 + win, j0 : j0 + win] += block_filt * w2
            wsum[i0 : i0 + win, j0 : j0 + win] += w2
    valid = wsum > 0
    out[valid] /= wsum[valid]
    return out.astype(np.complex64)


__all__ = [
    "closure_correct",
    "closure_refine_mcf",
    "compute_residues",
    "diagonal_ramp",
    "goldstein_prefilter",
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
