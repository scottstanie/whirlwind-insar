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
    (Goldstein & Werner 1998) before unwrapping. On a 40 MHz NISAR HH
    GSLC this raises agreement-with-SNAPHU from 80% → 99.5% within
    ±π/2 by stripping spatial-frequency phase noise that the Carballo
    cost's 7×7 box-filtered gradient estimate can't reach.

    To disable Goldstein (e.g. on already-filtered or synthetic data),
    pass ``goldstein_alpha=0``.

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
    if goldstein_alpha > 0:
        igram = goldstein(igram, alpha=goldstein_alpha, psize=goldstein_psize)
        if mask is not None:
            igram = igram.copy()
            igram[~mask] = 0
    return _unwrap_with_conncomp_native(
        igram,
        corr,
        nlooks,
        mask=mask,
        cost_threshold=cost_threshold,
        min_size_frac=min_size_frac,
        max_ncomps=max_ncomps,
    )


def goldstein(
    phase: "NDArray[np.complex64] | NDArray[np.float64]",
    alpha: float = 0.7,
    psize: int = 64,
) -> "NDArray[np.complex64]":
    """Goldstein adaptive phase filter (Goldstein & Werner 1998).

    Closely follows ``dolphin.goldstein`` (reflect padding,
    ``|F|^alpha`` magnitude shaping) with two changes that matter for
    *unwrapping* (vs visualisation):
    1. **Input is normalised to unit magnitude** before filtering. With
       raw SLC magnitudes, bright pixels (urban) dominate the FFT spectral
       peak, so ``|F|^alpha`` enhances *amplitude* structure instead of
       *phase* structure.
    2. **Hann (cosine) overlap-add window** instead of triangle. Triangle
       has discontinuous slope at the block centre, which leaks more
       cross-block phase artifacts into the gradient estimate.

    Ablation on the NISAR HH test scene (agreement with SNAPHU within
    ±π/2): raw dolphin = 87% → unit-mag + triangle = 93% → unit-mag +
    Hann = 99.5%. Goldstein is on by default in
    :func:`unwrap_with_conncomp`.

    Parameters
    ----------
    phase : complex64 or float
        Wrapped interferogram (complex) or wrapped phase (real, will be
        promoted to ``exp(i·phase)``).
    alpha : float, default 0.7
        ``[0, 1]``. 0 = identity, 1 = maximum filtering.
    psize : int, default 64
        Square FFT patch size.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if np.iscomplexobj(phase):
        data = phase.astype(np.complex64)
    else:
        data = np.exp(1j * phase).astype(np.complex64)
    empty_mask = np.isnan(data) | (data == 0)
    if np.all(empty_mask):
        return data
    # Normalise to unit magnitude — see docstring on why this matters for
    # unwrapping vs visualisation.
    mag = np.abs(data)
    data = np.where(mag > 0, data / np.maximum(mag, 1e-30), 0).astype(np.complex64)

    nrows, ncols = data.shape
    step = psize // 2
    pad_top = step
    pad_left = step
    pad_bottom = step + (step - (nrows % step)) % step
    pad_right = step + (step - (ncols % step)) % step
    data_padded = np.pad(
        data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect"
    )
    # Hann window — smoother taper than triangle. On NISAR HH the
    # triangle/Hann choice moves the diff from 92.9% → 99.5% within ±π/2.
    w1 = 0.5 * (1 - np.cos(2 * np.pi * np.arange(psize) / (psize - 1)))
    weight = (w1[:, None] * w1[None, :]).astype(np.float32)
    out = np.zeros(data_padded.shape, dtype=np.complex64)
    weight_sum = np.zeros(data_padded.shape, dtype=np.float32)
    pr, pc = data_padded.shape
    for i in range(0, pr - psize + 1, step):
        for j in range(0, pc - psize + 1, step):
            block = data_padded[i : i + psize, j : j + psize]
            f = np.fft.fft2(block)
            f = (np.abs(f) ** alpha) * f
            block_filt = np.fft.ifft2(f).astype(np.complex64)
            out[i : i + psize, j : j + psize] += weight * block_filt
            weight_sum[i : i + psize, j : j + psize] += weight
    valid = weight_sum > 0
    out[valid] = out[valid] / weight_sum[valid]
    out = out[pad_top : pad_top + nrows, pad_left : pad_left + ncols]
    out[empty_mask] = 0
    return out


__all__ = [
    "closure_correct",
    "closure_refine_mcf",
    "compute_residues",
    "diagonal_ramp",
    "goldstein",
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
