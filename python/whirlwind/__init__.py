"""whirlwind-rs: Rust-backed InSAR phase unwrapper."""

from __future__ import annotations

import os
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

# Single source of truth is Cargo.toml ([workspace.package].version); maturin
# stamps it into the installed distribution metadata, which we read back here.
__version__ = version("whirlwind-insar")

from ._native import (
    closure_correct,
    closure_refine_mcf,
    compute_residues,
    diagonal_ramp,
    goldstein,
    interpolate,
    num_threads,
    quality_map,
    quality_triangles,
    set_num_threads,
    simulate_ifg,
    unwrap_crlb,
    unwrap_crlb_grounded,
    unwrap_reuse,
    unwrap_sparse,
    wrap_phase,
)
from ._native import _unwrap_native, _unwrap_with_costs, label_components

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
    convenient Python wrapper - there is no shared state between IGs. For
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
        ``(E, m, n)`` (per-IG mask). ``False`` means an excluded pixel.
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
            assert mask.shape == (m, n), f"2D mask shape {mask.shape} != ({m}, {n})"
            per_ig_mask = None
        elif mask.ndim == 3:
            assert (
                mask.shape == igram_cube.shape
            ), f"3D mask shape {mask.shape} != igram cube shape {igram_cube.shape}"
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


# The coherence connected-component cost is round(CONNCOMP_COST_SCALE · Carballo
# LLR), where the LLR = log(p0/p1) is the log-odds of "one-cycle correction" (p1)
# vs "no correction" (p0) for an edge under the Lee-1994 multilook phase model.
CONNCOMP_COST_SCALE = 6


def cost_threshold_from_cycle_prob(cycle_prob: float) -> int:
    """Connected-component ``cost_threshold`` for a target per-edge one-cycle
    probability.

    An edge is cut (a component boundary) when its cost is ``<= cost_threshold``,
    which happens when its local one-cycle-correction probability is at least
    ``cycle_prob``. This is a local edge reliability, not a global
    residue-pairing probability. A lower ``cycle_prob`` raises the threshold and
    cuts more edges (stricter). The default ``cost_threshold=50`` corresponds to
    a ``cycle_prob`` of about 2.4e-4, roughly a 3.5-sigma Gaussian equivalent.
    This applies to the Carballo coherence connected components only; the CRLB
    inverse-variance cost path does not use this scaling.
    """
    import math

    p = min(max(cycle_prob, 1e-12), 1.0 - 1e-12)
    return round(CONNCOMP_COST_SCALE * math.log((1.0 - p) / p))


def unwrap(
    igram: "NDArray[np.complex64]",
    corr: "NDArray[np.float32]",
    nlooks: float,
    mask: "NDArray[np.bool_] | None" = None,
    *,
    bridge: bool = True,
    multilook: int = 1,
    cost_threshold: int = 50,
    conncomp_cycle_prob: "float | None" = None,
    conncomp_sigma: "float | None" = None,
    conncomp_coh_floor: "float | None" = None,
    min_size_px: int = 100,
    max_ncomps: int = 1024,
    goldstein_alpha: float = 0.0,
    goldstein_psize: int = 64,
) -> "tuple[NDArray[np.float32], NDArray[np.uint32]]":
    """Unwrap a wrapped interferogram with a minimum-cost-flow (MCF) solver.

    Main unwrapping entry point. Estimates the integer number of 2π cycles
    at each pixel, and adds them back to the wrapped phase, returning the
    continuous unwrapped phase together with connected-component labels.

    The ``conncomp`` output labels regions believed to be unwrapped
    self-consistently, with one positive integer per region and ``0`` for
    background or dropped pixels, analogous to SNAPHU's connected components.
    They are grown globally from the coherence cost.

    A fast default post-pass (``bridge``) repairs the relative 2π level of
    regions that the valid mask splits apart, such as land slabs separated by a
    low-coherence river.

    Parameters
    ----------
    igram : ndarray of complex64
        Wrapped interferometric phase.
        Out-of-bounds or nodata pixels should be ``0+0j``.
    corr : ndarray of float32
        Sample coherence in ``[0, 1]``, same shape as ``igram``.
    nlooks : float
        Effective number of looks used to estimate ``corr`` (at least 1). A
        higher number of looks means higher confidence in ``corr`` and sets the
        width of the coherence cost model.
    mask : ndarray of bool, optional
        Valid-pixel mask, ``True`` = valid. Defaults to ``igram != 0``.
    bridge : bool, default True
        Post-processing step that re-levels regions the valid mask splits into
        disconnected pieces (for example two land slabs separated by a
        low-coherence river). The MCF seeds each piece at an arbitrary 2π level,
        so the relative offset between pieces is under-determined. This pass
        snaps each region to a coarse 8x-downlooked anchor (shifts taken
        relative to the largest region), only where the coarse scale connects
        the regions and only when the offset rounds cleanly to an integer. A
        single-region or coherently-connected frame is left unchanged. Fixes the
        NISAR A_025 river frame (58 to about 100 percent) with no regression
        elsewhere. Disable with ``bridge=False`` or ``WHIRLWIND_NO_BRIDGE=1``.
    multilook : int, default 1
        Coarse-solve factor for noisy scenes. When greater than 1, the complex
        interferogram is coherently averaged into ``multilook x multilook``
        blocks, which suppresses the noise the linear cost otherwise mis-routes
        through, and that smaller, smoother frame is unwrapped to decide which
        2π cycle each block sits on. The coarse integer-cycle field is then
        transferred back onto the full-resolution wrapped phase
        (``k = round((coarse_up - wrapped) / 2π)``; ``unw = wrapped + 2π k``),
        so the output keeps every per-pixel wrapped value rather than becoming
        block-constant; only the integer cycle is borrowed from the coarse
        solve. The one thing lost is detail finer than the block scale, which
        genuinely aliases under the downlook. Use it for noisy or
        moderate-coherence scenes (for example Sentinel-1) where a
        full-resolution solve mis-routes; leave it at 1 for clean scenes.
    goldstein_alpha : float, default 0.0
        Goldstein adaptive-filter strength in ``[0, 1]``. 0 (default) disables
        filtering; a typical "on" value is 0.7. When enabled, the filter only
        informs the MCF; the integer cycle field it produces is applied to the
        original wrapped phase, so every per-pixel value the caller passed in is
        preserved.
    goldstein_psize : int, default 64
        Goldstein FFT patch size (only used when ``goldstein_alpha > 0``).

    Other Parameters
    ----------------
    cost_threshold : int, default 50
        Connected-component boundary threshold in raw cost units. An edge becomes
        a component boundary when its statistical cost is ``<= cost_threshold``.
        A larger value makes more boundaries and so smaller, safer components.
        Prefer the physical knobs below over tuning this directly.
    conncomp_sigma : float or None, optional
        Set ``cost_threshold`` from a Gaussian-equivalent noise level: an edge is
        cut when its one-cycle-correction probability exceeds
        ``0.5 * erfc(sigma / sqrt(2))``. A higher sigma is stricter and makes
        more boundaries. ``sigma`` of about 3.5 reproduces the default
        ``cost_threshold=50``. Takes precedence over ``cost_threshold`` and
        ``conncomp_cycle_prob`` when given.
    conncomp_cycle_prob : float or None, optional
        Set ``cost_threshold`` from a target per-edge one-cycle-correction
        probability (via :func:`cost_threshold_from_cycle_prob`). This is a
        local edge reliability, not a global residue-pairing probability. A
        lower ``cycle_prob`` is stricter and makes more boundaries; about 2.4e-4
        matches the default. Takes precedence over ``cost_threshold``, but
        ``conncomp_sigma`` wins if both are given.
    conncomp_coh_floor : float or None, optional
        After labelling, drop any pixel whose coherence is below this floor to
        background (label ``0``). Unlike ``cost_threshold``, a coherence floor
        cuts regardless of the local gradient, so it cleanly removes noisy
        low-coherence speckle that the cost threshold alone leaves behind.
    min_size_px : int, default 100
        Discard connected components smaller than this many pixels.
    max_ncomps : int, default 1024
        Maximum number of connected components to keep (largest first).

    Returns
    -------
    unwrapped : ndarray of float32, shape ``(m, n)``
        Unwrapped phase, in radians.
    conncomp : ndarray of uint32, shape ``(m, n)``
        Connected-component labels; ``0`` = background / dropped.
    """
    if conncomp_sigma is not None:
        import math

        conncomp_cycle_prob = 0.5 * math.erfc(conncomp_sigma / math.sqrt(2.0))
    if conncomp_cycle_prob is not None:
        cost_threshold = cost_threshold_from_cycle_prob(conncomp_cycle_prob)

    if goldstein_alpha <= 0:
        unw, cc = _unwrap_native(
            igram,
            corr,
            nlooks,
            mask=mask,
            tile_size=0,
            tile_overlap=0,
            multilook=multilook,
            cost_threshold=cost_threshold,
            min_size_px=min_size_px,
            max_ncomps=max_ncomps,
        )
    else:
        ig_filt = goldstein(igram, alpha=goldstein_alpha, psize=goldstein_psize)
        if mask is not None:
            ig_filt = ig_filt.copy()
            ig_filt[~mask] = 0
        unw_filt, cc = _unwrap_native(
            ig_filt,
            corr,
            nlooks,
            mask=mask,
            tile_size=0,
            tile_overlap=0,
            multilook=multilook,
            cost_threshold=cost_threshold,
            min_size_px=min_size_px,
            max_ncomps=max_ncomps,
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

    if bridge and os.environ.get("WHIRLWIND_NO_BRIDGE", "") not in (
        "1",
        "true",
        "True",
    ):
        unw = _bridge_components(unw, igram, corr, nlooks, mask)
    if conncomp_coh_floor:
        # Drop low-coherence pixels from their components (a quality floor):
        # noisy percolation goes to background predictably, since a coherence
        # floor - unlike cost_threshold - cuts regardless of the local gradient.
        cc = np.asarray(cc).copy()
        cc[np.clip(np.nan_to_num(corr), 0.0, 1.0) < conncomp_coh_floor] = 0
    return unw, cc


def _bridge_components(
    unw: "NDArray[np.float32]",
    igram: "NDArray[np.complex64]",
    corr: "NDArray[np.float32]",
    nlooks: float,
    mask: "NDArray[np.bool_] | None",
    *,
    multilook_factor: int = 8,
    gate_frac: float = 0.5,
    amb_band: float = 0.25,
    min_px: int = 500,
) -> "NDArray[np.float32]":
    """Integration-component gauge bridging post-pass (see ``unwrap(bridge=)``).

    The free 2π gauge is between INTEGRATION components - the connected
    components of the valid mask, which is the partition the MCF integrator seeds
    independently - NOT the (finer) connected-component labels. Each region is
    re-levelled to a coherent xL coarse anchor, with shifts taken *relative to
    the largest region*, gated to regions the coarse scale connects, and vetoed
    unless the offset is cleanly integer. A single-region (or coherently
    connected) frame yields no shifts and is byte-identical. Prototype +
    validation: ``scripts/proto_bridge_a025.py`` (A_025 58 → 99.99 %, zero
    regression on A_016/A_030/A_028/D_077/D_074).
    """
    tau = 2.0 * np.pi
    L = multilook_factor
    m, n = unw.shape
    if mask is None:
        mask = igram != 0  # masked convention = 0+0j (or angle 0)
    mask = np.ascontiguousarray(mask, dtype=bool)

    # Integration components = 4-connected components of the valid mask (native
    # BFS labeller, matching integrate_with_mask's partition; no scipy needed).
    region, n_region = label_components(mask)
    if n_region <= 1:
        return unw  # single integration component -> structural no-op

    def block_mean(a):
        mm, nn = a.shape[0] // L, a.shape[1] // L
        return a[: mm * L, : nn * L].reshape(mm, L, nn, L).mean(axis=(1, 3))

    def upsample(a):
        up = np.kron(a, np.ones((L, L), a.dtype))
        return np.pad(
            up,
            ((0, max(0, m - up.shape[0])), (0, max(0, n - up.shape[1]))),
            mode="edge",
        )[:m, :n]

    # Coherent xL coarse anchor: unit-magnitude complex (angle 0 in masked
    # pixels under either masking convention), block-averaged, unwrapped whole so
    # its single integration BFS gives one consistent gauge across the banks.
    wrapped = np.angle(igram).astype(np.float32)
    cig = block_mean(np.exp(1j * wrapped).astype(np.complex64)).astype(np.complex64)
    ccoh = block_mean(np.clip(np.nan_to_num(corr), 0, 1).astype(np.float32)).astype(
        np.float32
    )
    cmask = np.ascontiguousarray(block_mean(mask.astype(np.float32)) > 0.4)
    cunw, _ = unwrap(cig, ccoh, float(nlooks) * L * L, cmask, bridge=False)
    anchor = upsample(np.asarray(cunw, np.float32))
    coarse_region = upsample(label_components(cmask)[0].astype(np.int64))

    sizes = np.bincount(region.ravel())
    ref = int(np.argmax(sizes[1:]) + 1)  # largest integration component = reference
    ref_coarse = np.bincount(coarse_region[region == ref]).argmax()
    both = mask & np.isfinite(anchor) & np.isfinite(unw)
    ref_reg = (region == ref) & both
    ref_off = np.median((anchor[ref_reg] - unw[ref_reg]) / tau)

    out = unw.copy()
    for lab in range(1, n_region + 1):
        if lab == ref or sizes[lab] < min_px:
            continue
        reg = (region == lab) & both
        if reg.sum() < min_px:
            continue
        # data-support gate: region must share the reference's coarse component.
        if np.mean(coarse_region[reg] == ref_coarse) < gate_frac:
            continue
        rel = (
            np.median((anchor[reg] - unw[reg]) / tau) - ref_off
        )  # relative to reference
        s = int(np.rint(rel))
        if abs(rel - s) > amb_band:  # ambiguity-band veto -> decline (convention)
            continue
        if s != 0:
            out[region == lab] += tau * s
    return out


# goldstein() is the Rust-backed native binding re-exported from
# ``._native``. See crates/whirlwind-core/src/goldstein.rs for the
# implementation and the ww-specific choices (unit-magnitude
# normalisation, Hann window) that move agreement-with-SNAPHU from
# 87% → 99.5% within ±π/2 on the NISAR HH test scene.


# NOTE: the CRLB unwrappers (``unwrap_crlb``, ``unwrap_crlb_grounded``,
# ``unwrap_crlb_stack``) and the whole-image ``unwrap_reuse`` solver are
# intentionally NOT in ``__all__``. They are experimental / unvalidated: the
# CRLB paths are still WIP, and ``unwrap_reuse`` is redundant with the validated
# default (and reachable via ``WHIRLWIND_UNWRAP_SOLVER=reuse``). They remain
# importable for internal use and parity tests but are kept off the public API
# until validated.
__all__ = [
    "closure_correct",
    "closure_refine_mcf",
    "compute_residues",
    "cost_threshold_from_cycle_prob",
    "diagonal_ramp",
    "goldstein",
    "interpolate",
    "label_components",
    "num_threads",
    "quality_map",
    "quality_triangles",
    "set_num_threads",
    "simulate_ifg",
    "unwrap",
    "unwrap_sparse",
    "wrap_phase",
]
