"""whirlwind-rs: Rust-backed InSAR phase unwrapper."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

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


# The coherence connected-component cost is round(CONNCOMP_COST_SCALE · Carballo
# LLR), where the LLR = log(p0/p1) is the log-odds of "one-cycle correction" (p1)
# vs "no correction" (p0) for an edge under the Lee-1994 multilook phase model.
CONNCOMP_COST_SCALE = 6


def cost_threshold_from_cycle_prob(cycle_prob: float) -> int:
    """Connected-component ``cost_threshold`` for a target per-edge one-cycle
    probability.

    An edge is cut (a component boundary) when its cost ``<= cost_threshold``,
    i.e. when its **local one-cycle-correction probability** ``>= cycle_prob``.
    This is *local edge reliability*, NOT a global residue-pairing probability.
    Lower ``cycle_prob`` ⇒ higher threshold ⇒ MORE edges cut (stricter). The
    legacy ``cost_threshold=50`` ≈ ``cycle_prob ≈ 2.4e-4`` (≈ 3.5σ Gaussian-
    equivalent). Carballo/coherence conncomp only — the CRLB inverse-variance
    cost path does not use this scaling.
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
    tile_size: int = 0,
    tile_overlap: int = 0,
    cost_threshold: int = 50,
    conncomp_cycle_prob: "float | None" = None,
    conncomp_sigma: "float | None" = None,
    conncomp_coh_floor: "float | None" = None,
    min_size_px: int = 100,
    max_ncomps: int = 1024,
    goldstein_alpha: float = 0.0,
    goldstein_psize: int = 64,
) -> "tuple[NDArray[np.float32], NDArray[np.uint32]]":
    """MCF unwrap returning ``(unwrapped_phase, conn_components)``.

    The main entry point. By default the phase is solved with the **verified
    single-tile linear solver** (``unwrap_linear``: ww-orig-parity Carballo cost,
    capacity-1 MCF, adaptive PD/SSP fallback that drains heavily-masked frames),
    which matches Python ``ww-orig`` across the validated NISAR frame set. The
    older tiled robustness pipeline (auto-tile, multi-shift re-solve, coarse
    anchor + cascade, seam-repair) is **opt-in** — it is not yet validated on all
    NISAR frames (it can produce artifacts on fragmented scenes) — selected by
    ``multilook > 1``, an explicit ``tile_size``, or ``WHIRLWIND_UNWRAP_SOLVER=
    tiled``. The reuse (PHASS) whole-image solver (``=reuse``) is likewise opt-in
    and not yet validated. Connected components are grown SNAPHU-style globally
    from the Carballo coherence cost, independent of the phase solver.

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
    bridge : bool, default True
        Solver-aware integration-component gauge bridging (a fast post-pass).
        When the valid mask splits into several disconnected regions (e.g. two
        river/water-separated land slabs), the MCF integrator seeds each at an
        arbitrary 2π level, so their *relative* offset is under-determined. This
        pass re-levels each region to a coherent ×8 coarse anchor (shifts taken
        *relative to the largest region*), gated to regions the coarse scale
        actually connects and vetoed unless the offset is cleanly integer — so a
        single-region or coherently-connected frame is a strict no-op. Fixes the
        NISAR A_025 river frame (58 → ~100 %) with zero regression elsewhere.
        Set ``False`` (or ``WHIRLWIND_NO_BRIDGE=1``) to disable.
    multilook : int, default 1
        > 1 coherently down-looks first (noisy / moderate-coherence scenes),
        unwraps the coarse frame, then upsamples.
    tile_size, tile_overlap : int
        ``tile_size=0`` (default) uses the single-tile linear solver (whole
        image). Set ``tile_size`` ≥ 4 (with ``tile_overlap`` ≥ 2) to opt into
        the tiled pipeline at that tile size (auto-512/overlap-64 if a tiled
        path is otherwise requested, e.g. via ``multilook`` or the env knob).
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
    if conncomp_sigma is not None:
        import math
        conncomp_cycle_prob = 0.5 * math.erfc(conncomp_sigma / math.sqrt(2.0))
    if conncomp_cycle_prob is not None:
        cost_threshold = cost_threshold_from_cycle_prob(conncomp_cycle_prob)

    if goldstein_alpha <= 0:
        unw, cc = _unwrap_native(
            igram, corr, nlooks,
            mask=mask, tile_size=tile_size, tile_overlap=tile_overlap, multilook=multilook,
            cost_threshold=cost_threshold, min_size_px=min_size_px, max_ncomps=max_ncomps,
        )
    else:
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

    if bridge and os.environ.get("WHIRLWIND_NO_BRIDGE", "") not in ("1", "true", "True"):
        unw = _bridge_components(unw, igram, corr, nlooks, mask)
    if conncomp_coh_floor:
        # Drop low-coherence pixels from their components (a quality floor):
        # noisy percolation goes to background predictably, since a coherence
        # floor — unlike cost_threshold — cuts regardless of the local gradient.
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

    The free 2π gauge is between INTEGRATION components — the connected
    components of the valid mask, which is the partition the MCF integrator seeds
    independently — NOT the (finer) connected-component labels. Each region is
    re-levelled to a coherent ×L coarse anchor, with shifts taken *relative to
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
        return np.pad(up, ((0, max(0, m - up.shape[0])), (0, max(0, n - up.shape[1]))), mode="edge")[:m, :n]

    # Coherent ×L coarse anchor: unit-magnitude complex (angle 0 in masked
    # pixels under either masking convention), block-averaged, unwrapped whole so
    # its single integration BFS gives one consistent gauge across the banks.
    wrapped = np.angle(igram).astype(np.float32)
    cig = block_mean(np.exp(1j * wrapped).astype(np.complex64)).astype(np.complex64)
    ccoh = block_mean(np.clip(np.nan_to_num(corr), 0, 1).astype(np.float32)).astype(np.float32)
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
        rel = np.median((anchor[reg] - unw[reg]) / tau) - ref_off  # relative to reference
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
    "unwrap_crlb",
    "unwrap_crlb_grounded",
    "unwrap_crlb_stack",
    "unwrap_reuse",
    "unwrap_sparse",
    "wrap_phase",
]
