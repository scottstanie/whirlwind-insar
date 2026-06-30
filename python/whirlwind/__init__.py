"""whirlwind: Rust-backed InSAR phase unwrapper."""

from __future__ import annotations

import logging
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

# Version lives in Cargo.toml; maturin stamps it into the distribution metadata.
__version__ = version("whirlwind-insar")

from ._native import (
    bridge_components,
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
from ._native import (
    _unwrap_native,
    components_snaphu,
    label_components,
)

# `interpolate` is re-exported above as the public native binding. Alias it so
# the `interpolate=` keyword argument inside unwrap() (which shadows the name in
# that scope) can still reach the function.
_interpolate = interpolate

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Effective looks above this are capped when building the Lee (1994) cost LUTs.
# The PDF is already a near-delta spike there (the cost shape stops changing),
# and that many *independent* looks is not physically reachable from correlated
# multilook windows. Keep in sync with `MAX_COST_MODEL_NLOOKS` in
# crates/whirlwind-core/src/cost/lut.rs.
_MAX_COST_MODEL_NLOOKS = 80.0


def _validate_nlooks(nlooks: float) -> None:
    """Reject nonphysical ``nlooks`` early, and warn when it exceeds the cap.

    A clear ``ValueError`` here beats a Rust panic deep in the cost-LUT build
    (which is what ``nlooks < 1`` triggers via the Lee 1994 PDF assert). The
    ``not (nlooks >= 1)`` form also rejects ``NaN``.
    """
    if not (nlooks >= 1.0):
        raise ValueError(f"nlooks must be a finite value >= 1, got {nlooks!r}")
    if nlooks > _MAX_COST_MODEL_NLOOKS:
        logger.warning(
            "nlooks=%g exceeds the cost-model cap of %g; the Lee (1994) phase "
            "PDF is already a near-delta spike by then, so the cost model uses "
            "%g looks. (That many independent looks is rarely physical from "
            "correlated multilook windows.)",
            nlooks,
            _MAX_COST_MODEL_NLOOKS,
            _MAX_COST_MODEL_NLOOKS,
        )


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
    This applies only to the legacy ``conncomp_algorithm="linear"`` Carballo
    coherence connected components; the default SNAPHU ambiguity-wiggle path and
    the CRLB inverse-variance cost path do not use this scaling.
    """
    import math

    p = min(max(cycle_prob, 1e-12), 1.0 - 1e-12)
    return round(CONNCOMP_COST_SCALE * math.log((1.0 - p) / p))


# Internal scale relating the user-facing ``conncomp_reliability`` (in
# inverse-variance ``1 / sigma2`` units, so values are small and easy to type) to
# the native solver's raw reliability threshold. A clean edge's raw reliability is
# ``COST_SCALE * nshortcycle**2 / sigma2`` (= this constant / sigma2), so dividing
# by it puts the knob in ``1 / sigma2`` units.
CONNCOMP_RELIABILITY_UNIT = 100 * 100**2  # COST_SCALE * nshortcycle**2 = 1_000_000


def conncomp_reliability_from_coherence(coherence: float, nlooks: float) -> float:
    """``conncomp_reliability`` value that cuts conncomp edges below a target coherence.

    A guessable way to set ``conncomp_reliability``: pass a target minimum
    coherence and use the result, and the connected components keep edges whose
    coherence is roughly above ``coherence`` and drop (label ``0``) those below.

    ``conncomp_reliability`` is in inverse-variance (``1 / sigma2``) units, and a
    clean edge at coherence ``gamma`` and ``L=nlooks`` looks has reliability
    ``1 / sigma2(gamma)`` with the Just/Bamler phase variance
    ``sigma2 = (1 - gamma**2) / (2 * L * gamma**2)`` -- so this returns
    ``1 / sigma2(coherence)``. It is an estimate (an edge near a wrap line is less
    reliable than a clean one at the same coherence), so treat it as a starting
    point and check the result. ``coherence`` near 0 returns a tiny value (cut
    almost nothing); near 1, a large one (cut almost everything). Typical values
    are small, e.g. ``gamma=0.3 -> ~3.2``, ``gamma=0.5 -> ~11``.
    """
    g = min(max(coherence, 1e-3), 0.999)
    sigma2 = (1.0 - g * g) / (2.0 * nlooks * g * g)
    return 1.0 / sigma2


def conncomp_min_coherence_auto(nlooks: float) -> float:
    """Looks-aware default for ``conncomp_min_coherence``.

    Returns ``0.32 / sqrt(nlooks)``, clipped to ``[0.02, 0.30]`` -- about a third
    of the sample-coherence noise floor (the magnitude estimated from
    uncorrelated data, which falls as ``1 / sqrt(nlooks)``). A fixed coherence
    cutoff ignores this and so means different things at different looks. Equals
    ``0.08`` at 16 looks.
    """
    return min(max(0.32 / (nlooks**0.5), 0.02), 0.30)


def unwrap(
    igram: "NDArray[np.complex64]",
    corr: "NDArray[np.float32]",
    nlooks: float,
    mask: "NDArray[np.bool_] | None" = None,
    *,
    bridge: bool = True,
    downsample: int = 1,
    interpolate: bool = False,
    interp_cutoff: float = 0.1,
    interp_num_neighbors: int = 30,
    interp_max_radius: int = 101,
    interp_min_radius: int = 0,
    interp_alpha: float = 0.75,
    conncomp_algorithm: str = "snaphu",
    conncomp_min_coherence: "float | str | None" = "auto",
    conncomp_reliability: float = 0.0,
    cost_threshold: int = 50,
    conncomp_cycle_prob: "float | None" = None,
    conncomp_sigma: "float | None" = None,
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
    By default (``conncomp_algorithm="snaphu"``) they are grown by the
    SNAPHU-like ambiguity-wiggle reliability test on the unwrapped phase
    output (reproducing SNAPHU's ``GrowConnCompsMask``).
    ``conncomp_algorithm="linear"`` uses a global coherence-cost growing algorithm.

    A fast default post-pass (``bridge``) attmempts to minimize 2π offsets
    between regions that the valid mask splits apart, such as land slabs separated by a
    low-coherence river.

    Parameters
    ----------
    igram : ndarray of complex64
        Wrapped interferometric phase. Nodata pixels should be ``0+0j``; any
        ``NaN`` is treated as nodata (set to ``0`` with a warning).
    corr : ndarray of float32
        Sample coherence in ``[0, 1]``, same shape as ``igram``. ``NaN`` is
        treated as nodata (set to ``0`` with a warning).
    nlooks : float
        Effective number of looks used to estimate ``corr``. Must be at least
        ``1`` (values below raise ``ValueError``). A higher number of looks
        means higher confidence in ``corr`` and a narrower coherence cost model.
        Very large values (above ~80 looks) are capped for the cost model, with
        a warning, since the phase PDF has effectively converged by then.
    mask : ndarray of bool, optional
        Valid-pixel mask, ``True`` = valid. Defaults to ``(igram != 0) &
        (corr > 0)``, so exact-zero phase or zero-coherence pixels are excluded.
    bridge : bool, default True
        Post-processing step that re-levels regions the valid mask splits into
        disconnected pieces (for example two land slabs separated by a
        low-coherence river). The MCF seeds each piece at an arbitrary 2π level,
        so the relative offset between pieces is under-determined. This pass
        estimates each region's level from the unwrapped phase at the region
        boundaries and snaps it to an integer number of cycles, propagated along
        a minimum spanning tree rooted at the largest region.
    downsample : int, default 1
        Coarse-solve factor for noisy scenes. When greater than 1, the complex
        interferogram is coherently averaged into ``downsample x downsample``
        blocks and that smaller, smoother frame is unwrapped to decide which 2π
        cycle each block sits on. Only the integer cycle is borrowed back onto
        the full-resolution wrapped phase, so every per-pixel value is kept;
        detail finer than the block scale aliases under the downlook.
        May be useful for noisy or moderate-coherence scenes, provides little benefit
        for clean scenes. Note this coherently averages an existing
        interferogram, which is not the same as forming a multilooked
        interferogram from the SLCs.
    interpolate : bool, default False
        Spiral persistent-scatterer interpolation pre-pass (the Rust port of
        dolphin's ``interpolation.interpolate``, exposed standalone as
        :func:`interpolate`). When True, every valid pixel whose coherence is
        below ``interp_cutoff`` has its phase replaced by a Gaussian
        distance-weighted average of the nearest high-coherence pixels' unit
        phasors before the solve.
        The integer cycle field after unwrapping is applied back to the
        original wrapped phase, so every per-pixel value the caller passed in is
        preserved. Useful for scenes with isolated low-coherence speckle that
        seeds spurious residues.
    interp_cutoff : float, default 0.1
        Coherence below which a valid pixel is interpolated (only used when
        ``interpolate`` is True). ``corr`` is used as the weight map.
    interp_num_neighbors : int, default 30
        Number of nearest high-coherence pixels averaged per interpolated pixel.
    interp_max_radius : int, default 101
        Maximum search radius in pixels for the concentric-circle neighbor search.
    interp_min_radius : int, default 0
        Minimum search radius in pixels; closer neighbors are skipped.
    interp_alpha : float, default 0.75
        Gaussian distance-weighting falloff for the neighbor average.
    goldstein_alpha : float, default 0.0
        Goldstein adaptive-filter strength in ``[0, 1]``. 0 (default) disables
        filtering; a typical "on" value is 0.7. When enabled, the filter only
        informs the MCF; the integer cycle field it produces is applied to the
        original wrapped phase, so every per-pixel value the caller passed in is
        preserved.
    goldstein_psize : int, default 64
        Goldstein FFT patch size (only used when ``goldstein_alpha > 0``).
    conncomp_algorithm : {"snaphu", "linear"}, default "snaphu"
        Which connected-component grow to use. ``"snaphu"`` (default) is the
        SNAPHU-faithful ambiguity-wiggle on the unwrapped output (see
        :func:`whirlwind._native.components_snaphu`), tuned by
        ``conncomp_reliability``. ``"linear"`` is the older global
        coherence-cost grow, tuned by ``cost_threshold`` / ``conncomp_sigma`` /
        ``conncomp_cycle_prob``.
    conncomp_min_coherence : float or "auto" or None, default "auto"
        Coherence below which pixels are labeled ``0`` (background) in the default
        ("snaphu") connected components. ``"auto"`` uses
        :func:`conncomp_min_coherence_auto` (``0.32 / sqrt(nlooks)``); pass a float
        for a fixed cutoff, or ``None`` to disable the cutoff and use
        ``conncomp_reliability`` (``0`` labels every reliably unwrapped pixel).
        Takes precedence over ``conncomp_reliability``. Only used when
        ``conncomp_algorithm="snaphu"``.
    conncomp_reliability : float, default 0.0
        Lower-level conservativeness knob for the default ("snaphu") connected
        components, in inverse-variance (``1 / sigma2``) units, so values are
        small. Used only when ``conncomp_min_coherence`` is ``None``. An edge
        becomes a component boundary when a one-cycle ambiguity flip across it is
        no more expensive than the achieved flow; an edge of coherence ``gamma``
        is cut roughly when ``conncomp_reliability`` exceeds ``1 / sigma2(gamma)``.
        ``0`` labels essentially every reliably unwrapped pixel; raise it to cut
        more low-coherence interior edges. To pick one from a target minimum
        coherence use :func:`conncomp_reliability_from_coherence` (e.g.
        ``coherence=0.3`` -> about 3.2), or just set ``conncomp_min_coherence``.
        Only used when ``conncomp_algorithm="snaphu"``.
    cost_threshold : int, default 50
        Connected-component boundary threshold in raw cost units, for the
        ``"linear"`` algorithm only. An edge becomes a boundary when its
        statistical cost is ``<= cost_threshold``; a larger value makes more
        boundaries and so smaller, safer components. (No effect under the
        default ``"snaphu"`` algorithm.)
    conncomp_sigma : float or None, optional
        For ``conncomp_algorithm="linear"``: set ``cost_threshold`` from a
        Gaussian-equivalent noise level: an edge is cut when its
        one-cycle-correction probability exceeds ``0.5 * erfc(sigma / sqrt(2))``.
        A higher sigma is stricter. ``sigma`` of about 3.5 reproduces the default
        ``cost_threshold=50``. Takes precedence over ``cost_threshold`` and
        ``conncomp_cycle_prob`` when given.
    conncomp_cycle_prob : float or None, optional
        For ``conncomp_algorithm="linear"``: set ``cost_threshold`` from a target
        per-edge one-cycle-correction probability (via
        :func:`cost_threshold_from_cycle_prob`). A lower ``cycle_prob`` is
        stricter; about 2.4e-4 matches the default. Takes precedence over
        ``cost_threshold``, but ``conncomp_sigma`` wins if both are given.
    min_size_px : int, default 100
        Discard connected components smaller than this many pixels (both
        algorithms).
    max_ncomps : int, default 1024
        Maximum number of connected components to keep (largest first).

    Returns
    -------
    unwrapped : ndarray of float32, shape ``(m, n)``
        Unwrapped phase, in radians.
    conncomp : ndarray of uint32, shape ``(m, n)``
        Connected-component labels; ``0`` = background / dropped.
    """
    _validate_nlooks(nlooks)

    # NaN inputs are treated as nodata: zero them (so the default mask drops
    # them) and warn, rather than letting a NaN propagate through the solve.
    igram = np.ascontiguousarray(igram, dtype=np.complex64)
    corr = np.ascontiguousarray(corr, dtype=np.float32)
    ig_nan = np.isnan(igram)
    corr_nan = np.isnan(corr)
    n_nan = int(ig_nan.sum() + corr_nan.sum())
    if n_nan:
        logger.warning("NaN in %d input pixel(s); treating as nodata (0).", n_nan)
        igram = igram.copy()
        corr = corr.copy()
        igram[ig_nan] = 0
        corr[corr_nan] = 0

    if mask is None:
        mask = (igram != 0) & (corr > 0)
    mask = np.ascontiguousarray(mask, dtype=bool)

    if conncomp_sigma is not None:
        import math

        conncomp_cycle_prob = 0.5 * math.erfc(conncomp_sigma / math.sqrt(2.0))
    if conncomp_cycle_prob is not None:
        cost_threshold = cost_threshold_from_cycle_prob(conncomp_cycle_prob)

    # Build the phase fed to the MCF. Interpolation and Goldstein filtering both
    # only INFORM the solver; the integer 2π·k field they produce is transferred
    # back onto the ORIGINAL wrapped phase below, so every per-pixel value the
    # caller passed in is preserved.
    ig_solve = igram
    if interpolate:
        # Spiral PS interpolator: fill each valid pixel whose coherence is below
        # interp_cutoff from a Gaussian distance-weighted average of nearby
        # high-coherence phasors. `corr` is the weight map.
        weights = np.clip(np.nan_to_num(corr), 0.0, 1.0).astype(np.float32)
        ig_solve = np.ascontiguousarray(
            _interpolate(
                ig_solve,
                weights,
                interp_cutoff,
                interp_num_neighbors,
                interp_max_radius,
                interp_min_radius,
                interp_alpha,
            ),
            dtype=np.complex64,
        )
    if goldstein_alpha > 0:
        ig_solve = goldstein(ig_solve, alpha=goldstein_alpha, psize=goldstein_psize)
    if ig_solve is not igram:
        # Pre-pass produced a fresh array; zero masked pixels so the solver sees
        # the same nodata convention as the original phase.
        ig_solve = np.array(ig_solve, dtype=np.complex64, copy=True)
        if mask is not None:
            ig_solve[~mask] = 0

    if conncomp_algorithm not in ("snaphu", "linear"):
        raise ValueError(
            f"conncomp_algorithm must be 'snaphu' or 'linear', got {conncomp_algorithm!r}"
        )

    # `_unwrap_native` returns the phase plus the legacy linear-cost conncomp.
    # The linear grow is cheap (a few % of the solve), so we always take it; it
    # is the returned conncomp under ``conncomp_algorithm="linear"`` and is
    # discarded under the default "snaphu" path (replaced below).
    unw_solve, cc_linear = _unwrap_native(
        ig_solve,
        corr,
        nlooks,
        mask=mask,
        tile_size=0,
        tile_overlap=0,
        multilook=downsample,
        cost_threshold=cost_threshold,
        min_size_px=min_size_px,
        max_ncomps=max_ncomps,
    )

    if ig_solve is igram:
        unw = np.asarray(unw_solve, dtype=np.float32)
    else:
        # Transfer the integer 2π·k field from the interpolated/filtered unwrap
        # onto the *original* wrapped phase, rounding against the original (not
        # the modified) phase to avoid the dolphin-#364 artefact: any pixel where
        # the pre-pass moved phase across the ±π discontinuity would otherwise
        # pick up a spurious ±2π cycle, producing visible outlines along fringe
        # boundaries.
        tau = np.float32(2 * np.pi)
        phase_orig = np.angle(igram).astype(np.float32)
        k = np.round((np.asarray(unw_solve) - phase_orig) / tau).astype(np.float32)
        unw = (phase_orig + tau * k).astype(np.float32)
        if mask is not None:
            unw[~mask] = 0.0

    if bridge:
        unw = bridge_components(unw, mask)

    # Connected components. The default "snaphu" grow runs on the FINAL (bridged)
    # unwrapped phase via the convex-cost ambiguity wiggle; it is bridge-invariant
    # but using the final phase keeps it unambiguous. "linear" returns the legacy
    # coherence-cost grow already computed above.
    if conncomp_algorithm == "snaphu":
        # A target minimum coherence (the default, and the intuitive knob) takes
        # precedence over a raw 1/sigma2 reliability: components keep edges above
        # ~conncomp_min_coherence and drop (label 0) those below, so conncomp > 0
        # acts as a reliability mask. Pass conncomp_min_coherence=None to use the
        # raw conncomp_reliability instead (0 = label every unwrapped pixel).
        if conncomp_min_coherence is not None:
            gamma = (
                conncomp_min_coherence_auto(nlooks)
                if conncomp_min_coherence == "auto"
                else float(conncomp_min_coherence)
            )
            conncomp_reliability = conncomp_reliability_from_coherence(gamma, nlooks)
        # The public knob is in 1/sigma2 units; the native grow takes the raw
        # convex-cost reliability threshold (this * COST_SCALE * nshortcycle**2).
        reliability_raw = round(conncomp_reliability * CONNCOMP_RELIABILITY_UNIT)
        cc = components_snaphu(
            igram,
            corr,
            nlooks,
            unw,
            mask,
            reliability_raw,
            min_size_px,
            max_ncomps,
        )
    else:
        cc = cc_linear
    return unw, cc


# goldstein() is the Rust-backed native binding re-exported from ``._native``.
# See crates/whirlwind-core/src/goldstein.rs for the implementation and the
# unit-magnitude normalisation / Hann-window choices it makes.


# NOTE: several natives are imported above but intentionally not in ``__all__``.
# They remain importable for internal use, benchmarks, and parity tests, but are
# kept off the public API:
#   - the CRLB unwrappers (``unwrap_crlb``, ``unwrap_crlb_grounded``) and the
#     whole-image ``unwrap_reuse`` solver - experimental / unvalidated
#     (``unwrap_reuse`` is reachable via ``WHIRLWIND_UNWRAP_SOLVER=reuse``);
#   - the temporal-closure stack functions (``closure_correct``,
#     ``closure_refine_mcf``) and quality maps (``quality_map``,
#     ``quality_triangles``) - experimental 3D / diagnostic helpers;
#   - the synthetic-scene generators (``diagonal_ramp``, ``simulate_ifg``) -
#     test/benchmark utilities.
__all__ = [
    "bridge_components",
    "compute_residues",
    "conncomp_reliability_from_coherence",
    "conncomp_min_coherence_auto",
    "cost_threshold_from_cycle_prob",
    "goldstein",
    "interpolate",
    "label_components",
    "num_threads",
    "set_num_threads",
    "unwrap",
    "unwrap_sparse",
    "wrap_phase",
]
