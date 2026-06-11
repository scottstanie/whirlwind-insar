"""Type stubs for the Rust-backed `_native` extension module.

Hand-written; keep in sync with `crates/whirlwind-py/src/lib.rs`. Pyright cannot
introspect compiled extension modules, so these stubs are the source of truth
for editor type-checking.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

def _unwrap_native(
    igram: NDArray[np.complex64],
    corr: NDArray[np.float32],
    nlooks: float,
    mask: NDArray[np.bool_] | None = ...,
    tile_size: int = ...,
    tile_overlap: int = ...,
    multilook: int = ...,
    cost_threshold: int = ...,
    min_size_px: int = ...,
    max_ncomps: int = ...,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """Engine behind :func:`whirlwind.unwrap`: single-tile linear coherence-cost
    unwrap returning ``(phase, conn_components)``.

    Prefer the Python :func:`whirlwind.unwrap` wrapper - it adds the
    integration-component "bridge" gauge post-pass + the K-transfer back onto the
    original phase. This bare native call does neither (and no Goldstein, which
    is OFF by default in the wrapper too).

    Phase: ``tile_size=0`` (default) is single-tile linear MCF on the WHOLE frame
    - the verified ww-orig-parity path (Carballo Lee-1994 cost, capacity-1
    min-cost-flow, adaptive PD→SSP fallback for masked frames); matches ww-orig
    on all 13 validated NISAR GUNW frames. The TILED pipeline is opt-in and
    experimental (not validated on most scenes): select it only with explicit
    ``tile_size>=4`` (``2<=tile_overlap<tile_size``), ``multilook>1``, or
    ``WHIRLWIND_UNWRAP_SOLVER=tiled``. ``WHIRLWIND_UNWRAP_SOLVER`` =
    ``linear|tiled|reuse`` (default ``linear``); tiled/reuse are experimental,
    not production. ``multilook=L`` (L>1) coherently down-looks xL first and
    routes through the opt-in tiled path.

    Components: grown GLOBALLY from the Carballo cost grid, independent of the
    (tiled) phase solve - a pixel edge is a cut when an underlying arc is
    mask-forbidden or its raw cost is ≤ ``cost_threshold``. uint32, 0 =
    background (cut/masked/below ``min_size_px``), renumbered 1..=K by size,
    capped at ``max_ncomps``. ``min_size_px`` (default 100, ≈0.8 km at 80 m) is
    the absolute, scene-size-invariant speckle floor.
    """

def unwrap_reuse(
    igram: NDArray[np.complex64],
    corr: NDArray[np.float32],
    nlooks: float = ...,
    mask: NDArray[np.bool_] | None = ...,
) -> NDArray[np.float32]:
    """PHASS-style flow-reuse solver - experimental/research, NOT the default
    (the public ``unwrap`` default is single-tile linear; reuse is opt-in via
    ``WHIRLWIND_UNWRAP_SOLVER=reuse``).

    Same Carballo coherence cost as :func:`whirlwind.unwrap`, but arcs carry
    multiple units of flow at zero marginal cost after the first push.
    """

def unwrap_crlb(
    igram: NDArray[np.complex64],
    variance: NDArray[np.float32],
    mask: NDArray[np.bool_] | None = ...,
    coherence: NDArray[np.float32] | None = ...,
    tile_size: int = ...,
    tile_overlap: int = ...,
    cost_threshold: int = ...,
    min_size_px: int = ...,
    max_ncomps: int = ...,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """CRLB-weighted unwrap (phase-linked IGs) → ``(phase, conn_components)``.

    The phase-linked twin of :func:`whirlwind.unwrap`. **Experimental / not
    validated.** This CRLB path rides the same experimental tiled pipeline:
    ``tile_size=0`` tiles frames > 512 px + a gated multi-shift winding fix on a
    variance-derived pseudo-coherence; components are grown globally from the
    CRLB cost grid. A verified single-tile CRLB path (reusing the coherence
    default kernel) is future work.
    """

def unwrap_crlb_grounded(
    igram: NDArray[np.complex64],
    variance: NDArray[np.float32],
    mask: NDArray[np.bool_] | None = ...,
    ground_cost: int = ...,
) -> NDArray[np.float32]:
    """CRLB 2D unwrap with a virtual ground node on every boundary residue.

    Fixes the capacity-1 stacking limit of ``unwrap_crlb`` for inputs with
    boundary-only wrap-lines (e.g. smooth ramps; tile-boundary residues).
    ``ground_cost = 0`` makes ground free; positive cost biases MCF toward
    internal pairing for the bulk of residues on noisy data.
    """

def unwrap_sparse(
    points: NDArray[np.float64],
    wrapped_phase: NDArray[np.float32],
    variance: NDArray[np.float32],
    max_edge_length: float | None = ...,
) -> NDArray[np.float32]:
    """Sparse / irregular-grid unwrap over a Delaunay triangulation of the
    supplied valid pixels.

    ``points`` is a float64 ``(n, 2)`` array of ``(x, y)`` coordinates;
    ``wrapped_phase`` and ``variance`` are length-``n`` per-pixel arrays.
    Edges with non-finite variance at either endpoint are pre-saturated so
    MCF cannot route flow through them. ``max_edge_length`` carves out
    triangulation edges longer than the cutoff as outer-face boundary
    edges (integration BFS skips them); pixels reachable only via long
    edges come back as ``NaN`` in the output.
    """

def compute_residues(wrapped_phase: NDArray[np.float32]) -> NDArray[np.int32]:
    """Compute the integer residue grid from a wrapped-phase array."""

def diagonal_ramp(m: int, n: int) -> NDArray[np.float32]: ...
def wrap_phase(unw: NDArray[np.float32]) -> NDArray[np.float32]: ...
def label_components(mask: NDArray[np.bool_]) -> tuple[NDArray[np.int32], int]:
    """4-connected component labels of a boolean mask -> ``(labels, n)``."""

def cli_main(argv: list[str]) -> int:
    """Run the `whirlwind` CLI in-process (argv excludes the program name);
    returns the exit code. Backs the ``whirlwind`` console script."""

def bridge_components(
    unw: NDArray[np.float32],
    mask: NDArray[np.bool_] | None = ...,
    *,
    radius: int = ...,
    min_px: int = ...,
    max_boundary: int = ...,
) -> NDArray[np.float32]:
    """Re-level the disconnected regions of an unwrapped phase image.

    Integration-component gauge bridging (the ``unwrap(bridge=True)``
    post-pass): MST over closest-boundary distances rooted at the largest
    region, each child region shifted by an integer number of cycles read from
    boundary-local medians. See ``crates/whirlwind-core/src/bridge.rs``.
    """

def interpolate(
    ifg: NDArray[np.complex64],
    weights: NDArray[np.float32],
    weight_cutoff: float = ...,
    num_neighbors: int = ...,
    max_radius: int = ...,
    min_radius: int = ...,
    alpha: float = ...,
) -> NDArray[np.complex64]:
    """Spiral PS phase interpolator - Rust port of dolphin ``interpolation.interpolate``.

    Replaces low-weight pixels' phase with a Gaussian-distance-weighted average of
    the nearest ``num_neighbors`` high-weight pixels; amplitude preserved.
    """

def simulate_ifg(
    truth: NDArray[np.float32],
    gamma: NDArray[np.float32],
    nlooks: int,
    seed: int,
) -> tuple[NDArray[np.complex64], NDArray[np.float32]]: ...

class ClosureResult(TypedDict):
    corrected: NDArray[np.float32]  # (n_edges, m, n)
    corrections: NDArray[np.int16]  # (n_edges, m, n)
    date_phases: NDArray[np.float32]  # (n_dates, m, n)
    closure_rms: NDArray[np.float32]  # (m, n)

def closure_correct(
    unw_stack: NDArray[np.float32],
    edges_from: NDArray[np.uint32],
    edges_to: NDArray[np.uint32],
    n_dates: int,
    reference: int,
    tree_priority: NDArray[np.float32] | None = ...,
) -> ClosureResult:
    """Closure-correct a stack of unwrapped IGs over a temporal graph.

    Uses Prim's algorithm on `tree_priority` to pick a minimum-variance
    spanning tree of the temporal graph. Per pixel: propagate phase along
    the tree from `reference`; for every non-tree edge, snap its residual
    to the nearest integer multiple of 2π.

    Returns a dict; see ClosureResult.
    """

class RefineResult(TypedDict):
    corrected: NDArray[np.float32]  # (E, m, n)
    corrections: NDArray[np.int16]  # (E, m, n) - additive on top of input
    residual_violations: NDArray[np.uint16]  # (m, n)
    iterations: NDArray[np.uint8]  # (m, n)

def closure_refine_mcf(
    unw_stack: NDArray[np.float32],
    edges_from: NDArray[np.uint32],
    edges_to: NDArray[np.uint32],
    n_dates: int,
    reference: int,
    crlb_per_date: NDArray[np.float32],
    tree_priority: NDArray[np.float32] | None = ...,
    max_iter: int = ...,
) -> RefineResult:
    """Cycle-greedy MCF refinement: doesn't implicitly trust the tree.

    Routes integer cycle violations to the highest-σ² edge in each cycle.
    """

def quality_map(
    unw_stack: NDArray[np.float32],
    edges_from: NDArray[np.uint32],
    edges_to: NDArray[np.uint32],
    n_dates: int,
    reference: int,
    tree_priority: NDArray[np.float32] | None = ...,
) -> NDArray[np.uint16]:
    """Per-pixel max |K| over the *fundamental* temporal cycle basis.

    K = round(cycle_residual / 2π); 0 on perfectly consistent pixels.
    Cycles are length up to D-1 (tree path); errors accumulate. Prefer
    `quality_triangles` for phase-linked stacks with triangle redundancy.
    """

def quality_triangles(
    unw_stack: NDArray[np.float32],
    edges_from: NDArray[np.uint32],
    edges_to: NDArray[np.uint32],
    n_dates: int,
) -> NDArray[np.uint16]:
    """Per-pixel max |K| over all temporal triangles (3-cycles).

    Local 3-cycle check - recommended over `quality_map` for phase-linked
    stacks where short-baseline triangles are the natural redundancy.
    """

def goldstein(
    igram: NDArray[np.complex64],
    alpha: float = ...,
    psize: int = ...,
) -> NDArray[np.complex64]:
    """Goldstein adaptive phase filter (block-parallel Rust + rustfft).

    Bit-equivalent to the prior numpy implementation but typically
    5–10x faster on large scenes. Two ww-specific choices baked in
    (see ``whirlwind-core/src/goldstein.rs``):
    1. Input normalised to unit magnitude before filtering.
    2. Hann overlap-add window (smoother than triangle).
    """

def set_num_threads(n: int) -> None:
    """Set the rayon thread pool size used for all parallel ww work.

    Must be called *before* the first parallel ww function. Raises
    ``RuntimeError`` if the pool is already initialised (by an earlier
    call, by env vars, or by an earlier rayon call). Precedence:
    ``WHIRLWIND_NUM_THREADS`` > ``RAYON_NUM_THREADS`` > this function
    > rayon default (all logical CPUs).
    """

def num_threads() -> int:
    """Return the current rayon thread pool size used by ww."""
