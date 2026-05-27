"""Type stubs for the Rust-backed `_native` extension module.

Hand-written; keep in sync with `crates/whirlwind-py/src/lib.rs`. Pyright cannot
introspect compiled extension modules, so these stubs are the source of truth
for editor type-checking.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


def unwrap(
    igram: NDArray[np.complex64],
    corr: NDArray[np.float32],
    nlooks: float,
    mask: NDArray[np.bool_] | None = ...,
) -> NDArray[np.float32]:
    """2D phase unwrap with the Carballo/SNAPHU-style coherence cost."""

def unwrap_crlb(
    igram: NDArray[np.complex64],
    variance: NDArray[np.float32],
    mask: NDArray[np.bool_] | None = ...,
    tile_size: int = ...,
    tile_overlap: int = ...,
) -> NDArray[np.float32]:
    """2D phase unwrap with the CRLB-weighted Gaussian cost (for phase-linked IGs).

    If ``tile_size > 0`` and the image is larger than one tile, the image is
    split into overlapping tiles, each tile is unwrapped independently, and
    they are stitched together by CRLB-weighted overlap-median 2π
    reconciliation. Bounds per-IG MCF memory use to tile-size scale.
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


def unwrap_with_conncomp(
    igram: NDArray[np.complex64],
    corr: NDArray[np.float32],
    nlooks: float,
    mask: NDArray[np.bool_] | None = ...,
    cost_threshold: int = ...,
    min_size_frac: float = ...,
    max_ncomps: int = ...,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """Carballo unwrap + SNAPHU-style connected components from one MCF solve.

    Returns ``(unwrapped_phase, components)``. ``components`` is uint32 with
    0 = background (cut off, masked, or smaller than ``min_size_frac``);
    valid components are renumbered 1..=K by descending size, capped at
    ``max_ncomps``. A pixel edge is a cut when an underlying MCF arc is
    mask-forbidden, carries flow (branch cut), or has raw cost
    ≤ ``cost_threshold``.
    """


def unwrap_crlb_with_conncomp(
    igram: NDArray[np.complex64],
    variance: NDArray[np.float32],
    mask: NDArray[np.bool_] | None = ...,
    cost_threshold: int = ...,
    min_size_frac: float = ...,
    max_ncomps: int = ...,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """CRLB unwrap + SNAPHU-style connected components. See
    :func:`unwrap_with_conncomp` for component semantics.
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

def simulate_ifg(
    truth: NDArray[np.float32],
    gamma: NDArray[np.float32],
    nlooks: int,
    seed: int,
) -> tuple[NDArray[np.complex64], NDArray[np.float32]]: ...


class ClosureResult(TypedDict):
    corrected:   NDArray[np.float32]   # (n_edges, m, n)
    corrections: NDArray[np.int16]     # (n_edges, m, n)
    date_phases: NDArray[np.float32]   # (n_dates, m, n)
    closure_rms: NDArray[np.float32]   # (m, n)


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
    corrected:           NDArray[np.float32]  # (E, m, n)
    corrections:         NDArray[np.int16]    # (E, m, n) — additive on top of input
    residual_violations: NDArray[np.uint16]   # (m, n)
    iterations:          NDArray[np.uint8]    # (m, n)


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

    Local 3-cycle check — recommended over `quality_map` for phase-linked
    stacks where short-baseline triangles are the natural redundancy.
    """


def goldstein(
    igram: NDArray[np.complex64],
    alpha: float = ...,
    psize: int = ...,
) -> NDArray[np.complex64]:
    """Goldstein adaptive phase filter (block-parallel Rust + rustfft).

    Bit-equivalent to the prior numpy implementation but typically
    5–10× faster on large scenes. Two ww-specific choices baked in
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
