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
) -> NDArray[np.float32]:
    """2D phase unwrap with the CRLB-weighted Gaussian cost (for phase-linked IGs)."""

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
