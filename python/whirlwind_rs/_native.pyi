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
