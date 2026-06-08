"""Integration-component gauge bridging post-pass for :func:`whirlwind.unwrap`.

Extracted from ``__init__.py`` to keep the package entry point small. The single
public helper, :func:`_bridge_components`, repairs the relative 2π level of
regions that the valid mask splits apart (for example two land slabs separated by
a low-coherence river), which the MCF integrator seeds at an arbitrary level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._native import label_components

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    connected) frame yields no shifts and is byte-identical.
    """
    # Lazy import to avoid a circular import: unwrap() calls _bridge_components(),
    # and _bridge_components() recursively calls unwrap() for the coarse anchor.
    from . import unwrap

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
