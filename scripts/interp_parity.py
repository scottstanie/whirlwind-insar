"""Parity check: whirlwind.interpolate (Rust) vs dolphin's numba interpolate, on
a synthetic complex ifg + random weights (with some masked pixels). Run in an env
that has BOTH the new whirlwind wheel AND dolphin+numba.

Usage: python scripts/interp_parity.py
"""

import sys

import numpy as np

sys.path.insert(0, "/Users/staniewi/repos/dolphin/src")
from dolphin.interpolation import interpolate as interp_numba  # noqa: E402
import whirlwind as ww  # noqa: E402

rng = np.random.default_rng(0)
m, n = 300, 320
phase = rng.uniform(-np.pi, np.pi, (m, n)).astype(np.float32)
amp = rng.uniform(0.5, 2.0, (m, n)).astype(np.float32)
ifg = (amp * np.exp(1j * phase)).astype(np.complex64)
ifg[rng.random((m, n)) < 0.05] = 0  # ~5% masked
weights = rng.random((m, n)).astype(np.float32)

kw = dict(weight_cutoff=0.5, num_neighbors=20, max_radius=51, min_radius=0, alpha=0.75)
out_np = np.asarray(interp_numba(ifg, weights, **kw))
out_rs = np.asarray(ww.interpolate(ifg, weights, 0.5, 20, 51, 0, 0.75))

valid = ifg != 0
dphase = np.angle(out_rs[valid] * np.conj(out_np[valid]))  # wrapped phase diff
damp = np.abs(np.abs(out_rs[valid]) - np.abs(out_np[valid]))
print(f"valid px: {valid.sum()}")
print(
    f"max |Δphase| = {np.abs(dphase).max():.2e} rad   mean = {np.abs(dphase).mean():.2e}"
)
print(f"max |Δamp|   = {damp.max():.2e}")
print(
    f"PARITY (phase<1e-3, amp<1e-4): {np.abs(dphase).max() < 1e-3 and damp.max() < 1e-4}"
)
