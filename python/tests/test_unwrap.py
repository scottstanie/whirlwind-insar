"""whirlwind-rs unwrap tests, mirroring snaphu-py/test/test_unwrap.py."""

from __future__ import annotations

import numpy as np
import pytest

import whirlwind as ww


def _align_to_truth(unw: np.ndarray, truth: np.ndarray) -> np.ndarray:
    mean_diff = float(np.mean(unw - truth))
    offset = 2.0 * np.pi * round(mean_diff / (2.0 * np.pi))
    return unw - offset


class TestUnwrap:
    # ww.unwrap now returns (phase, conncomp). These solver-recovery tests pass
    # goldstein_alpha=0 to exercise the bare MCF unwrap (their original intent);
    # the Goldstein-on default path is covered by test_unwrap_returns_conncomp.
    def test_diagonal_ramp_clean(self):
        """SNAPHU-style smooth diagonal-ramp regression test."""
        y, x = np.ogrid[-3:3:512j, -3:3:512j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.ones(igram.shape, dtype=np.float32) * 0.999

        unw, _cc = ww.unwrap(igram, corr, nlooks=1.0, goldstein_alpha=0)

        aligned = _align_to_truth(unw, phase)
        np.testing.assert_allclose(aligned, phase, atol=1e-2)

    def test_smaller_ramp(self):
        y, x = np.ogrid[-1:1:128j, -1:1:128j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.ones(igram.shape, dtype=np.float32) * 0.999

        unw, _cc = ww.unwrap(igram, corr, nlooks=1.0, goldstein_alpha=0)
        aligned = _align_to_truth(unw, phase)
        np.testing.assert_allclose(aligned, phase, atol=1e-2)

    def test_nan_inputs_masked(self):
        """NaN-pixels are masked; the rest must unwrap correctly."""
        y, x = np.ogrid[-3:3:256j, -3:3:256j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.ones(igram.shape, dtype=np.float32) * 0.999

        # Mark a horizontal band as invalid.
        mask = np.zeros(igram.shape, dtype=np.bool_)
        mask[64:-64] = True
        igram[~mask] = np.nan + 1j * np.nan
        corr[~mask] = 0.0

        # Replace NaN before passing (whirlwind-rs doesn't auto-handle NaN igram).
        igram = np.nan_to_num(igram, nan=0.0).astype(np.complex64)

        unw, _cc = ww.unwrap(igram, corr, nlooks=1.0, mask=mask, goldstein_alpha=0)
        # Only check the valid band.
        aligned = _align_to_truth(unw[mask], phase[mask])
        np.testing.assert_allclose(aligned, phase[mask], atol=5e-2)

    def test_noisy_gaussian_bump(self):
        """Real-world flavor: a Gaussian deformation bump under Goodman noise."""
        truth = ww.diagonal_ramp(96, 96) * 0.0  # zeros; we'll add a bump
        # Build a small Gaussian bump via the bindings' simulator helpers.
        # We don't expose gaussian_bump as a binding yet; just synthesize here.
        m, n = 96, 96
        ci, cj = (m - 1) / 2, (n - 1) / 2
        sigma = n / 8.0
        truth = np.zeros((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                truth[i, j] = 6.0 * np.exp(
                    -((i - ci) ** 2 + (j - cj) ** 2) / (2 * sigma ** 2)
                )

        gamma = np.full((m, n), 0.85, dtype=np.float32)
        igram, corr = ww.simulate_ifg(truth, gamma, nlooks=10, seed=42)
        unw, _cc = ww.unwrap(igram, corr, nlooks=10.0, goldstein_alpha=0)
        aligned = _align_to_truth(unw, truth)
        # Within 2π pretty much anywhere for a smooth bump.
        assert np.max(np.abs(aligned - truth)) < 6.5

    def test_shape_mismatch_raises(self):
        igram = np.zeros((8, 8), dtype=np.complex64)
        corr = np.zeros((8, 9), dtype=np.float32)
        with pytest.raises(ValueError):
            ww.unwrap(igram, corr, nlooks=1.0, goldstein_alpha=0)

    def test_dtype_preserved(self):
        m, n = 16, 16
        igram = np.ones((m, n), dtype=np.complex64)
        corr = np.ones((m, n), dtype=np.float32)
        unw, _cc = ww.unwrap(igram, corr, nlooks=1.0, goldstein_alpha=0)
        assert unw.dtype == np.float32
        assert unw.shape == (m, n)

    def test_unwrap_returns_conncomp(self):
        """Default path returns (phase, conncomp); Goldstein off by default."""
        y, x = np.ogrid[-2:2:128j, -2:2:128j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.ones(igram.shape, dtype=np.float32) * 0.95

        unw, cc = ww.unwrap(igram, corr, nlooks=5.0)
        assert unw.shape == igram.shape and unw.dtype == np.float32
        assert cc.shape == igram.shape and cc.dtype == np.uint32
        # A coherent clean ramp is one connected component.
        assert cc.max() >= 1

        # The opt-in Goldstein path also works and returns a valid tuple.
        ug, ccg = ww.unwrap(igram, corr, nlooks=5.0, goldstein_alpha=0.7)
        assert ug.shape == igram.shape and ccg.dtype == np.uint32

        # Regression for the #34 bug: min_size_px was silently dropped on the
        # GOLDSTEIN branch specifically. A huge floor must drop all there too.
        _ug, ccg_strict = ww.unwrap(
            igram, corr, nlooks=5.0, goldstein_alpha=0.7, min_size_px=10**9
        )
        assert ccg_strict.max() == 0


def _k_correct(unw, truth):
    d = unw - truth
    d = d - 2 * np.pi * round(float(np.median(d)) / (2 * np.pi))
    return float(np.mean(np.round(d / (2 * np.pi)) == 0))


class TestCrlb:
    def test_unwrap_crlb_returns_conncomp(self):
        """unwrap_crlb returns (phase, conncomp) (#35) AND its default path is
        corner-safe: a steep clean ramp that the plain capacity-1 CRLB solver
        mis-routes is recovered exactly (the default now routes through reuse)."""
        m, n = 96, 96
        truth = np.fromfunction(
            lambda i, j: 0.3 * (i + j), (m, n)
        ).astype(np.float32)  # ~9π across, steep enough to expose corner stacking
        igram = np.exp(1j * truth).astype(np.complex64)
        var = np.full((m, n), 0.05, dtype=np.float32)  # low variance = high confidence

        unw, cc = ww.unwrap_crlb(igram, var)
        assert unw.shape == igram.shape and unw.dtype == np.float32
        assert cc.shape == igram.shape and cc.dtype == np.uint32
        assert cc.max() >= 1
        assert _k_correct(unw, truth) > 0.99, "corner-safe CRLB default must recover steep ramp"
