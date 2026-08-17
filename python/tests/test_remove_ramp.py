"""Tests for the ``remove_ramp`` pre-pass and the ramp helpers.

The de-ramp is congruence-preserving: unwrapping ``igram·exp(-i·ramp)`` and
adding ``ramp`` back is a valid unwrapping of ``igram`` for *any* plane. So the
core guarantees are (1) the result stays per-pixel congruent to the wrapped
input, and (2) flattening a steep ramp lets the bridge pass re-level subswaths
that a mask splits apart.
"""

from __future__ import annotations

import numpy as np
import pytest

import whirlwind as ww

TAU = 2.0 * np.pi


def _ramp_igram(m: int, n: int, a: float, b: float) -> np.ndarray:
    ii, jj = np.mgrid[0:m, 0:n]
    return np.exp(1j * (a * ii + b * jj)).astype(np.complex64)


def _congruence_err(unw: np.ndarray, igram: np.ndarray, mask: np.ndarray) -> float:
    """Max |wrap(unw - angle(igram))| over valid pixels (0 if congruent)."""
    resid = np.angle(np.exp(1j * (unw - np.angle(igram)))).astype(np.float64)
    return float(np.max(np.abs(resid[mask])))


def _align_to_truth(unw: np.ndarray, truth: np.ndarray) -> np.ndarray:
    mean_diff = float(np.mean(unw - truth))
    return unw - TAU * round(mean_diff / TAU)


class TestRampHelpers:
    def test_fit_ramp_recovers_slopes(self):
        a, b = 0.20, -0.13
        ig = _ramp_igram(200, 180, a, b)
        row_slope, col_slope = ww.fit_ramp(ig)
        assert row_slope == pytest.approx(a, abs=1e-3)
        assert col_slope == pytest.approx(b, abs=1e-3)

    def test_deramp_flattens(self):
        a, b = 0.31, 0.07
        ig = _ramp_igram(150, 160, a, b)
        row_slope, col_slope = ww.fit_ramp(ig)
        flat = ww.deramp(ig, row_slope, col_slope)
        # No dominant slope left after de-ramping.
        r2, c2 = ww.fit_ramp(flat)
        assert abs(r2) < 1e-3 and abs(c2) < 1e-3

    def test_deramp_preserves_nodata_and_magnitude(self):
        ig = _ramp_igram(48, 40, 0.1, 0.2)
        ig[5, 6] *= 3.0  # vary amplitude
        ig[10, 10] = 0  # nodata hole
        row_slope, col_slope = ww.fit_ramp(ig)
        out = ww.deramp(ig, row_slope, col_slope)
        assert out[10, 10] == 0
        np.testing.assert_allclose(np.abs(out), np.abs(ig), atol=1e-4)

    def test_deramp_add_ramp_roundtrip(self):
        # add_ramp is the unwrapped-domain inverse of deramp.
        a, b = 0.05, 0.09
        m, n = 64, 48
        ii, jj = np.mgrid[0:m, 0:n]
        truth = (a * ii + b * jj + 0.5 * np.sin(ii / 10.0)).astype(np.float32)
        ig = np.exp(1j * truth).astype(np.complex64)
        row_slope, col_slope = ww.fit_ramp(ig)
        flat_phase = np.angle(ww.deramp(ig, row_slope, col_slope)).astype(np.float32)
        restored = ww.add_ramp(flat_phase, row_slope, col_slope)
        # restored ≡ truth (mod 2π) everywhere.
        resid = np.angle(np.exp(1j * (restored - truth)))
        assert np.max(np.abs(resid)) < 1e-3

    def test_fit_ramp_dtypes(self):
        ig = _ramp_igram(16, 16, 0.1, 0.1)
        row_slope, col_slope = ww.fit_ramp(ig)
        assert isinstance(row_slope, float) and isinstance(col_slope, float)
        assert ww.deramp(ig, row_slope, col_slope).dtype == np.complex64
        assert ww.add_ramp(np.zeros((16, 16), np.float32), 0.1, 0.1).dtype == np.float32


class TestUnwrapRemoveRamp:
    def test_congruent_and_recovers_clean_ramp(self):
        """remove_ramp on a steep clean ramp stays congruent and recovers truth."""
        m = n = 128
        a = b = 0.25  # ~10 fringes across each axis
        ii, jj = np.mgrid[0:m, 0:n]
        truth = (a * ii + b * jj).astype(np.float32)
        igram = np.exp(1j * truth).astype(np.complex64)
        corr = np.full((m, n), 0.95, np.float32)
        mask = np.ones((m, n), dtype=bool)

        unw, cc = ww.unwrap(igram, corr, nlooks=5.0, remove_ramp=True)
        assert unw.shape == (m, n) and unw.dtype == np.float32
        assert cc.dtype == np.uint32
        # Per-pixel congruent to the wrapped input.
        assert _congruence_err(unw, igram, mask) < 1e-3
        # And it recovers the true ramp up to a global 2π offset.
        np.testing.assert_allclose(_align_to_truth(unw, truth), truth, atol=1e-2)

    def test_flat_scene_noop(self):
        """A ramp-free scene: remove_ramp fits ≈0 and must not corrupt the unwrap."""
        m = n = 96
        ii, jj = np.mgrid[0:m, 0:n]
        truth = (TAU * 2 * (ii + jj) / n).astype(np.float32)  # gentle, still a ramp
        igram = np.exp(1j * truth).astype(np.complex64)
        corr = np.full((m, n), 0.95, np.float32)
        off, _ = ww.unwrap(igram, corr, nlooks=1.0, remove_ramp=False)
        on, _ = ww.unwrap(igram, corr, nlooks=1.0, remove_ramp=True)
        # Both valid unwraps; compare up to a global gauge offset.
        a_off = _align_to_truth(np.nan_to_num(off), truth)
        a_on = _align_to_truth(np.nan_to_num(on), truth)
        np.testing.assert_allclose(a_on, a_off, atol=1e-2)

    def test_remove_ramp_relevels_split_subswaths(self):
        """The motivating case: a steep ramp split by a masked gap into two
        subswaths. Flattening the ramp lets the bridge pass set the correct
        relative 2π level between the pieces."""
        m = n = 128
        a = b = 0.25
        ii, jj = np.mgrid[0:m, 0:n]
        truth = (a * ii + b * jj).astype(np.float32)
        igram = np.exp(1j * truth).astype(np.complex64)
        corr = np.full((m, n), 0.95, np.float32)
        mask = np.ones((m, n), dtype=bool)
        mask[:, 62:66] = False  # a masked gap -> two subswaths
        igram[~mask] = 0

        unw, _ = ww.unwrap(
            igram, corr, nlooks=5.0, mask=mask, remove_ramp=True, bridge=True
        )
        unw = np.asarray(unw, np.float32)

        left = mask.copy()
        left[:, 62:] = False
        right = mask.copy()
        right[:, :66] = False

        def rel_cycles(u):
            al = np.median(np.round((u[left] - truth[left]) / TAU))
            ar = np.median(np.round((u[right] - truth[right]) / TAU))
            return ar - al

        # The two subswaths are consistently leveled (matching the shared ramp),
        # and each pixel stays congruent to the wrapped input.
        assert rel_cycles(unw) == 0
        assert _congruence_err(unw, igram, mask) < 1e-3

    def test_composes_with_prepasses(self):
        """remove_ramp composes with interpolate / goldstein / downsample and
        still returns a valid, congruent unwrap."""
        m = n = 96
        a = b = 0.18
        ii, jj = np.mgrid[0:m, 0:n]
        truth = (a * ii + b * jj).astype(np.float32)
        igram, corr = ww.simulate_ifg(truth, np.full((m, n), 0.8, np.float32), 10, 7)
        mask = np.ones((m, n), dtype=bool)

        for kwargs in (
            {"interpolate": True, "interp_cutoff": 0.3},
            {"goldstein_alpha": 0.7},
            {"downsample": 2},
        ):
            unw, cc = ww.unwrap(igram, corr, nlooks=10.0, remove_ramp=True, **kwargs)
            assert unw.shape == (m, n) and unw.dtype == np.float32
            assert cc.dtype == np.uint32
            assert _congruence_err(unw, igram, mask) < 1e-2, kwargs
