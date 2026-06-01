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


def _bowl(shape, g_edge):
    """Paraboloid with edge fringe rate ``g_edge`` rad/pixel."""
    m, n = shape
    ci, cj = (m - 1) / 2, (n - 1) / 2
    i, j = np.ogrid[:m, :n]
    r = np.sqrt((i - ci) ** 2 + (j - cj) ** 2)
    r_max = float(np.sqrt(ci**2 + cj**2))
    a = g_edge / (2.0 * r_max)
    return (a * r * r).astype(np.float32)


def _cone(shape, g):
    m, n = shape
    ci, cj = (m - 1) / 2, (n - 1) / 2
    i, j = np.ogrid[:m, :n]
    r = np.sqrt((i - ci) ** 2 + (j - cj) ** 2)
    return (g * r).astype(np.float32)


def _k_correct(unw, truth):
    d = unw - truth
    d = d - 2 * np.pi * round(float(np.median(d)) / (2 * np.pi))
    return float(np.mean(np.round(d / (2 * np.pi)) == 0))


class TestPyramid:
    def test_shape_and_dtype(self):
        truth = _cone((96, 96), 0.2 * np.pi)
        ig, corr = ww.simulate_ifg(truth, np.full(truth.shape, 0.9, np.float32), 8, 0)
        unw = ww.unwrap_pyramid(ig, corr, nlooks=8.0, base_factor=4)
        assert unw.dtype == np.float32
        assert unw.shape == (96, 96)

    def test_recovers_steep_bowl_that_multilook_destroys(self):
        """The core motivation. A steep but unaliased bowl (edge rate 0.5π) is
        recovered by the pyramid (modest base) but destroyed by a single ×8
        multilook, which aliases at edge rates above π/8."""
        truth = _bowl((192, 192), 0.5 * np.pi)
        ig, corr = ww.simulate_ifg(truth, np.full(truth.shape, 0.95, np.float32), 8, 0)

        ml8, _cc = ww.unwrap(ig, corr, nlooks=8.0, multilook=8, goldstein_alpha=0)
        pyr = ww.unwrap_pyramid(ig, corr, nlooks=8.0, base_factor=2)

        assert _k_correct(ml8, truth) < 0.3, "multilook ×8 should alias the steep bowl"
        assert _k_correct(pyr, truth) > 0.75, "pyramid should recover the steep bowl"

    def test_beats_fullres_in_heavy_noise(self):
        """In the mild-rate / heavy-noise regime that motivates multilooking,
        the pyramid's coarse prediction is noise-robust where a full-res solve
        drowns in residues — while not aliasing the way single-shot ml8 does."""
        truth = _cone((192, 192), 0.2 * np.pi)
        ig, corr = ww.simulate_ifg(truth, np.full(truth.shape, 0.25, np.float32), 4, 0)

        # Baseline pinned to LINEAR full-res: ww.unwrap now defaults to the
        # corner-safe reuse solver, which is itself competitive in heavy noise
        # (≈0.94 vs pyramid ≈0.95 here), so the pyramid's margin is measured
        # against the weak linear cost the multigrid prediction rescues.
        full = ww.unwrap_pyramid(ig, corr, nlooks=4.0, base_factor=1, solver="linear")
        pyr = ww.unwrap_pyramid(ig, corr, nlooks=4.0, base_factor=4)

        assert _k_correct(pyr, truth) > _k_correct(full, truth) + 0.1

    def test_base_factor_one_linear_matches_plain_unwrap(self):
        # base=1 + reuse degenerates to a single plain (no-Goldstein) unwrap.
        truth = _cone((64, 64), 0.15 * np.pi)
        ig, corr = ww.simulate_ifg(truth, np.full(truth.shape, 0.9, np.float32), 8, 0)
        a = ww.unwrap_pyramid(ig, corr, nlooks=8.0, base_factor=1, solver="reuse")
        b, _cc = ww.unwrap(ig, corr, nlooks=8.0, goldstein_alpha=0)
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_reuse_solver_fixes_clean_bowl_corners(self):
        # The linear cost mis-routes the corners of a clean steep bowl; the
        # default reuse solver does not (capacity-1 boundary-stacking fix).
        truth = _bowl((192, 192), 0.6 * np.pi)
        ig = np.exp(1j * truth).astype(np.complex64)
        corr = np.full(truth.shape, 0.999, np.float32)
        lin = ww.unwrap_pyramid(ig, corr, nlooks=1.0, base_factor=1, solver="linear")
        reu = ww.unwrap_pyramid(ig, corr, nlooks=1.0, base_factor=1, solver="reuse")
        assert _k_correct(reu, truth) > 0.99
        assert _k_correct(reu, truth) > _k_correct(lin, truth) + 0.05

    def test_auto_base_factor(self):
        # base_factor=0 -> automatic (Itoh-violation-rate probe). On a steep
        # noisy bowl it recovers the surface, and in the heavy-noise gentle-rate
        # regime it beats full-res by downsampling as far as it safely can.
        steep = _bowl((256, 256), 0.6 * np.pi)
        ig, corr = ww.simulate_ifg(steep, np.full(steep.shape, 0.9, np.float32), 8, 0)
        assert _k_correct(ww.unwrap_pyramid(ig, corr, nlooks=8.0, base_factor=0), steep) > 0.9

        noisy = _cone((256, 256), 0.2 * np.pi)
        ig, corr = ww.simulate_ifg(noisy, np.full(noisy.shape, 0.25, np.float32), 4, 0)
        ka = _k_correct(ww.unwrap_pyramid(ig, corr, nlooks=4.0, base_factor=0), noisy)
        # vs LINEAR full-res (ww.unwrap now defaults to corner-safe reuse, which
        # is competitive in heavy noise — see test_beats_fullres_in_heavy_noise).
        kf = _k_correct(ww.unwrap_pyramid(ig, corr, nlooks=4.0, base_factor=1, solver="linear"), noisy)
        assert ka > kf + 0.05, f"auto-base {ka} should beat linear full-res {kf} in heavy noise"

    def test_tiled_finest_level_matches_untiled(self):
        # In-regime (base unaliased): tiling the finest levels must not change
        # the answer beyond seam-level noise.
        truth = _bowl((384, 384), 0.3 * np.pi)
        ig, corr = ww.simulate_ifg(truth, np.full(truth.shape, 0.85, np.float32), 8, 0)
        untiled = ww.unwrap_pyramid(ig, corr, nlooks=8.0, base_factor=4, tile_size=0)
        tiled = ww.unwrap_pyramid(ig, corr, nlooks=8.0, base_factor=4, tile_size=128)
        assert _k_correct(tiled, truth) > 0.9
        assert abs(_k_correct(tiled, truth) - _k_correct(untiled, truth)) < 0.05

    def test_unknown_solver_raises(self):
        ig = np.ones((16, 16), dtype=np.complex64)
        corr = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            ww.unwrap_pyramid(ig, corr, nlooks=1.0, solver="bogus")


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
