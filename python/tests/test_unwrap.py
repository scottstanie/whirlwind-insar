"""whirlwind unwrap tests, mirroring snaphu-py/test/test_unwrap.py."""

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
        """NaN pixels (a nodata hole) are masked; the rest must unwrap correctly."""
        y, x = np.ogrid[-3:3:256j, -3:3:256j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.ones(igram.shape, dtype=np.float32) * 0.999

        # Punch a circular nodata hole in the middle (e.g. a lake): the valid
        # area stays one connected region around it.
        ii, jj = np.ogrid[:256, :256]
        hole = (ii - 128) ** 2 + (jj - 128) ** 2 < 40**2
        mask = ~hole
        igram[~mask] = np.nan + 1j * np.nan
        corr[~mask] = 0.0

        # unwrap() sanitizes NaN inputs to nodata (0) with a warning, so the
        # NaN hole can be passed straight through.
        unw, _cc = ww.unwrap(igram, corr, nlooks=1.0, mask=mask, goldstein_alpha=0)
        # Only check the valid pixels.
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
                    -((i - ci) ** 2 + (j - cj) ** 2) / (2 * sigma**2)
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

    @pytest.mark.parametrize("bad", [0.0, 0.5, -1.0, np.nan])
    def test_nlooks_below_one_raises(self, bad):
        """Nonphysical nlooks (< 1 or NaN) must raise early, not panic in Rust."""
        igram = np.ones((8, 8), dtype=np.complex64)
        corr = np.ones((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="nlooks"):
            ww.unwrap(igram, corr, nlooks=bad)

    def test_huge_nlooks_warns_but_succeeds(self, caplog):
        """A huge effective-looks value (e.g. a 100x100 multilook) must not
        crash or produce NaN; it is capped for the cost model with a warning."""
        y, x = np.ogrid[-2:2:64j, -2:2:64j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.full(igram.shape, 0.99, dtype=np.float32)

        with caplog.at_level("WARNING", logger="whirlwind"):
            unw, cc = ww.unwrap(igram, corr, nlooks=10_000.0)
        assert np.all(np.isfinite(unw)), "huge nlooks produced non-finite phase"
        assert cc.dtype == np.uint32
        assert any("cost-model cap" in r.message for r in caplog.records)

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

    def test_conncomp_algorithm_default_and_selector(self):
        """Default conncomp is the SNAPHU-faithful grow; `linear` opts out; the
        choice never changes the unwrapped phase; bad values raise."""
        y, x = np.ogrid[-2:2:128j, -2:2:128j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.ones(igram.shape, dtype=np.float32) * 0.95

        unw_snaphu, cc_snaphu = ww.unwrap(igram, corr, nlooks=5.0)  # default
        unw_linear, cc_linear = ww.unwrap(
            igram, corr, nlooks=5.0, conncomp_algorithm="linear"
        )
        assert cc_snaphu.dtype == np.uint32 and cc_linear.dtype == np.uint32
        # A clean coherent ramp is one component under either grow.
        assert cc_snaphu.max() >= 1 and cc_linear.max() >= 1
        # The conncomp choice must not perturb the unwrapped phase.
        np.testing.assert_array_equal(unw_snaphu, unw_linear)

        with pytest.raises(ValueError):
            ww.unwrap(igram, corr, nlooks=5.0, conncomp_algorithm="bogus")

    def test_phase_grad_window_default_and_effect(self):
        """`phase_grad_window` mirrors SNAPHU's KPARDPSI/KPERPDPSI. The default
        (7, 7) must reproduce the no-argument result bit-for-bit, a non-default
        window must be able to change the solved phase, the parallel/perpendicular
        orientation must matter on an anisotropic scene, and a zero extent must
        raise cleanly rather than panic in Rust."""
        rng = np.random.default_rng(3)
        m, n = 160, 160
        yy, xx = np.mgrid[0:m, 0:n].astype(np.float32)
        # Anisotropic wrapping phase: fast vertical fringes + a y-dependent bend.
        true = 0.8 * xx + 0.002 * (yy - n / 2) ** 2
        wrapped = np.angle(np.exp(1j * true)).astype(np.float32)
        noise = rng.normal(0, 0.7, (m, n)).astype(np.float32)
        igram = np.exp(1j * (wrapped + noise)).astype(np.complex64)
        corr = np.full((m, n), 0.35, np.float32)

        base, _ = ww.unwrap(igram, corr, nlooks=4.0)
        same, _ = ww.unwrap(igram, corr, nlooks=4.0, phase_grad_window=(7, 7))
        np.testing.assert_array_equal(base, same)

        # A different window can change the integer-cycle field, and the
        # parallel/perpendicular orientation is not symmetric here.
        a, _ = ww.unwrap(igram, corr, nlooks=4.0, phase_grad_window=(25, 3))
        b, _ = ww.unwrap(igram, corr, nlooks=4.0, phase_grad_window=(3, 25))
        assert not np.array_equal(np.nan_to_num(a), np.nan_to_num(base))
        assert not np.array_equal(np.nan_to_num(a), np.nan_to_num(b))

        for bad in [(0, 7), (7, 0), (-1, 7)]:
            with pytest.raises(ValueError, match="phase_grad_window"):
                ww.unwrap(igram, corr, nlooks=4.0, phase_grad_window=bad)

    def test_conncomp_reliability_raises_to_more_conservative(self):
        """Raising `conncomp_reliability` cuts more edges (more conservative):
        an absurdly high threshold cuts every edge, so all pixels are isolated
        and dropped below the size floor, labeling nothing. `conncomp_min_coherence`
        is set to None here to exercise the raw reliability knob directly."""
        y, x = np.ogrid[-2:2:128j, -2:2:128j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.ones(igram.shape, dtype=np.float32) * 0.95

        _u, cc0 = ww.unwrap(
            igram,
            corr,
            nlooks=5.0,
            conncomp_min_coherence=None,
            conncomp_reliability=0.0,
        )
        _u, cc_hi = ww.unwrap(
            igram,
            corr,
            nlooks=5.0,
            conncomp_min_coherence=None,
            conncomp_reliability=1e12,
        )
        assert cc0.max() >= 1  # raw reliability 0 labels the coherent ramp
        assert cc_hi.max() == 0  # everything cut -> nothing survives the floor

    def test_conncomp_min_coherence_auto_default(self):
        """The default conncomp_min_coherence='auto' is the looks-aware floor
        (0.32/sqrt(nlooks)); at nlooks=16 it is exactly the validated 0.08."""
        assert ww.conncomp_min_coherence_auto(16.0) == pytest.approx(0.08)
        # Looks-aware: fewer looks -> higher floor, more looks -> lower.
        assert ww.conncomp_min_coherence_auto(4.0) > ww.conncomp_min_coherence_auto(
            64.0
        )
        y, x = np.ogrid[-2:2:128j, -2:2:128j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.full(igram.shape, 0.9, np.float32)
        # The "auto" default at nlooks=16 must match an explicit 0.08.
        _u, cc_auto = ww.unwrap(igram, corr, nlooks=16.0)
        _u, cc_008 = ww.unwrap(igram, corr, nlooks=16.0, conncomp_min_coherence=0.08)
        np.testing.assert_array_equal(cc_auto, cc_008)

    def test_conncomp_min_coherence_gates_and_matches_reliability(self):
        """`conncomp_min_coherence` (the default knob) is more conservative as it
        rises, and is exactly equivalent to passing the mapped raw reliability."""
        y, x = np.ogrid[-2:2:128j, -2:2:128j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.full(igram.shape, 0.9, np.float32)

        # The gentle 0.08 default keeps the coherent ramp labeled.
        _u, cc_def = ww.unwrap(igram, corr, nlooks=16.0)
        assert cc_def.max() >= 1
        # A high target min-coherence is strictly more conservative than off.
        _u, cc_all = ww.unwrap(igram, corr, nlooks=16.0, conncomp_min_coherence=None)
        _u, cc_strict = ww.unwrap(igram, corr, nlooks=16.0, conncomp_min_coherence=0.99)
        assert (cc_strict > 0).mean() < (cc_all > 0).mean()
        # min_coherence == passing the mapped raw reliability with min_coh None.
        r = ww.conncomp_reliability_from_coherence(0.5, 16.0)
        _u, cc_mc = ww.unwrap(igram, corr, nlooks=16.0, conncomp_min_coherence=0.5)
        _u, cc_rl = ww.unwrap(
            igram,
            corr,
            nlooks=16.0,
            conncomp_min_coherence=None,
            conncomp_reliability=r,
        )
        np.testing.assert_array_equal(cc_mc, cc_rl)

    def test_conncomp_reliability_from_coherence(self):
        """The coherence->reliability helper is monotonic and matches the
        documented `1 / sigma2(gamma)` mapping (so a user can pick a value)."""
        f = ww.conncomp_reliability_from_coherence
        # Higher coherence -> larger value (stricter): monotone increasing.
        vals = [f(g, 16.0) for g in (0.1, 0.2, 0.3, 0.5, 0.7)]
        assert vals == sorted(vals)
        # gamma=0.3, L=16: sigma2=(1-0.09)/(2*16*0.09)=0.31597 -> 1/sigma2 ~= 3.16.
        assert f(0.3, 16.0) == 1.0 / ((1 - 0.3**2) / (2 * 16.0 * 0.3**2))
        assert abs(f(0.3, 16.0) - 3.165) < 0.01


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
        truth = np.fromfunction(lambda i, j: 0.3 * (i + j), (m, n)).astype(
            np.float32
        )  # ~9π across, steep enough to expose corner stacking
        igram = np.exp(1j * truth).astype(np.complex64)
        var = np.full((m, n), 0.05, dtype=np.float32)  # low variance = high confidence

        unw, cc = ww.unwrap_crlb(igram, var)
        assert unw.shape == igram.shape and unw.dtype == np.float32
        assert cc.shape == igram.shape and cc.dtype == np.uint32
        assert cc.max() >= 1
        assert (
            _k_correct(unw, truth) > 0.99
        ), "corner-safe CRLB default must recover steep ramp"


class TestBridge:
    """Integration-component gauge bridging (``unwrap(bridge=)``)."""

    def test_label_components_splits_mask(self):
        mask = np.ones((6, 9), dtype=np.bool_)
        mask[:, 4] = False  # a masked column splits the frame in two
        labels, n = ww.label_components(np.ascontiguousarray(mask))
        assert n == 2
        assert labels[mask].min() >= 1 and (labels[~mask] == 0).all()
        # the two sides carry distinct labels
        assert labels[0, 0] != labels[0, 8]

    def test_bridge_noop_single_region(self):
        # One connected valid region -> bridging must be a strict no-op.
        m = n = 96
        ii, jj = np.mgrid[0:m, 0:n]
        phase = (ii + jj).astype(np.float32) / n * (2 * np.pi * 3)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.full((m, n), 0.95, np.float32)
        off, _ = ww.unwrap(igram, corr, nlooks=1.0, bridge=False)
        on, _ = ww.unwrap(igram, corr, nlooks=1.0, bridge=True)
        np.testing.assert_array_equal(off, on)

    def test_bridge_fixes_disconnected_gauge(self):
        # A gentle ramp split by a thin masked strip: the integrator seeds each
        # side independently, so the far side picks up an integer-cycle gauge
        # error. The MST bridge re-levels it (matching isce3), while bridge=False
        # leaves the error.
        m = n = 128
        tau = 2 * np.pi
        ii, jj = np.mgrid[0:m, 0:n]
        truth = (ii + jj).astype(np.float32) / n * (tau * 3)  # ~3 gentle cycles
        igram = np.exp(1j * truth).astype(np.complex64)
        corr = np.full((m, n), 0.95, np.float32)
        mask = np.ones((m, n), dtype=np.bool_)
        mask[:, 62:65] = False  # 3-px masked river -> two regions
        igram[~mask] = 0

        off, _ = ww.unwrap(igram, corr, nlooks=1.0, mask=mask, bridge=False)
        on, _ = ww.unwrap(igram, corr, nlooks=1.0, mask=mask, bridge=True)
        off = np.asarray(off, np.float32)
        on = np.asarray(on, np.float32)

        left = mask.copy()
        left[:, 62:] = False
        right = mask.copy()
        right[:, :65] = False

        def rel_cycles(u):
            al = np.median(np.round((u[left] - truth[left]) / tau))
            ar = np.median(np.round((u[right] - truth[right]) / tau))
            return ar - al

        assert abs(rel_cycles(off)) >= 1, "expected an unbridged integer gauge error"
        assert rel_cycles(on) == 0, "bridging must re-level the disconnected region"

    def test_bridge_components_public_direct(self):
        # The public bridge_components operates on an unwrapped phase + mask
        # alone. Build a clean ramp, split it by a masked strip, and inject a
        # +2-cycle gauge error into the right region; the bridge must remove it.
        tau = 2 * np.pi
        m = n = 128
        ii, jj = np.mgrid[0:m, 0:n]
        truth = (ii + jj).astype(np.float32) / n * (tau * 3)
        mask = np.ones((m, n), dtype=np.bool_)
        mask[:, 62:65] = False  # masked strip -> two integration regions
        unw = np.where(mask, truth, 0.0).astype(np.float32)
        unw[:, 65:] += tau * 2  # right region offset by +2 cycles

        bridged = np.asarray(ww.bridge_components(unw, mask), np.float32)

        left = mask.copy()
        left[:, 62:] = False
        right = mask.copy()
        right[:, :65] = False

        def rel_cycles(u):
            al = np.median(np.round((u[left] - truth[left]) / tau))
            ar = np.median(np.round((u[right] - truth[right]) / tau))
            return ar - al

        # Bridging makes the regions mutually consistent (the absolute level is a
        # free gauge, so it need not equal truth): the +2-cycle relative jump goes.
        assert rel_cycles(unw) == 2
        assert rel_cycles(bridged) == 0
        # A single-region frame is returned unchanged.
        full = np.ones((m, n), dtype=np.bool_)
        same = np.asarray(ww.bridge_components(truth.copy(), full), np.float32)
        np.testing.assert_array_equal(same, truth)


class TestSolveCoherenceGate:
    """PHASS-style solve-domain gating (``solve_min_coherence``)."""

    @staticmethod
    def _islands_scene(rng=None):
        """Steep ramp with a pure-noise channel splitting two coherent islands.

        The channel's sample coherence sits at the zero-coherence floor for
        L=25 (0.177), i.e. below the auto gate, and its phase is uniform
        noise: exactly the indistinguishable-from-noise ocean the gate is for.
        """
        rng = np.random.default_rng(1234) if rng is None else rng
        m, n = 256, 256
        y, x = np.ogrid[0:m, 0:n]
        truth = (0.10 * x + 0.04 * y).astype(np.float32)
        phase = truth.copy()
        corr = np.full((m, n), 0.9, dtype=np.float32)
        channel = slice(m // 2 - 20, m // 2 + 20)
        phase[channel, :] = rng.uniform(-np.pi, np.pi, (40, n)).astype(np.float32)
        corr[channel, :] = 0.14
        igram = np.exp(1j * phase).astype(np.complex64)
        chan_mask = np.zeros((m, n), dtype=bool)
        chan_mask[channel, :] = True
        return igram, corr, truth, chan_mask

    def test_gate_noop_on_clean_scene(self):
        y, x = np.ogrid[-1:1:128j, -1:1:128j]
        phase = (np.pi * (x + y)).astype(np.float32)
        igram = np.exp(1j * phase).astype(np.complex64)
        corr = np.full(igram.shape, 0.9, dtype=np.float32)
        gated, cc_g = ww.unwrap(igram, corr, nlooks=25.0)
        ungated, cc_u = ww.unwrap(igram, corr, nlooks=25.0, solve_min_coherence=None)
        np.testing.assert_array_equal(gated, ungated)
        np.testing.assert_array_equal(cc_g, cc_u)

    def test_gated_pixels_relabeled_and_rewrap_exact(self):
        igram, corr, truth, chan = self._islands_scene()
        unw, cc = ww.unwrap(igram, corr, nlooks=25.0, min_size_px=10)
        # Gated pixels: labeled 0, finite, and rewrap-exact.
        assert cc[chan].max() == 0
        assert np.isfinite(unw).all()
        resid = np.angle(np.exp(1j * (unw - np.angle(igram))))
        assert np.abs(resid).max() < 1e-3
        # Both islands are recovered exactly (up to one global 2pi level).
        keep = ~chan
        aligned = _align_to_truth(unw[keep], truth[keep])
        # Each island may sit on its own level; check per island instead.
        top = _align_to_truth(unw[:108, :], truth[:108, :])
        bot = _align_to_truth(unw[148:, :], truth[148:, :])
        np.testing.assert_allclose(top, truth[:108, :], atol=1e-2)
        np.testing.assert_allclose(bot, truth[148:, :], atol=1e-2)
        del aligned

    def test_gate_threshold_knob(self):
        igram, corr, truth, chan = self._islands_scene()
        # A gate below the channel coherence (0.14) solves the channel too:
        # identical to disabling the gate.
        loose, _ = ww.unwrap(igram, corr, nlooks=25.0, solve_min_coherence=0.05)
        off, _ = ww.unwrap(igram, corr, nlooks=25.0, solve_min_coherence=None)
        np.testing.assert_array_equal(loose, off)

    def test_gate_all_pixels_falls_back(self):
        igram, corr, truth, _ = self._islands_scene()
        unw, _cc = ww.unwrap(igram, corr, nlooks=25.0, solve_min_coherence=0.99)
        off, _ = ww.unwrap(igram, corr, nlooks=25.0, solve_min_coherence=None)
        np.testing.assert_array_equal(unw, off)

    def test_auto_formula(self):
        assert ww.solve_min_coherence_auto(25.0) == pytest.approx(0.17725, abs=1e-4)
        assert ww.solve_min_coherence_auto(1e6) == 0.02
        assert ww.solve_min_coherence_auto(1.0) == 0.18
