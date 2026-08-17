"""Tests for the ``connect_gaps`` pre-pass.

The point of connecting is to make a gapped frame a single component so the
solve picks the per-region 2*pi levels itself. So the tests assert the thing
that actually matters -- the regions come out on the right relative level -- and
each one checks the baseline FAILS the same case, because an assertion that
passes with the feature switched off is not testing the feature.
"""

from __future__ import annotations

import numpy as np
import pytest

import whirlwind as ww

TWOPI = 2.0 * np.pi


def striped_scene(
    m: int = 256,
    n: int = 256,
    n_stripes: int = 5,
    gap: int = 10,
    ramp_cycles: float = 6.0,
    curvature_cycles: float = 0.0,
    coherence: float = 0.8,
    nlooks: int = 40,
    seed: int = 3,
):
    """A scene cut into subswath stripes, plus its truth and mask.

    ``curvature_cycles`` adds a smooth NON-planar surface. That matters because
    a plane fit can absorb a pure ramp, so a ramp-only scene cannot tell a
    method that models curvature apart from one that does not.
    """
    ii, jj = np.mgrid[0:m, 0:n]
    x = jj / (n - 1)
    y = ii / (m - 1)
    truth = TWOPI * ramp_cycles * x
    if curvature_cycles:
        truth = truth + TWOPI * curvature_cycles * (
            np.sin(2.2 * np.pi * x + 0.6) * np.cos(1.7 * np.pi * y - 0.3)
            + 0.5 * np.sin(1.1 * np.pi * (x + y))
        )
    truth = truth.astype(np.float32)

    mask = np.zeros((m, n), dtype=bool)
    width = (n - (n_stripes - 1) * gap) // n_stripes
    for s in range(n_stripes):
        a = s * (width + gap)
        mask[:, a : a + width] = True
    gamma = np.full((m, n), coherence, np.float32)
    igram, corr = ww.simulate_ifg(truth, gamma, nlooks, seed)
    igram = np.asarray(igram, np.complex64)
    corr = np.asarray(corr, np.float32)
    igram[~mask] = 0
    corr = np.where(mask, corr, 0).astype(np.float32)
    return igram, corr, mask, truth, float(nlooks)


def on_cycle_fraction(unw, truth, mask) -> float:
    """Fraction of valid pixels on the correct cycle, after one global shift."""
    d = (np.asarray(unw, np.float64) - truth)[mask]
    d = d - TWOPI * np.round(np.median(d) / TWOPI)
    return float(np.mean(np.abs(d) < np.pi))


class TestConnectGapsHelper:
    def test_connects_only_interior_gaps(self):
        mask = np.ones((8, 80), dtype=bool)
        mask[:, :6] = False  # frame edge: only one side, must NOT be crossed
        mask[:, 40:44] = False  # interior gap: must be crossed
        igram = np.exp(1j * np.zeros((8, 80))).astype(np.complex64)
        igram[~mask] = 0
        _out, added = ww.connect_gaps(igram, mask, edge=8)
        assert added[:, 40:44].all()
        assert not added[:, :6].any()
        assert not added[mask].any()

    def test_skips_gaps_without_enough_data_to_extrapolate_from(self):
        """A sliver of a region cannot define a fringe rate, so leave the gap."""
        mask = np.ones((8, 40), dtype=bool)
        mask[:, :30] = False  # leaves a 4 px sliver before the gap
        mask[:, 34:36] = False
        igram = np.exp(1j * np.zeros((8, 40))).astype(np.complex64)
        igram[~mask] = 0
        _out, added = ww.connect_gaps(igram, mask, edge=8)
        assert not added.any()

    def test_respects_max_gap(self):
        mask = np.ones((8, 40), dtype=bool)
        mask[:, 18:30] = False  # 12 px wide
        igram = np.exp(1j * np.zeros((8, 40))).astype(np.complex64)
        igram[~mask] = 0
        _o, wide = ww.connect_gaps(igram, mask, max_gap=6, edge=8)
        assert not wide.any(), "gap wider than max_gap must be left alone"
        _o, narrow = ww.connect_gaps(igram, mask, max_gap=20, edge=8)
        assert narrow[:, 18:30].all()

    def test_carries_the_fringe_rate_across(self):
        """The synthesised path continues the fringes; it does not flatten them."""
        m, n, gap_a, gap_b = 32, 120, 55, 65
        cycles = 4.0
        truth = TWOPI * cycles * np.arange(n) / (n - 1)
        igram = np.exp(1j * np.tile(truth, (m, 1))).astype(np.complex64)
        mask = np.ones((m, n), dtype=bool)
        mask[:, gap_a:gap_b] = False
        igram[~mask] = 0
        out, added = ww.connect_gaps(igram, mask, edge=32)
        got = np.unwrap(np.angle(out[m // 2]))
        got -= got[0] - truth[0]
        # Inside the gap the path should track the true ramp, not short-circuit
        # it: a flattening fill would be off by the phase the gap spans.
        err = np.abs(got[gap_a:gap_b] - truth[gap_a:gap_b]) / TWOPI
        assert err.max() < 0.25, f"max {err.max():.2f} cycles off across the gap"
        assert added[:, gap_a:gap_b].all()

    def test_rejects_bad_arguments(self):
        igram = np.ones((4, 4), np.complex64)
        mask = np.ones((4, 4), bool)
        with pytest.raises(ValueError, match="must match"):
            ww.connect_gaps(igram, np.ones((5, 5), bool))
        with pytest.raises(ValueError, match="max_gap"):
            ww.connect_gaps(igram, mask, max_gap=0)
        with pytest.raises(ValueError, match="edge"):
            ww.connect_gaps(igram, mask, edge=2)


class TestUnwrapConnectGaps:
    def test_levels_subswaths_that_bridging_gets_wrong(self):
        """The motivating case, with the baseline contrast that gives it teeth."""
        igram, corr, mask, truth, nlooks = striped_scene()

        off, _ = ww.unwrap(igram, corr, nlooks, mask)
        on, _ = ww.unwrap(igram, corr, nlooks, mask, connect_gaps=True)

        score_off = on_cycle_fraction(off, truth, mask)
        score_on = on_cycle_fraction(on, truth, mask)
        # Guard: if the baseline ever starts passing this scene, the test has
        # stopped exercising the feature and the scene needs a steeper ramp.
        assert score_off < 0.9, f"baseline unexpectedly fine ({score_off:.2f})"
        assert score_on > 0.99, f"connect_gaps left {1 - score_on:.1%} off-cycle"

    def test_synthesised_pixels_are_not_returned(self):
        igram, corr, mask, _truth, nlooks = striped_scene()
        _z, added = ww.connect_gaps(igram, mask)
        assert added.any(), "scene must actually have crossable gaps"
        unw, cc = ww.unwrap(igram, corr, nlooks, mask, connect_gaps=True)
        assert unw.shape == mask.shape
        # The synthesised pixels were scaffolding; they carry no measurement, so
        # they must come back empty in both outputs rather than as plausible
        # phase a caller could mistake for data.
        assert np.all(np.asarray(unw)[added] == 0)
        assert np.all(np.asarray(cc)[added] == 0)

    def test_noop_without_gaps(self):
        """A frame with no interior gaps must be untouched."""
        m = n = 96
        ii, jj = np.mgrid[0:m, 0:n]
        truth = (TWOPI * 2 * (ii + jj) / n).astype(np.float32)
        igram, corr = ww.simulate_ifg(truth, np.full((m, n), 0.9, np.float32), 20, 5)
        mask = np.ones((m, n), dtype=bool)
        off, _ = ww.unwrap(igram, corr, 20.0, mask)
        on, _ = ww.unwrap(igram, corr, 20.0, mask, connect_gaps=True)
        np.testing.assert_array_equal(np.asarray(off), np.asarray(on))

    def test_wide_gaps_left_to_bridging(self):
        """Below max_gap the pass acts; above it the frame is untouched."""
        igram, corr, mask, truth, nlooks = striped_scene(gap=10)
        narrow, _ = ww.unwrap(igram, corr, nlooks, mask, connect_gaps=True)
        wide, _ = ww.unwrap(
            igram, corr, nlooks, mask, connect_gaps=True, connect_gaps_max_px=4
        )
        plain, _ = ww.unwrap(igram, corr, nlooks, mask)
        assert on_cycle_fraction(narrow, truth, mask) > 0.99
        np.testing.assert_array_equal(np.asarray(wide), np.asarray(plain))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"remove_ramp": True},
            {"goldstein_alpha": 0.5},
            {"interpolate": True, "interp_cutoff": 0.3},
            {"remove_ramp": True, "interpolate": True, "interp_cutoff": 0.3},
        ],
    )
    def test_composes_with_other_prepasses(self, kwargs):
        igram, corr, mask, truth, nlooks = striped_scene()
        unw, _ = ww.unwrap(igram, corr, nlooks, mask, connect_gaps=True, **kwargs)
        assert on_cycle_fraction(unw, truth, mask) > 0.99, kwargs

    def test_composes_with_a_steep_ramp_and_a_curved_surface(self):
        """Regression: the synthesised path must live in the SOLVER's domain.

        ``remove_ramp`` de-ramps the phase the solver sees. Restoring the
        synthesised pixels from the original igram after that put them a full
        fitted-ramp offset away from their neighbours, tearing the frame at
        every gap edge -- and it did so worst on steep ramps, which is exactly
        when someone reaches for ``remove_ramp``. A mild ramp hides this, so
        this scene uses a steep one plus curvature no plane can absorb.
        """
        igram, corr, mask, truth, nlooks = striped_scene(
            m=384, n=384, n_stripes=7, gap=16, ramp_cycles=40.0, curvature_cycles=6.0
        )
        alone, _ = ww.unwrap(igram, corr, nlooks, mask, connect_gaps=True)
        both, _ = ww.unwrap(
            igram, corr, nlooks, mask, connect_gaps=True, remove_ramp=True
        )
        s_alone = on_cycle_fraction(alone, truth, mask)
        s_both = on_cycle_fraction(both, truth, mask)
        assert s_alone > 0.99, f"connect_gaps alone regressed ({s_alone:.3f})"
        # The bug drove this to ~0.14 (one stripe of seven) while `alone` stayed
        # at 1.0, so adding a pre-pass made the result far worse than omitting
        # it. Composition must not cost anything here.
        assert s_both > 0.99, f"remove_ramp broke the connection ({s_both:.3f})"

    def test_ramp_is_fitted_to_measured_pixels_only(self, monkeypatch):
        """Synthesised phase must not feed back into the plane fit.

        Asserted on the mask `fit_ramp` actually receives, because the effect on
        the fitted slopes is far too small to detect from the output.
        """
        igram, corr, mask, _truth, nlooks = striped_scene(
            n_stripes=7, gap=16, ramp_cycles=20.0
        )
        _joined, added = ww.connect_gaps(igram, mask, max_gap=300)
        assert added.any(), "scene must actually have crossable gaps"

        seen: list[np.ndarray] = []
        real_fit_ramp = ww.fit_ramp

        def spy(ig, mask=None):
            seen.append(np.array(mask, copy=True))
            return real_fit_ramp(ig, mask=mask)

        monkeypatch.setattr(ww, "fit_ramp", spy)
        ww.unwrap(igram, corr, nlooks, mask, connect_gaps=True, remove_ramp=True)

        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0], mask)
        assert not (seen[0] & added).any(), "plane was fitted to invented phase"
