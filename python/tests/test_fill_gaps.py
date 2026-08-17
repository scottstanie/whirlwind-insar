"""Tests for the ``fill_gaps`` pre-pass.

The point of filling is to make a gapped frame connected so the solver picks the
per-region 2*pi levels itself. So the tests assert the thing that actually
matters -- the regions come out on the right relative level -- and each one
checks the baseline FAILS the same case, because an assertion that passes with
the feature switched off is not testing the feature.
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
    coherence: float = 0.8,
    nlooks: int = 40,
    seed: int = 3,
):
    """A ramped scene cut into subswath stripes, plus its truth and mask."""
    ii, jj = np.mgrid[0:m, 0:n]
    truth = (TWOPI * ramp_cycles * jj / (n - 1)).astype(np.float32)
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


class TestFillGapsHelper:
    def test_fills_only_interior_gaps(self):
        mask = np.ones((8, 80), dtype=bool)
        mask[:, :6] = False  # frame edge: only one side, must NOT be filled
        mask[:, 40:44] = False  # interior gap: must be filled
        igram = np.exp(1j * np.zeros((8, 80))).astype(np.complex64)
        igram[~mask] = 0
        _out, filled = ww.fill_gaps(igram, mask, edge=8)
        assert filled[:, 40:44].all()
        assert not filled[:, :6].any()
        assert not filled[mask].any()

    def test_skips_gaps_without_enough_data_to_extrapolate_from(self):
        """A sliver of a region cannot define a fringe rate, so leave the gap."""
        mask = np.ones((8, 40), dtype=bool)
        mask[:, :30] = False  # leaves a 4 px sliver before the gap
        mask[:, 34:36] = False
        igram = np.exp(1j * np.zeros((8, 40))).astype(np.complex64)
        igram[~mask] = 0
        _out, filled = ww.fill_gaps(igram, mask, edge=8)
        assert not filled.any()

    def test_respects_max_gap(self):
        mask = np.ones((8, 40), dtype=bool)
        mask[:, 18:30] = False  # 12 px wide
        igram = np.exp(1j * np.zeros((8, 40))).astype(np.complex64)
        igram[~mask] = 0
        _o, wide = ww.fill_gaps(igram, mask, max_gap=6, edge=8)
        assert not wide.any(), "gap wider than max_gap must be left alone"
        _o, narrow = ww.fill_gaps(igram, mask, max_gap=20, edge=8)
        assert narrow[:, 18:30].all()

    def test_carries_the_fringe_rate_across(self):
        """The fill continues the fringes; it does not flatten them."""
        m, n, gap_a, gap_b = 32, 120, 55, 65
        cycles = 4.0
        truth = TWOPI * cycles * np.arange(n) / (n - 1)
        igram = np.exp(1j * np.tile(truth, (m, 1))).astype(np.complex64)
        mask = np.ones((m, n), dtype=bool)
        mask[:, gap_a:gap_b] = False
        igram[~mask] = 0
        out, filled = ww.fill_gaps(igram, mask, edge=32)
        got = np.unwrap(np.angle(out[m // 2]))
        got -= got[0] - truth[0]
        # Inside the gap the fill should track the true ramp, not short-circuit
        # it: a flattening fill would be off by the phase the gap spans.
        err = np.abs(got[gap_a:gap_b] - truth[gap_a:gap_b]) / TWOPI
        assert err.max() < 0.25, f"max {err.max():.2f} cycles off across the gap"
        assert filled[:, gap_a:gap_b].all()

    def test_rejects_bad_arguments(self):
        igram = np.ones((4, 4), np.complex64)
        mask = np.ones((4, 4), bool)
        with pytest.raises(ValueError, match="must match"):
            ww.fill_gaps(igram, np.ones((5, 5), bool))
        with pytest.raises(ValueError, match="max_gap"):
            ww.fill_gaps(igram, mask, max_gap=0)
        with pytest.raises(ValueError, match="edge"):
            ww.fill_gaps(igram, mask, edge=2)


class TestUnwrapFillGaps:
    def test_levels_subswaths_that_bridging_gets_wrong(self):
        """The motivating case, with the baseline contrast that gives it teeth."""
        igram, corr, mask, truth, nlooks = striped_scene()

        off, _ = ww.unwrap(igram, corr, nlooks, mask)
        on, _ = ww.unwrap(igram, corr, nlooks, mask, fill_gaps=True)

        score_off = on_cycle_fraction(off, truth, mask)
        score_on = on_cycle_fraction(on, truth, mask)
        # Guard: if the baseline ever starts passing this scene, the test has
        # stopped exercising the feature and the scene needs a steeper ramp.
        assert score_off < 0.9, f"baseline unexpectedly fine ({score_off:.2f})"
        assert score_on > 0.99, f"fill_gaps left {1 - score_on:.1%} off-cycle"

    def test_filled_pixels_are_not_returned(self):
        igram, corr, mask, _truth, nlooks = striped_scene()
        _z, filled = ww.fill_gaps(igram, mask)
        assert filled.any(), "scene must actually have fillable gaps"
        unw, cc = ww.unwrap(igram, corr, nlooks, mask, fill_gaps=True)
        assert unw.shape == mask.shape
        # The synthesised pixels were scaffolding; they carry no measurement, so
        # they must come back empty in both outputs rather than as plausible
        # phase a caller could mistake for data.
        assert np.all(np.asarray(unw)[filled] == 0)
        assert np.all(np.asarray(cc)[filled] == 0)

    def test_noop_without_gaps(self):
        """A frame with no interior gaps must be untouched."""
        m = n = 96
        ii, jj = np.mgrid[0:m, 0:n]
        truth = (TWOPI * 2 * (ii + jj) / n).astype(np.float32)
        igram, corr = ww.simulate_ifg(truth, np.full((m, n), 0.9, np.float32), 20, 5)
        mask = np.ones((m, n), dtype=bool)
        off, _ = ww.unwrap(igram, corr, 20.0, mask)
        on, _ = ww.unwrap(igram, corr, 20.0, mask, fill_gaps=True)
        np.testing.assert_array_equal(np.asarray(off), np.asarray(on))

    def test_wide_gaps_left_to_bridging(self):
        """Below max_gap the fill acts; above it the frame is untouched."""
        igram, corr, mask, truth, nlooks = striped_scene(gap=10)
        narrow, _ = ww.unwrap(igram, corr, nlooks, mask, fill_gaps=True)
        wide, _ = ww.unwrap(
            igram, corr, nlooks, mask, fill_gaps=True, fill_gaps_max_px=4
        )
        plain, _ = ww.unwrap(igram, corr, nlooks, mask)
        assert on_cycle_fraction(narrow, truth, mask) > 0.99
        np.testing.assert_array_equal(np.asarray(wide), np.asarray(plain))

    def test_composes_with_other_prepasses(self):
        igram, corr, mask, truth, nlooks = striped_scene()
        for kwargs in (
            {"remove_ramp": True},
            {"goldstein_alpha": 0.5},
            {"interpolate": True, "interp_cutoff": 0.3},
        ):
            unw, _ = ww.unwrap(igram, corr, nlooks, mask, fill_gaps=True, **kwargs)
            assert on_cycle_fraction(unw, truth, mask) > 0.99, kwargs
