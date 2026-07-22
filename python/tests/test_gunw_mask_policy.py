"""The GUNW ``mask`` policies in ``aws-batch/compare_gunw.py``.

Pure numpy, no product download -- unlike ``test_nisar_gunw_integration.py``,
these run in CI.

The NISAR GUNW ``mask`` layer packs two independent exclusions into the low
byte of a 3-digit decimal code ``[water][subswath_ref][subswath_sec]``: water,
and samples invalid in either RSLC. The four policies are every combination of
applying them, so they are testable exhaustively against the handful of codes
that actually occur.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("matplotlib")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "aws-batch"))
import compare_gunw as cg  # noqa: E402

# Every code that occurs in practice: [water][ref_subswath][sec_subswath],
# plus the 255 _FillValue. A 0 in a subswath digit means an invalid sample in
# that RSLC; a 1 in the water digit means water.
CODES = np.array([[0, 1, 10, 11, 12, 100, 101, 110, 111, 112, 255]], dtype=np.uint32)
SHAPE = CODES.shape


def _valid(policy: str) -> np.ndarray:
    return cg.mask_to_bool(CODES, policy, SHAPE).ravel()


def test_subswath_keeps_water_but_drops_invalid_samples():
    """Default policy: both subswath digits nonzero, water irrelevant."""
    # Keeps 11, 12 (valid, land) and 111, 112 (valid, water). Drops every code
    # with a 0 subswath digit, and the fill value.
    expected = [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0]
    assert _valid("subswath").tolist() == expected


def test_water_only_keeps_invalid_samples():
    expected = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert _valid("water_only").tolist() == expected


def test_water_and_subswath_is_the_conjunction():
    """The combined policy must be exactly both single-axis policies ANDed.

    This is the invariant the name promises. If someone edits one branch, this
    catches the drift.
    """
    combined = _valid("water_and_subswath")
    assert combined.tolist() == (_valid("subswath") & _valid("water_only")).tolist()
    assert combined.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]


def test_ignore_keeps_everything_including_fill():
    assert _valid("ignore").all()


def test_water_flag_is_independent_of_subswath_validity():
    expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    assert cg.gunw_water_mask(CODES, SHAPE).ravel().tolist() == expected


@pytest.mark.parametrize("policy", ["subswath", "water_only", "water_and_subswath"])
def test_fill_value_is_never_valid(policy: str):
    """255 is the _FillValue, not the code 2/5/5."""
    assert not _valid(policy)[-1]


@pytest.mark.parametrize(
    "policy", ["subswath", "water_only", "water_and_subswath", "ignore"]
)
def test_high_bits_do_not_leak_into_the_decimal_code(policy: str):
    """Newer products widened ``mask`` to uint32 and pack anomaly flags into
    bits 8-23 and the ionosphere-fill flag into bit 24. Those must not reach
    the ``// 100`` water digit -- bit 24 alone is 16777216, which would read as
    a water code and silently drop most of a frame."""
    with_flags = CODES | (1 << 24) | (1 << 16) | (1 << 8)
    assert (
        cg.mask_to_bool(with_flags, policy, SHAPE)
        == cg.mask_to_bool(CODES, policy, SHAPE)
    ).all()


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="Unknown mask policy"):
        cg.mask_to_bool(CODES, "nisar_land", SHAPE)


def test_none_mask_is_all_valid():
    assert cg.mask_to_bool(None, "subswath", SHAPE).all()
    assert not cg.gunw_water_mask(None, SHAPE).any()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Mask shape"):
        cg.mask_to_bool(CODES, "subswath", (5, 5))
