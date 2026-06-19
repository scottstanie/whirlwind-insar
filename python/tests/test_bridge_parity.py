"""Bit-for-bit parity: native ``whirlwind.bridge_components`` vs the frozen
numpy oracle (``bridge_reference.py``).

The Rust implementation (``crates/whirlwind-core/src/bridge.rs``) is canonical;
the oracle is the retired numpy original it was ported from. Every quantity
feeding the cycle rounding was matched deliberately (label order, boundary
subsampling stride, argmin tie order, f32 medians, round-half-even ``rint``,
the single f64->f32 cast of the 2π·k shift), so the comparison is exact
equality - any drift is a behaviour change, not float noise.

All scenes use explicit ``np.random.default_rng`` seeds, so pytest-randomly
cannot perturb them.
"""

import numpy as np
import pytest

import whirlwind as ww
from bridge_reference import bridge_components_reference


def make_carved_scene(seed: int, nodata: float):
    """83x97 frame carved into a grid of regions by masked gap rows/columns.

    Smooth ramp + noise, with a per-region integer-cycle gauge offset (the
    raster-first region is left alone) and NaN speckles inside the valid area.
    Returns ``(unw, mask)``.
    """
    rng = np.random.default_rng(seed)
    m, n = 83, 97
    yy, xx = np.mgrid[0:m, 0:n].astype(np.float32)
    unw = (0.02 * yy + 0.03 * xx + rng.normal(0.0, 0.25, (m, n))).astype(np.float32)

    mask = np.ones((m, n), dtype=bool)
    for x in rng.choice(np.arange(10, n - 10), size=2, replace=False):
        mask[:, x : x + int(rng.integers(1, 4))] = False
    for y in rng.choice(np.arange(10, m - 10), size=2, replace=False):
        mask[y : y + int(rng.integers(1, 4)), :] = False

    labels, n_lab = ww.label_components(mask)
    assert n_lab >= 4, "carving must produce a multi-region scene"
    for lab in range(2, n_lab + 1):
        cycles = np.float32(int(rng.integers(-3, 4)))
        unw[labels == lab] += np.float32(2 * np.pi) * cycles

    speckle = (rng.random((m, n)) < 0.002) & mask
    unw[speckle] = np.nan
    unw[~mask] = nodata
    return unw, mask


@pytest.mark.parametrize("seed", range(5))
def test_parity_explicit_mask(seed):
    unw, mask = make_carved_scene(seed, nodata=0.0)
    # Garbage under the mask must be ignored - and passed through - identically.
    rng = np.random.default_rng(1000 + seed)
    unw[~mask] = rng.normal(0.0, 100.0, int((~mask).sum())).astype(np.float32)

    got = np.asarray(ww.bridge_components(unw, mask, radius=20, min_px=30))
    want = bridge_components_reference(unw, mask, radius=20, min_px=30)
    assert got.dtype == np.float32
    assert np.array_equal(got, want, equal_nan=True)
    # Non-vacuity: a no-op bridge would make this parity check meaningless.
    assert not np.array_equal(got, unw, equal_nan=True)


@pytest.mark.parametrize("nodata", [0.0, np.nan], ids=["zero", "nan"])
@pytest.mark.parametrize("seed", range(5))
def test_parity_default_mask(seed, nodata):
    # mask=None: regions come from the finite, nonzero pixels of unw itself.
    unw, _ = make_carved_scene(seed, nodata=nodata)
    got = np.asarray(ww.bridge_components(unw, radius=20, min_px=30))
    want = bridge_components_reference(unw, radius=20, min_px=30)
    assert np.array_equal(got, want, equal_nan=True)


@pytest.mark.parametrize("seed", range(3))
def test_parity_strided_boundary_subsample(seed):
    # max_boundary far below the true boundary count forces the stride
    # subsample path on both sides.
    unw, mask = make_carved_scene(seed, nodata=0.0)
    got = np.asarray(
        ww.bridge_components(unw, mask, radius=20, min_px=30, max_boundary=17)
    )
    want = bridge_components_reference(unw, mask, radius=20, min_px=30, max_boundary=17)
    assert np.array_equal(got, want, equal_nan=True)


@pytest.mark.parametrize("seed", range(3))
def test_parity_default_params(seed):
    # No kwargs: the native defaults are single-sourced from the Rust consts
    # and must equal the values the reference froze (500 / 500 / 2000). The
    # 83x97 frame engages both the min_px=500 region filter and the
    # scene-relative radius clamp.
    unw, mask = make_carved_scene(100 + seed, nodata=0.0)
    got = np.asarray(ww.bridge_components(unw, mask))
    want = bridge_components_reference(unw, mask)
    assert np.array_equal(got, want, equal_nan=True)


def test_parity_single_region_noop():
    rng = np.random.default_rng(7)
    unw = rng.normal(0.0, 1.0, (40, 50)).astype(np.float32)
    unw[unw == 0] = np.float32(0.5)  # keep every pixel in the default mask
    got = np.asarray(ww.bridge_components(unw))
    assert np.array_equal(got, unw)
    assert np.array_equal(got, bridge_components_reference(unw))


def test_float64_input_rejected():
    # dtype-strict contract: no silent casting, reject loudly.
    unw = np.zeros((8, 8), dtype=np.float64)
    with pytest.raises(TypeError):
        ww.bridge_components(unw)  # type: ignore[arg-type]
