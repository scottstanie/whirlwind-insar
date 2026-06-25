"""Opt-in integration test: whirlwind vs a real NISAR L2 GUNW product.

This is **skipped by default** (CI does not download hundreds of MB). To run it,
point it at one or more real GUNW ``.h5`` files:

    # one file
    WHIRLWIND_TEST_GUNW=/path/NISAR_..._001.h5 \
        uv run python -m pytest python/tests/test_nisar_gunw_integration.py -v

    # a directory of files
    WHIRLWIND_TEST_GUNW_DIR=/path/to/gunws \
        uv run python -m pytest python/tests/test_nisar_gunw_integration.py -v

Download a product first with ``aws-batch/compare_gunw.py`` (it accepts a granule
name or ASF URL), then pass the resulting ``.h5`` here.

The canonical sample granules (also in ``aws-batch/sample_granules.txt``) are
listed in ``SAMPLE_GRANULES`` below. ``A_140_cryo`` is a fringe-rich cryosphere
scene -- a good unwrap stress test.

Tunable via env: ``WW_GUNW_CROP`` (center-crop px, default 1024; ``full`` for the
whole frame) and ``WW_GUNW_MATCH_FLOOR`` (per-component ambiguity-match floor,
default 0.7).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

import whirlwind as ww


def _test_files() -> list[Path]:
    files: list[Path] = []
    if one := os.environ.get("WHIRLWIND_TEST_GUNW"):
        files.append(Path(one))
    if d := os.environ.get("WHIRLWIND_TEST_GUNW_DIR"):
        files.extend(sorted(Path(d).glob("*.h5")))
    return [f for f in files if f.exists()]


_FILES = _test_files()

# Opt-in: skip the whole module (before importing h5py/matplotlib, which the
# default test environment does not install) unless GUNW files are configured.
if not _FILES:
    pytest.skip(
        "Set WHIRLWIND_TEST_GUNW=<.h5> or WHIRLWIND_TEST_GUNW_DIR=<dir> to run "
        "the NISAR GUNW integration test (no downloads in CI).",
        allow_module_level=True,
    )

import h5py  # noqa: E402

# Reuse the exact comparison helpers the AWS-Batch tool uses, so this test and
# the benchmark stay in lockstep.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "aws-batch"))
import compare_gunw as cg  # noqa: E402

# ASF download URLs for the validated sample set. Resolve with earthaccess/CMR
# or fetch directly; see aws-batch/sample_granules.txt.
_BASE = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_BETA_V1"
SAMPLE_GRANULES = {
    "D_071": f"{_BASE}/NISAR_L2_PR_GUNW_009_156_D_071_010_4000_SH_20260108T004405_20260108T004440_20260120T004406_20260120T004440_X05010_N_F_J_001/NISAR_L2_PR_GUNW_009_156_D_071_010_4000_SH_20260108T004405_20260108T004440_20260120T004406_20260120T004440_X05010_N_F_J_001.h5",
    "A_019": f"{_BASE}/NISAR_L2_PR_GUNW_009_148_A_019_010_4000_SH_20260107T105515_20260107T105550_20260119T105516_20260119T105551_X05010_N_F_J_001/NISAR_L2_PR_GUNW_009_148_A_019_010_4000_SH_20260107T105515_20260107T105550_20260119T105516_20260119T105551_X05010_N_F_J_001.h5",
    "A_140_cryo": f"{_BASE}/NISAR_L2_PR_GUNW_009_163_A_140_010_7700_SH_20260108T130215_20260108T130251_20260120T130216_20260120T130252_X05010_N_P_J_001/NISAR_L2_PR_GUNW_009_163_A_140_010_7700_SH_20260108T130215_20260108T130251_20260120T130216_20260120T130252_X05010_N_P_J_001.h5",
}


@pytest.mark.parametrize("h5_path", _FILES, ids=lambda p: p.name[:48])
def test_gunw_unwrap_agrees_with_production(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        paths = cg.gunw_paths(f, None)
        prod_unw = cg.read_array(f[paths["unw"]], np.float32)
        coh = cg.read_array(f[paths["coh_unw"]], np.float32)
        prod_cc = f[paths["cc"]][()].astype(np.int64, copy=False)
        mask_arr = f[paths["mask"]][()] if paths["mask"] in f else None

    crop_env = os.environ.get("WW_GUNW_CROP", "1024")
    size = "full" if crop_env == "full" else min(int(crop_env), *prod_unw.shape)
    ys, xs, _ = cg.center_crop_slices(prod_unw.shape, size)

    base_mask = cg.mask_to_bool(mask_arr, "water_only", prod_unw.shape)
    base_mask &= np.isfinite(prod_unw) & np.isfinite(coh)
    ig_full = cg.wrap_phase(prod_unw).astype(np.float32)

    ig = np.ascontiguousarray(ig_full[ys, xs])
    coh_c = np.ascontiguousarray(coh[ys, xs])
    prod_unw_c = np.ascontiguousarray(prod_unw[ys, xs])
    prod_cc_c = np.ascontiguousarray(prod_cc[ys, xs])
    mask = np.ascontiguousarray(base_mask[ys, xs])
    assert mask.sum() > 0, "crop has no valid pixels"

    ig_solver = np.where(mask, ig, 0.0).astype(np.float32)
    coh_solver = np.where(mask, np.clip(np.nan_to_num(coh_c), 0.0, 1.0), 0.0).astype(
        np.float32
    )
    igram = np.exp(1j * ig_solver).astype(np.complex64)

    unw, cc = ww.unwrap(igram, coh_solver, 16.0, mask)
    unw = np.asarray(unw, dtype=np.float32)
    cc = np.asarray(cc)

    # whirlwind must return finite phase and at least one connected component.
    assert np.isfinite(unw[mask]).all()
    assert cc.shape == unw.shape
    assert int(cc.max()) >= 1

    stats, *_ = cg.compute_compare_stats(
        ig=ig,
        coh=coh_c,
        mask=mask,
        prod_unw=prod_unw_c,
        prod_cc=prod_cc_c,
        ww_unw=unw,
        ww_cc=cc,
        runtime_s=0.0,
        rss_delta_mb=None,
    )
    floor = float(os.environ.get("WW_GUNW_MATCH_FLOOR", "0.7"))
    match = stats["ambiguity_match_frac_percomp"]
    assert match == pytest.approx(match, nan_ok=False), "no production components"
    assert match > floor, (
        f"{h5_path.name}: per-component ambiguity match {match:.3f} below floor "
        f"{floor:.2f} (residual_wrapped_rmse={stats['residual_wrapped_rmse_rad']:.3f} rad)"
    )
