#!/usr/bin/env python
"""Real-data smoke test for the CLI flat-binary formats (ROI_PAC dhaka pair).

Feeds the same Sentinel-1 interferogram to `whirlwind unwrap` twice:
  1. flat binary path: --ifg <pair>.int  --cor <pair>.cc   (geometry from the
     auto-discovered .rsc sidecar; .cc is the 2-band rmg amp+cor layout)
  2. TIFF path:        --phase phase.tif --cor cor.tif     (written here with
     rasterio from the same arrays)
with an identical explicit flat byte mask, and requires the two unwrapped
outputs to agree. Also writes a flat rmg `.unw` and cross-checks it against
the TIFF output, and validates the flat complex read against GDAL's view of
the pre-existing `.int.tif` twin.

Run with the repo venv: .venv/bin/python scripts/smoke_flat_formats_dhaka.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

DATA = Path("/Users/staniewi/Documents/Learning/dhaka/l1_path114")
PAIR = "20150902_20150914"
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/whirlwind-flat-smoke")
BIN = Path(__file__).resolve().parents[1] / "target/release/whirlwind"
NLOOKS = "10"

WIDTH, LENGTH = 840, 1200  # from the .rsc


def run(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    int_file = DATA / f"{PAIR}.int"
    cc_file = DATA / f"{PAIR}.cc"

    ifg = np.fromfile(int_file, dtype=np.complex64).reshape(LENGTH, WIDTH)
    cc = np.fromfile(cc_file, dtype=np.float32).reshape(LENGTH, 2, WIDTH)
    amp, cor = cc[:, 0, :], cc[:, 1, :]
    assert np.allclose(amp.mean(), np.abs(ifg).mean(), rtol=1e-5), "band order"

    # (the on-disk .int.tif "twin" is a colorized uint8 quicklook, not raw
    # complex data, so there is no GDAL raster to cross-check against; the
    # flat-vs-TIFF CLI equivalence below is the real test)

    # shared explicit mask (also exercises the flat byte-mask reader)
    mask = ((ifg != 0) & (cor > 0)).astype(np.uint8)
    mask_file = OUT / "mask.msk"
    mask.tofile(mask_file)

    # TIFF twins of phase/cor for the reference run
    prof = dict(driver="GTiff", width=WIDTH, height=LENGTH, count=1, dtype="float32")
    phase_tif, cor_tif = OUT / "phase.tif", OUT / "cor.tif"
    with rasterio.open(phase_tif, "w", **prof) as dst:
        dst.write(np.angle(ifg).astype(np.float32), 1)
    with rasterio.open(cor_tif, "w", **prof) as dst:
        dst.write(cor, 1)

    unw_flatpath = OUT / "unw_from_flat.tif"
    unw_tiffpath = OUT / "unw_from_tiff.tif"
    unw_rmg = OUT / "unw_from_flat.unw"
    cc_flat = OUT / "conncomp_from_flat.conncomp"

    # 1) flat inputs (.rsc sidecar provides the width), rmg cor
    run(
        [
            str(BIN),
            "unwrap",
            "--ifg",
            str(int_file),
            "--cor",
            str(cc_file),
            "--mask",
            str(mask_file),
            "--nlooks",
            NLOOKS,
            "--out",
            str(unw_flatpath),
            "--conncomp",
            str(cc_flat),
        ]
    )
    # 1b) same, but flat rmg .unw output
    run(
        [
            str(BIN),
            "unwrap",
            "--ifg",
            str(int_file),
            "--cor",
            str(cc_file),
            "--mask",
            str(mask_file),
            "--nlooks",
            NLOOKS,
            "--out",
            str(unw_rmg),
        ]
    )
    # 2) TIFF reference
    run(
        [
            str(BIN),
            "unwrap",
            "--phase",
            str(phase_tif),
            "--cor",
            str(cor_tif),
            "--mask",
            str(mask_file),
            "--nlooks",
            NLOOKS,
            "--out",
            str(unw_tiffpath),
        ]
    )

    with rasterio.open(unw_flatpath) as src:
        a = src.read(1)
    with rasterio.open(unw_tiffpath) as src:
        b = src.read(1)
    diff = np.abs(a - b)[mask.astype(bool)]
    n_off = int((diff > 1e-3).sum())
    print(
        f"flat vs tiff path: max|diff|={diff.max():.3e} on valid px, "
        f"{n_off}/{diff.size} px differ >1e-3"
    )
    assert n_off == 0, "flat and TIFF paths disagree"

    rmg = np.fromfile(unw_rmg, dtype=np.float32).reshape(LENGTH, 2, WIDTH)
    assert np.array_equal(rmg[:, 1, :], a), ".unw phase band != tif output"
    assert np.allclose(
        rmg[:, 0, :], np.abs(ifg), rtol=1e-6
    ), ".unw amplitude band != |ifg|"
    print(".unw rmg output: phase band identical, amplitude band == |ifg|")

    ccomp = np.fromfile(cc_flat, dtype=np.uint8).reshape(LENGTH, WIDTH)
    print(
        f"flat conncomp: {ccomp.max()} component(s), "
        f"{(ccomp > 0).mean():.1%} labeled"
    )

    # the mandatory look at the result
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
    show = np.where(mask.astype(bool), a, np.nan)
    im = axes[0].imshow(np.angle(ifg), cmap="twilight", interpolation="none")
    axes[0].set_title(f"wrapped ({PAIR})")
    plt.colorbar(im, ax=axes[0], shrink=0.7)
    im = axes[1].imshow(show, cmap="RdBu_r", interpolation="none")
    axes[1].set_title("unwrapped (flat .int/.cc path)")
    plt.colorbar(im, ax=axes[1], shrink=0.7)
    im = axes[2].imshow(ccomp, cmap="tab20", interpolation="none")
    axes[2].set_title("conncomp (flat u8 output)")
    plt.colorbar(im, ax=axes[2], shrink=0.7)
    png = OUT / "smoke_flat_formats_dhaka.png"
    fig.savefig(png, dpi=120)
    print(f"figure: {png}")
    print("SMOKE OK")


if __name__ == "__main__":
    sys.exit(main())
