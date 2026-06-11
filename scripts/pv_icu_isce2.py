#!/usr/bin/env python
"""Run isce2's mroipac ICU (Giangi's C extension, NOT isce3's) over the Palos
Verdes Capella interferogram stack, writing dolphin-convention outputs so the
result slots into the same timeseries inversion as the other engines.

ICU estimates its own correlation (PHASESIGMA) from local phase variance, so it
gets only the unit-magnitude wrapped interferogram. Pixels where the input
interferogram is exactly 0 (nodata) are filled with RANDOM phase -- a constant
region reads as perfectly coherent and corrupts the unwrap (see
icu_isce2_run.py for the NISAR version of this lesson).

Outputs per pair: <pair>.unw.tif (float32 radians, 0 where ICU gave up) and
<pair>.unw.conncomp.tif (uint16 labels from ICU's own .conncomp file).

Must run in the test-isce2 env (imports isce2 + gdal):
    /Users/staniewi/miniforge3/envs/test-isce2/bin/python scripts/pv_icu_isce2.py [--limit 1]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from osgeo import gdal

import isce  # noqa: F401  -- injects mroipac/isceobj onto sys.path
import isceobj
from mroipac.icu.Icu import Icu

gdal.UseExceptions()

BASE = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes/"
    "Palos_Verdes_C13_RO23_SP/e2e_output_20260519"
)
IFG_DIR = BASE / "dolphin/interferograms"
OUT_DIR = BASE / "unwrap_compare/icu_isce2/unwrapped"
GTIFF_OPTS = ["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES", "BIGTIFF=YES"]


def write_gtiff(
    path: Path, arr: np.ndarray, like: gdal.Dataset, dtype, nodata: float | None = None
) -> None:
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), arr.shape[1], arr.shape[0], 1, dtype, GTIFF_OPTS)
    ds.SetGeoTransform(like.GetGeoTransform())
    ds.SetProjection(like.GetProjection())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(arr)
    ds = None


def run_one(ifg_path: Path, scratch: Path) -> tuple[float, float]:
    src = gdal.Open(str(ifg_path))
    ifg = src.GetRasterBand(1).ReadAsArray()
    length, width = ifg.shape

    nodata = ifg == 0
    rng = np.random.default_rng(0)
    phase = np.where(nodata, rng.uniform(-np.pi, np.pi, ifg.shape), np.angle(ifg))
    igram = np.exp(1j * phase).astype(np.complex64)

    int_file = str(scratch / "pair.int")
    unw_file = str(scratch / "pair.unw")
    for f in (int_file, unw_file, int_file + ".conncomp"):
        if os.path.exists(f):
            os.remove(f)
    igram.tofile(int_file)  # raw single-band CFLOAT, BIP

    objInt = isceobj.createIntImage()
    objInt.initImage(int_file, "read", width)
    objInt.setLength(length)
    objInt.renderHdr()
    objInt.createImage()

    objUnw = isceobj.createImage()
    objUnw.setFilename(unw_file)
    objUnw.setWidth(width)
    objUnw.dataType = "FLOAT"
    objUnw.bands = 2
    objUnw.scheme = "BIL"
    objUnw.imageType = "unw"
    objUnw.setAccessMode("write")
    objUnw.createImage()

    icuObj = Icu()
    icuObj.filteringFlag = False
    icuObj.useAmplitudeFlag = False
    icuObj.singlePatch = True
    icuObj.unwrappingFlag = True

    t0 = time.perf_counter()
    icuObj.icu(intImage=objInt, unwImage=objUnw)
    dt = time.perf_counter() - t0

    objInt.finalizeImage()
    objUnw.finalizeImage()

    raw = np.fromfile(unw_file, dtype=np.float32).reshape(length, 2, width)
    icu_done = raw[:, 0, :] != 0.0
    unw = np.where(icu_done, raw[:, 1, :], 0.0).astype(np.float32)

    # ICU writes its own connected-component labels next to the input .int
    cc_file = icuObj.conncompFilename
    assert cc_file and os.path.exists(cc_file), f"no conncomp output at {cc_file!r}"
    cc = np.fromfile(cc_file, dtype=np.uint8).reshape(length, width).astype(np.uint16)

    pair = ifg_path.name.split(".")[0]
    write_gtiff(OUT_DIR / f"{pair}.unw.tif", unw, src, gdal.GDT_Float32, nodata=0)
    write_gtiff(OUT_DIR / f"{pair}.unw.conncomp.tif", cc, src, gdal.GDT_UInt16, nodata=65535)
    cov = (icu_done & ~nodata).sum() / max((~nodata).sum(), 1)
    return dt, cov


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    ifgs = sorted(IFG_DIR.glob("2*.int.tif"))
    assert len(ifgs) == 150, f"expected 150 ifgs, found {len(ifgs)}"
    if args.limit:
        ifgs = ifgs[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scratch = OUT_DIR / "scratch"
    scratch.mkdir(exist_ok=True)

    t_all = time.perf_counter()
    for i, f in enumerate(ifgs):
        pair = f.name.split(".")[0]
        if (OUT_DIR / f"{pair}.unw.tif").exists():
            print(f"[{i + 1}/{len(ifgs)}] {pair} exists, skipping", flush=True)
            continue
        dt, cov = run_one(f, scratch)
        print(
            f"[{i + 1}/{len(ifgs)}] {pair} icu(isce2) {dt:6.1f}s coverage={cov * 100:4.1f}%",
            flush=True,
        )
    print(f"DONE {len(ifgs)} pairs in {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
