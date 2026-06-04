"""Run isce2's mroipac ICU unwrapper (Giangi's C extension, NOT isce3's ICU) on
one NISAR GUNW frame and score per-connected-component ambiguity match vs the
production GUNW unwrap -- the SAME metric tophu_compare.percomp_match uses, so
ICU is judged on the exact footing as the whirlwind/ww-orig/PHASS sweep numbers.

Self-contained for the test-isce2 env (needs isce2 + numpy + h5py only). ICU
estimates its own correlation (PHASESIGMA) from the phase, so we feed it only the
unit-magnitude wrapped interferogram (no amplitude, no filtering).

Usage (must be the isce2 env, which injects mroipac onto sys.path via `import isce`):
    /Users/staniewi/miniforge3/envs/test-isce2/bin/python scripts/icu_isce2_run.py A_013
"""
import sys
import glob
import time

import numpy as np
import h5py

import isce  # noqa: F401  -- injects mroipac/isceobj onto sys.path
import isceobj
from mroipac.icu.Icu import Icu

TWOPI = 2.0 * np.pi
wrap = lambda x: (x + np.pi) % TWOPI - np.pi
SCRATCH = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/icu_scratch"

frame = sys.argv[1] if len(sys.argv) > 1 else "A_013"
h5path = glob.glob(f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5")[0]
base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
with h5py.File(h5path, "r") as h:
    grp = h[base]
    pol = sorted(k for k, v in grp.items() if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"})[0]
    prod_unw = h[f"{base}/{pol}/unwrappedPhase"][()].astype(np.float32)
    coh = h[f"{base}/{pol}/coherenceMagnitude"][()].astype(np.float32)
    prod_cc = h[f"{base}/{pol}/connectedComponents"][()].astype(np.int64)
    mask_arr = h[f"{base}/mask"][()] if "mask" in grp else None

mask = (mask_arr != 255) & ((mask_arr // 100) % 10 == 0) if mask_arr is not None else np.ones(prod_unw.shape, bool)
mask &= np.isfinite(prod_unw) & np.isfinite(coh)
# ICU estimates coherence internally from local phase variance (PHASESIGMA), so
# we cannot hand it an external mask. Zeroing masked/water phase makes a giant
# CONSTANT (=perfectly coherent) region that ICU seeds + references the whole
# frame to -> corrupts the land. Fill masked pixels with RANDOM phase instead so
# ICU reads them as decorrelated and skips them.
rng = np.random.default_rng(0)
wrapped = np.where(mask, wrap(prod_unw), rng.uniform(-np.pi, np.pi, prod_unw.shape)).astype(np.float32)
igram = np.exp(1j * wrapped).astype(np.complex64)
length, width = igram.shape
print(f"{frame}: shape=({length},{width}) valid={mask.mean()*100:.1f}%", flush=True)

import os
os.makedirs(SCRATCH, exist_ok=True)
int_file = f"{SCRATCH}/{frame}.int"
unw_file = f"{SCRATCH}/{frame}.unw"
igram.tofile(int_file)  # raw single-band CFLOAT, BIP (real,imag interleaved)

# Input interferogram image. isce2's GDAL accessor reads via a .vrt/.xml
# sidecar, so render the header from the metadata before opening for read.
objInt = isceobj.createIntImage()
objInt.initImage(int_file, "read", width)
objInt.setLength(length)
objInt.renderHdr()  # writes <int_file>.xml + .vrt describing the raw CFLOAT
objInt.createImage()

# Output unwrapped image: 2-band BIL (band0=amplitude, band1=unwrapped phase).
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
icuObj.filteringFlag = False        # no PS/LP prefilter -- unwrap raw wrapped phase
icuObj.useAmplitudeFlag = False     # no amplitude provided/needed
icuObj.singlePatch = True           # whole frame in one patch (no azimuth tiling)
icuObj.unwrappingFlag = True

t0 = time.perf_counter()
icuObj.icu(intImage=objInt, unwImage=objUnw)
dt = time.perf_counter() - t0

objInt.finalizeImage()
objUnw.finalizeImage()

# Read back: BIL 2-band -> (length, 2, width); band 1 = unwrapped phase.
raw = np.fromfile(unw_file, dtype=np.float32).reshape(length, 2, width)
icu_amp = raw[:, 0, :]
icu_unw = raw[:, 1, :]
# band0 (amp) is ICU's connected flag (=1.0 where it grew a component, 0 else).
icu_done = icu_amp != 0.0
cov = (mask & icu_done).sum() / max(mask.sum(), 1)


def percomp_match(test_unw, valid):
    amb = np.rint((test_unw - wrapped) / TWOPI) - np.rint((prod_unw - wrapped) / TWOPI)
    in_comp = valid & (prod_cc > 0)
    if not in_comp.any():
        return float("nan")
    off = np.zeros(amb.shape, np.float64)
    for lab in np.unique(prod_cc[in_comp]):
        m = valid & (prod_cc == lab)
        off[m] = np.rint(np.median(amb[m]))
    return float(np.mean((amb - off)[in_comp] == 0))


valid_strict = mask & np.isfinite(icu_unw)
valid_done = valid_strict & icu_done
pc_strict = percomp_match(icu_unw, valid_strict)
pc_done = percomp_match(icu_unw, valid_done)
print(f"{frame}: icu(isce2) {dt:6.1f}s  per-comp(strict mask)={pc_strict*100:5.1f}%  "
      f"per-comp(ICU-connected only)={pc_done*100:5.1f}%  coverage={cov*100:4.1f}%", flush=True)
# Sweep-compatible line: score ICU like the other engines (per-comp over the
# pixels it actually connected — gaps excluded, as for PHASS), + coverage note.
print(f"{frame}: icu        {dt:6.1f}s  per-comp-match-vs-prod={pc_done*100:5.1f}%  "
      f"ncc=0  shape={prod_unw.shape}  coverage={cov*100:.0f}%", flush=True)
