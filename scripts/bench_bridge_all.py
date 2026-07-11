"""13-frame NISAR GUNW bench for the bridging post-pass: per-frame before/after
per-comp-match + how many pixels the bridge moved. One HEAVY unwrap per frame
(bridge off), then the cheap bridge post-pass applied on top -- so before/after
is exact without doubling the heavy solves. Sequential (one heavy unwrap at a
time). Confirms 005_A_025 improves and every other frame is byte-identical (0 moved).

Usage (base miniforge3 env): python scripts/bench_bridge_all.py
"""

import glob
import re
import time

import numpy as np
import h5py

import sys

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase, percomp_match

import whirlwind as ww

H5DIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw"
h5s = sorted(glob.glob(f"{H5DIR}/*.h5"))
print(
    f"{'frame':7s} {'base%':>7s} {'bridged%':>9s} {'Δ':>6s} {'moved_px':>9s} {'ncomp':>6s} {'unw_s':>6s} {'br_s':>5s}",
    flush=True,
)

rows = []
for h5path in h5s:
    frame = re.search(r"_([AD]_\d{3})_", h5path).group(1)
    with h5py.File(h5path, "r") as h:
        pol, prod_unw, coh, prod_cc, mask_arr = gunw_layers(h)
    mask = (
        water_only_mask(mask_arr, prod_unw.shape)
        & np.isfinite(prod_unw)
        & np.isfinite(coh)
    )
    wrapped = np.where(mask, wrap_phase(prod_unw), 0.0).astype(np.float32)
    ig = np.exp(1j * wrapped).astype(np.complex64)
    coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)

    t0 = time.perf_counter()
    unw_off, cc = ww.unwrap(ig, coh_in, 16.0, mask, bridge=False)  # HEAVY (single)
    t_unw = time.perf_counter() - t0
    unw_off = np.asarray(unw_off, np.float32)

    t1 = time.perf_counter()
    unw_on = ww.bridge_components(unw_off, mask)
    t_br = time.perf_counter() - t1
    unw_on = np.asarray(unw_on, np.float32)

    moved = int(((unw_off != unw_on) & mask).sum())
    valid = mask & np.isfinite(unw_off)
    pc_off = percomp_match(unw_off, prod_unw, wrapped, prod_cc, valid)
    pc_on = percomp_match(
        unw_on, prod_unw, wrapped, prod_cc, mask & np.isfinite(unw_on)
    )
    rows.append((frame, pc_off, pc_on, moved))
    print(
        f"{frame:7s} {pc_off*100:7.2f} {pc_on*100:9.2f} {(pc_on-pc_off)*100:+6.2f} {moved:9d} "
        f"{int(cc.max()):6d} {t_unw:6.1f} {t_br:5.1f}",
        flush=True,
    )

print("\n--- summary ---", flush=True)
regress = [r for r in rows if r[2] < r[1] - 1e-6]
improved = [r for r in rows if r[2] > r[1] + 1e-6]
noop = [r for r in rows if r[3] == 0]
print(
    f"improved: {[(r[0], round(r[1]*100,1), round(r[2]*100,1)) for r in improved]}",
    flush=True,
)
print(
    f"REGRESSED: {[(r[0], round(r[1]*100,1), round(r[2]*100,1)) for r in regress]}",
    flush=True,
)
print(f"byte-identical (0 px moved): {len(noop)}/{len(rows)} frames", flush=True)
