"""Cache fine + coarse unwrap arrays for the bridging prototype, so proto_bridge
iterates with ZERO heavy compute. This is the ONLY script that calls ww.unwrap.

Per frame: reuse the cached fine (unw, cc) from bridge_cache/<FRAME>.npz if present
(written by plot_unwrap_compare.py), else run ww.unwrap once (heavy). Then build the
L=8 coherent-multilook coarse anchor (light: ~1/64 pixels) and save everything to
bridge_cache/<FRAME>.npz. Unwraps run SEQUENTIALLY in one process -> one heavy
unwrap at a time (concurrency rule).

Usage (base miniforge3 env): python scripts/cache_bridge_arrays.py [FRAMES...]
"""

import sys
import glob
import os

import numpy as np
import h5py

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase

import whirlwind as ww

CACHE = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/bridge_cache"
L = 8
frames = sys.argv[1:] or [
    "005_A_025",
    "005_A_016",
    "005_A_030",
    "005_A_028",
    "005_D_077",
    "005_D_074",
]
os.makedirs(CACHE, exist_ok=True)


def block_mean(a, L):
    mm, nn = a.shape[0] // L, a.shape[1] // L
    return a[: mm * L, : nn * L].reshape(mm, L, nn, L).mean(axis=(1, 3))


for frame in frames:
    h5path = glob.glob(
        f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5"
    )[0]
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

    npz_path = f"{CACHE}/{frame}.npz"
    cached = np.load(npz_path) if os.path.exists(npz_path) else None
    if cached is not None and "unw" in cached and "cunw" in cached:
        print(f"{frame}: already fully cached (fine+coarse), skipping", flush=True)
        continue
    if cached is not None and "unw" in cached:
        unw = cached["unw"].astype(np.float32)
        cc = cached["cc"].astype(np.int32)
        print(f"{frame}: reusing cached fine unwrap (ncc={int(cc.max())})", flush=True)
    else:
        unw, cc = ww.unwrap(ig, coh_in, 16.0, mask)  # HEAVY (single, sequential)
        unw = np.asarray(unw, np.float32)
        cc = np.asarray(cc).astype(np.int32)
        print(f"{frame}: fine unwrap done (ncc={int(cc.max())})", flush=True)

    # Coarse L=8 coherent-multilook anchor (light).
    cig = block_mean(ig, L).astype(np.complex64)
    ccoh = block_mean(coh_in, L).astype(np.float32)
    cmask = block_mean(mask.astype(np.float32), L) > 0.4
    cunw, _ = ww.unwrap(cig, ccoh, 16.0 * L * L, cmask)
    cunw = np.asarray(cunw, np.float32)

    np.savez_compressed(
        npz_path,
        unw=unw,
        cc=cc,
        coh=coh_in,
        prod=prod_unw.astype(np.float32),
        prod_cc=prod_cc.astype(np.int32),
        mask=mask,
        wrapped=wrapped,
        cunw=cunw,
        ccoh=ccoh,
        cmask=cmask,
        L=L,
    )
    print(
        f"{frame}: cached fine+coarse -> {npz_path}  (coarse {cunw.shape})", flush=True
    )
