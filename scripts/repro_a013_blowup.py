"""Minimal repro: run single-tile unwrap_linear on a heavily-masked GUNW frame
(A_013) in isolation (no bench overhead) to diagnose the memory/time blowup.

Run with WHIRLWIND_DEBUG=1 RUST_BACKTRACE=1 to see PD/SSP progress + any crash.
Usage: python scripts/repro_a013_blowup.py [FRAME]   (default A_013)
"""
import sys, glob, time
import h5py
import numpy as np
import whirlwind as ww

tau = 2 * np.pi
frame = sys.argv[1] if len(sys.argv) > 1 else "A_013"
h5path = glob.glob(f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5")[0]

unw_path = mask_path = coh_path = None
with h5py.File(h5path, "r") as f:
    def visit(name, obj):
        global unw_path, mask_path, coh_path
        if isinstance(obj, h5py.Dataset):
            if name.endswith("unwrappedPhase") and unw_path is None:
                unw_path = name
            if name.endswith("/mask") and mask_path is None and obj.ndim == 2:
                mask_path = name
            if name.endswith("coherenceMagnitude") and coh_path is None:
                coh_path = name
    f.visititems(visit)
    unw = f[unw_path][()].astype(np.float32)
    mcode = f[mask_path][()]
    coh = f[coh_path][()].astype(np.float32)

valid = (mcode != 255) & ((mcode // 100) % 10 == 0)
wrapped = np.where(valid, ((unw + np.pi) % tau) - np.pi, 0.0).astype(np.float32)
igram = np.exp(1j * wrapped).astype(np.complex64)
corr_in = np.where(valid, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)

print(f"{frame}: shape={igram.shape} valid={valid.mean()*100:.1f}%", flush=True)
t0 = time.time()
out = ww._native.unwrap_linear(igram, corr_in, 16.0, valid)
print(f"DONE {time.time()-t0:.1f}s range=[{np.nanmin(out):.1f},{np.nanmax(out):.1f}]", flush=True)
