"""Run unwrap_linear once on FRAME and report per-comp match to production.
Honors WHIRLWIND_SEA_COST (read once per process by the Rust cost). Loop the
env var in bash to sweep the de-degeneration knob, one heavy solve per value.

Usage: WHIRLWIND_SEA_COST=2 python scripts/diag_seacost.py D_074
"""
import os, sys, glob, time
import h5py
import numpy as np
import whirlwind as ww

tau = 2 * np.pi
wrap = lambda x: ((x + np.pi) % tau) - np.pi
frame = sys.argv[1]
sea = os.environ.get("WHIRLWIND_SEA_COST", "0")

h5 = glob.glob(f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5")[0]
base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
with h5py.File(h5, "r") as h:
    grp = h[base]
    pol = sorted(k for k, v in grp.items() if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"})[0]
    prod_unw = h[f"{base}/{pol}/unwrappedPhase"][()].astype(np.float32)
    coh = h[f"{base}/{pol}/coherenceMagnitude"][()].astype(np.float32)
    prod_cc = h[f"{base}/{pol}/connectedComponents"][()].astype(np.int32)
    mask_arr = h[f"{base}/mask"][()] if "mask" in grp else None

mask = (mask_arr != 255) & ((mask_arr // 100) % 10 == 0) if mask_arr is not None else np.ones(prod_unw.shape, bool)
mask &= np.isfinite(prod_unw) & np.isfinite(coh)
ig = np.exp(1j * np.where(mask, wrap(prod_unw), 0.0)).astype(np.complex64)
coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)


def percomp(unw):
    unw = np.asarray(unw)
    in_c = mask & np.isfinite(unw) & (prod_cc > 0)
    if not in_c.any():
        return float("nan")
    amb = np.rint((unw[in_c] - prod_unw[in_c]) / tau)
    cc = prod_cc[in_c]
    ok = tot = 0
    for lab in np.unique(cc):
        m = cc == lab
        off = np.median(amb[m])
        ok += int((np.abs(amb[m] - off) < 0.5).sum())
        tot += int(m.sum())
    return ok / tot


t = time.time()
u = ww._native.unwrap_linear(ig, coh_in, 16.0, mask)
print(f"{frame}: SEA_COST={sea}  percomp={percomp(u):.3f}  ({time.time()-t:.0f}s)", flush=True)
