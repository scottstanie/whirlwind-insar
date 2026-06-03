"""Head-to-head on IDENTICAL inputs: Python ww-orig vs Rust unwrap_linear.

Builds one set of inputs the run_whirlwind_orig.py way (water_only & finite mask,
masked phase zeroed, coherence cleaned), runs BOTH solvers sequentially (one
heavy unwrap at a time), and scores each vs the production GUNW unwrap
(per-component aligned). Tells us definitively whether a "problem" frame is a
Rust<->Python divergence (ww-orig good, Rust bad) or a genuinely hard frame
(both bad).

Usage: python scripts/headtohead_orig_vs_rust.py FRAME   (e.g. D_074)
"""
import sys, glob, time
import h5py
import numpy as np
import whirlwind as ww
from whirlwind_orig._unwrap import unwrap as ww_orig_unwrap

tau = 2 * np.pi
frame = sys.argv[1]
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
wrap = lambda x: ((x + np.pi) % tau) - np.pi
wrapped = np.where(mask, wrap(prod_unw), 0.0).astype(np.float32)
ig = np.exp(1j * wrapped).astype(np.complex64)
coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)
print(f"{frame}: shape={ig.shape} valid={mask.mean()*100:.1f}%", flush=True)


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


# ww-orig takes mask=INVALID (True=masked); Rust unwrap_linear takes mask=VALID.
t = time.time(); u_orig = np.asarray(ww_orig_unwrap(ig, coh_in, 16.0, mask=~mask)); t_orig = time.time() - t
print(f"{frame}: ww-orig       per-comp={percomp(u_orig):.3f}  ({t_orig:.0f}s)", flush=True)
t = time.time(); u_rust = np.asarray(ww._native.unwrap_linear(ig, coh_in, 16.0, mask)); t_rust = time.time() - t
print(f"{frame}: unwrap_linear per-comp={percomp(u_rust):.3f}  ({t_rust:.0f}s)", flush=True)

both = mask & np.isfinite(u_orig) & np.isfinite(u_rust)
dcyc = np.rint((u_rust[both] - u_orig[both]) / tau)
print(f"{frame}: rust-vs-orig integer-cycle diff: nonzero_frac={(dcyc != 0).mean():.3f} "
      f"range=[{dcyc.min():.0f},{dcyc.max():.0f}]", flush=True)
