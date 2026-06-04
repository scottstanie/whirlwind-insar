"""Characterize the A_025 (low-coherence river) failure: is it a clean
relative-2pi-offset split across the river (a bridging problem), or scattered
within-region 2pi errors? And is the river cheap/free to cross (so the offset is
under-determined)? Produces a 6-panel diagnostic figure + prints structure.

Usage: python scripts/diag_a025.py [FRAME=A_025]
"""
import sys
import glob

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import whirlwind as ww

tau = 2 * np.pi
wrap = lambda x: ((x + np.pi) % tau) - np.pi
frame = sys.argv[1] if len(sys.argv) > 1 else "A_025"
h5 = glob.glob(f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5")[0]
base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
with h5py.File(h5, "r") as h:
    grp = h[base]
    pol = sorted(k for k, v in grp.items() if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"})[0]
    prod = h[f"{base}/{pol}/unwrappedPhase"][()].astype(np.float32)
    coh = h[f"{base}/{pol}/coherenceMagnitude"][()].astype(np.float32)
    prod_cc = h[f"{base}/{pol}/connectedComponents"][()].astype(np.int32)
    mask_arr = h[f"{base}/mask"][()] if "mask" in grp else None

mask = (mask_arr != 255) & ((mask_arr // 100) % 10 == 0) if mask_arr is not None else np.ones(prod.shape, bool)
mask &= np.isfinite(prod) & np.isfinite(coh)
wrapped = np.where(mask, wrap(prod), 0.0).astype(np.float32)
ig = np.exp(1j * wrapped).astype(np.complex64)
coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)
print(f"{frame}: shape={ig.shape} valid={mask.mean()*100:.1f}%", flush=True)

unw, cc = ww.unwrap(ig, coh_in, 16.0, mask)
unw = np.asarray(unw, np.float32)

# Raw ambiguity-diff vs production (NOT per-component aligned) -> spatial structure.
both = mask & np.isfinite(unw) & np.isfinite(prod)
amb = np.full(prod.shape, np.nan, np.float32)
amb[both] = np.rint((unw[both] - prod[both]) / tau)
# Remove a single global offset (unobservable) so the map is centered.
g = np.rint(np.nanmedian(amb[both]))
amb_c = amb - g
vals, counts = np.unique(amb_c[np.isfinite(amb_c)], return_counts=True)
print(f"{frame}: ambiguity-diff (global-aligned) value distribution:")
for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])[:8]:
    print(f"    k={v:+.0f}: {c/both.sum()*100:5.1f}% of valid")

# How many production components, and per-component how mixed is the ambiguity?
print(f"{frame}: production has {int(prod_cc.max())} connected components")
in_c = both & (prod_cc > 0)
mixed = 0
for lab in np.unique(prod_cc[in_c]):
    m = in_c & (prod_cc == lab)
    a = amb_c[m]
    frac_majority = np.bincount((a - a.min()).astype(int)).max() / m.sum()
    if m.sum() > 5000 and frac_majority < 0.95:
        mixed += 1
print(f"{frame}: production components >5k px that are INTERNALLY split (>5% off the "
      f"component's majority cycle): {mixed}  <- these are the bridging failures")

# Coherence structure of the "river": fraction of valid pixels at low coherence.
lowcoh = mask & (coh_in < 0.3)
print(f"{frame}: valid pixels with coh<0.3 (decorrelation bands/river): {lowcoh.sum()/mask.sum()*100:.1f}%")

# Figure.
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
def show(a, arr, title, **kw):
    d = np.where(mask, arr, np.nan)
    im = a.imshow(d, **kw); a.set_title(title); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046)
show(ax[0, 0], wrapped, "wrapped phase", cmap="twilight", vmin=-np.pi, vmax=np.pi)
show(ax[0, 1], coh_in, "coherence", cmap="gray", vmin=0, vmax=1)
show(ax[0, 2], prod_cc.astype(float), f"production conncomp ({int(prod_cc.max())})", cmap="tab20")
pu = np.where(both, prod, np.nan); wu = np.where(both, unw, np.nan)
vlo, vhi = np.nanpercentile(pu, [2, 98])
show(ax[1, 0], pu, "production unwrap", cmap="viridis", vmin=vlo, vmax=vhi)
show(ax[1, 1], wu, "whirlwind unwrap", cmap="viridis", vmin=vlo, vmax=vhi)
show(ax[1, 2], amb_c, "ambiguity diff (ww-prod)/2π", cmap="RdBu", vmin=-2, vmax=2)
fig.suptitle(f"{frame}: A-025 river/bridging characterization (whirlwind per-comp ≈ 58%)")
fig.tight_layout()
out = f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_sweep/{frame}_characterize.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"{frame}: figure -> {out}", flush=True)
