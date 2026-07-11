"""Prototype the 'minimum-jumpy' bridge for 005_A_025: anchor whirlwind's regions to
a coarse multilook unwrap (which averages the decorrelation river into coherence,
so its relative bank offset is well-determined). Snap each whirlwind connected
component to the coarse anchor's integer 2pi level. Reports per-comp before/after.

Usage: python scripts/proto_anchor_a025.py [FRAME=005_A_025] [L=8]
"""

import sys, glob
import h5py, numpy as np
import whirlwind as ww

tau = 2 * np.pi
wrap = lambda x: ((x + np.pi) % tau) - np.pi
frame = sys.argv[1] if len(sys.argv) > 1 else "005_A_025"
L = int(sys.argv[2]) if len(sys.argv) > 2 else 8
h5 = glob.glob(
    f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5"
)[0]
base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
with h5py.File(h5, "r") as h:
    grp = h[base]
    pol = sorted(
        k
        for k, v in grp.items()
        if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"}
    )[0]
    prod = h[f"{base}/{pol}/unwrappedPhase"][()].astype(np.float32)
    coh = h[f"{base}/{pol}/coherenceMagnitude"][()].astype(np.float32)
    prod_cc = h[f"{base}/{pol}/connectedComponents"][()].astype(np.int32)
    mask_arr = h[f"{base}/mask"][()] if "mask" in grp else None
mask = (
    (mask_arr != 255) & ((mask_arr // 100) % 10 == 0)
    if mask_arr is not None
    else np.ones(prod.shape, bool)
)
mask &= np.isfinite(prod) & np.isfinite(coh)
ig = np.exp(1j * np.where(mask, wrap(prod), 0.0)).astype(np.complex64)
coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)
m, n = ig.shape


def percomp(u):
    in_c = mask & np.isfinite(u) & (prod_cc > 0)
    amb = np.rint((u[in_c] - prod[in_c]) / tau)
    ccp = prod_cc[in_c]
    ok = tot = 0
    for lab in np.unique(ccp):
        mm = ccp == lab
        off = np.median(amb[mm])
        ok += int((np.abs(amb[mm] - off) < 0.5).sum())
        tot += int(mm.sum())
    return ok / tot


# Fine single-tile unwrap (the default) + its conncomp.
unw, cc = ww.unwrap(ig, coh_in, 16.0, mask)
unw = np.asarray(unw, np.float32)
cc = np.asarray(cc)
print(
    f"{frame}: fine whirlwind per-comp={percomp(unw)*100:.1f}%  (ncc={int(cc.max())})",
    flush=True,
)

# Coarse multilook: COHERENT-average the complex igram + coherence over LxL blocks.
mm, nn = m // L, n // L


def bmean(a):
    return a[: mm * L, : nn * L].reshape(mm, L, nn, L).mean(axis=(1, 3))


cig = bmean(ig).astype(np.complex64)
ccoh = bmean(coh_in).astype(np.float32)
cmask = bmean(mask.astype(np.float32)) > 0.4  # majority-valid coarse pixels
cunw, _ = ww.unwrap(cig, ccoh, 16.0 * L * L, cmask)
cunw = np.asarray(cunw, np.float32)
# Upsample (block-replicate) to full res, padding the ragged last block by edge.
anchor = np.kron(cunw, np.ones((L, L), np.float32))
anchor = np.pad(
    anchor,
    ((0, max(0, m - anchor.shape[0])), (0, max(0, n - anchor.shape[1]))),
    mode="edge",
)[:m, :n]
print(
    f"{frame}: coarse ({mm}x{nn}, L={L}) coh>0.3 in river now {(cmask & (ccoh>0.3)).sum()/max(cmask.sum(),1)*100:.0f}% of coarse-valid",
    flush=True,
)

# Snap each fine conncomp to the coarse anchor's integer 2pi level - but ONLY
# when the anchor is UNANIMOUS about the shift (>= THRESH of region pixels vote
# the same nonzero shift). On a confident coherent frame the coarse anchor is
# noisier than the fine unwrap, so its votes scatter -> gate skips -> no
# regression; across a true low-coherence river the votes are unanimous -> snap.
import os

THRESH = float(os.environ.get("ANCHOR_THRESH", "0.75"))
both = mask & np.isfinite(unw) & np.isfinite(anchor)
for snap_all in (False, True):  # gated vs ungated, for comparison
    unw_fix = unw.copy()
    nshift = 0
    for lab in np.unique(cc[cc > 0]):
        reg = (cc == lab) & both
        if reg.sum() < 200:
            continue
        votes = np.rint((anchor[reg] - unw[reg]) / tau).astype(int)
        vals, cnts = np.unique(votes, return_counts=True)
        s = int(vals[cnts.argmax()])
        frac = cnts.max() / len(votes)
        if s != 0 and (snap_all or frac >= THRESH):
            unw_fix[cc == lab] = unw[cc == lab] + tau * s
            nshift += 1
    tag = "ungated" if snap_all else f"gated@{THRESH}"
    print(
        f"{frame}: anchor-snap {tag:11s} per-comp={percomp(unw_fix)*100:.1f}%  (shifted {nshift} regions)",
        flush=True,
    )
    if not snap_all:
        unw_gated = unw_fix.copy()

# --- Codex's region definition: connected components of the HIGH-COHERENCE land
# (NOT the fine conncomps). 005_D_077 is then ~one coherent region -> one global
# shift (no regression); 005_A_025 is the banks cut by the low-coh river -> per-bank
# anchor vote. Low-coh pixels (river) keep their fine value.
from scipy import ndimage

COH_T = float(os.environ.get("COH_T", "0.3"))
land = mask & (coh_in > COH_T)
regions, nreg = ndimage.label(land)
unw_cr = unw.copy()
sizes = ndimage.sum(np.ones_like(regions), regions, index=np.arange(1, nreg + 1))
nshift = 0
for lab in range(1, nreg + 1):
    if sizes[lab - 1] < 500:
        continue
    reg = (regions == lab) & both
    if reg.sum() < 200:
        continue
    s = int(np.rint(np.nanmedian((anchor[reg] - unw[reg]) / tau)))
    if s != 0:
        unw_cr[regions == lab] = unw[regions == lab] + tau * s
        nshift += 1
print(
    f"{frame}: coh-region snap (coh>{COH_T}, {nreg} regions) per-comp={percomp(unw_cr)*100:.1f}%  "
    f"(shifted {nshift})",
    flush=True,
)

# --- ROBUST region definition: connected components of COARSE coherent land.
# At x8 the speckle that fragmented the fine coh-mask averages into coherence,
# while a WIDE river/water gap persists as low-coh. So a coherent frame (005_A_030)
# collapses to ~one coarse region (one global shift = unobservable, no
# regression), while 005_A_025's banks stay separate (real barrier -> data-supported
# anchor). This IS Codex's "same coarse component => data-supported" test.
coarse_land = cmask & (ccoh > COH_T)
clabels, ncr = ndimage.label(coarse_land)
clab_up = np.kron(clabels, np.ones((L, L), int))
clab_up = np.pad(
    clab_up,
    ((0, max(0, m - clab_up.shape[0])), (0, max(0, n - clab_up.shape[1]))),
    mode="edge",
)[:m, :n]
csizes = ndimage.sum(np.ones_like(clabels), clabels, index=np.arange(1, ncr + 1))
unw_cc = unw.copy()
nshift = 0
for lab in range(1, ncr + 1):
    if csizes[lab - 1] < 50:  # skip tiny coarse regions
        continue
    reg = (clab_up == lab) & both
    if reg.sum() < 500:
        continue
    s = int(np.rint(np.nanmedian((anchor[reg] - unw[reg]) / tau)))
    if s != 0:
        unw_cc[clab_up == lab] = unw[clab_up == lab] + tau * s
        nshift += 1
big = int((csizes >= 50).sum())
print(
    f"{frame}: COARSE-region snap ({ncr} coarse regions, {big} big) per-comp={percomp(unw_cc)*100:.1f}%  "
    f"(shifted {nshift})",
    flush=True,
)

# Oracle ceiling: best achievable by snapping each fine conncomp to PRODUCTION.
unw_or = unw.copy()
for lab in np.unique(cc[cc > 0]):
    reg = (cc == lab) & (mask & np.isfinite(unw) & (prod_cc > 0))
    if reg.sum() < 200:
        continue
    off = np.rint(np.nanmedian((prod[reg] - unw[reg]) / tau))
    unw_or[cc == lab] = unw[cc == lab] + tau * off
print(
    f"{frame}: oracle ceiling (snap fine cc -> production)  per-comp={percomp(unw_or)*100:.1f}%",
    flush=True,
)
