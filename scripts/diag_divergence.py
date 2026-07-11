"""Bisect the Rust `unwrap_linear` vs Python ww-orig divergence, stage by stage.

On the "problem" NISAR frames (005_D_074, 005_D_075, 006_A_035) Rust `unwrap_linear`
(capacity-1, parity path) scores far below Python ww-orig per-component, even
though both implement the *same* capacity-1 Carballo MCF algorithm and match to
99.5% on 005_D_077. This script isolates WHICH stage diverges by feeding identical
intermediate state across the Rust<->Python boundary:

  STAGE 0  residues   : ww._native.compute_residues  vs  ww_orig._lib.residue
  STAGE 1  cost/solver: ww._native.unwrap_linear_ext_costs(ORIG costs) vs
                        ww_orig.unwrap  -> if ext_costs ~= ww_orig, the COST
                        differs (rust-parity != orig); if ext_costs ~= linear,
                        the SOLVER differs (same cost, different flow).

NOTE: `unwrap_reuse` is deliberately NOT run here. It is a DIFFERENT algorithm
(multi-unit PHASS flow-reuse, production Carballo costs, 50 PD iters), not the
ww-orig parity path, so it cannot tell us whether `unwrap_linear` matches
ww-orig. The goal of this script is to prove parity / locate the divergence in
the capacity-1 parity path only.

Heavy unwraps run SEQUENTIALLY (one NISAR-scale solve at a time; see memory).
A synthetic self-test guards the Python->Rust cost-layout conversion before any
heavy run, so a layout bug can't masquerade as a solver divergence.

Usage: python scripts/diag_divergence.py 005_D_074
"""

import sys
import glob
import time

import h5py
import numpy as np

import whirlwind as ww
from whirlwind_orig._unwrap import unwrap as ww_orig_unwrap
from whirlwind_orig._lib import residue as orig_residue
from whirlwind_orig._cost import compute_carballo_costs as orig_costs

tau = 2 * np.pi
wrap = lambda x: ((x + np.pi) % tau) - np.pi


def py_costs_to_rust(py_cost, m_phase, n_phase):
    """Convert ww-orig cost layout [UP, LEFT, DOWN, RIGHT] to Rust
    [DOWN, UP, RIGHT, LEFT]. See whirlwind_core::unwrap_linear_ext_costs."""
    n_v = m_phase * (n_phase + 1)
    n_h = (m_phase + 1) * n_phase
    assert py_cost.size == 2 * n_v + 2 * n_h, (py_cost.size, 2 * n_v + 2 * n_h)
    up = py_cost[0:n_v]
    lt = py_cost[n_v : n_v + n_h]
    dn = py_cost[n_v + n_h : 2 * n_v + n_h]
    rt = py_cost[2 * n_v + n_h :]
    return np.ascontiguousarray(np.concatenate([dn, up, rt, lt]).astype(np.int32))


def self_test_conversion():
    """Tiny clean ramp: orig costs -> Rust ext_costs must unwrap it correctly.
    If the layout conversion were transposed/misordered, a smooth ramp fails."""
    m = n = 48
    yy, xx = np.mgrid[0:m, 0:n]
    truth = 0.35 * (yy + xx)  # gentle ramp, no residues
    ig = np.exp(1j * wrap(truth)).astype(np.complex64)
    corr = np.full((m, n), 0.95, np.float32)
    valid = np.ones((m, n), bool)
    pc = orig_costs(ig, corr, 16.0, ~valid)  # orig wants INVALID mask
    rc = py_costs_to_rust(np.asarray(pc), m, n)
    u = np.asarray(ww._native.unwrap_linear_ext_costs(ig, valid, rc))
    off = np.round((u - truth) / tau).astype(int)
    u_al = u - tau * np.median(off)
    err = np.nanmax(np.abs(u_al - truth))
    assert err < 1e-2, f"cost-conversion self-test FAILED: max err {err:.3f} rad"
    print(f"[self-test] cost conversion OK (ramp max err {err:.2e} rad)", flush=True)


frame = sys.argv[1]
self_test_conversion()

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
    prod_unw = h[f"{base}/{pol}/unwrappedPhase"][()].astype(np.float32)
    coh = h[f"{base}/{pol}/coherenceMagnitude"][()].astype(np.float32)
    prod_cc = h[f"{base}/{pol}/connectedComponents"][()].astype(np.int32)
    mask_arr = h[f"{base}/mask"][()] if "mask" in grp else None

mask = (
    (mask_arr != 255) & ((mask_arr // 100) % 10 == 0)
    if mask_arr is not None
    else np.ones(prod_unw.shape, bool)
)
mask &= np.isfinite(prod_unw) & np.isfinite(coh)
wrapped = np.where(mask, wrap(prod_unw), 0.0).astype(np.float32)
ig = np.exp(1j * wrapped).astype(np.complex64)
coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)
m_phase, n_phase = ig.shape
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
        msk = cc == lab
        off = np.median(amb[msk])
        ok += int((np.abs(amb[msk] - off) < 0.5).sum())
        tot += int(msk.sum())
    return ok / tot


def dcyc(a, b):
    both = mask & np.isfinite(a) & np.isfinite(b)
    d = np.rint((a[both] - b[both]) / tau)
    return float((d != 0).mean()), float(d.min()), float(d.max())


# ----- STAGE 0: residues (boundary-zeroed, as both unwrap()s do) -----
def bz(r):
    r = r.copy()
    r[0, :] = 0
    r[-1, :] = 0
    r[:, 0] = 0
    r[:, -1] = 0
    return r


# Use the SAME phase array for both so STAGE 0 isolates the residue *algorithm*,
# not a float wrap difference. `unwrap_linear` internally does `igram.arg()`, so
# feed `np.angle(ig)` to both (not the pre-wrap `wrapped`).
phase = np.angle(ig).astype(np.float32)
r_rust = bz(np.asarray(ww._native.compute_residues(phase)))
r_orig = bz(np.asarray(orig_residue(phase)))
res_mism = int((r_rust != r_orig).sum())
print(
    f"{frame}: STAGE0 residues  rust_nnz={int((r_rust!=0).sum())} "
    f"orig_nnz={int((r_orig!=0).sum())}  mismatches={res_mism}",
    flush=True,
)

# ----- build ORIG costs once, convert to Rust layout -----
t = time.time()
pc = orig_costs(ig, coh_in, 16.0, ~mask)
rc = py_costs_to_rust(np.asarray(pc), m_phase, n_phase)
print(
    f"{frame}: orig costs built+converted ({time.time()-t:.0f}s) "
    f"nnz={int((rc!=0).sum())} min={rc.min()} max={rc.max()}",
    flush=True,
)

# ----- heavy solves, SEQUENTIAL -----
results = {}
t = time.time()
results["ww_orig"] = np.asarray(ww_orig_unwrap(ig, coh_in, 16.0, mask=~mask))
to = time.time() - t
print(
    f"{frame}: ww_orig            percomp={percomp(results['ww_orig']):.3f}  ({to:.0f}s)",
    flush=True,
)

t = time.time()
results["linear"] = np.asarray(ww._native.unwrap_linear(ig, coh_in, 16.0, mask))
tl = time.time() - t
print(
    f"{frame}: unwrap_linear      percomp={percomp(results['linear']):.3f}  ({tl:.0f}s)",
    flush=True,
)

t = time.time()
results["ext_orig"] = np.asarray(ww._native.unwrap_linear_ext_costs(ig, mask, rc))
te = time.time() - t
print(
    f"{frame}: ext_costs(orig)    percomp={percomp(results['ext_orig']):.3f}  ({te:.0f}s)  "
    f"<- ORIG cost + RUST solver",
    flush=True,
)

# ----- pairwise integer-cycle diffs -----
print(f"{frame}: --- integer-cycle diffs (nonzero_frac, range) ---", flush=True)
for a, b in [("linear", "ww_orig"), ("ext_orig", "ww_orig"), ("ext_orig", "linear")]:
    nz, lo, hi = dcyc(results[a], results[b])
    print(
        f"{frame}:   {a:9s} vs {b:9s}: nonzero_frac={nz:.3f} range=[{lo:.0f},{hi:.0f}]",
        flush=True,
    )

print(
    f"{frame}: VERDICT GUIDE: ext_orig~=ww_orig => COST differs (rust-parity LUT); "
    f"ext_orig~=linear => SOLVER differs (tie/SSP/integration)",
    flush=True,
)
