"""Compare the MCF OBJECTIVE (total flow cost) + balance between ww-orig and
Rust unwrap_linear on the SAME inputs. Decides the pivotal question:

  * Rust total_cost == ww-orig total_cost  -> equal-cost DEGENERATE optima;
    the cost under-determines the answer across the masked sea; matching
    ww-orig is matching a tie-break (fix = de-degenerate, e.g. short-path pref).
  * Rust total_cost  >  ww-orig total_cost  -> Rust is SUB-OPTIMAL (solver bug);
    fix = make the solver reach the optimum like ww-orig.
  * remaining_excess != 0 -> the flow never balanced (incomplete).

ww-orig's cost via Network.total_cost(); Rust's via the [pd_full] FINAL debug
line (set WHIRLWIND_DEBUG=1 when running this script).

Usage: WHIRLWIND_DEBUG=1 python scripts/diag_cost_compare.py 005_D_074
"""

import sys
import glob
import time

import h5py
import numpy as np

import whirlwind as ww
from whirlwind_orig._cost import compute_carballo_costs as orig_costs
from whirlwind_orig._lib import residue as orig_residue
from whirlwind_orig.graph import RectangularGridGraph
from whirlwind_orig.network import Network, primal_dual

tau = 2 * np.pi
wrap = lambda x: ((x + np.pi) % tau) - np.pi
frame = sys.argv[1]

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
print(f"{frame}: shape={ig.shape} valid={mask.mean()*100:.1f}%", flush=True)

# --- ww-orig: build network manually, run PD x8, read objective + balance ---
phase = np.angle(ig).astype(np.float32)
res = np.asarray(orig_residue(phase))
res[0, :] = 0
res[-1, :] = 0
res[:, 0] = 0
res[:, -1] = 0
surplus = res.flatten()
cost = orig_costs(ig, coh_in, 16.0, ~mask)
graph = RectangularGridGraph(*res.shape)
network = Network(graph, surplus, cost, capacity=1)
t = time.time()
primal_dual(network, maxiter=8)
print(
    f"{frame}: ww_orig  total_cost={network.total_cost()}  "
    f"balanced={network.is_balanced()}  total_excess={network.total_excess()}  "
    f"({time.time()-t:.0f}s)",
    flush=True,
)

# --- Rust: run unwrap_linear; the [pd_full] FINAL line prints to stderr ---
print(
    f"{frame}: running Rust unwrap_linear (FINAL cost on stderr below) ...", flush=True
)
t = time.time()
_ = ww._native.unwrap_linear(ig, coh_in, 16.0, mask)
print(
    f"{frame}: unwrap_linear done ({time.time()-t:.0f}s) - see [pd_full] FINAL above",
    flush=True,
)
