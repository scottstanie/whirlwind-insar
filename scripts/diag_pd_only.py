"""Probe ww-orig's PD vs SSP split on FRAME, using only the exposed API.

C++ `primal_dual(net, maxiter=0)` runs the PD loop until balanced with NO SSP
(the `iter==maxiter` break never triggers; it returns when excess hits 0). So:
  * maxiter=0 : PD-only, run to convergence (or until PD makes no progress).
  * maxiter=8 : 8 PD iters then SSP (the real ww-orig unwrap path).
Comparing total_cost / is_balanced / total_excess across maxiter tells us whether
ww-orig's PD alone solves the frame or leans on SSP - locating where Rust's PD
(which strands 4 sources) diverges. Pure diagnostic; no rebuild needed.

Usage: python scripts/diag_pd_only.py 005_D_074
"""

import sys, glob, time
import h5py, numpy as np
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
    prod = h[f"{base}/{pol}/unwrappedPhase"][()].astype(np.float32)
    coh = h[f"{base}/{pol}/coherenceMagnitude"][()].astype(np.float32)
    ma = h[f"{base}/mask"][()] if "mask" in grp else None
mask = (
    (ma != 255) & ((ma // 100) % 10 == 0)
    if ma is not None
    else np.ones(prod.shape, bool)
)
mask &= np.isfinite(prod) & np.isfinite(coh)
ig = np.exp(1j * np.where(mask, wrap(prod), 0.0)).astype(np.complex64)
coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)
phase = np.angle(ig).astype(np.float32)


def build():
    res = np.asarray(orig_residue(phase)).copy()
    res[0, :] = 0
    res[-1, :] = 0
    res[:, 0] = 0
    res[:, -1] = 0
    cost = orig_costs(ig, coh_in, 16.0, ~mask)
    g = RectangularGridGraph(*res.shape)
    return Network(g, res.flatten(), cost, capacity=1)


import signal


class Timeout(Exception):
    pass


def _alarm(*a):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)

print(f"{frame}: shape={ig.shape} valid={mask.mean()*100:.1f}%", flush=True)
for mx in (
    8,
    0,
):  # maxiter=0 = PD-only (no SSP); guarded - C++ has no no-progress break
    net = build()
    tag = "PD-only(maxiter=0)" if mx == 0 else "PD8+SSP(maxiter=8)"
    t = time.time()
    signal.alarm(240 if mx == 0 else 0)
    try:
        primal_dual(net, mx)
        signal.alarm(0)
        print(
            f"{frame}: {tag:20s} total_cost={net.total_cost()} balanced={net.is_balanced()} "
            f"total_excess={net.total_excess()} ({time.time()-t:.0f}s)",
            flush=True,
        )
    except Timeout:
        print(
            f"{frame}: {tag:20s} DID NOT CONVERGE in 240s (PD alone cannot balance → relies on SSP)",
            flush=True,
        )
