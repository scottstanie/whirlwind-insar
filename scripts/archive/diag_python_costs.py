"""
Diagnostic: feed Python ww-orig costs directly into the Rust PD solver.

If score reaches ~99.49% → the cost function is the bottleneck.
If score stays ~94%       → the solver has a deeper structural issue.

Layout conversion from Python _cost.compute_carballo_costs output
  [UP(n_v), LEFT(n_h), DOWN(n_v), RIGHT(n_h)]
to Rust arc order
  [DOWN(n_v), UP(n_v), RIGHT(n_h), LEFT(n_h)].
"""
import sys
import time
import numpy as np
import scipy.ndimage
from scipy.interpolate import RegularGridInterpolator
import whirlwind as ww

# Load _cost.py directly without triggering whirlwind_orig.__init__
# (the compiled _lib extension isn't importable in this env).
_WWORIG_DIR = "/Users/staniewi/repos/whirlwind/src/whirlwind_orig"

def _load_rgi(path):
    data = np.load(path, allow_pickle=False)
    grid = (data["grid_0"], data["grid_1"], data["grid_2"])
    fill_value = float(data["fill_value"]) if data["fill_value"] is not None else None
    return RegularGridInterpolator(
        points=grid, values=data["values"], method=str(data["method"]),
        bounds_error=bool(data["bounds_error"]), fill_value=fill_value,
    )

def compute_carballo_costs(igram, corr, nlooks, mask=None, batch_size=1000):
    dy_igram = igram[1:, :] * igram[:-1, :].conj()
    dx_igram = igram[:, 1:] * igram[:, :-1].conj()
    phase_dy = np.angle(dy_igram)
    phase_dx = np.angle(dx_igram)
    phase_dy_smooth = scipy.ndimage.uniform_filter(phase_dy, size=(7, 7), mode="nearest")
    phase_dx_smooth = scipy.ndimage.uniform_filter(phase_dx, size=(7, 7), mode="nearest")

    corr = np.asanyarray(corr)
    corr_dy = np.minimum(corr[1:, :], corr[:-1, :])
    corr_dx = np.minimum(corr[:, 1:], corr[:, :-1])

    spline_pdf0 = _load_rgi(f"{_WWORIG_DIR}/carballo-pdf-0-spline.npz")
    spline_pdf1 = _load_rgi(f"{_WWORIG_DIR}/carballo-pdf-1-spline.npz")

    def compute_cost(phase_diff, min_corr):
        total_size = phase_diff.size
        costs = np.empty_like(phase_diff)
        for start_idx in range(0, total_size, batch_size):
            end_idx = min(start_idx + batch_size, total_size)
            phase_batch = phase_diff.ravel()[start_idx:end_idx]
            corr_batch = min_corr.ravel()[start_idx:end_idx]
            p1 = spline_pdf1((phase_batch, corr_batch, nlooks))
            p0 = spline_pdf0((phase_batch, corr_batch, nlooks))
            costs.ravel()[start_idx:end_idx] = -np.log(p1 / p0)
        return costs

    cost_up = compute_cost(-phase_dx_smooth, corr_dx)
    cost_lt = compute_cost(phase_dy_smooth, corr_dy)
    cost_dn = compute_cost(phase_dx_smooth, corr_dx)
    cost_rt = compute_cost(-phase_dy_smooth, corr_dy)

    if mask is not None:
        mask = np.asanyarray(mask)
        mask_dy = np.logical_and(mask[1:, :], mask[:-1, :])
        mask_dx = np.logical_and(mask[:, 1:], mask[:, :-1])
        cost_dn[mask_dx] = np.nan
        cost_up[mask_dx] = np.nan
        cost_rt[mask_dy] = np.nan
        cost_lt[mask_dy] = np.nan

    cost = np.ascontiguousarray(np.concatenate([
        np.pad(cost_up, pad_width=[(0, 0), (1, 1)]).flatten(),
        np.pad(cost_lt, pad_width=[(1, 1), (0, 0)]).flatten(),
        np.pad(cost_dn, pad_width=[(0, 0), (1, 1)]).flatten(),
        np.pad(cost_rt, pad_width=[(1, 1), (0, 0)]).flatten(),
    ]))
    cost[np.isnan(cost)] = 0.0
    return (100.0 * cost).astype(np.int32)

WWORIG_NPZ = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_ref/D_077_wworig.npz"
OUTDIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_ref"
tau = 2 * np.pi

d = np.load(WWORIG_NPZ)
unw_py   = d["unw"]
unw_prod = d["prod_unw"]
conn_prod = d["prod_cc"].astype(np.int32)
mask      = d["mask"].astype(bool)
corr      = d["coh"].astype(np.float32)

wrapped = np.where(mask, ((unw_prod + np.pi) % tau) - np.pi, 0.0).astype(np.float32)
igram   = np.exp(1j * wrapped).astype(np.complex64)
corr_in = np.where(mask, np.clip(np.nan_to_num(corr), 0, 1), 0.0).astype(np.float32)
nlooks  = 16.0

m_phase, n_phase = igram.shape
print(f"Shape: {igram.shape}, mask {mask.mean()*100:.1f}% valid, nlooks={nlooks}")

# --- Compute Python costs ---
# Python _cost.compute_carballo_costs expects mask=~valid_mask (invalid pixels)
print("Computing Python costs (may take a minute)...")
t0 = time.time()
py_costs = compute_carballo_costs(igram, corr_in, nlooks, mask=~mask)
print(f"  Python costs computed in {time.time()-t0:.1f}s")
print(f"  Cost shape: {py_costs.shape}, dtype: {py_costs.dtype}")
print(f"  Cost range: [{py_costs.min()}, {py_costs.max()}]  nonzero: {(py_costs != 0).mean()*100:.1f}%")

# Layout: Python [UP(n_v), LEFT(n_h), DOWN(n_v), RIGHT(n_h)]
n_v = m_phase * (n_phase + 1)
n_h = (m_phase + 1) * n_phase
assert py_costs.shape == (2 * (n_v + n_h),), f"Expected {2*(n_v+n_h)}, got {py_costs.shape}"

# Remap to Rust order [DOWN(n_v), UP(n_v), RIGHT(n_h), LEFT(n_h)]
py_up    = py_costs[0          : n_v]
py_left  = py_costs[n_v        : n_v + n_h]
py_down  = py_costs[n_v + n_h  : 2*n_v + n_h]
py_right = py_costs[2*n_v + n_h: 2*n_v + 2*n_h]
rust_costs = np.ascontiguousarray(
    np.concatenate([py_down, py_up, py_right, py_left]), dtype=np.int32
)
print(f"  Rust-order costs: {rust_costs.shape}, range [{rust_costs.min()}, {rust_costs.max()}]")


def score(unw_ww, unw_ref, conn_ref, mask):
    comps = np.unique(conn_ref[conn_ref > 0])
    total = correct = 0
    for c in comps:
        pm = (conn_ref == c) & mask
        if not pm.any():
            continue
        diff = (unw_ww[pm] - unw_ref[pm]) / tau
        offset = np.round(np.median(diff))
        residual = np.abs(diff - offset)
        total += pm.sum()
        correct += (residual < 0.5).sum()
    return correct / total if total > 0 else 0.0


# --- Rust solver with Python costs ---
print("\nRunning Rust solver with Python costs...")
t0 = time.time()
unw_ext = ww._native.unwrap_linear_ext_costs(igram, mask, rust_costs)
t_ext = time.time() - t0
print(f"  Done in {t_ext:.1f}s, range [{np.nanmin(unw_ext):.1f}, {np.nanmax(unw_ext):.1f}]")

# --- Rust solver with Rust (Carballo) costs for comparison ---
print("\nRunning Rust solver with Rust (Carballo) costs...")
t0 = time.time()
unw_lin = ww._native.unwrap_linear(igram, corr_in, nlooks, mask)
t_lin = time.time() - t0
print(f"  Done in {t_lin:.1f}s, range [{np.nanmin(unw_lin):.1f}, {np.nanmax(unw_lin):.1f}]")

sc_ext = score(unw_ext, unw_prod, conn_prod, mask)
sc_lin = score(unw_lin, unw_prod, conn_prod, mask)
sc_py  = score(unw_py,  unw_prod, conn_prod, mask)

print(f"\n--- Per-component scores vs production (snaphu) ---")
print(f"  Python ww-orig (reference):              {sc_py*100:.2f}%")
print(f"  Rust solver + Python costs:              {sc_ext*100:.2f}%  ({t_ext:.1f}s)")
print(f"  Rust solver + Rust Carballo costs:       {sc_lin*100:.2f}%  ({t_lin:.1f}s)")

# Check cost distribution differences
print(f"\n--- Cost distribution comparison ---")
rust_carb = ww._native  # can't directly get costs, just show Python stats
print(f"  Python costs: p50={np.percentile(py_costs[py_costs>0], 50):.0f}  p90={np.percentile(py_costs[py_costs>0], 90):.0f}  p99={np.percentile(py_costs[py_costs>0], 99):.0f}  max={py_costs.max()}")
