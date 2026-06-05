"""
Compare Rust unwrap_linear (capacity-1, maxiter=8) vs unwrap_reuse on D_077 full frame.
Goal: unwrap_linear should match Python ww-orig at ~99.49%.
"""

import numpy as np
import time
import whirlwind as ww

WWORIG_NPZ = (
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_ref/D_077_wworig.npz"
)
OUTDIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_ref"

tau = 2 * np.pi

# --- reconstruct inputs from saved npz (same as run_whirlwind_orig.py did) ---
d = np.load(WWORIG_NPZ)
unw_py = d["unw"]
unw_prod = d["prod_unw"]
conn_prod = d["prod_cc"].astype(np.int32)
mask = d["mask"].astype(bool)
corr = d["coh"].astype(np.float32)

# Reconstruct igram exactly as run_whirlwind_orig.py did
wrapped = np.where(mask, ((unw_prod + np.pi) % tau) - np.pi, 0.0).astype(np.float32)
igram = np.exp(1j * wrapped).astype(np.complex64)
corr_in = np.where(mask, np.clip(np.nan_to_num(corr), 0, 1), 0.0).astype(np.float32)

nlooks = 16.0  # matches run_whirlwind_orig.py
print(f"Loaded from {WWORIG_NPZ}")
print(f"Shape: {igram.shape}, mask {mask.mean()*100:.1f}% valid, nlooks={nlooks}")
print(f"Python ww-orig range: [{unw_py.min():.1f}, {unw_py.max():.1f}] rad")


# --- scoring function (per-component) ---
def score(unw_ww, unw_ref, conn_ref, mask):
    """Per-component aligned score: fraction of valid pixels where round((ww-ref)/2pi)==0."""
    comps = np.unique(conn_ref[conn_ref > 0])
    total = 0
    correct = 0
    for c in comps:
        pm = (conn_ref == c) & mask
        if not pm.any():
            continue
        diff = (unw_ww[pm] - unw_ref[pm]) / tau
        offset = np.round(np.median(diff))
        residual = np.abs(diff - offset)
        n = pm.sum()
        nc = (residual < 0.5).sum()
        total += n
        correct += nc
    return correct / total if total > 0 else 0.0


# --- run Rust linear (now uses parity costs: 100x scale, both-invalid mask) ---
print("\nRunning unwrap_linear (parity costs, maxiter=8)...")
t0 = time.time()
unw_linear = ww._native.unwrap_linear(igram, corr_in, float(nlooks), mask)
t_linear = time.time() - t0
print(
    f"  Done in {t_linear:.1f}s, range [{np.nanmin(unw_linear):.1f}, {np.nanmax(unw_linear):.1f}]"
)

# --- score ---
sc_linear = score(unw_linear, unw_prod, conn_prod, mask)
sc_py = score(unw_py, unw_prod, conn_prod, mask)

print(f"\n--- Per-component scores vs production (snaphu) ---")
print(f"  Python ww-orig (capacity-1, maxiter=8):          {sc_py*100:.2f}%")
print(
    f"  Rust unwrap_linear (parity costs, maxiter=8):    {sc_linear*100:.2f}%  ({t_linear:.1f}s)"
)

# --- save linear result ---
np.savez_compressed(
    f"{OUTDIR}/D_077_rust_linear.npz",
    unw=unw_linear,
    score=sc_linear,
)
print(f"\nSaved to {OUTDIR}/D_077_rust_linear.npz")

# --- comparison plot ---
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Re-reference to same pixel before plotting.
largest_comp = max(
    np.unique(conn_prod[conn_prod > 0]), key=lambda c: (conn_prod == c).sum()
)
ref_mask = (conn_prod == largest_comp) & mask
ref_idx = tuple(int(x) for x in np.argwhere(ref_mask)[len(np.argwhere(ref_mask)) // 2])


def reref(data, ref, ref_idx):
    offset = np.round((data[ref_idx] - ref[ref_idx]) / tau) * tau
    return data - offset


unw_linear_ref = reref(unw_linear, unw_prod, ref_idx)
unw_py_ref = reref(unw_py, unw_prod, ref_idx)

vmin, vmax = np.nanpercentile(unw_prod[mask], [1, 99])

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle(
    f"D_077 single-tile comparison (parity costs)\n"
    f"Rust linear={sc_linear*100:.2f}%  Python ww-orig={sc_py*100:.2f}%",
    fontsize=11,
)

for ax, data, title in zip(
    axes,
    [unw_prod, unw_linear_ref, unw_py_ref],
    [
        "Production (snaphu)",
        f"Rust linear parity ({sc_linear*100:.1f}%)",
        f"Python ww-orig ({sc_py*100:.1f}%)",
    ],
):
    d_show = np.where(mask, data, np.nan)
    im = ax.imshow(d_show, vmin=vmin, vmax=vmax, cmap="RdYlBu_r", aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

fig.colorbar(im, ax=axes[-1], label="phase (rad)", shrink=0.7)
plt.tight_layout()
out = f"{OUTDIR}/D_077_linear_vs_reuse_vs_py.png"
plt.savefig(out, dpi=120)
print(f"Plot: {out}")
