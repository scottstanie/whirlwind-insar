"""Show the spiral interpolator on a river-divided frame: wrapped phase before
vs after interpolating pixels with coherence < cutoff (default 0.1). The
decorrelated banks/river get filled with a smoothed phase from coherent
neighbors, while the amplitude (here unit) is preserved.

Usage: python scripts/plot_interp_river.py [FRAMES...] [--cutoff 0.1]
"""

import sys
import glob

import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase
import whirlwind as ww

args = [a for a in sys.argv[1:] if not a.startswith("--")]
cutoff = 0.1
if "--cutoff" in sys.argv:
    cutoff = float(sys.argv[sys.argv.index("--cutoff") + 1])
frames = args or ["005_A_025", "005_A_016"]
OUT = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final"

for frame in frames:
    h5 = glob.glob(
        f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5"
    )[0]
    with h5py.File(h5, "r") as h:
        pol, prod, coh, pcc, marr = gunw_layers(h)
    mask = water_only_mask(marr, prod.shape) & np.isfinite(prod) & np.isfinite(coh)
    wrapped = np.where(mask, wrap_phase(prod), 0.0).astype(np.float32)
    ig = np.exp(1j * wrapped).astype(np.complex64)
    ig[~mask] = 0  # masked/nodata -> interpolator skips
    weights = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)

    interp = np.asarray(ww.interpolate(ig, weights, cutoff, 20, 51, 0, 0.75))
    n_interp = int((mask & (weights < cutoff)).sum())

    before = np.where(mask, np.angle(ig), np.nan)
    after = np.where(mask, np.angle(interp), np.nan)
    lowcoh = np.where(mask, weights < cutoff, np.nan)

    fig, ax = plt.subplots(1, 3, figsize=(17, 6))
    for a, arr, title, cmap, vmm in [
        (ax[0], before, "wrapped phase (before)", "twilight", (-np.pi, np.pi)),
        (
            ax[1],
            after,
            f"after interp (coh<{cutoff}: {n_interp/max(mask.sum(),1)*100:.0f}% of valid)",
            "twilight",
            (-np.pi, np.pi),
        ),
        (ax[2], lowcoh, f"interpolated pixels (coh<{cutoff})", "Reds", (0, 1)),
    ]:
        im = a.imshow(arr, cmap=cmap, vmin=vmm[0], vmax=vmm[1])
        a.set_title(title, fontsize=11)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{frame}: spiral PS interpolation of low-coherence pixels", fontsize=13
    )
    fig.tight_layout()
    out = f"{OUT}/{frame}_interp.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"{frame}: interpolated {n_interp} px (coh<{cutoff}) -> {out}", flush=True)
