"""Demonstrate the conncomp_coh_floor knob: it fixes the noisy "percolation"
(label-1 leakage + spurious islands) that cost_threshold alone can't, because a
coherence floor cuts regardless of the local gradient. Uses cached (cc, coh, mask)
so it's a fast pure-numpy post-mask — no heavy unwrap.

Usage: python scripts/conncomp_demo.py [FRAME=A_030] [floor=0.3]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/bridge_cache"
frame = sys.argv[1] if len(sys.argv) > 1 else "A_030"
floor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

d = np.load(f"{CACHE}/{frame}.npz")
cc = d["cc"].astype(np.int64)
coh = d["coh"].astype(np.float32)
mask = d["mask"]

cc_floor = cc.copy()
cc_floor[np.clip(np.nan_to_num(coh), 0, 1) < floor] = 0  # the conncomp_coh_floor post-mask
dropped = int(((cc > 0) & (cc_floor == 0)).sum())

# Pick the noisiest window (lowest mean coherence among valid 400x400 tiles).
m, n = coh.shape
W = 400
best, bij = 1e9, (0, 0)
for i in range(0, m - W, 200):
    for j in range(0, n - W, 200):
        sub = mask[i:i + W, j:j + W]
        if sub.sum() < W * W * 0.5:
            continue
        mc = coh[i:i + W, j:j + W][sub].mean()
        if mc < best:
            best, bij = mc, (i, j)
i0, j0 = bij


def disp(a, m_):
    return np.where(m_, a.astype(float), np.nan)


# Show LABELED (cc>0, green) vs BACKGROUND (cc==0, red) so the percolation cleanup
# is obvious; the coherence panel shows where the noise is.
lab0 = disp((cc > 0).astype(float), mask)
lab1 = disp((cc_floor > 0).astype(float), mask)
fig, ax = plt.subplots(2, 3, figsize=(16, 10))
sl = (slice(i0, i0 + W), slice(j0, j0 + W))
panels = [
    (ax[0, 0], lab0, "labeled cc>0 (default)", "RdYlGn", (0, 1)),
    (ax[0, 1], lab1, f"coh_floor={floor}: cc>0 (dropped {dropped/max(mask.sum(),1)*100:.1f}%)", "RdYlGn", (0, 1)),
    (ax[0, 2], disp(coh, mask), "coherence", "gray", (0, 1)),
    (ax[1, 0], lab0[sl], "default (zoom: noisiest window)", "RdYlGn", (0, 1)),
    (ax[1, 1], lab1[sl], f"coh_floor={floor} (zoom)", "RdYlGn", (0, 1)),
    (ax[1, 2], disp(coh, mask)[sl], "coherence (zoom)", "gray", (0, 1)),
]
zpanels = []
for a, arr, title, cmap, vmm in panels + zpanels:
    im = a.imshow(arr, cmap=cmap, vmin=(vmm[0] if vmm else None), vmax=(vmm[1] if vmm else None))
    a.set_title(title, fontsize=10); a.axis("off")
fig.suptitle(f"{frame}: conncomp_coh_floor cleans noisy percolation (cost_threshold can't)", fontsize=13)
fig.tight_layout()
out = f"{CACHE}/{frame}_conncomp_demo.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"{frame}: coh_floor={floor} dropped {dropped} px; noisiest window at {bij} (mean coh {best:.2f}) -> {out}", flush=True)
