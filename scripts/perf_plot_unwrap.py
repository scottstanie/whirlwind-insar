#!/usr/bin/env python3
"""Plot wrapped input vs unwrapped output for a prepared npz frame.

Usage: python perf_plot_unwrap.py <frame.npz> <out.png>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import whirlwind as ww

npz, out_png = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(npz)
ig_complex = np.ascontiguousarray(d["ig_complex"])
coh = np.ascontiguousarray(d["coh"])
mask = np.ascontiguousarray(d["mask"])

unw = np.asarray(ww._native.unwrap_linear(ig_complex, coh, 16.0, mask))

fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
wrapped = np.where(mask, np.angle(ig_complex), np.nan)
im0 = axes[0].imshow(wrapped, cmap="twilight", interpolation="nearest")
axes[0].set_title("wrapped input")
fig.colorbar(im0, ax=axes[0], shrink=0.8, label="rad")
im1 = axes[1].imshow(unw, cmap="RdBu_r", interpolation="nearest")
axes[1].set_title("whirlwind unwrap_linear (optimized build)")
fig.colorbar(im1, ax=axes[1], shrink=0.8, label="rad")
fig.suptitle(npz.stem)
fig.savefig(out_png, dpi=110)
print(f"wrote {out_png.resolve()}")
