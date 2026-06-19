"""Internal-only: runtime of the current Rust whirlwind vs the original Python
prototype ("wworig"), per NISAR GUNW frame, from the sweep results.csv.

This is the single place the original prototype is compared; it is kept out of
the public docs because end users only care about the shipped library. The
quality numbers match (same 2pi levels); the point here is the speedup.

Usage: python scripts/plot_speed_vs_original.py [results.csv] [out.png]
"""

import csv
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "docs/nisar_4way_results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "experiments/figures/speed_vs_original.png"

rt = defaultdict(dict)  # frame -> engine -> runtime_s
with open(CSV) as f:
    for row in csv.DictReader(f):
        try:
            rt[row["frame"]][row["engine"]] = float(row["runtime_s"])
        except (ValueError, KeyError):
            continue

frames = [fr for fr in sorted(rt) if "whirlwind" in rt[fr] and "wworig" in rt[fr]]
ww = np.array([rt[fr]["whirlwind"] for fr in frames])
orig = np.array([rt[fr]["wworig"] for fr in frames])
speedup = orig / ww

x = np.arange(len(frames))
w = 0.4
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w / 2, ww, w, label="whirlwind (Rust)", color="#1f77b4")
ax.bar(x + w / 2, orig, w, label="original prototype (Python)", color="#7f7f7f")
ax.set_ylabel("runtime (s)")
ax.set_xticks(x)
ax.set_xticklabels(frames, rotation=45, ha="right")
ax.set_title(
    f"Runtime vs the original Python prototype "
    f"(median {np.median(speedup):.1f}x faster, range {speedup.min():.1f}-{speedup.max():.1f}x)"
)
ax.legend()
fig.tight_layout()
fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"speed-vs-original figure -> {OUT}")
print(
    f"median speedup {np.median(speedup):.1f}x, range {speedup.min():.1f}-{speedup.max():.1f}x"
)
