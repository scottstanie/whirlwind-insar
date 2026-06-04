"""NISAR-summary headline figure from the 4-way sweep results.csv: per-frame
per-component match (whirlwind vs ww-orig vs PHASS) + runtime. Highlights A_025.

Usage: python scripts/plot_nisar_summary.py [results.csv] [out.png]
"""
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final/results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final/nisar_summary.png"

pc = defaultdict(dict)   # frame -> engine -> percomp
rt = defaultdict(dict)   # frame -> engine -> runtime
with open(CSV) as f:
    for row in csv.DictReader(f):
        try:
            pc[row["frame"]][row["engine"]] = float(row["percomp"])
            rt[row["frame"]][row["engine"]] = float(row["runtime_s"])
        except (ValueError, KeyError):
            continue

frames = sorted(pc)
engines = [("whirlwind", "#1f77b4"), ("wworig", "#7f7f7f"), ("phass", "#d62728")]
labels = {"whirlwind": "whirlwind (default)", "wworig": "ww-orig (ref)", "phass": "PHASS"}
x = np.arange(len(frames))
w = 0.26

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[2, 1])

for k, (eng, color) in enumerate(engines):
    vals = [pc[fr].get(eng, np.nan) for fr in frames]
    ax0.bar(x + (k - 1) * w, vals, w, label=labels[eng], color=color)
ax0.set_ylabel("per-component match vs SNAPHU (%)")
ax0.set_ylim(0, 105)
ax0.set_xticks(x); ax0.set_xticklabels(frames, rotation=45, ha="right")
ax0.axhline(100, color="k", lw=0.5, ls=":")
ax0.legend(loc="lower left", ncol=3)
ax0.set_title("Whirlwind 2D unwrapping — NISAR GUNW 13-frame comparison (per-component match vs the production SNAPHU unwrap)")
if "A_025" in frames:
    ax0.annotate("A_025 river:\nbridge fixes 58→100%", xy=(frames.index("A_025"), 100),
                 xytext=(frames.index("A_025"), 40), ha="center", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#1f77b4"))

for k, (eng, color) in enumerate(engines):
    vals = [rt[fr].get(eng, np.nan) for fr in frames]
    ax1.bar(x + (k - 1) * w, vals, w, label=labels[eng], color=color)
ax1.axhline(590, color="k", lw=1, ls="--")
ax1.text(0.2, 600, "single-tile SNAPHU ≈ 590 s", fontsize=8, va="bottom")
ax1.set_ylabel("runtime (s, log)")
ax1.set_yscale("log")
ax1.set_xticks(x); ax1.set_xticklabels(frames, rotation=45, ha="right")

fig.tight_layout()
fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"summary figure -> {OUT}", flush=True)
print("\nper-comp table (%):")
print(f"{'frame':8s} {'whirl':>7s} {'wworig':>7s} {'phass':>7s}")
for fr in frames:
    print(f"{fr:8s} {pc[fr].get('whirlwind', float('nan')):7.2f} {pc[fr].get('wworig', float('nan')):7.2f} {pc[fr].get('phass', float('nan')):7.2f}")
