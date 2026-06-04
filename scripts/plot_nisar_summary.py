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
mem = defaultdict(dict)  # frame -> engine -> peak RSS (GB)
with open(CSV) as f:
    for row in csv.DictReader(f):
        try:
            pc[row["frame"]][row["engine"]] = float(row["percomp"])
            rt[row["frame"]][row["engine"]] = float(row["runtime_s"])
            mem[row["frame"]][row["engine"]] = float(row["peak_rss_bytes"]) / 1e9
        except (ValueError, KeyError):
            continue

frames = sorted(pc)
# Recognizable / published engines on the headline figure. ww-orig stays in
# results.csv (for readers who know it) but is off the figure by default — set
# WW_ORIG=1 to include it.
import os as _os
engines = [
    ("whirlwind", "#1f77b4"),
    ("snaphu", "#2ca02c"),       # SNAPHU single-tile
    ("snaphu9x9", "#98df8a"),    # SNAPHU 9x9 tiles + reoptimize (production path)
    ("phass", "#d62728"),
    ("icu", "#9467bd"),          # isce2 mroipac ICU
]
if _os.environ.get("WW_ORIG") == "1":
    engines.append(("wworig", "#7f7f7f"))
labels = {"whirlwind": "whirlwind (default)", "snaphu": "SNAPHU (1 tile)",
          "snaphu9x9": "SNAPHU (9×9+reopt)", "phass": "PHASS", "icu": "ICU (isce2)",
          "wworig": "ww-orig"}
# Drop engines with no data yet.
engines = [(e, c) for (e, c) in engines if any(e in pc[fr] for fr in frames)]
ne = len(engines)
x = np.arange(len(frames))
w = 0.8 / max(ne, 1)

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 11), height_ratios=[2, 1.2, 1.2])
off = (np.arange(ne) - (ne - 1) / 2) * w

# Panel 0 — per-component match vs the production SNAPHU unwrap.
for k, (eng, color) in enumerate(engines):
    ax0.bar(x + off[k], [pc[fr].get(eng, np.nan) for fr in frames], w, label=labels[eng], color=color)
ax0.set_ylabel("per-component match (%)")
ax0.set_ylim(0, 105)
ax0.set_xticks(x); ax0.set_xticklabels([])
ax0.axhline(100, color="k", lw=0.5, ls=":")
ax0.legend(loc="lower left", ncol=len(engines), fontsize=9)
ax0.set_title("Whirlwind 2D unwrapping — NISAR GUNW 13-frame comparison (quality / runtime / peak memory)")
if "A_025" in frames:
    ax0.annotate("A_025 river:\nbridge fixes 58→100%", xy=(frames.index("A_025"), 100),
                 xytext=(frames.index("A_025"), 38), ha="center", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#1f77b4"))

# Panel 1 — runtime (log).
for k, (eng, color) in enumerate(engines):
    ax1.bar(x + off[k], [rt[fr].get(eng, np.nan) for fr in frames], w, color=color)
ax1.set_ylabel("runtime (s, log)")
ax1.set_yscale("log")
ax1.set_xticks(x); ax1.set_xticklabels([])

# Panel 2 — peak resident memory (GB).
for k, (eng, color) in enumerate(engines):
    ax2.bar(x + off[k], [mem[fr].get(eng, np.nan) for fr in frames], w, color=color)
ax2.set_ylabel("peak RSS (GB)")
ax2.set_xticks(x); ax2.set_xticklabels(frames, rotation=45, ha="right")

fig.tight_layout()
fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"summary figure -> {OUT}", flush=True)

# Printed table: every engine, all three metrics.
hdr = " | ".join(f"{e:>10s}" for e, _ in engines)
for metric, d, fmt in [("per-comp %", pc, "{:10.1f}"), ("runtime s", rt, "{:10.1f}"), ("peak GB", mem, "{:10.2f}")]:
    print(f"\n--- {metric} ---\n{'frame':8s} | {hdr}")
    for fr in frames:
        print(f"{fr:8s} | " + " | ".join(fmt.format(d[fr].get(e, float('nan'))) for e, _ in engines))
