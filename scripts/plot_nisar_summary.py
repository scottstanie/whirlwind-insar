"""NISAR-summary headline figure from the sweep results.csv: per-frame
per-component match, runtime, and peak memory for whirlwind vs SNAPHU (1-tile and
9x9), PHASS, and ICU. Highlights A_025.

Usage: python scripts/plot_nisar_summary.py [results.csv] [out.png]
"""

import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final/results.csv"
)
OUT = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final/nisar_summary.png"
)
# The 3x3-tiled SNAPHU runtime+memory come from a SEPARATE tree-sampled sweep
# (scripts/sweep_snaphu_tiled_mem.sh): /usr/bin/time in the main sweep undercounts
# SNAPHU's forked tile workers, so results.csv's `snaphu9x9` peak RSS (~3 GB) is a
# per-process undercount. The tree-summed numbers (~6-13 GB, the figures the doc
# quotes) live in this CSV under engine `snaphu_par`.
TILED_CSV = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "/Users/staniewi/Documents/Learning/snaphu_3x3_recheck/snaphu_tiled.csv"
)

pc = defaultdict(dict)  # frame -> engine -> percomp
rt = defaultdict(dict)  # frame -> engine -> runtime
mem = defaultdict(dict)  # frame -> engine -> peak RSS (GB)
with open(CSV) as f:
    for row in csv.DictReader(f):
        try:
            pc[row["frame"]][row["engine"]] = float(row["percomp"])
            rt[row["frame"]][row["engine"]] = float(row["runtime_s"])
            mem[row["frame"]][row["engine"]] = float(row["peak_rss_bytes"]) / 1e9
        except (ValueError, KeyError):
            continue

# Inject the tree-summed 3x3-tiled SNAPHU as engine `snaphu3x3`: runtime + peak
# memory from the tree-sampled recheck, per-comp carried over from the in-sweep
# tiled run (`snaphu9x9`, falling back to single-tile `snaphu`) since the tiled
# result self-matches the production SNAPHU unwrap regardless of how it's measured.
try:
    tiled_file = open(TILED_CSV)
except FileNotFoundError:
    print(f"WARNING: tiled recheck CSV not found ({TILED_CSV}); "
          "3x3 tiled SNAPHU will be absent from the figure.", flush=True)
else:
    with tiled_file as f:
        for row in csv.DictReader(f):
            if row.get("engine") != "snaphu_par":
                continue
            fr = row["frame"]
            try:
                rt[fr]["snaphu3x3"] = float(row["runtime_s"])
                mem[fr]["snaphu3x3"] = float(row["tree_peak_bytes"]) / 1e9
            except (ValueError, KeyError):
                continue
            src = pc[fr].get("snaphu9x9", pc[fr].get("snaphu"))
            if src is not None:
                pc[fr]["snaphu3x3"] = src

frames = sorted(pc)
# Recognizable / published engines on the headline figure. ww-orig stays in
# results.csv (for readers who know it) but is off the figure by default - set
# WW_ORIG=1 to include it.
import os as _os

engines = [
    ("whirlwind", "#1f77b4"),
    ("snaphu", "#2ca02c"),  # SNAPHU single-tile
    ("snaphu3x3", "#98df8a"),  # SNAPHU 3x3 tiles + reoptimize (9 tiles, all parallel)
    ("phass", "#d62728"),
    ("icu", "#9467bd"),  # isce2 mroipac ICU
]
if _os.environ.get("WW_ORIG") == "1":
    engines.append(("wworig", "#7f7f7f"))
labels = {
    "whirlwind": "whirlwind (default)",
    "snaphu": "SNAPHU (1 tile)",
    "snaphu3x3": "SNAPHU (3x3+reopt)",
    "phass": "PHASS",
    "icu": "ICU (isce2)",
    "wworig": "ww-orig",
}
# Drop engines with no data yet.
engines = [(e, c) for (e, c) in engines if any(e in pc[fr] for fr in frames)]
ne = len(engines)
x = np.arange(len(frames))
w = 0.8 / max(ne, 1)

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 11), height_ratios=[2, 1.2, 1.2])
off = (np.arange(ne) - (ne - 1) / 2) * w

# Panel 0 - per-component match vs the production SNAPHU unwrap.
for k, (eng, color) in enumerate(engines):
    ax0.bar(
        x + off[k],
        [pc[fr].get(eng, np.nan) for fr in frames],
        w,
        label=labels[eng],
        color=color,
    )
ax0.set_ylabel("per-component match (%)")
ax0.set_ylim(0, 105)
ax0.set_xticks(x)
ax0.set_xticklabels([])
ax0.axhline(100, color="k", lw=0.5, ls=":")
ax0.legend(loc="lower left", ncol=len(engines), fontsize=9)
ax0.set_title(
    "Whirlwind 2D unwrapping - NISAR GUNW 13-frame comparison (quality / runtime / peak memory)"
)

# Panel 1 - runtime (log).
for k, (eng, color) in enumerate(engines):
    ax1.bar(x + off[k], [rt[fr].get(eng, np.nan) for fr in frames], w, color=color)
ax1.set_ylabel("runtime (s, log)")
ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels([])

# Panel 2 - peak resident memory (GB).
for k, (eng, color) in enumerate(engines):
    ax2.bar(x + off[k], [mem[fr].get(eng, np.nan) for fr in frames], w, color=color)
ax2.set_ylabel("peak RSS (GB)")
ax2.set_xticks(x)
ax2.set_xticklabels(frames, rotation=45, ha="right")

fig.tight_layout()
fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"summary figure -> {OUT}", flush=True)

# Printed table: every engine, all three metrics.
hdr = " | ".join(f"{e:>10s}" for e, _ in engines)
for metric, d, fmt in [
    ("per-comp %", pc, "{:10.1f}"),
    ("runtime s", rt, "{:10.1f}"),
    ("peak GB", mem, "{:10.2f}"),
]:
    print(f"\n--- {metric} ---\n{'frame':8s} | {hdr}")
    for fr in frames:
        print(
            f"{fr:8s} | "
            + " | ".join(fmt.format(d[fr].get(e, float("nan"))) for e, _ in engines)
        )
