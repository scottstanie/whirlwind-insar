#!/usr/bin/env python3
"""Compare two perf_frame_runner sweep outputs (baseline vs optimized).

Usage: python perf_sweep_compare.py baseline.txt optimized.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAT = re.compile(
    r"frame=(\S+) shape=\(\d+, \d+\) runtime=([\d.]+)s peak_rss=([\d.]+)GiB "
    r"\(pre-solve rss=[\d.]+GiB\) sha1\(valid\)=([0-9a-f]+)"
)


def parse(path: Path) -> dict[str, tuple[float, float, str]]:
    out = {}
    for line in path.read_text().splitlines():
        m = PAT.search(line)
        if m:
            frame = m.group(1).replace("NISAR_L2_PR_GUNW_003_00", "")
            out[frame] = (float(m.group(2)), float(m.group(3)), m.group(4))
    return out


base = parse(Path(sys.argv[1]))
opt = parse(Path(sys.argv[2]))
assert base.keys() == opt.keys(), (base.keys(), opt.keys())

print(f"{'frame':24s} {'time (s)':>16s} {'peak RSS (GiB)':>18s}  parity")
tsum_b = tsum_o = 0.0
n_match = 0
for k in sorted(base):
    tb, rb, hb = base[k]
    to, ro, ho = opt[k]
    tsum_b += tb
    tsum_o += to
    match = "IDENTICAL" if hb == ho else f"DIFFERS ({hb} vs {ho})"
    n_match += hb == ho
    print(f"{k:24s} {tb:6.2f} -> {to:6.2f} {rb:8.2f} -> {ro:5.2f}   {match}")
rss_b = [v[1] for v in base.values()]
rss_o = [v[1] for v in opt.values()]
print(
    f"\ntotals: runtime {tsum_b:.1f}s -> {tsum_o:.1f}s "
    f"({100 * (tsum_o / tsum_b - 1):+.1f}%), "
    f"mean peak RSS {sum(rss_b) / len(rss_b):.2f} -> {sum(rss_o) / len(rss_o):.2f} GiB "
    f"({100 * (sum(rss_o) / sum(rss_b) - 1):+.1f}%), "
    f"checksums identical {n_match}/{len(base)}"
)
