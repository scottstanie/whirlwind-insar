"""Profile ww.unwrap peak memory on a SYNTHETIC NISAR-sized frame (no external
data needed). Memory peak is array-size driven, not data driven, so this is a
faithful A/B target for the flow_count / cost-drop memory changes.

Usage: python scripts/profile_ww_synth.py <m> <n> [nlooks]
Wrap in scripts/peak_rss_tree.py for tree peak RSS.
"""

import sys
import time

import numpy as np

import whirlwind as ww

m = int(sys.argv[1])
n = int(sys.argv[2])
nlooks = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0
coh = float(sys.argv[4]) if len(sys.argv) > 4 else 0.85

# Smooth multi-cycle truth ramp (many wraps -> real residue/flow work). Lower
# `coh` injects noise -> many residues -> a real, sustained MCF/Dijkstra phase
# (the phase the memory changes target), like a noisy NISAR frame.
yy, xx = np.mgrid[0:m, 0:n]
truth = (2 * np.pi * (xx / 40.0 + yy / 55.0)).astype(np.float32)
gamma = np.full((m, n), coh, np.float32)
igram, corr = ww.simulate_ifg(truth, gamma, int(nlooks), 1234)

t0 = time.perf_counter()
unw, cc = ww.unwrap(igram, corr, nlooks=nlooks)
dt = time.perf_counter() - t0
print(
    f"synth {m}x{n} ({m * n / 1e6:.1f} Mpx): ww.unwrap {dt:.1f}s  "
    f"ncc={int(np.asarray(cc).max())}  finite%={np.isfinite(unw).mean() * 100:.1f}",
    flush=True,
)
