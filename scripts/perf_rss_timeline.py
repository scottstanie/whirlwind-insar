#!/usr/bin/env python3
"""Run unwrap_linear on a prepared npz while polling RSS from a thread.

Prints a t,rss_mb timeline so the peak can be attributed to a solver phase
(cross-reference the WHIRLWIND_DEBUG stderr timestamps).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import whirlwind as ww

npz = Path(sys.argv[1])
d = np.load(npz)
ig_complex = np.ascontiguousarray(d["ig_complex"])
coh = np.ascontiguousarray(d["coh"])
mask = np.ascontiguousarray(d["mask"])
del d

proc = psutil.Process()
samples: list[tuple[float, float]] = []
stop = threading.Event()


def poll() -> None:
    t0 = time.perf_counter()
    while not stop.is_set():
        samples.append((time.perf_counter() - t0, proc.memory_info().rss / 2**20))
        time.sleep(0.2)


th = threading.Thread(target=poll, daemon=True)
th.start()
t0 = time.perf_counter()
unw = ww._native.unwrap_linear(ig_complex, coh, 16.0, mask)
dt = time.perf_counter() - t0
stop.set()
th.join()

peak_t, peak_rss = max(samples, key=lambda s: s[1])
print(f"runtime={dt:.2f}s peak_rss={peak_rss:.0f}MB at t={peak_t:.1f}s")
for t, r in samples[:: max(1, len(samples) // 60)]:
    print(f"  t={t:6.1f}s rss={r:7.0f}MB")
