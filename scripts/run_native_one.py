"""Run ONE native unwrapper (whirlwind or ww-orig) on ONE GUNW frame, timed,
printing the SAME per-comp-match line format as tophu_compare.py so the
all-unwrappers sweep can score every engine identically.

Usage: python scripts/run_native_one.py <h5path> {whirlwind|wworig}
Wrap in `/usr/bin/time -l` to also capture peak RSS.
"""

import sys
import re
import time

import numpy as np
import h5py

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase, percomp_match

h5path, engine = sys.argv[1], sys.argv[2]
frame = re.search(r"_([AD]_\d{3})_", h5path).group(1)
with h5py.File(h5path, "r") as h:
    pol, prod_unw, coh, prod_cc, mask_arr = gunw_layers(h)
mask = (
    water_only_mask(mask_arr, prod_unw.shape) & np.isfinite(prod_unw) & np.isfinite(coh)
)
# Canonical input (matches run_whirlwind_orig.py / the diag scripts): ZERO the
# masked/nodata phase before building the igram. prod_unw has NaN at nodata; if
# left in, exp(1j*NaN) -> NaN igram, and ww-orig's integration propagates NaN
# across the whole frame (whirlwind/phass tolerate it, ww-orig does not). Masked
# pixels are excluded from per-comp anyway, so this only fixes NaN propagation.
wrapped = np.where(mask, wrap_phase(prod_unw), 0.0).astype(np.float32)
ig = np.exp(1j * wrapped).astype(np.complex64)
coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)

if engine == "whirlwind":
    import whirlwind as ww  # default = single-tile linear (adaptive fallback)

    t0 = time.perf_counter()
    unw, cc = ww.unwrap(ig, coh_in, 16.0, mask)
    dt = time.perf_counter() - t0
    unw = np.asarray(unw, np.float32)
    cc = np.asarray(cc).astype(np.int64)
    ncc = int(cc.max()) if cc.size else 0
elif engine == "wworig":
    from whirlwind_orig._unwrap import unwrap as wwo  # Python reference; mask=INVALID

    t0 = time.perf_counter()
    unw = np.asarray(wwo(ig, coh_in, 16.0, mask=~mask), np.float32)
    dt = time.perf_counter() - t0
    ncc = 0  # ww-orig returns no conncomp
else:
    raise SystemExit(f"unknown engine {engine!r}")

valid = mask & np.isfinite(unw)
pc = percomp_match(unw, prod_unw, wrapped, prod_cc, valid)
print(
    f"{frame}: {engine:9s} {dt:6.1f}s  per-comp-match-vs-prod={pc * 100:5.1f}%  "
    f"ncc={ncc}  shape={ig.shape}",
    flush=True,
)
