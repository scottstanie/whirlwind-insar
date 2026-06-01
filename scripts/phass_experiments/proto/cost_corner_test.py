"""Corner-bug test: clean 6pi steep diagonal ramp (512), tiled 256, under the
per-tile solver selected by WHIRLWIND_TILE_SOLVER. linear should fail (corner
boundary-stacking ~12 rad err); reuse/convex should be ~0.
"""
import os
import numpy as np
import whirlwind as ww

S = os.environ.get("WHIRLWIND_TILE_SOLVER", "linear")
y, x = np.ogrid[-3:3:512j, -3:3:512j]
phase = (np.pi * (x + y)).astype(np.float32)
ig = np.exp(1j * phase).astype(np.complex64)
coh = np.ones((512, 512), np.float32) * 0.999
u = ww.unwrap(ig, coh, 1.0, tile_size=256, tile_overlap=32)
o = u[0, 0] - phase[0, 0]
err = float(np.nanmax(np.abs((u - o) - phase)))
print(f"  {S:7s} tiled256  max|err|={err:.3f} rad  {'PASS' if err < 0.5 else 'FAIL'}", flush=True)
