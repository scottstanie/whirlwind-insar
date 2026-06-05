"""Fresh tiling attempt on the VALIDATED single-tile-linear kernel, stitched with
the bridging integer-gauge idea (tiles share overlaps -> data-supported offsets).

Goal (the tiling-revisit bar): memory-bounded (never build the whole-image graph)
yet NEARLY ALWAYS MATCH the single-tile result. Each tile is unwrapped
independently with ww.unwrap (single-tile linear, bridge off) on its sub-array;
adjacent tiles are reconciled by the integer 2π offset estimated over their
overlap (round of the median phase difference), solved globally on a
max-overlap-confidence spanning tree (Kruskal + BFS propagate). Composite is
center-priority. Many SMALL tile solves -> low peak memory, sequential.

Usage: python scripts/proto_tile_linear.py [FRAME] [TILE=1024] [OVERLAP=128]
"""

import glob
import sys
import time

import numpy as np
import h5py

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase, percomp_match
import whirlwind as ww

tau = 2 * np.pi
frame = sys.argv[1] if len(sys.argv) > 1 else "A_030"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
O = int(sys.argv[3]) if len(sys.argv) > 3 else 128

h5 = glob.glob(
    f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5"
)[0]
with h5py.File(h5, "r") as h:
    pol, prod, coh, pcc, marr = gunw_layers(h)
mask = water_only_mask(marr, prod.shape) & np.isfinite(prod) & np.isfinite(coh)
wr = np.where(mask, wrap_phase(prod), 0.0).astype(np.float32)
ig = np.exp(1j * wr).astype(np.complex64)
ci = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)
m, n = mask.shape
stride = T - O
i0s = list(range(0, max(1, m - O), stride))
j0s = list(range(0, max(1, n - O), stride))


def tile_box(i0, j0):
    return i0, min(i0 + T, m), j0, min(j0 + T, n)


# 1. Unwrap each tile independently (small, memory-bounded).
t0 = time.perf_counter()
U = {}  # (ti,tj) -> unwrapped tile (global-arbitrary gauge), NaN where invalid
boxes = {}
for ti, i0 in enumerate(i0s):
    for tj, j0 in enumerate(j0s):
        a, b, c, d = tile_box(i0, j0)
        mt = mask[a:b, c:d]
        if mt.sum() < 50:
            continue
        ut, _ = ww.unwrap(
            np.ascontiguousarray(ig[a:b, c:d]),
            np.ascontiguousarray(ci[a:b, c:d]),
            16.0,
            np.ascontiguousarray(mt),
            bridge=False,
        )
        ut = np.asarray(ut, np.float32)
        ut = np.where(mt, ut, np.nan)
        U[(ti, tj)] = ut
        boxes[(ti, tj)] = (a, b, c, d)
t_unw = time.perf_counter() - t0

# 2. Global x8 coarse anchor (small, memory-bounded - never the whole-image graph).
L = 8


def block_mean(arr):
    mm, nn = arr.shape[0] // L, arr.shape[1] // L
    return arr[: mm * L, : nn * L].reshape(mm, L, nn, L).mean(axis=(1, 3))


cig = block_mean(np.exp(1j * np.angle(ig)).astype(np.complex64)).astype(np.complex64)
ccoh = block_mean(ci).astype(np.float32)
cmask = np.ascontiguousarray(block_mean(mask.astype(np.float32)) > 0.4)
cunw, _ = ww.unwrap(cig, ccoh, 16.0 * L * L, cmask, bridge=False)
cunw = np.asarray(cunw, np.float32)
anchor = np.kron(cunw, np.ones((L, L), np.float32))
anchor = np.pad(
    anchor,
    ((0, max(0, m - anchor.shape[0])), (0, max(0, n - anchor.shape[1]))),
    mode="edge",
)[:m, :n]

# 3+4. Composite: snap EACH (tile ∩ integration-component) piece to the global
# anchor (the shared reference) - the bridging idea at tile granularity, so no
# tile-pair MST is needed and multi-component tiles are handled per piece.
canvas = np.full((m, n), np.nan, np.float32)
best = np.full((m, n), np.inf, np.float32)
for k, ut in U.items():
    a, b, c, d = boxes[k]
    mt = mask[a:b, c:d]
    lab, nl = ww.label_components(np.ascontiguousarray(mt))
    asub = anchor[a:b, c:d]
    val = ut.copy()
    for l in range(1, nl + 1):
        piece = (lab == l) & np.isfinite(ut) & np.isfinite(asub)
        if piece.sum() < 30:
            continue
        s = np.rint(np.median((asub[piece] - ut[piece]) / tau))
        val[lab == l] = ut[lab == l] + tau * s
    th, tw = b - a, d - c
    pr = np.maximum(
        np.abs(np.arange(th) - th / 2)[:, None], np.abs(np.arange(tw) - tw / 2)[None, :]
    )
    sub_best = best[a:b, c:d]
    take = np.isfinite(val) & (pr < sub_best)
    canvas[a:b, c:d] = np.where(take, val, canvas[a:b, c:d])
    best[a:b, c:d] = np.where(take, pr, sub_best)

# Score vs production + agreement vs single-tile.
us, _ = ww.unwrap(ig, ci, 16.0, mask, bridge=False)
us = np.asarray(us, np.float32)
v_tile = mask & np.isfinite(canvas)
pc_tile = percomp_match(canvas, prod, wr, pcc, v_tile)
pc_single = percomp_match(us, prod, wr, pcc, mask & np.isfinite(us))
ambd = np.round((canvas - us) / tau)
ambd -= np.median(ambd[v_tile])
agree = (np.abs(ambd)[v_tile] < 0.5).mean()
print(
    f"{frame} T={T} O={O}: {len(U)} tiles, unwrap {t_unw:.1f}s  "
    f"tiled per-comp={pc_tile*100:.1f}%  single={pc_single*100:.1f}%  agree-vs-single={agree*100:.1f}%",
    flush=True,
)
