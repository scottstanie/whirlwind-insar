"""CRLB-path eval on the Mexico City (Capella) dolphin scene — baseline for the
#35 anchor/cascade refactor.

Runs `ww.unwrap_crlb` on one phase-linked IG (variance = crlb_a + crlb_b),
reports K-match vs dolphin's spurt reference + the coherent-cut rate (tears
through high-coherence terrain = the block-2π artifacts the anchor/cascade
fix). Save BEFORE the refactor and AFTER; the script compares two labels.

    env -u CONDA_PREFIX uv run --with rasterio python scripts/diag_crlb_eval.py before
    # refactor + maturin develop --release
    env -u CONDA_PREFIX uv run --with rasterio python scripts/diag_crlb_eval.py after
    env -u CONDA_PREFIX uv run --with rasterio python scripts/diag_crlb_eval.py compare
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

D = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/mexico_city/e2e_output/dolphin")
IGDIR = D / "interferograms"
UNWDIR = D / "unwrapped"
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/crlb_eval")
STEM = "20240626_20240629"  # first IG; variance = crlb_<a> + crlb_<b>
COH_CUT_THR = 0.7
TAU = np.float32(2 * np.pi)


def _read(path, dt):
    import rasterio
    with rasterio.open(path) as s:
        return s.read(1).astype(dt)


def coherent_cut_rate(unw, wrapped, coh, valid):
    nvalid = int(valid.sum())
    tot = 0.0
    for ax in (1, 0):  # right edges, then down edges
        a = unw
        w = wrapped
        if ax == 1:
            cE = np.minimum(coh[:, :-1], coh[:, 1:]); vE = valid[:, :-1] & valid[:, 1:] & (cE > COH_CUT_THR)
            dphi = (w[:, 1:] - w[:, :-1]); flow = np.abs(np.round((a[:, 1:] - a[:, :-1] - _wrap(dphi)) / TAU))
        else:
            cE = np.minimum(coh[:-1, :], coh[1:, :]); vE = valid[:-1, :] & valid[1:, :] & (cE > COH_CUT_THR)
            dphi = (w[1:, :] - w[:-1, :]); flow = np.abs(np.round((a[1:, :] - a[:-1, :] - _wrap(dphi)) / TAU))
        tot += float(np.where(vE & (flow > 0), flow * cE, 0.0).sum())
    return tot / max(nvalid, 1)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def run(label):
    import whirlwind as ww
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = STEM.split("_")
    ig = np.ascontiguousarray(_read(IGDIR / f"{STEM}.int.tif", np.complex64))
    var = (_read(IGDIR / f"crlb_{a}.tif", np.float32) + _read(IGDIR / f"crlb_{b}.tif", np.float32))
    var = np.ascontiguousarray(np.nan_to_num(var, nan=0.0), dtype=np.float32)
    coh = np.ascontiguousarray(_read(IGDIR / f"{STEM}.int.cor.tif", np.float32))
    mask = np.isfinite(coh) & (coh > 0) & (np.abs(ig) > 0) & (var > 0)
    import time
    t0 = time.perf_counter()
    unw, cc = ww.unwrap_crlb(ig, var)
    dt = time.perf_counter() - t0

    wrapped = np.angle(np.where(mask, ig, 0)).astype(np.float32)
    valid = mask & np.isfinite(unw)
    rate = coherent_cut_rate(unw, wrapped, np.clip(np.where(mask, coh, 0), 0, 1), valid)

    # K-match vs spurt reference (on spurt cc>0 ∩ ww cc>0).
    ref = _read(UNWDIR / f"{STEM}.unw.tif", np.float32)
    refcc = _read(UNWDIR / f"{STEM}.unw.conncomp.tif", np.float32)
    common = valid & (cc > 0) & np.isfinite(ref) & (refcc > 0)
    km = float("nan")
    if common.sum() > 0:
        ref_k = np.round((ref[common] - wrapped[common]) / TAU).astype(np.int64)
        ww_k = np.round((unw[common] - wrapped[common]) / TAU).astype(np.int64)
        dk = ww_k - ref_k
        center = int(np.bincount((dk - dk.min()).astype(np.int64)).argmax() + dk.min())
        km = float((dk - center == 0).sum()) / dk.size * 100

    np.savez(OUT / f"{label}.npz", unw=unw, cc=cc)
    print(f"[{label}] {STEM}  shape={ig.shape}  {dt:.1f}s  n_cc={int(cc.max())}  "
          f"coverage={(cc>0).mean()*100:.1f}%")
    print(f"[{label}] coherent_cut_rate(coh>0.7) = {rate:.3e}   K-match vs spurt = {km:.2f}%  "
          f"(n_common={int(common.sum()):,})")


def compare():
    a, b = np.load(OUT / "before.npz"), np.load(OUT / "after.npz")
    print("unw bit-identical:", np.array_equal(a["unw"], b["unw"], equal_nan=True))
    print("(see the per-run coherent_cut_rate / K-match printed above for quality delta)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "before"
    compare() if arg == "compare" else run(arg)
