"""NISAR tiled unwrap: global coarse-anchor (new) vs no-anchor (old), measured
against SNAPHU 9x9 on THREE regions:

  * mainland  = (snaphu_cc == 1) & mask   — the historical metric / regression
    tripwire. The visible low-coherence blocks are EXCLUDED here, so a fix must
    NOT move this number; if it does, the anchor reached into the mainland.
  * reliable  = (snaphu_cc  > 0) & mask    — everywhere SNAPHU trusts.
  * full      = mask                        — everywhere both unwrapped. This is
    the metric that registers the visible rectangular artifacts (they live in
    cc<1 regions that the mainland metric never counts).

Runs anchor and no-anchor SEQUENTIALLY in one process (one heavy unwrap at a
time, per the laptop concurrency limit). Saves full-res arrays for plotting.

  python scripts/phass_experiments/run_nisar_anchor.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
TAU = np.float32(2 * np.pi)
TS, OV = 512, 64
NLOOKS = 100.0


def modal(d: np.ndarray) -> int:
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def match_pct(kw: np.ndarray, ks: np.ndarray, region: np.ndarray) -> tuple[float, float, float]:
    d = (kw - ks)[region]
    d = d[np.isfinite(d)]
    d = d - modal(d)
    n = d.size
    m0 = float((d == 0).sum()) / n * 100
    m1 = float((np.abs(d) == 1).sum()) / n * 100
    m2 = float((np.abs(d) >= 2).sum()) / n * 100
    return m0, m1, m2


def main() -> None:
    import whirlwind as ww

    ig = rasterio.open(N / "20251224_20260117.int.looked.tif").read(1).astype(np.complex64)
    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    sunw = rasterio.open(N / "20251224_20260117.snaphu_9x9.unw.tif").read(1).astype(np.float32)
    scc = rasterio.open(N / "20251224_20260117.snaphu_9x9.cc.tif").read(1).astype(np.uint32)

    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig[~mask] = 0
    coh = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    sk = np.round((sunw - wrapped) / TAU)

    mainland = (scc == 1) & mask
    reliable = (scc > 0) & mask
    full = mask
    print(f"shape={ig.shape}  valid={mask.sum():,}  "
          f"mainland(cc=1)={mainland.sum():,}  reliable(cc>0)={reliable.sum():,}", flush=True)

    results = {}
    for label, no_anchor in [("anchor", False), ("no_anchor", True)]:
        if no_anchor:
            os.environ["WHIRLWIND_NO_ANCHOR"] = "1"
        else:
            os.environ.pop("WHIRLWIND_NO_ANCHOR", None)
        t0 = time.perf_counter()
        unw = ww.unwrap(ig, coh, nlooks=NLOOKS, mask=mask, tile_size=TS, tile_overlap=OV)
        dt = time.perf_counter() - t0
        kw = np.round((unw - wrapped) / TAU)
        kw[~mask] = np.nan
        row = {
            "elapsed": dt,
            "mainland": match_pct(kw, sk, mainland),
            "reliable": match_pct(kw, sk, reliable),
            "full": match_pct(kw, sk, full),
        }
        results[label] = row
        print(f"\n[{label}] {dt:.1f}s", flush=True)
        for reg in ("mainland", "reliable", "full"):
            m0, m1, m2 = row[reg]
            print(f"   {reg:9s}  match={m0:6.2f}%  |dK|=1={m1:5.2f}%  |dK|>=2={m2:5.2f}%", flush=True)
        np.save(OUT / f"nisar_{label}_unw.npy", unw.astype(np.float32))

    np.save(OUT / "nisar_anchor_sk.npy", sk.astype(np.float32))
    np.save(OUT / "nisar_anchor_scc.npy", scc)
    np.save(OUT / "nisar_anchor_mask.npy", mask)
    np.save(OUT / "nisar_anchor_wrapped.npy", wrapped)

    print("\n=== SUMMARY (match% on each region) ===", flush=True)
    print(f"{'region':9s} {'no_anchor':>12s} {'anchor':>12s} {'delta':>8s}", flush=True)
    for reg in ("mainland", "reliable", "full"):
        na = results["no_anchor"][reg][0]
        an = results["anchor"][reg][0]
        print(f"{reg:9s} {na:11.2f}% {an:11.2f}% {an-na:+7.2f}%", flush=True)


if __name__ == "__main__":
    main()
