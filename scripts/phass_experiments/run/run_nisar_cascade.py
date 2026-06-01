"""Run NISAR tiled+anchor with the multi-scale refine cascade (f=16,8,4) and
compare to the single-f=8 anchor result (already saved). Metrics on mainland
(tripwire) and full frame.
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


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def match_pct(kw, ks, region):
    d = (kw - ks)[region]; d = d[np.isfinite(d)]; d = d - modal(d)
    n = d.size
    return (float((d == 0).sum()) / n * 100, float((np.abs(d) >= 2).sum()) / n * 100)


def main() -> None:
    import whirlwind as ww
    ig = rasterio.open(N / "20251224_20260117.int.looked.tif").read(1).astype(np.complex64)
    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    sk = np.load(OUT / "nisar_anchor_sk.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
    ig[~mask] = 0
    coh = np.clip(np.where(mask, coh, 0), 0, 1).astype(np.float32)
    wrapped = np.angle(ig).astype(np.float32)
    mainland = (scc == 1) & mask

    os.environ["WHIRLWIND_REFINE_CASCADE"] = "1"
    os.environ.pop("WHIRLWIND_NO_ANCHOR", None)
    t0 = time.perf_counter()
    unw, _cc = ww.unwrap(ig, coh, nlooks=100.0, mask=mask, tile_size=512, tile_overlap=64)
    dt = time.perf_counter() - t0
    kw = np.round((unw - wrapped) / TAU); kw[~mask] = np.nan
    m_main = match_pct(kw, sk, mainland)
    m_full = match_pct(kw, sk, mask)
    print(f"[anchor+cascade] {dt:.1f}s", flush=True)
    print(f"   mainland match={m_main[0]:.2f}%  |dK|>=2={m_main[1]:.2f}%", flush=True)
    print(f"   full     match={m_full[0]:.2f}%  |dK|>=2={m_full[1]:.2f}%", flush=True)
    np.save(OUT / "nisar_cascade_unw.npy", unw.astype(np.float32))

    # vs the saved single-f=8 anchor
    una = np.load(OUT / "nisar_anchor_unw.npy")
    ka = np.round((una - wrapped) / TAU); ka[~mask] = np.nan
    print(f"[anchor single] mainland match={match_pct(ka, sk, mainland)[0]:.2f}%  "
          f"full match={match_pct(ka, sk, mask)[0]:.2f}%", flush=True)
    chg = int(np.nansum(np.abs(kw - ka) > 0.5))
    print(f"cascade changed {chg:,} px vs single-f8 ({100*chg/mask.sum():.2f}%)", flush=True)


if __name__ == "__main__":
    main()
