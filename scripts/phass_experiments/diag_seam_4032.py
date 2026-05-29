"""Localize the NISAR col-4032 seam strip across saved pipeline stages.

The audit found a 2px-wide strip at cols 4032-4033 sitting -1 cycle below both
neighbors on coh~0.73 mainland, exactly on a tile seam (9*448=4032, step =
tile_size 512 - overlap 64). heal_thin_lines (1px-only) cannot touch it.

This traces the strip WITHOUT a new heavy unwrap, using saved stage arrays:
  nisar_no_anchor_unw.npy  = composite + anchorless-vote + heal
  nisar_anchor_unw.npy     = composite + anchor-cascade + heal
  nisar_cascade_unw.npy    = shipped (anchor-cascade + heal)
For each: per-column median (K_ww - K_snaphu - global_modal) over the strip
rows, for cols 4024..4041. If all three show -1 at 4032/4033, the strip is
introduced UPSTREAM of refine (composite / per-tile / reconcile) and refine
neither created nor fixed it -> fix belongs in the composite / seam stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
TAU = float(2 * np.pi)
J0, J1 = 4024, 4042  # column band around the seam at 4032


def modal(d: np.ndarray) -> int:
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def main() -> None:
    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    mask = np.load(OUT / "nisar_anchor_mask.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    sk = np.load(OUT / "nisar_anchor_sk.npy")
    wrapped = np.load(OUT / "nisar_anchor_wrapped.npy")
    mainland = (scc == 1) & mask

    stages = {
        "no_anchor (vote+heal)": OUT / "nisar_no_anchor_unw.npy",
        "anchor (cascade+heal)": OUT / "nisar_anchor_unw.npy",
        "cascade (shipped)": OUT / "nisar_cascade_unw.npy",
    }

    # Use the shipped field to find the strip rows (cols 4032/4033 vs neighbors).
    ship = np.load(OUT / "nisar_cascade_unw.npy")
    ks = np.round((ship - wrapped) / TAU)
    ks[~mask] = np.nan
    dks = ks - sk
    gmod = modal(dks[mainland])
    dks = dks - gmod
    # strip rows: where col 4032 AND 4033 are -1 but 4031 AND 4034 are 0, on mainland
    col = lambda dk, j: np.where(mainland[:, j], dk[:, j], np.nan)
    strip_rows = np.where(
        (col(dks, 4032) == -1) & (col(dks, 4033) == -1)
        & (col(dks, 4031) == 0) & (col(dks, 4034) == 0)
    )[0]
    print(f"strip rows (4032&4033==-1, 4031&4034==0 on mainland): {strip_rows.size}", flush=True)
    if strip_rows.size:
        print(f"  row span {strip_rows.min()}..{strip_rows.max()}", flush=True)
        cohband = coh[strip_rows][:, J0:J1]
        mb = mainland[strip_rows][:, J0:J1]
        print(f"  mean coh on strip band (mainland): {np.nanmean(np.where(mb, cohband, np.nan)):.3f}", flush=True)

    if strip_rows.size == 0:
        print("No strip found on shipped field with strict definition; widening.", flush=True)
        strip_rows = np.where((col(dks, 4032) == -1))[0]
        print(f"  rows where col4032==-1: {strip_rows.size}", flush=True)

    rows = strip_rows
    print(f"\nPer-column median (K_ww - K_snaphu - modal) over {rows.size} strip rows:")
    hdr = "stage".ljust(26) + "".join(f"{j:>5d}" for j in range(J0, J1))
    print(hdr, flush=True)
    for name, path in stages.items():
        u = np.load(path)
        k = np.round((u - wrapped) / TAU)
        k[~mask] = np.nan
        dk = k - sk
        dk = dk - modal(dk[mainland])
        med = []
        for j in range(J0, J1):
            v = np.where(mainland[rows, j], dk[rows, j], np.nan)
            v = v[np.isfinite(v)]
            med.append(int(np.median(v)) if v.size else 9)
        print(name.ljust(26) + "".join(f"{m:>5d}" for m in med), flush=True)

    # also report whether the three stages differ anywhere in the band
    print("\nStage diffs in band (rows x cols 4024..4041):", flush=True)
    arrs = {n: np.load(p) for n, p in stages.items()}
    names = list(arrs)
    for a, b in [(0, 1), (1, 2)]:
        d = arrs[names[a]][rows][:, J0:J1] - arrs[names[b]][rows][:, J0:J1]
        d = d[np.isfinite(d)]
        print(f"  max|{names[a]} - {names[b]}| = {np.abs(d).max() if d.size else 0:.4f}", flush=True)


if __name__ == "__main__":
    main()
