"""K-match analysis for the Atlanta S-1 scene vs the OPERA (SNAPHU) reference.

The OPERA "displacement" GeoTIFF is the SNAPHU-derived unwrapped solution
converted to LOS displacement in meters. We reconstruct its phase and compute
the integer-ambiguity field K_ref = round((phase_ref - wrapped)/2pi), then
compare each whirlwind mode's K against it.

Two K-match numbers are reported per mode:
  * global : a single modal 2pi offset removed over the whole comparison region
             (apples-to-apples with analyze.py's NISAR/PV numbers).
  * per-cc : a separate modal 2pi offset removed PER OPERA connected component.
             The gap between global and per-cc isolates how much of any
             disagreement is just whole-region absolute-offset (which PHASS-style
             per-component integration would fix) vs genuine misrouting.

Usage:  analyze_atlanta.py [mode ...]   (default: baseline reuse)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio

OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
ATL = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
LAMBDA_S1 = 0.05546576  # m, Sentinel-1 C-band
TAU = np.float32(2 * np.pi)

modes = sys.argv[1:] or ["baseline", "reuse", "convex"]


def modal_offset(dk: np.ndarray) -> int:
    """Most-common integer in dk (the global 2pi offset to remove)."""
    if dk.size == 0:
        return 0
    return int(np.bincount(dk - dk.min()).argmax() + dk.min())


def load_ref():
    with rasterio.open(ATL / "opera.int.phs.tif") as src:
        phase = src.read(1).astype(np.float32)
    with rasterio.open(ATL / "opera.int.cor.tif") as src:
        coh = src.read(1).astype(np.float32)
    with rasterio.open(ATL / "opera.displacement.tif") as src:
        disp = src.read(1).astype(np.float32)
    with rasterio.open(ATL / "opera.conncomp.tif") as src:
        cc = src.read(1).astype(np.int32)

    wrapped = np.angle(np.exp(1j * phase)).astype(np.float32)
    mask = (
        np.isfinite(phase) & np.isfinite(coh) & np.isfinite(disp)
        & (coh > 0) & (coh < 1.0)
    )
    # Displacement -> phase. Auto-detect the sign convention by congruence:
    # the correct sign makes (phase_ref - wrapped)/2pi cluster on integers.
    best = None
    for s in (+1.0, -1.0):
        phase_ref = s * disp * (4.0 * np.pi / LAMBDA_S1)
        resid = (phase_ref - wrapped) / TAU
        frac = resid - np.round(resid)
        spread = float(np.nanstd(frac[mask]))
        if best is None or spread < best[0]:
            best = (spread, s, phase_ref)
    spread, s, phase_ref = best
    print(f"[atlanta] displacement->phase sign = {s:+.0f}  "
          f"(congruence frac-std={spread:.4f} cycles; lower=better)", flush=True)
    k_ref = np.round((phase_ref - wrapped) / TAU).astype(np.int32)
    return dict(wrapped=wrapped, coh=coh, mask=mask, k_ref=k_ref, cc=cc)


def main() -> None:
    ref = load_ref()
    mask, k_ref, cc = ref["mask"], ref["k_ref"], ref["cc"]
    # Comparison region: OPERA's largest component (its trusted area) ∩ mask.
    labels, counts = np.unique(cc[cc > 0], return_counts=True)
    main_label = int(labels[np.argmax(counts)]) if labels.size else 0
    region = mask & (cc == main_label)
    n_region = int(region.sum())
    print(f"[atlanta] OPERA cc={main_label} mainland = {n_region:,} px "
          f"({n_region/cc.size*100:.1f}% of frame); mask={int(mask.sum()):,}",
          flush=True)

    header = ("mode", "wall", "K=match%", "|dK|=1%", "|dK|>=2%",
              "per-cc K=match%", "n_cc(opera)")
    rows = [header, tuple("---" for _ in header)]
    for mode in modes:
        path = OUT / f"atlanta_{mode}.npz"
        if not path.exists():
            rows.append((mode, "—", "—", "—", "—", "—", "—"))
            continue
        d = np.load(path)
        k_ww = d["k"].astype(np.int32)
        elapsed = float(d["elapsed"]) if "elapsed" in d else float("nan")

        # --- global single-offset metric (matches analyze.py) ---
        dk = k_ww[region] - k_ref[region]
        c = modal_offset(dk)
        dk_c = dk - c
        m0 = (dk_c == 0).mean() * 100
        m1 = (np.abs(dk_c) == 1).mean() * 100
        m2 = (np.abs(dk_c) >= 2).mean() * 100

        # --- per-OPERA-component offset metric ---
        # Remove an independent modal offset within each OPERA component that
        # overlaps the mask; aggregate the match fraction over all of them.
        matched = 0
        total = 0
        cc_in = np.unique(cc[mask & (cc > 0)])
        for lab in cc_in:
            sel = mask & (cc == lab)
            n = int(sel.sum())
            if n < 200:  # PHASS min region size
                continue
            dkl = k_ww[sel] - k_ref[sel]
            cl = modal_offset(dkl)
            matched += int((dkl - cl == 0).sum())
            total += n
        m0_cc = (matched / total * 100) if total else float("nan")

        rows.append((mode, f"{elapsed:.1f}s", f"{m0:.2f}", f"{m1:.2f}",
                     f"{m2:.2f}", f"{m0_cc:.2f}", str(len(cc_in))))

    w = [max(len(str(r[i])) for r in rows) for i in range(len(header))]
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)), flush=True)


if __name__ == "__main__":
    main()
