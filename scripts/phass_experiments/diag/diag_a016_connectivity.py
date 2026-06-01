"""How do A_016's correct-left and drifted-right connect? Determines the fix shape.

- Are they the SAME valid connected-component (joined by a bridge) or separate?
- If same: find the articulation / narrowest cross-section (the bridge) — the fix
  must SPLIT there and re-level (within-component articulation problem).
- If separate: a region-graph secondary that levels separate components suffices.

Also: does cutting low-coherence edges (threshold sweep) split left from right while
keeping a clean dense scene as one piece? (feasibility of coherence-based segmentation)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import label

LEARN = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")
A016 = "NISAR_L2_PR_GUNW_003_005_A_016_004_4000_SH_20251017T125003_20251017T125038_20251029T125004_20251029T125039_X05010_N_F_J_001"
TAU = 2 * np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def comp_stats(m):
    lab, n = label(m)
    if n == 0:
        return lab, n, np.array([])
    sizes = np.bincount(lab.ravel())[1:]
    return lab, n, sizes


def main() -> None:
    d = np.load(LEARN / "ww_gunw_bench" / A016 / "full_arrays.npz")
    mask = d["mask"]; prod = d["prod_unw"]; pcc = d["prod_cc"]; coh = d["coh"]
    unw = d["ww_unw"].astype(np.float64)
    H, W = unw.shape
    reg = mask & (pcc > 0) & np.isfinite(unw)
    a = np.rint((unw - prod) / TAU); a = a - modal(a[reg])
    correct = reg & (np.abs(a) < 0.5)
    drift = reg & (np.abs(a) >= 0.5)
    cols = np.arange(W)[None, :]
    print(f"A_016 valid={mask.sum():,} ({mask.mean():.1%})  correct={correct.sum():,}  drift={drift.sum():,}", flush=True)

    # 1) connectivity of the VALID mask (theta=0)
    lab, n, sizes = comp_stats(mask)
    big = np.argsort(sizes)[::-1][:5] + 1
    print(f"\nvalid-mask components: {n}; top sizes {sorted(sizes)[::-1][:5]}", flush=True)
    # Is the largest component spanning both correct(left) and drift(right)?
    lc = (lab == (np.argmax(sizes) + 1))
    print(f"  largest comp: {lc.sum():,} px; contains correct={np.sum(lc & correct):,}, drift={np.sum(lc & drift):,}", flush=True)
    if np.sum(lc & correct) > 1000 and np.sum(lc & drift) > 1000:
        print("  => SAME component spans correct+drift -> joined by a BRIDGE (within-component articulation).", flush=True)
    else:
        print("  => correct and drift are in DIFFERENT components -> separate-region leveling suffices.", flush=True)

    # 2) per-column valid count: locate the narrow bridge (the thin cross-section)
    colcount = mask.sum(0)
    # focus on the drift frontier ~col 2075
    fr = slice(1900, 2300)
    seg = colcount[fr]
    nar = np.argmin(seg) + 1900
    print(f"\n  valid px per column near frontier: min={seg.min()} at col~{nar} "
          f"(left avg {colcount[1500:1900].mean():.0f}, right avg {colcount[2300:2700].mean():.0f})", flush=True)

    # 3) coherence-threshold segmentation feasibility: cut edges where coh<theta,
    #    does left split from right? (count components, and whether the frontier separates)
    print("\n  coherence-cut segmentation (keep pixel iff coh>=theta), components spanning frontier:", flush=True)
    for theta in (0.3, 0.4, 0.5, 0.6, 0.7):
        hi = mask & (coh >= theta)
        lab2, n2, sz2 = comp_stats(hi)
        if n2 == 0:
            continue
        lc2 = (lab2 == (np.argmax(sz2) + 1))
        spans = np.sum(lc2 & correct) > 500 and np.sum(lc2 & drift) > 500
        print(f"    theta={theta}: {n2} comps, largest={sz2.max():,}px, "
              f"largest-spans-both={spans}, kept={hi.sum()/mask.sum():.0%} of valid", flush=True)


if __name__ == "__main__":
    main()
