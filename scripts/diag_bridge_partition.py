"""Decisive partition check for the solver-aware bridging design (no heavy compute
-- reads the cached bridge_cache/<FRAME>.npz from plot_unwrap_compare.py).

Tests FACT 1 from the design synthesis: the free 2pi gauge lives between
INTEGRATION components (4-connected components of the valid MASK), not between
conncomp labels. Crux for 005_A_025: are the two river banks the SAME integration
component (river valid-but-low-coh -> gauge already pinned by MCF flow) or
DIFFERENT integration components (river masked -> genuine free gauge to bridge)?

Usage: python scripts/diag_bridge_partition.py [005_A_025 005_D_077 005_A_016 005_D_074]
"""

import sys

import numpy as np
from scipy import ndimage

tau = 2 * np.pi
CACHE = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/bridge_cache"
S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # 4-connectivity
frames = sys.argv[1:] or ["005_A_025", "005_D_077", "005_A_016", "005_D_074"]

for frame in frames:
    d = np.load(f"{CACHE}/{frame}.npz")
    unw, cc, mask = d["unw"], d["cc"].astype(np.int64), d["mask"]
    prod, prod_cc, coh = d["prod"], d["prod_cc"].astype(np.int64), d["coh"]
    m, n = mask.shape
    nvalid = int(mask.sum())

    # Integration partition = 4-connected components of the valid mask.
    R, nR = ndimage.label(mask, structure=S4)
    ncc = int(cc.max())
    Rsizes = np.bincount(R.ravel())[1:]  # drop background label 0
    big = np.argsort(Rsizes)[::-1]
    top = [(int(b + 1), int(Rsizes[b]), Rsizes[b] / nvalid) for b in big[:4]]

    # FACT-1 coarseness: every conncomp label must sit inside exactly ONE
    # integration component (cc finer than R).
    v = mask & (cc > 0)
    pair = np.unique(np.stack([cc[v], R[v]], 1), axis=0)
    ncc_labels, counts = np.unique(pair[:, 0], return_counts=True)
    max_R_per_cc = int(counts.max())  # ==1 means cc strictly refines R

    # Error slab: pixels where whirlwind disagrees with production beyond the
    # global integer offset (the thing bridging would fix).
    both = mask & np.isfinite(unw) & np.isfinite(prod)
    ambv = np.rint((unw[both] - prod[both]) / tau)
    gmed = np.median(ambv)
    err = np.zeros(mask.shape, bool)
    err[both] = np.rint((unw[both] - prod[both]) / tau) != gmed
    err_frac = err.sum() / max(both.sum(), 1)

    # Does the error live WITHIN the dominant integration component, or in a
    # separate one? (within -> flow error; separate -> mask-gap gauge.)
    err_R = R[err]
    if err_R.size:
        eR_lab, eR_cnt = np.unique(err_R, return_counts=True)
        dom_err_R = int(eR_lab[eR_cnt.argmax()])
        dom_err_R_share = eR_cnt.max() / err_R.size
    else:
        dom_err_R, dom_err_R_share = -1, 0.0
    main_R = top[0][0]

    # prod_cc straddle: does any production component span >1 integration
    # component? (If so, the river is mask-connected and per-comp shifts won't
    # reach the oracle; if production cc itself splits at the river, it's a gap.)
    vp = mask & (prod_cc > 0)
    ppair = np.unique(np.stack([prod_cc[vp], R[vp]], 1), axis=0)
    plab, pcnt = np.unique(ppair[:, 0], return_counts=True)
    straddle = int((pcnt > 1).sum())
    n_prod = int(plab.size)

    # Coherence of the error region (is the slab decorrelated land or masked?).
    err_coh_med = float(np.median(coh[err])) if err.any() else float("nan")
    lowcoh_valid = (mask & (coh < 0.3)).sum() / max(nvalid, 1)

    print(
        f"\n=== {frame}  shape=({m},{n}) valid={nvalid/(m*n)*100:.1f}% ===", flush=True
    )
    print(
        f"  conncomps ncc={ncc}   integration comps nR={nR}   (cc-within-R max={max_R_per_cc} -> {'OK coarser' if max_R_per_cc==1 else 'VIOLATION'})",
        flush=True,
    )
    print(
        f"  top integration comps (label,size,frac valid): {[(l, s, round(f,3)) for l,s,f in top]}",
        flush=True,
    )
    print(
        f"  error slab: {err_frac*100:.1f}% of valid; dominant in integration comp {dom_err_R} ({dom_err_R_share*100:.0f}% of error); main comp={main_R}",
        flush=True,
    )
    print(
        f"    -> error is {'WITHIN the main integration comp (flow error, NOT a mask-gap gauge)' if dom_err_R==main_R else 'in a SEPARATE integration comp (mask-gap gauge -> formulation applies)'}",
        flush=True,
    )
    print(
        f"  error-region median coherence={err_coh_med:.3f}; valid px coh<0.3 = {lowcoh_valid*100:.1f}%",
        flush=True,
    )
    print(
        f"  production: {n_prod} comps, {straddle} STRADDLE >1 integration comp",
        flush=True,
    )
