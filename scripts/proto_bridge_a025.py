"""Solver-aware bridging prototype (v2) on CACHED fine unwraps -- the only unwraps
here are TINY coarse anchors (~550 px, negligible), never a heavy full-res solve.

FACT 1 (verified, scripts/diag_bridge_partition.py): the free 2pi gauge lives
between INTEGRATION components (4-connected comps of the valid MASK), NOT conncomp
labels. Node set = scipy.ndimage.label(mask). A single-integration-component frame
(005_D_077, 005_D_074) is a STRUCTURAL no-op. 005_A_025's integration-component oracle = 100%.

v1 found: the L=8 coarse mask is itself split (ncR=4) on 005_A_025 -> snapping to it
does nothing; and an UNVETOED snap regresses 005_A_028 (confidently-wrong anchor).
v2 fixes both:
  - CONNECTED anchor: morphologically CLOSE the coarse mask so its single
    integration BFS gauge spans the banks (bridges the masked river at x8).
  - AMBIGUITY-BAND veto: accept a component's integer shift only if the unrounded
    median (anchor-unw)/2pi is within AMB_BAND of an integer (else the anchor is
    untrustworthy there -> decline, tag convention).
  - SAME-COARSE-COMPONENT gate: the component must share the reference's coarse
    integration component (data-supported relative gauge).

Modes per frame: baseline | open-gated-veto | closed-gated-veto | oracle.
Usage: python scripts/proto_bridge_a025.py [FRAMES...]
"""

import sys
import os

import numpy as np
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import whirlwind as ww  # TINY coarse anchors only; fine unwrap always from cache

tau = 2 * np.pi
CACHE = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/bridge_cache"
S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
MIN_PX = 500
GATE_FRAC = 0.5
AMB_BAND = 0.25
L = 8
CLOSE_ITERS = 4
frames = sys.argv[1:] or [
    "005_A_025",
    "005_A_016",
    "005_A_030",
    "005_A_028",
    "005_D_077",
    "005_D_074",
]


def block_mean(a, L):
    mm, nn = a.shape[0] // L, a.shape[1] // L
    return a[: mm * L, : nn * L].reshape(mm, L, nn, L).mean(axis=(1, 3))


def kron_up(a, L, m, n):
    up = np.kron(a, np.ones((L, L), a.dtype))
    return np.pad(
        up, ((0, max(0, m - up.shape[0])), (0, max(0, n - up.shape[1]))), mode="edge"
    )[:m, :n]


def coarse_anchor(wrapped, coh_in, mask, L, close_iters, m, n):
    """Coherent x L multilook -> unwrap the (optionally CLOSED) coarse mask whole.
    Closing the coarse mask bridges the masked river so the integration BFS gives
    ONE consistent gauge across the banks. Returns (anchor, coarse-comp map up)."""
    ig = np.exp(1j * wrapped).astype(np.complex64)
    cig = block_mean(ig, L).astype(np.complex64)
    ccoh = block_mean(coh_in, L).astype(np.float32)
    cmask = block_mean(mask.astype(np.float32), L) > 0.4
    if close_iters > 0:
        cmask = ndimage.binary_closing(cmask, iterations=close_iters)
        ccoh = np.where(cmask, np.maximum(ccoh, 0.05), 0.0).astype(
            np.float32
        )  # keep river barely valid
    cunw, _ = ww.unwrap(cig, ccoh, 16.0 * L * L, cmask)
    cunw = np.asarray(cunw, np.float32)
    cR, ncR = ndimage.label(cmask, structure=S4)
    return kron_up(cunw, L, m, n), kron_up(cR.astype(np.int64), L, m, n), ncR


for frame in frames:
    npz = f"{CACHE}/{frame}.npz"
    if not os.path.exists(npz):
        print(f"{frame}: no cache -- run scripts/cache_bridge_arrays.py", flush=True)
        continue
    d = np.load(npz)
    unw, cc, mask = d["unw"].astype(np.float32), d["cc"].astype(np.int64), d["mask"]
    prod, prod_cc = d["prod"].astype(np.float32), d["prod_cc"].astype(np.int64)
    coh, wrapped = d["coh"].astype(np.float32), d["wrapped"].astype(np.float32)
    m, n = mask.shape

    def percomp(u):
        in_c = mask & np.isfinite(u) & (prod_cc > 0)
        amb = np.rint((u[in_c] - prod[in_c]) / tau)
        ccp = prod_cc[in_c]
        ok = tot = 0
        for lab in np.unique(ccp):
            mm = ccp == lab
            off = np.median(amb[mm])
            ok += int((np.abs(amb[mm] - off) < 0.5).sum())
            tot += int(mm.sum())
        return ok / max(tot, 1)

    R, nR = ndimage.label(mask, structure=S4)
    Rsizes = np.bincount(R.ravel())
    ref_lab = int(np.argmax(Rsizes[1:]) + 1)
    labels = [l for l in range(1, nR + 1) if Rsizes[l] >= MIN_PX]

    def snap(anchor, cR_up, gated, veto):
        ref_cR = np.bincount(cR_up[R == ref_lab]).argmax()
        both = mask & np.isfinite(anchor) & np.isfinite(unw)
        refreg = (R == ref_lab) & both
        med_ref = np.median((anchor[refreg] - unw[refreg]) / tau)  # reference gauge
        u = unw.copy()
        nshift = nconv = 0
        for lab in labels:
            if lab == ref_lab:
                continue
            reg = (R == lab) & both
            if reg.sum() < MIN_PX:
                continue
            if gated and np.mean(cR_up[reg] == ref_cR) < GATE_FRAC:
                continue
            rel = (
                np.median((anchor[reg] - unw[reg]) / tau) - med_ref
            )  # RELATIVE to reference
            s = int(np.rint(rel))
            if veto and abs(rel - s) > AMB_BAND:
                nconv += 1
                continue
            if s != 0:
                u[R == lab] += tau * s
                nshift += 1
        return u, nshift, nconv

    def oracle():
        u = unw.copy()
        for lab in labels:
            reg = (R == lab) & mask & np.isfinite(unw) & (prod_cc > 0)
            if reg.sum() < MIN_PX:
                continue
            s = int(np.rint(np.median((prod[reg] - unw[reg]) / tau)))
            u[R == lab] += tau * s
        return u

    anchor_o, cRo, ncR_o = coarse_anchor(wrapped, coh, mask, L, 0, m, n)

    # DIAGNOSTIC: for the big integration comps, does the open anchor IMPLY the
    # oracle shift? (oracle = round((prod-unw)); anchor = round((anchor-unw)); both
    # relative to the reference comp.) anchor==oracle means the river is genuinely
    # data-bridgeable at xL (not a convention).
    bothc = mask & np.isfinite(anchor_o) & np.isfinite(unw) & (prod_cc > 0)
    rref = (R == ref_lab) & bothc
    oref = np.median((prod[rref] - unw[rref]) / tau)
    aref = np.median((anchor_o[rref] - unw[rref]) / tau)
    bigs = [l for l in labels if Rsizes[l] >= 0.04 * mask.sum()]
    diag = []
    for l in bigs:
        reg = (R == l) & bothc
        if reg.sum() < MIN_PX:
            continue
        os_ = int(np.rint(np.median((prod[reg] - unw[reg]) / tau) - oref))
        as_ = int(np.rint(np.median((anchor_o[reg] - unw[reg]) / tau) - aref))
        cohm = float(np.median(coh[(R == l) & mask]))
        diag.append((l, int(Rsizes[l]), os_, as_, round(cohm, 2)))
    print(
        f"  [{frame}] big comps (lab,size,ORACLE_shift,ANCHOR_shift,medcoh) ref={ref_lab}: {diag}",
        flush=True,
    )

    base = percomp(unw)
    # PRIMARY = open x8 anchor (the closed variant fabricated a micro-bridge that
    # regressed 005_A_030; open already connects 005_A_025's narrow river -> data-supported).
    u_o, no, ncv = snap(anchor_o, cRo, True, True)
    u_or = oracle()
    pc_o, pc_or = percomp(u_o), percomp(u_or)
    u_c = u_o  # figure uses the primary (open) result

    flag = ""
    if frame != "005_A_025" and pc_o < base - 1e-9:
        flag = f"  <-- REGRESSION {(pc_o-base)*100:+.3f}"
    elif frame == "005_A_025" and pc_o > base + 1e-9:
        flag = f"  <-- FIXED {(pc_o-base)*100:+.1f}"
    print(
        f"{frame}: nR={nR} ncR={ncR_o}  base={base*100:.1f}%  "
        f"bridged={pc_o*100:.2f}%(n{no},conv{ncv})  oracle={pc_or*100:.1f}%{flag}",
        flush=True,
    )

    pu = np.where(mask, prod, np.nan)
    vlo, vhi = np.nanpercentile(pu, [2, 98])

    def galign(u):
        v = mask & np.isfinite(u)
        g = np.rint(np.nanmedian(np.rint((u[v] - prod[v]) / tau)))
        return np.where(mask, u - tau * g, np.nan)

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    v_all = mask & np.isfinite(unw)
    ambc = np.where(
        mask,
        np.rint((u_c - prod) / tau)
        - np.rint(np.nanmedian(np.rint((u_c[v_all] - prod[v_all]) / tau))),
        np.nan,
    )
    panels = [
        (
            np.where(mask, np.log1p(R.astype(float)), np.nan),
            f"integration comps (nR={nR})",
            "tab20",
            None,
            None,
        ),
        (pu, f"production ({int(prod_cc.max())} cc)", "viridis", vlo, vhi),
        (galign(unw), f"whirlwind baseline {base*100:.1f}%", "viridis", vlo, vhi),
        (galign(anchor_o), f"coarse x{L} anchor (ncR={ncR_o})", "viridis", vlo, vhi),
        (galign(u_o), f"bridged {pc_o*100:.1f}% (n{no})", "viridis", vlo, vhi),
        (ambc, "bridged ambiguity diff (cyc)", "RdBu", -2, 2),
    ]
    for a, (arr, title, cmap, lo, hi) in zip(ax.ravel(), panels):
        im = a.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(title, fontsize=10)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{frame}: solver-aware bridging (integration gauge + x{L} anchor + veto)  base {base*100:.1f}% -> {pc_o*100:.2f}% (oracle {pc_or*100:.1f}%)",
        fontsize=13,
    )
    fig.tight_layout()
    out = f"{CACHE}/{frame}_bridge.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"{frame}: figure -> {out}", flush=True)
