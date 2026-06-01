"""Seam-repair cleanup: after the (multi-shift) result, find residual clusters of
HIGH-coherence branch cuts (the water-tile block, leftover seam strips), re-solve a
window around each one SEAM-FREE (single tile), and snap the integer-disagreeing
high-coherence pixels to the re-solve. Leaves genuinely-ambiguous LOW-coherence
islands alone (their cuts are low-coh -> not clusters).

Validate on A_016: block fixed, island untouched, match up.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import label, binary_dilation

import whirlwind as ww

TAU = 2 * np.pi
COH_THR = 0.7
MIN_CLUSTER = 500   # high-coh-cut pixels per cluster; below this is speckle
MIN_BLOCK = 4000    # only snap a connected disagreement this large (a real block)
MARGIN = 220        # window margin around a cluster bbox (~ tile_size/2)


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min()) if x.size else 0


def hicut_pixels(unw, ig, coh, valid):
    H, W = unw.shape
    fh = np.rint((unw[:, 1:] - unw[:, :-1] - wrap(ig[:, 1:] - ig[:, :-1])) / TAU)
    fv = np.rint((unw[1:, :] - unw[:-1, :] - wrap(ig[1:, :] - ig[:-1, :])) / TAU)
    ch = np.minimum(coh[:, :-1], coh[:, 1:]); cv = np.minimum(coh[:-1, :], coh[1:, :])
    cut = np.zeros((H, W), bool)
    h = valid[:, :-1] & valid[:, 1:] & (fh != 0) & (ch > COH_THR)
    cut[:, :-1] |= h; cut[:, 1:] |= h
    v = valid[:-1, :] & valid[1:, :] & (fv != 0) & (cv > COH_THR)
    cut[:-1, :] |= v; cut[1:, :] |= v
    return cut


def seam_repair(unw, ig, coh, valid):
    H, W = unw.shape
    cut = hicut_pixels(unw, ig, coh, valid)
    lab, n = label(binary_dilation(cut, iterations=3))
    out = unw.copy()
    nrep = 0
    sizes = np.bincount(lab.ravel())[1:] if n else []
    for k in np.argsort(sizes)[::-1] if n else []:
        if sizes[k] < MIN_CLUSTER:
            break
        comp = (lab == k + 1)
        ys, xs = np.where(comp)
        r0, r1 = max(0, ys.min() - MARGIN), min(H, ys.max() + MARGIN)
        c0, c1 = max(0, xs.min() - MARGIN), min(W, xs.max() + MARGIN)
        win = (slice(r0, r1), slice(c0, c1))
        mk = valid[win]
        if mk.sum() < 1000:
            continue
        igc = np.ascontiguousarray(np.exp(1j * ig[win]), np.complex64)
        cohw = np.ascontiguousarray(np.clip(np.where(mk, coh[win], 0), 0, 1), np.float32)
        fresh = np.asarray(ww.unwrap_reuse(igc, cohw, 16.0, np.ascontiguousarray(mk, bool)), np.float64)
        # align fresh to current over the window's valid pixels
        cur = out[win]
        off = modal(np.rint((fresh - cur)[mk & np.isfinite(fresh) & np.isfinite(cur)] / TAU))
        fresh -= off * TAU
        # snap: where current disagrees with fresh by an integer in HIGH-coh pixels, take fresh
        dis = mk & np.isfinite(fresh) & np.isfinite(cur) & (np.abs(np.rint((cur - fresh) / TAU)) >= 1) & (coh[win] > COH_THR)
        if dis.sum() < MIN_CLUSTER:
            continue
        # Snap only the LARGEST CONNECTED single-integer disagreement (a coherent
        # block at the wrong cycle), not scattered multi-integer speckle.
        koff = np.rint((cur - fresh) / TAU)
        dlab, dn = label(dis)
        if dn == 0:
            continue
        dsz = np.bincount(dlab.ravel())[1:]
        big_d = dlab == (np.argmax(dsz) + 1)
        kvals = koff[big_d]
        if big_d.sum() < MIN_BLOCK or np.mean(kvals == np.median(kvals)) < 0.9:
            continue
        cand = cur.copy()
        cand[big_d] = fresh[big_d]
        # monotonic: must reduce the window's high-coherence-cut count.
        if int(hicut_pixels(cand, ig[win], coh[win], mk).sum()) < int(hicut_pixels(cur, ig[win], coh[win], mk).sum()):
            out[win] = cand
            nrep += int(big_d.sum())
    return out, nrep


def main() -> None:
    d = np.load(Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/a016_diag/a016_default_fixed.npz"))
    unw = d["unw"].astype(np.float64); prod = d["prod"].astype(np.float64); pcc = d["pcc"]
    coh = d["coh"].astype(np.float64); mask = d["mask"]; ig = d["ig"].astype(np.float64)
    reg = mask & (pcc > 0) & np.isfinite(unw) & np.isfinite(prod)

    def match(u):
        a = np.rint((u - prod) / TAU)[reg]; a = a[np.isfinite(a)]; return 100 * np.mean(np.abs(a - modal(a)) < 0.5)

    out, nrep = seam_repair(unw, ig, coh, mask & np.isfinite(unw))
    a0 = np.rint((unw - prod) / TAU); a0 = a0 - modal(a0[reg])
    a1 = np.rint((out - prod) / TAU); a1 = a1 - modal(a1[reg])
    blk = np.zeros_like(mask); blk[1600:1803, 2834:2944] = True
    isl = np.zeros_like(mask); isl[943:1736, 1133:1722] = True
    print(f"match before={match(unw):.2f}%  after={match(out):.2f}%  (pixels snapped={nrep:,})")
    print(f"  block off: {int((blk&reg&(np.abs(a0)>=1)).sum()):,} -> {int((blk&reg&(np.abs(a1)>=1)).sum()):,}")
    print(f"  island off: {int((isl&reg&(np.abs(a0)>=1)).sum()):,} -> {int((isl&reg&(np.abs(a1)>=1)).sum()):,} (should stay)")


if __name__ == "__main__":
    main()
