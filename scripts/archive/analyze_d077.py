#!/usr/bin/env python3
"""Decisive real-frame analysis for D_077. Loads snaphu (this run), ww-convex
whole-image, ww-tiled (default), and production, then:
  (A) computes the convex MAP objective J(phi) for each field under ONE
      consistent per-edge cost (snaphu's exact smooth cost: offset = ns*(dpsi -
      avgdpsi), sigsq from coherence). The field with the LOWEST J is the
      cost-optimal surface. If the runaway (ww-convex) has the lowest J, the
      cost's optimum != truth (the COST is the lever, not the solver). If
      production/snaphu has lower J, ww failed to reach the optimum (solver/scale).
  (B) cross per-component matches: snaphu-vs-prod, ww-vs-prod, ww-vs-snaphu.
  (C) a 5-panel comparison figure.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

TWOPI = 2.0 * np.pi
NS = 100.0
WD = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")


def boxcar(a, k=7):
    """Separable box mean, edge-replicate; numpy-only."""
    from numpy.lib.stride_tricks import sliding_window_view as swv

    pad = k // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    a = swv(ap, k, axis=0).mean(-1)
    ap = np.pad(a, ((0, 0), (pad, pad)), mode="edge")
    a = swv(ap, k, axis=1).mean(-1)
    return a


def snaphu_sigsq_edge(rho, nlooks):
    """snaphu smooth sigsq per edge (cost.c:1107-1124), in short-units^2."""
    L = nlooks
    rho0 = 1.3 / L + 0.14
    thresh = 1.2 * rho0
    rhopow = 2 * 0.4 + 0.35 * np.log(L) + 0.06 * L
    r = np.where(rho < thresh, 0.0, rho)
    sigsqrho = (2.0 / 12.0) * (1 - r) ** rhopow + 0.05
    sigsq = sigsqrho * (NS**2)  # weight=1, costscale folded into relative compare
    return np.maximum(sigsq, 1.0)


def map_objective(phi, wrapped, coh, mask, nlooks):
    """Convex MAP objective J = sum_edges (ns*(grad_cycles - avgdpsi))^2 / sigsq,
    over edges with both endpoints valid. (snaphu smooth cost; ww's is the same
    structure with Lee variance - qualitative cost ranking is identical.)"""
    # NaN-safe: zero invalid samples (validity mask excludes them from the sum;
    # the shared avgdpsi bias near masked regions cancels in the cross-field ranking).
    phi = np.nan_to_num(phi, nan=0.0)
    wrapped = np.nan_to_num(wrapped, nan=0.0)
    coh = np.nan_to_num(coh, nan=0.0)
    # wrapped gradients (cycles), wrapped to [-0.5,0.5)
    dxw = ((wrapped[:, 1:] - wrapped[:, :-1] + np.pi) % TWOPI - np.pi) / TWOPI
    dyw = ((wrapped[1:, :] - wrapped[:-1, :] + np.pi) % TWOPI - np.pi) / TWOPI
    # boxcar of wrapped gradients -> avgdpsi (pad gradients back to full grid for boxcar)
    dxw_full = np.zeros_like(wrapped)
    dxw_full[:, :-1] = dxw
    dyw_full = np.zeros_like(wrapped)
    dyw_full[:-1, :] = dyw
    avgx = boxcar(dxw_full)[:, :-1]
    avgy = boxcar(dyw_full)[:-1, :]
    # unwrapped gradients (cycles) of the candidate field
    gx = (phi[:, 1:] - phi[:, :-1]) / TWOPI
    gy = (phi[1:, :] - phi[:-1, :]) / TWOPI
    # per-edge coherence (min of endpoints) and sigsq
    cx = np.minimum(coh[:, 1:], coh[:, :-1])
    cy = np.minimum(coh[1:, :], coh[:-1, :])
    sx = snaphu_sigsq_edge(cx, nlooks)
    sy = snaphu_sigsq_edge(cy, nlooks)
    vx = mask[:, 1:] & mask[:, :-1] & np.isfinite(gx)
    vy = mask[1:, :] & mask[:-1, :] & np.isfinite(gy)
    jx = (NS * (gx - avgx)) ** 2 / sx
    jy = (NS * (gy - avgy)) ** 2 / sy
    return float(np.sum(jx[vx]) + np.sum(jy[vy]))


def percomp_match(test, prod_unw, wrapped, prod_cc, valid):
    amb = np.rint((test - wrapped) / TWOPI) - np.rint((prod_unw - wrapped) / TWOPI)
    in_comp = valid & (prod_cc > 0)
    if not in_comp.any():
        return float("nan")
    off = np.zeros(amb.shape)
    for lab in np.unique(prod_cc[in_comp]):
        m = valid & (prod_cc == lab)
        off[m] = np.rint(np.median(amb[m]))
    return float(np.mean((amb - off)[in_comp] == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlooks", type=float, default=16.0)
    args = ap.parse_args()

    sn = np.load(WD / "snaphu_ref" / "D_077.npz")
    prod_unw = sn["prod_unw"]
    prod_cc = sn["prod_cc"].astype(np.int64)
    coh = sn["coh"]
    mask = sn["mask"]
    wrapped = sn["wrapped"]
    snaphu_unw = sn["snaphu_unw"]
    print(
        f"snaphu: per-comp-vs-prod={sn['percomp']*100:.2f}%  runtime={sn['runtime_s']:.0f}s"
    )

    cvx = np.load(glob.glob(str(WD / "ww_gunw_convex/*D_077*/full_arrays.npz"))[0])
    ww_convex = cvx["ww_unw"]
    til = np.load(glob.glob(str(WD / "ww_gunw_expand/*D_077*/full_arrays.npz"))[0])
    ww_tiled = til["ww_unw"]

    fields = {
        "production": prod_unw,
        "snaphu": snaphu_unw,
        "ww_tiled": ww_tiled,
        "ww_convex_whole": ww_convex,
    }

    print("\n=== (A) convex MAP objective J (lower=cost-optimal) ===")
    Js = {}
    for name, phi in fields.items():
        Js[name] = map_objective(
            phi.astype(np.float64), wrapped, coh, mask, args.nlooks
        )
    base = Js["production"]
    for name, j in sorted(Js.items(), key=lambda kv: kv[1]):
        print(f"  {name:18s} J={j:.4e}   (x production = {j/base:.3f})")

    print("\n=== (B) cross per-component matches ===")
    valid = mask & np.isfinite(snaphu_unw)
    for name, phi in fields.items():
        if name == "production":
            continue
        v = mask & np.isfinite(phi)
        m_prod = percomp_match(phi, prod_unw, wrapped, prod_cc, v)
        print(f"  {name:18s} vs production per-comp = {m_prod*100:5.2f}%")
    # ww vs snaphu (does ww match snaphu's winding better than production's?)
    sn_cc_proxy = prod_cc  # align within production components
    for name in ("ww_tiled", "ww_convex_whole"):
        phi = fields[name]
        v = mask & np.isfinite(phi) & np.isfinite(snaphu_unw)
        m_sn = percomp_match(phi, snaphu_unw, wrapped, sn_cc_proxy, v)
        print(f"  {name:18s} vs snaphu     per-comp = {m_sn*100:5.2f}%")

    # (C) figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = 4
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))

    def sh(a, d, t, **kw):
        v = d[::st, ::st]
        im = a.imshow(v, **kw)
        a.set_title(t, fontsize=10)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)

    sh(
        ax[0, 0],
        wrapped,
        "wrapped input (rad)",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    sh(ax[0, 1], np.where(mask, coh, np.nan), "coherence", cmap="gray", vmin=0, vmax=1)
    pm = np.where(mask & (prod_cc > 0), prod_unw, np.nan)
    lo, hi = np.nanpercentile(pm, [2, 98])
    sh(ax[0, 2], pm, "production (NISAR GUNW=snaphu)", cmap="viridis", vmin=lo, vmax=hi)
    sh(
        ax[1, 0],
        np.where(valid, snaphu_unw, np.nan),
        f"snaphu re-run ({sn['percomp']*100:.0f}%)",
        cmap="viridis",
        vmin=lo,
        vmax=hi,
    )
    sh(ax[1, 1], np.where(mask, ww_tiled, np.nan), "ww tiled (default)", cmap="viridis")
    sh(
        ax[1, 2],
        np.where(mask, ww_convex, np.nan),
        "ww convex whole-image",
        cmap="viridis",
    )
    fig.suptitle("D_077: production vs snaphu vs whirlwind (tiled / convex-whole)")
    fig.tight_layout()
    out = WD / "snaphu_ref" / "D_077_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
