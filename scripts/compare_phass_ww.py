#!/usr/bin/env python3
"""Visual comparison: production (=snaphu) vs whirlwind-insar vs whirlwind-orig vs PHASS.

Top row = unwrapped phase (shared color scale; seams/runaway show as sharp color
discontinuities / wrong large-scale gradient). Bottom row = connected components
(fragmentation visible). Judge ARTIFACTS by eye, not per-comp %.

Sources:
  ww (insar)   : ww_gunw_lk16cap/<frame>/full_arrays.npz
  ww (orig)    : phass_ref/<frame>_wworig.npz  (run_whirlwind_orig.py)
  PHASS        : phass_ref/<frame>_phass.npz   (tophu_compare.py --save-dir)
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

TWOPI = 2.0 * np.pi
WD = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning")


def align(field, prod, mask):
    """Remove a global integer-2π offset so phase color scales are comparable."""
    v = mask & np.isfinite(field) & np.isfinite(prod)
    if not v.any():
        return field
    off = np.round(np.median((field[v] - prod[v]) / TWOPI)) * TWOPI
    return field - off


def percomp_match(test_unw, prod_unw, wrapped, prod_cc, mask):
    amb = np.rint((test_unw - wrapped) / TWOPI) - np.rint((prod_unw - wrapped) / TWOPI)
    in_comp = mask & (prod_cc > 0)
    if not in_comp.any():
        return float("nan")
    off = np.zeros(amb.shape)
    for lab in np.unique(prod_cc[in_comp]):
        m = mask & (prod_cc == lab)
        off[m] = np.rint(np.median(amb[m]))
    return float(np.mean((amb - off)[in_comp] == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", default=["D_077", "D_078", "A_035"])
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fr in args.frames:
        wwg = glob.glob(str(WD / f"ww_gunw_lk16cap/*_{fr}_*/full_arrays.npz"))
        phg = glob.glob(str(WD / f"phass_ref/*{fr}*_phass.npz"))
        orig_g = glob.glob(str(WD / f"phass_ref/{fr}_wworig.npz"))

        if not wwg:
            print(f"{fr}: missing ww-insar npz"); continue

        ww = np.load(wwg[0])
        prod = ww["prod_unw"].astype(np.float32)
        mask = ww["mask"]
        wrapped = np.where(mask, (prod + np.pi) % TWOPI - np.pi, 0.0).astype(np.float32)
        prod_cc = ww["prod_cc"]
        ww_f = align(ww["ww_aligned"].astype(np.float32), prod, mask)
        ww_cc = ww["ww_cc"]

        # Determine layout: 4 columns if both PHASS and orig are present, else 3 or 2
        have_phass = bool(phg) and np.load(phg[0])["unw"].shape == prod.shape
        have_orig = bool(orig_g)
        n_cols = 1 + 1 + int(have_orig) + int(have_phass)  # prod + ww + [orig] + [phass]

        cols_data = [("production\n(=snaphu)", prod, prod_cc)]
        cols_data.append(("whirlwind-insar\n(tiled, default)", ww_f, ww_cc))

        if have_orig:
            or_d = np.load(orig_g[0])
            orig_f = align(or_d["unw"].astype(np.float32), prod, mask)
            orig_cc = np.zeros(prod.shape, np.int32)  # no cc from orig, show as single comp
            orig_cc[mask & np.isfinite(orig_f)] = 1
            cols_data.append(("whirlwind-orig\n(whole-image)", orig_f, orig_cc))

        if have_phass:
            ph = np.load(phg[0])
            ph_f = align(ph["unw"].astype(np.float32), prod, mask)
            cols_data.append(("PHASS\n(region-grow)", ph_f, ph["cc"]))

        # Compute per-comp matches for title
        matches = []
        for label, field, cc in cols_data:
            if "production" in label:
                matches.append("ref")
            else:
                pc = percomp_match(field, prod, wrapped, prod_cc, mask)
                ncc = int(np.asarray(cc)[mask].max()) if mask.any() else 0
                matches.append(f"{pc*100:.1f}% / {ncc}cc" if not np.isnan(pc) else "?")

        st = args.stride
        def m(a):
            return np.where(mask, a, np.nan)[::st, ::st]
        lo, hi = np.nanpercentile(np.where(mask & (prod_cc > 0), prod, np.nan), [2, 98])

        fig, ax = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
        for col_i, (label, field, cc) in enumerate(cols_data):
            a_top = ax[0, col_i]
            im = a_top.imshow(m(field), cmap="viridis", vmin=lo, vmax=hi)
            a_top.set_title(f"{label}\n{matches[col_i]}", fontsize=9)
            a_top.axis("off")
            fig.colorbar(im, ax=a_top, fraction=0.046)

            a_bot = ax[1, col_i]
            lab = np.where(mask & (cc > 0), ((cc - 1) % 20) + 1, 0).astype(float)
            lab[lab == 0] = np.nan
            a_bot.imshow(lab[::st, ::st], cmap="tab20", vmin=1, vmax=20)
            n = int(np.asarray(cc)[mask].max()) if mask.any() else 0
            a_bot.set_title(f"conncomps (n={n})", fontsize=9)
            a_bot.axis("off")

        fig.suptitle(f"{fr}: judge artifacts by eye (per-comp-match vs production)")
        fig.tight_layout()
        out = WD / "phass_ref" / f"compare_{fr}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"{fr} -> {out}")


if __name__ == "__main__":
    main()
