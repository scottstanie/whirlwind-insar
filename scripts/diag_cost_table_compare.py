#!/usr/bin/env python3
"""A/B a NISAR frame's unwrap under two Carballo cost-table sets.

Isolates the effect of the cost LOOKUP TABLE only: same frame, same Rust solver
(`ww._native.unwrap_linear_ext_costs`), same smoothing/coherence/nlooks - the ONLY thing that
changes is which `carballo-pdf-{0,1}-spline.npz` is used to build the arc costs.

  table A (shipping) : whirlwind/src/whirlwind_orig/  (the embedded ww-orig blobs)
  table B (script)   : output of scripts/generate_carballo_tables.py (full model)

Reports, per frame: cost differences, A-vs-B ambiguity disagreement, and each
table's ambiguity match against the production GUNW unwrapped phase.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage
from scipy.interpolate import RegularGridInterpolator

import whirlwind as ww

TWOPI = 2.0 * np.pi
UNW_BASE = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"


def load_rgi(npz: Path) -> RegularGridInterpolator:
    d = np.load(npz, allow_pickle=False)
    return RegularGridInterpolator(
        (d["grid_0"], d["grid_1"], d["grid_2"]),
        d["values"],
        method=str(d["method"]),
        bounds_error=bool(d["bounds_error"]),
        fill_value=float(d["fill_value"]),
    )


def compute_carballo_costs(igram, corr, nlooks, mask, p0_npz, p1_npz):
    """Port of whirlwind_orig._cost.compute_carballo_costs, table dir swappable."""
    dy = igram[1:, :] * igram[:-1, :].conj()
    dx = igram[:, 1:] * igram[:, :-1].conj()
    phase_dy = scipy.ndimage.uniform_filter(np.angle(dy), size=(7, 7), mode="nearest")
    phase_dx = scipy.ndimage.uniform_filter(np.angle(dx), size=(7, 7), mode="nearest")

    corr_dy = np.minimum(corr[1:, :], corr[:-1, :])
    corr_dx = np.minimum(corr[:, 1:], corr[:, :-1])
    s0, s1 = load_rgi(p0_npz), load_rgi(p1_npz)

    def cost(phase_diff, min_corr):
        pd, mc = phase_diff.ravel(), min_corr.ravel()
        p1 = s1((pd, mc, nlooks))
        p0 = s0((pd, mc, nlooks))
        return (-np.log(p1 / p0)).reshape(phase_diff.shape)

    cost_up = cost(-phase_dx, corr_dx)
    cost_lt = cost(phase_dy, corr_dy)
    cost_dn = cost(phase_dx, corr_dx)
    cost_rt = cost(-phase_dy, corr_dy)

    if mask is not None:
        mdy = mask[1:, :] & mask[:-1, :]
        mdx = mask[:, 1:] & mask[:, :-1]
        for c, m in ((cost_dn, mdx), (cost_up, mdx), (cost_rt, mdy), (cost_lt, mdy)):
            c[~m] = np.nan

    flat = np.concatenate(
        [
            np.pad(cost_up, [(0, 0), (1, 1)]).flatten(),
            np.pad(cost_lt, [(1, 1), (0, 0)]).flatten(),
            np.pad(cost_dn, [(0, 0), (1, 1)]).flatten(),
            np.pad(cost_rt, [(1, 1), (0, 0)]).flatten(),
        ]
    )
    flat[np.isnan(flat)] = 0.0
    return (100.0 * flat).astype(np.int32)


def remap_py_to_rust(costs_py: np.ndarray, m: int, n: int) -> np.ndarray:
    """Python compute_carballo_costs layout [UP,LEFT,DOWN,RIGHT] -> Rust arc
    order [DOWN,UP,RIGHT,LEFT] expected by unwrap_linear_ext_costs."""
    n_v, n_h = m * (n + 1), (m + 1) * n
    up = costs_py[0:n_v]
    lt = costs_py[n_v : n_v + n_h]
    dn = costs_py[n_v + n_h : 2 * n_v + n_h]
    rt = costs_py[2 * n_v + n_h : 2 * n_v + 2 * n_h]
    return np.ascontiguousarray(np.concatenate([dn, up, rt, lt]), dtype=np.int32)


def load_frame(h5_path: Path, size: int):
    with h5py.File(h5_path, "r") as h5:
        grp = h5[UNW_BASE]
        pol = sorted(
            k
            for k, v in grp.items()
            if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"}
        )[0]
        unw = h5[f"{UNW_BASE}/{pol}/unwrappedPhase"][()].astype(np.float64)
        coh = h5[f"{UNW_BASE}/{pol}/coherenceMagnitude"][()].astype(np.float64)
        maskcode = h5[f"{UNW_BASE}/mask"][()]
    ny, nx = unw.shape
    if size and (size < ny and size < nx):
        y0, x0 = (ny - size) // 2, (nx - size) // 2
        sl = (slice(y0, y0 + size), slice(x0, x0 + size))
        unw, coh, maskcode = unw[sl], coh[sl], maskcode[sl]
    # nisar_land mask: non-water, valid in both subswaths
    water = (maskcode // 100) % 10
    ref_sub, sec_sub = (maskcode // 10) % 10, maskcode % 10
    mask = (maskcode != 255) & (water == 0) & (ref_sub > 0) & (sec_sub > 0)
    bad = ~np.isfinite(unw) | ~np.isfinite(coh)
    mask &= ~bad
    unw = np.where(np.isfinite(unw), unw, 0.0)
    coh = np.clip(np.where(np.isfinite(coh), coh, 0.0), 0.0, 1.0)
    wrapped = np.angle(np.exp(1j * unw))  # re-wrap production phase
    igram = np.exp(1j * wrapped).astype(np.complex64)
    return igram, coh, mask, unw


def amb_match_vs_prod(ww_unw, prod_unw, igram_phase, valid):
    off = int(np.rint(np.nanmedian((ww_unw[valid] - prod_unw[valid]) / TWOPI)))
    a = ww_unw - off * TWOPI
    prod_amb = np.rint((prod_unw - igram_phase) / TWOPI)
    ww_amb = np.rint((a - igram_phase) / TWOPI)
    return float(np.mean(prod_amb[valid] == ww_amb[valid]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--table-a",
        type=Path,
        default=Path("/Users/staniewi/repos/whirlwind/src/whirlwind_orig"),
    )
    ap.add_argument("--table-b", type=Path, required=True)
    ap.add_argument("--nlooks", type=float, default=16.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for h5_path in args.h5:
        tag = (
            h5_path.name.split("_GUNW_")[1][:18].replace("/", "_")
            if "_GUNW_" in h5_path.name
            else h5_path.stem
        )
        igram, coh, mask, prod_unw = load_frame(h5_path, args.size)
        ph = np.angle(igram).astype(np.float64)

        cA = compute_carballo_costs(
            igram,
            coh,
            args.nlooks,
            mask,
            args.table_a / "carballo-pdf-0-spline.npz",
            args.table_a / "carballo-pdf-1-spline.npz",
        )
        cB = compute_carballo_costs(
            igram,
            coh,
            args.nlooks,
            mask,
            args.table_b / "carballo-pdf-0-spline.npz",
            args.table_b / "carballo-pdf-1-spline.npz",
        )
        m, n = igram.shape
        rcA, rcB = remap_py_to_rust(cA, m, n), remap_py_to_rust(cB, m, n)
        # REAL shipping solver (capacity-1 linear / unwrap_linear), external costs,
        # + the same default bridge post-pass ww.unwrap applies (load-bearing on
        # bridging-dependent frames like A_025).
        unwA = np.asarray(ww._native.unwrap_linear_ext_costs(igram, mask, rcA)).astype(
            np.float32
        )
        unwB = np.asarray(ww._native.unwrap_linear_ext_costs(igram, mask, rcB)).astype(
            np.float32
        )
        unwA = np.asarray(ww.bridge_components(unwA, mask)).astype(np.float64)
        unwB = np.asarray(ww.bridge_components(unwB, mask)).astype(np.float64)
        # Sanity: true public ww.unwrap (embedded table A, trilinear) should match unwA.
        unw_ship = np.asarray(ww.unwrap(igram, coh, args.nlooks, mask=mask)[0]).astype(
            np.float64
        )

        valid = mask & np.isfinite(unwA) & np.isfinite(unwB) & np.isfinite(prod_unw)
        ds = unw_ship - unwA
        ds -= np.rint(np.nanmedian(ds[valid]) / TWOPI) * TWOPI
        ship_vs_extA = float(np.mean(np.rint(ds[valid] / TWOPI) != 0))
        # A vs B
        d = unwB - unwA
        d -= np.rint(np.nanmedian(d[valid]) / TWOPI) * TWOPI
        amb = np.rint(d / TWOPI)
        disagree = float(np.mean(amb[valid] != 0))
        rms_wrapped = float(np.sqrt(np.nanmean(np.angle(np.exp(1j * d[valid])) ** 2)))
        mA = amb_match_vs_prod(unwA, prod_unw, ph, valid)
        mB = amb_match_vs_prod(unwB, prod_unw, ph, valid)
        dcost = np.abs(cB - cA)

        print(
            f"\n=== {tag}  ({valid.sum()} valid px, nlooks={args.nlooks}) ===",
            flush=True,
        )
        print(
            f"  SANITY ext-costs(A) vs true ww.unwrap: {100*ship_vs_extA:.3f}% cycle diff (want ~0)",
            flush=True,
        )
        print(
            f"  cost |B-A|: median={np.median(dcost):.0f} mean={dcost.mean():.0f} max={dcost.max()}",
            flush=True,
        )
        print(
            f"  A-vs-B ambiguity disagreement: {100*disagree:.3f}% of valid px",
            flush=True,
        )
        print(f"  A-vs-B wrapped-residual RMS:   {rms_wrapped:.4f} rad", flush=True)
        print(
            f"  ambiguity match vs production:  A(shipping)={100*mA:.3f}%   B(script)={100*mB:.3f}%",
            flush=True,
        )

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        v = np.where(valid, unwA, np.nan)
        im0 = ax[0].imshow(v, cmap="twilight")
        ax[0].set_title(f"{tag}\nA (shipping)")
        plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(np.where(valid, unwB, np.nan), cmap="twilight")
        ax[1].set_title("B (script)")
        plt.colorbar(im1, ax=ax[1])
        im2 = ax[2].imshow(np.where(valid, amb, np.nan), cmap="RdBu", vmin=-2, vmax=2)
        ax[2].set_title(f"B-A ambiguity (cycles)\ndisagree {100*disagree:.2f}%")
        plt.colorbar(im2, ax=ax[2])
        out = args.out_dir / f"cost_table_compare_{tag}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  figure: {out}")


if __name__ == "__main__":
    main()
