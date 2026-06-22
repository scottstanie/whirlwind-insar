#!/usr/bin/env python3
"""NISAR GUNW connected-component comparison: whirlwind vs production SNAPHU.

THIS IS THE ENTRY POINT for reproducing the NISAR GUNW conncomp comparison
figures the NISAR team reviews. One 8-panel figure per frame, written to
``./nisar-pngs/<YYYY-MM-DD>/``.

What each figure shows (2 rows x 4 columns)
-------------------------------------------
  1. wrapped phase            - the solver INPUT (re-wrapped production unwrap)
  2. coherence                - GUNW coherenceMagnitude (the cost weight)
  3. NISAR GUNW unwrapped      - production unwrappedPhase layer
  4. NISAR GUNW conncomps      - production ``connectedComponents`` layer.
        *** This is read straight from the GUNW HDF5, i.e. it IS the production
        SNAPHU result, NOT a re-run of SNAPHU on our side. ***  (team question 2)
  5. whirlwind unwrapped       - ww.unwrap phase, globally aligned to production
  6. whirlwind conncomps OLD   - the legacy linear-cost `components_only` labels
        used before 0.3.0 (tends to splinter into many tiny components)
  7. whirlwind conncomps NEW   - the default 0.3.0 SNAPHU-faithful
        `components_snaphu` ambiguity-wiggle labels (few large components, like
        SNAPHU) (team q 3/4)
  8. ambiguity diff (cycles)   - per-pixel 2pi disagreement ww vs production

Where the conncomps come from (provenance, team question 2)
-----------------------------------------------------------
* Panel 4 (the "SNAPHU" reference): the GUNW ``connectedComponents`` dataset.
  NISAR production unwrapping IS SNAPHU (cost=smooth, init=mcf), so this layer is
  the authoritative SNAPHU conncomp. We never plot our own tophu/snaphu re-run
  here (those live only in ``tophu_compare.py --save-dir`` and are not used by
  this figure).
* Panels 6/7 (whirlwind): grown on OUR side from the unwrapped phase output.

Water masking (team question 3, related)
----------------------------------------
Water IS masked out BEFORE the conncomp grows: ``water_only_mask`` (GUNW ``mask``
water flag) is folded into ``mask``, every conncomp edge touching a masked pixel
is cut, and masked pixels stay label 0. The NEW path additionally keys off the
*unwrapped phase output* directly (SNAPHU's ``GrowConnCompsMask`` reliability
test), which is what the team asked about.

Title / naming (team question 1)
--------------------------------
Each figure title and output filename carry the TRACK NUMBER (read from the GUNW
``/science/LSAR/identification``), e.g. ``T005_A_016`` = track 5, Ascending,
frame 16. (All 13 sample frames are track 5 except A_035, which is track 6.)

Data flow / how to reproduce from scratch
-----------------------------------------
  Stage 1 (heavy, once): scripts/plot_nisar_per_frame.py runs ww.unwrap on each
      GUNW frame and caches <frame>_panels.npz (wrapped, coh, mask, prod_unw,
      prod_cc, ww_unw, ww_cc) in CACHE_DIR. Heavy unwraps run ONE AT A TIME.
  Stage 2 (this script): reads those cached arrays, computes the NEW conncomp,
      and writes the 8-panel comparison PNGs. No re-unwrapping needed, so it is
      fast and safe to re-run. Pass --reunwrap to force stage 1 for a frame whose
      cache is missing or stale.

Usage
-----
  .venv/bin/python scripts/nisar_conncomp_compare.py            # all 13 frames
  .venv/bin/python scripts/nisar_conncomp_compare.py A_016 D_077
  .venv/bin/python scripts/nisar_conncomp_compare.py --reunwrap A_016

Requires a whirlwind build with `components_snaphu` (this branch):
  RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
      cargo build --release -p whirlwind-py
  cp target/release/lib_native.dylib python/whirlwind/_native.abi3.so
"""

from __future__ import annotations

import datetime as dt
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tophu_compare import gunw_layers, water_only_mask, wrap_phase, percomp_match

import whirlwind as ww

TWOPI = 2.0 * np.pi
H5DIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw"
# Stage-1 cache produced by scripts/plot_nisar_per_frame.py.
CACHE_DIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final"
NLOOKS = 16.0

# SNAPHU-faithful conncomp params (calibrated against the production GUNW
# connectedComponents: the reliability threshold barely moves the partition from
# 0 -> 5e4, so the calibration-free physical default of 0 is used; the 100-px
# floor matches SNAPHU's minregionsize convention).
RELIABILITY_THRESHOLD = 0
MIN_SIZE_PX = 100
MAX_NCOMPS = 4096

ALL_FRAMES = [
    "A_013",
    "A_016",
    "A_018",
    "A_020",
    "A_022",
    "A_025",
    "A_028",
    "A_030",
    "A_035",
    "D_074",
    "D_075",
    "D_077",
    "D_078",
]


def find_h5(frame: str) -> str:
    hits = glob.glob(f"{H5DIR}/*_{frame}_*.h5")
    if not hits:
        raise FileNotFoundError(f"no GUNW h5 for {frame} in {H5DIR}")
    return hits[0]


def identification(h5path: str) -> dict:
    """Authoritative track/frame/direction from the GUNW identification group."""
    with h5py.File(h5path, "r") as f:
        g = f["/science/LSAR/identification"]

        def s(key):
            v = g[key][()]
            return v.decode() if isinstance(v, bytes) else v

        track = int(g["trackNumber"][()])
        frame_no = int(g["frameNumber"][()])
        direction = s("orbitPassDirection")
        return {
            "track": track,
            "frame_no": frame_no,
            "direction": direction,
            "granule": s("granuleId"),
            "tag": f"T{track:03d}_{direction[0]}_{frame_no:03d}",
        }


def labels_for_show(cc: np.ndarray) -> np.ndarray:
    """0 (background/dropped) -> nan; remaining labels cycled into 1..20 so a
    categorical colormap stays readable regardless of the raw label count."""
    cc = np.asarray(cc).astype(np.int64)
    return np.where(cc > 0, ((cc - 1) % 20) + 1, np.nan).astype(float)


def ncc(cc: np.ndarray, valid: np.ndarray) -> int:
    v = cc[valid]
    return int(np.unique(v[v > 0]).size)


def load_or_unwrap(frame: str, reunwrap: bool) -> dict:
    """Return the per-frame arrays, from the stage-1 cache when available."""
    cache = Path(CACHE_DIR) / f"{frame}_panels.npz"
    if cache.exists() and not reunwrap:
        d = np.load(cache)
        return {
            "wrapped": d["wrapped"],
            "coh": d["coh"],
            "mask": d["mask"].astype(bool),
            "prod_unw": d["prod_unw"].astype(np.float32),
            "prod_cc": d["prod_cc"].astype(np.int64),
            "ww_unw": d["ww_unw"].astype(np.float32),  # already globally aligned
            "ww_cc_old": d["ww_cc"].astype(np.int64),
            "source": "cache",
        }

    # Stage 1 fallback: run the heavy unwrap (ONE AT A TIME on this laptop).
    h5 = find_h5(frame)
    with h5py.File(h5, "r") as h:
        _pol, prod_unw, coh, prod_cc, mask_arr = gunw_layers(h)
    mask = (
        water_only_mask(mask_arr, prod_unw.shape)
        & np.isfinite(prod_unw)
        & np.isfinite(coh)
    )
    wrapped = np.where(mask, wrap_phase(prod_unw), 0.0).astype(np.float32)
    ig = np.exp(1j * wrapped).astype(np.complex64)
    coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)
    unw, cc = ww.unwrap(ig, coh_in, NLOOKS, mask)
    unw = np.asarray(unw, np.float32)
    valid = mask & np.isfinite(unw)
    off = int(np.rint(np.nanmedian((unw[valid] - prod_unw[valid]) / TWOPI)))
    return {
        "wrapped": wrapped,
        "coh": coh_in,
        "mask": mask,
        "prod_unw": prod_unw.astype(np.float32),
        "prod_cc": prod_cc.astype(np.int64),
        "ww_unw": (unw - off * TWOPI).astype(np.float32),
        "ww_cc_old": np.asarray(cc).astype(np.int64),
        "source": "unwrap",
    }


def plot_frame(
    frame: str, out_dir: Path, reunwrap: bool = False, force: bool = False
) -> dict | None:
    ident = identification(find_h5(frame))
    out_png = out_dir / f"{ident['tag']}.png"

    a = load_or_unwrap(frame, reunwrap)
    wrapped, coh, mask = a["wrapped"], a["coh"], a["mask"]
    prod_unw, prod_cc = a["prod_unw"], a["prod_cc"]
    ww_unw, ww_cc_old = a["ww_unw"], a["ww_cc_old"]
    valid = mask & np.isfinite(ww_unw)

    # NEW SNAPHU-faithful conncomp: from correlation + unwrapped output only.
    # Water is already masked (mask) so masked edges are cut / masked px stay 0.
    ig = np.exp(1j * wrapped).astype(np.complex64)
    corr = np.clip(np.nan_to_num(coh), 0, 1).astype(np.float32)
    ww_cc_new = ww._native.components_snaphu(
        ig,
        corr,
        NLOOKS,
        ww_unw,
        mask,
        RELIABILITY_THRESHOLD,
        MIN_SIZE_PX,
        MAX_NCOMPS,
    ).astype(np.int64)

    # Headline metric + ambiguity-diff panel (relative to the shared wrapped input).
    pc = percomp_match(ww_unw, prod_unw, wrapped, prod_cc, valid)
    amb = np.rint((ww_unw - wrapped) / TWOPI) - np.rint((prod_unw - wrapped) / TWOPI)
    amb = np.where(valid, amb, np.nan)

    # Shared unwrapped-phase color scale from the production layer.
    pv = prod_unw[valid]
    lo, hi = np.nanpercentile(pv, [2, 98]) if pv.size else (-np.pi, np.pi)

    def m(x):
        return np.where(valid, x, np.nan)

    amax = float(
        max(np.nanpercentile(np.abs(amb), 99) if np.isfinite(amb).any() else 1.0, 1.0)
    )
    panels = [
        (m(wrapped), "1. wrapped phase (rad)", "twilight", -np.pi, np.pi),
        (m(coh), "2. coherence", "gray", 0.0, 1.0),
        (m(prod_unw), "3. NISAR GUNW unwrapped (rad)", "viridis", lo, hi),
        (
            labels_for_show(np.where(valid, prod_cc, 0)),
            f"4. NISAR GUNW conncomps = production SNAPHU (n={ncc(prod_cc, valid)})",
            "tab20",
            0,
            20,
        ),
        (m(ww_unw), "5. whirlwind unwrapped (rad)", "viridis", lo, hi),
        (
            labels_for_show(np.where(valid, ww_cc_old, 0)),
            f"6. whirlwind conncomps OLD linear (n={ncc(ww_cc_old, valid)})",
            "tab20",
            0,
            20,
        ),
        (
            labels_for_show(np.where(valid, ww_cc_new, 0)),
            f"7. whirlwind conncomps NEW SNAPHU-wiggle (n={ncc(ww_cc_new, valid)})",
            "tab20",
            0,
            20,
        ),
        (amb, "8. ambiguity diff ww-prod (cycles)", "RdBu", -amax, amax),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True)
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"Track {ident['track']}, {ident['direction']}, frame {ident['frame_no']} "
        f"({ident['tag']})   -   whirlwind vs NISAR GUNW (production SNAPHU)\n"
        f"{ident['granule']}\n"
        f"per-component 2pi match = {pc * 100:.1f}%   |   nlooks={NLOOKS:.0f}   |   "
        f"conncomp: production={ncc(prod_cc, valid)}  ww-old={ncc(ww_cc_old, valid)}  "
        f"ww-new={ncc(ww_cc_new, valid)}",
        fontsize=12,
    )
    if force or not out_png.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(
        f"{ident['tag']}: [{a['source']}] per-comp={pc * 100:.1f}%  "
        f"conncomp prod={ncc(prod_cc, valid)} old={ncc(ww_cc_old, valid)} "
        f"new={ncc(ww_cc_new, valid)} -> {out_png}",
        flush=True,
    )
    return {
        "frame": frame,
        "tag": ident["tag"],
        "track": ident["track"],
        "direction": ident["direction"],
        "frame_no": ident["frame_no"],
        "granule": ident["granule"],
        "percomp_match_pct": round(pc * 100, 2),
        "ncc_production_snaphu": ncc(prod_cc, valid),
        "ncc_ww_old_linear": ncc(ww_cc_old, valid),
        "ncc_ww_new_snaphu": ncc(ww_cc_new, valid),
        "source": a["source"],
    }


def main():
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    frames = [a for a in sys.argv[1:] if not a.startswith("--")] or ALL_FRAMES
    today = dt.date.today().isoformat()
    out_dir = Path("nisar-pngs") / today
    print(f"Writing {len(frames)} comparison figures to {out_dir}/", flush=True)
    rows = []
    for fr in frames:
        row = plot_frame(
            fr, out_dir, reunwrap="--reunwrap" in flags, force="--force" in flags
        )
        if row is not None:
            rows.append(row)
    if rows:
        import csv

        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "conncomp_summary.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {csv_path}", flush=True)
    print(f"DONE -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
