"""One 6-panel figure per NISAR GUNW frame comparing the production unwrap to
whirlwind's. Panels:

  1. wrapped phase (the re-wrapped production unwrappedPhase = solver input)
  2. coherence
  3. NISAR GUNW unwrapped phase (production layer)
  4. NISAR connected-component labels (production layer)
  5. whirlwind unwrapped phase (default ``ww.unwrap``, globally aligned to prod)
  6. whirlwind connected-component labels

Runs the default public ``ww.unwrap`` (single-tile linear + bridge), which returns
conncomp labels. Heavy unwraps run STRICTLY ONE AT A TIME (see the laptop memory
note). Resume-friendly: a frame whose PNG already exists is skipped unless
``--force``. The full per-frame arrays are saved next to each PNG as
``<frame>_panels.npz`` so the figure can be regenerated without re-unwrapping.

Usage: python scripts/plot_nisar_per_frame.py [FRAMES...] [--force]
"""

import glob
import sys
import time

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase, percomp_match

import whirlwind as ww

TWOPI = 2.0 * np.pi
H5DIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw"
OUT = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final"
NLOOKS = 16.0

ALL_FRAMES = [
    "005_A_013",
    "005_A_016",
    "005_A_018",
    "005_A_020",
    "005_A_022",
    "005_A_025",
    "005_A_028",
    "005_A_030",
    "006_A_035",
    "005_D_074",
    "005_D_075",
    "005_D_077",
    "005_D_078",
]


def find_h5(frame):
    hits = glob.glob(f"{H5DIR}/*_{frame}_*.h5")
    if not hits:
        raise FileNotFoundError(f"no GUNW h5 for {frame} in {H5DIR}")
    return hits[0]


def labels_for_show(cc):
    """0 (background/dropped) -> nan; remaining labels cycled into 1..20 so a
    categorical colormap stays readable regardless of the raw label count."""
    cc = np.asarray(cc).astype(np.int64)
    out = np.where(cc > 0, ((cc - 1) % 20) + 1, np.nan).astype(float)
    return out


def plot_frame(frame, force=False):
    out_png = f"{OUT}/{frame}_panels.png"
    out_npz = f"{OUT}/{frame}_panels.npz"
    if not force and glob.glob(out_png):
        print(f"{frame}: exists, skip (--force to redo)", flush=True)
        return

    h5 = find_h5(frame)
    with h5py.File(h5, "r") as h:
        pol, prod_unw, coh, prod_cc, mask_arr = gunw_layers(h)
    mask = (
        water_only_mask(mask_arr, prod_unw.shape)
        & np.isfinite(prod_unw)
        & np.isfinite(coh)
    )
    wrapped = np.where(mask, wrap_phase(prod_unw), 0.0).astype(np.float32)
    ig = np.exp(1j * wrapped).astype(np.complex64)
    coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)

    t0 = time.perf_counter()
    unw, cc = ww.unwrap(ig, coh_in, NLOOKS, mask)
    dt = time.perf_counter() - t0
    unw = np.asarray(unw, np.float32)
    cc = np.asarray(cc).astype(np.int64)

    valid = mask & np.isfinite(unw)
    pc = percomp_match(unw, prod_unw, wrapped, prod_cc, valid)
    # Remove a single global 2pi offset so the two unwrapped panels share a scale.
    off = int(np.rint(np.nanmedian((unw[valid] - prod_unw[valid]) / TWOPI)))
    unw_aligned = unw - off * TWOPI

    np.savez_compressed(
        out_npz,
        wrapped=wrapped,
        coh=coh_in,
        mask=mask,
        prod_unw=prod_unw.astype(np.float32),
        prod_cc=prod_cc.astype(np.int64),
        ww_unw=unw_aligned.astype(np.float32),
        ww_cc=cc,
    )

    # Shared unwrapped-phase color scale from the production layer.
    pv = prod_unw[valid]
    lo, hi = np.nanpercentile(pv, [2, 98]) if pv.size else (-np.pi, np.pi)

    def m(a):
        return np.where(valid, a, np.nan)

    panels = [
        (m(wrapped), "1. wrapped phase (rad)", "twilight", -np.pi, np.pi),
        (m(coh_in), "2. coherence", "gray", 0.0, 1.0),
        (m(prod_unw), "3. NISAR GUNW unwrapped (rad)", "viridis", lo, hi),
        (
            labels_for_show(np.where(valid, prod_cc, 0)),
            f"4. NISAR conncomps (n={int(np.unique(prod_cc[prod_cc>0]).size)})",
            "tab20",
            0,
            20,
        ),
        (m(unw_aligned), "5. whirlwind unwrapped (rad)", "viridis", lo, hi),
        (
            labels_for_show(np.where(valid, cc, 0)),
            f"6. whirlwind conncomps (n={int(cc.max())})",
            "tab20",
            0,
            20,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"{frame}  (pol={pol}, nlooks={NLOOKS:.0f})  -  whirlwind vs NISAR GUNW: "
        f"per-comp match {pc * 100:.1f}%, {dt:.1f}s",
        fontsize=13,
    )
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(
        f"{frame}: {dt:.1f}s  per-comp={pc * 100:.1f}%  ncc={int(cc.max())} -> {out_png}",
        flush=True,
    )


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    frames = args or ALL_FRAMES
    for fr in frames:
        plot_frame(fr, force=force)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
