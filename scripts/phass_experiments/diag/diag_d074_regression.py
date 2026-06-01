"""WHY does clean D_074 crater 98%->81% at tile>=1024 (both costs)? This is the
blocker to using bigger tiles (which fix A_016). Run D_074 at tile512 (good) and
tile1024 (bad), save both, and plot ambiguity-diff + coherence + conncomps to
locate the breakage (region flip? seam? anchor mis-level?).

Outputs a PNG; path printed at the end.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import whirlwind as ww

GUNW = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw")
FN = "NISAR_L2_PR_GUNW_003_005_D_074_004_4000_SH_20251017T132342_20251017T132345_20251029T132342_20251029T132346_X05010_N_P_J_001.h5"
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/d074_diag")
OUT.mkdir(parents=True, exist_ok=True)
TAU = 2 * np.pi
UNW = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def modal(x):
    x = x[np.isfinite(x)].astype(np.int64)
    return int(np.bincount(x - x.min()).argmax() + x.min())


def main() -> None:
    with h5py.File(GUNW / FN, "r") as h5:
        grp = h5[UNW]
        pol = sorted(k for k, v in grp.items() if isinstance(v, h5py.Group) and k.upper() not in {"MASK", "METADATA"})[0]
        prod = h5[f"{UNW}/{pol}/unwrappedPhase"][()].astype(np.float32)
        coh = h5[f"{UNW}/{pol}/coherenceMagnitude"][()].astype(np.float32)
        pcc = h5[f"{UNW}/{pol}/connectedComponents"][()].astype(np.int64)
        maskd = h5[f"{UNW}/mask"][()] if f"{UNW}/mask" in h5 else None
    valid = (maskd != 127) if maskd is not None else np.ones(prod.shape, bool)
    valid &= np.isfinite(prod) & np.isfinite(coh)
    ig = wrap(prod).astype(np.float32)
    igc = np.ascontiguousarray(np.exp(1j * ig), dtype=np.complex64)
    cohc = np.ascontiguousarray(coh, dtype=np.float32)
    maskc = np.ascontiguousarray(valid, dtype=bool)
    reg = valid & (pcc > 0)
    print(f"D_074 {prod.shape}  valid_frac={valid.mean():.3f}  prod_ncc={int(pcc[reg].max())}", flush=True)

    amb = {}
    for size in (512, 1024):
        unw_phase, _cc = ww.unwrap(igc, cohc, 16.0, maskc, tile_size=size, tile_overlap=64)
        unw = np.asarray(unw_phase, np.float64)
        a = np.rint((unw - prod) / TAU)
        g = modal(a[reg])
        a = a - g
        match = 100.0 * float(np.mean(np.abs(a[reg]) < 0.5))
        amb[size] = np.where(valid, a, np.nan)
        print(f"  tile{size}: match={match:.2f}%  modal_offset={g}", flush=True)
        # per-column wrong fraction to locate the breakage
        wrong = (np.abs(a) >= 0.5) & reg
        colw = wrong.sum(0) / np.maximum(1, reg.sum(0))
        bad = np.where(colw > 0.5)[0]
        if bad.size:
            print(f"    tile{size}: >50%-wrong column span {bad.min()}..{bad.max()} ({bad.size} cols)", flush=True)

    s = (slice(None, None, 3), slice(None, None, 3))
    fig, ax = plt.subplots(2, 2, figsize=(15, 14), constrained_layout=True)
    im = ax[0, 0].imshow(np.where(valid, coh, np.nan)[s], cmap="gray", vmin=0, vmax=1)
    ax[0, 0].set_title("coherence"); fig.colorbar(im, ax=ax[0, 0], shrink=0.7)
    im = ax[0, 1].imshow(np.where(reg, pcc, np.nan)[s], cmap="tab10")
    ax[0, 1].set_title(f"prod conncomps (n={int(pcc[reg].max())})"); fig.colorbar(im, ax=ax[0, 1], shrink=0.7)
    im = ax[1, 0].imshow(amb[512][s], cmap="RdBu", vmin=-2, vmax=2)
    ax[1, 0].set_title("ambiguity diff: tile512 (98%)"); fig.colorbar(im, ax=ax[1, 0], shrink=0.7)
    im = ax[1, 1].imshow(amb[1024][s], cmap="RdBu", vmin=-2, vmax=2)
    ax[1, 1].set_title("ambiguity diff: tile1024 (81% — what flipped?)"); fig.colorbar(im, ax=ax[1, 1], shrink=0.7)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    p = OUT / "d074_regression.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"PLOT: {p}", flush=True)


if __name__ == "__main__":
    main()
