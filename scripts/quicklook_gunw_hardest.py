"""Quicklook the wrapped phase + coherence of GUNW .h5 files.

Usage::

    python scripts/quicklook_gunw_hardest.py FILE.h5 [FILE2.h5 ...] --out-dir DIR

Reads the 20 m wrappedInterferogram and unwrapped-grid coherenceMagnitude at a
decimation that keeps each panel around 1500 px wide, and writes one PNG per
product. Meant for eyeballing whether a slow/failing frame is garbled noise or
real signal.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

WRAP = "/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/{pol}/wrappedInterferogram"
COH = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}/coherenceMagnitude"
CC = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}/connectedComponents"


def decim(dset, target=1500):
    step = max(1, dset.shape[1] // target)
    return dset[::step, ::step], step


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--pol", default="HH")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for f in args.files:
        with h5py.File(f, "r") as h5:
            wrapped, step_w = decim(h5[WRAP.format(pol=args.pol)])
            coh, _ = decim(h5[COH.format(pol=args.pol)])
            cc, _ = decim(h5[CC.format(pol=args.pol)])

        phase = np.angle(wrapped)
        mag = np.abs(wrapped)
        phase[mag == 0] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
        im0 = axes[0].imshow(phase, cmap="twilight_shifted", interpolation="nearest")
        axes[0].set_title(f"wrapped phase (20 m, every {step_w}px)")
        fig.colorbar(im0, ax=axes[0], shrink=0.8)
        im1 = axes[1].imshow(coh, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title("coherence (80 m unwrap grid)")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)
        im2 = axes[2].imshow(cc != 0, cmap="viridis", interpolation="nearest")
        axes[2].set_title("production conncomp != 0")
        fig.colorbar(im2, ax=axes[2], shrink=0.8)
        fig.suptitle(f.stem[:80], fontsize=9)

        out = args.out_dir / f"quicklook_{f.stem[:60]}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(out.resolve())


if __name__ == "__main__":
    main()
