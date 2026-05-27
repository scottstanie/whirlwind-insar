"""Sanity-check that Goldstein-default-on doesn't break clean Palos Verdes
phase-linked interferograms.

Loads a handful of Dolphin IGs, runs ``unwrap_with_conncomp`` with
``goldstein_alpha=0`` (legacy behaviour) and with the new default
(``alpha=0.7, psize=64``), compares outputs.

If Goldstein hurts on clean data we'd expect:
  * Coverage drops (conncomp threshold sees more 'cut' arcs)
  * Visible halos near sharp features
  * Edge-effect smearing
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio


def read_window(path: Path, window):
    with rasterio.open(path) as src:
        return src.read(1, window=window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dolphin",
        type=Path,
        default=Path(
            "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes"
            "/opera-cslc-s1/Palos_Verdes_Landslides_D071/dolphin"
        ),
    )
    ap.add_argument("--out", type=Path, default=Path("/tmp/pv-goldstein-check"))
    ap.add_argument("--row0", type=int, default=0)
    ap.add_argument("--col0", type=int, default=0)
    ap.add_argument("--size", type=int, default=0, help="0 = whole raster")
    ap.add_argument("--n-igs", type=int, default=3)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    win = (
        rasterio.windows.Window(args.col0, args.row0, args.size, args.size)
        if args.size > 0 else None
    )
    ig_dir = args.dolphin / "interferograms"
    pairs = sorted({p.stem.split(".")[0] for p in ig_dir.glob("*.int.tif")})[: args.n_igs]
    print(f"[pv] testing {len(pairs)} IGs at row {args.row0}:{args.row0+args.size} "
          f"col {args.col0}:{args.col0+args.size}")

    import whirlwind_rs as ww

    for pair in pairs:
        ig = read_window(ig_dir / f"{pair}.int.tif", win).astype(np.complex64)
        cor = read_window(ig_dir / f"{pair}.int.cor.tif", win).astype(np.float32)
        mask = np.isfinite(cor) & (cor > 0) & (np.abs(ig) > 0)
        ig[~mask] = 0
        cor[~mask] = 0
        cor = np.clip(cor, 0.0, 1.0)
        print(f"\n[{pair}] valid {int(mask.sum())}/{ig.size}  median coh={float(np.median(cor[mask])):.3f}")

        t0 = time.perf_counter()
        unw_off, cc_off = ww.unwrap_with_conncomp(
            ig, cor, 100.0, mask=mask, cost_threshold=10, goldstein_alpha=0.0
        )
        t_off = time.perf_counter() - t0

        t0 = time.perf_counter()
        unw_on, cc_on = ww.unwrap_with_conncomp(
            ig, cor, 100.0, mask=mask, cost_threshold=10, goldstein_alpha=0.7
        )
        t_on = time.perf_counter() - t0

        em = (cc_off > 0) & (cc_on > 0) & mask
        diff = unw_on - unw_off
        if em.any():
            k = int(np.round(float(np.nanmedian(diff[em])) / (2 * np.pi)))
            diff -= 2 * np.pi * k
            v = diff[em]
            print(f"  off → on  cov {100*((cc_off>0)&mask).mean():.1f}% → "
                  f"{100*((cc_on>0)&mask).mean():.1f}%"
                  f"  t {t_off:.1f}s → {t_on:.1f}s")
            print(f"  pairwise diff: RMS {float(np.sqrt(np.mean(v**2))):.3f} rad  "
                  f"within±π/2 {100*float((np.abs(v) < np.pi/2).mean()):.2f}%  "
                  f"max |diff| {float(np.abs(v).max()):.2f} rad")

        # Save figures for visual inspection
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
            axes[0].imshow(np.where(mask, np.angle(ig), np.nan), cmap="twilight",
                           vmin=-np.pi, vmax=np.pi, interpolation="none")
            axes[0].set_title(f"{pair} wrapped phase")
            ref_off = np.nanmedian(unw_off[em]) if em.any() else 0
            ref_on = np.nanmedian(unw_on[em]) if em.any() else 0
            axes[1].imshow(np.where(cc_off > 0, unw_off - ref_off, np.nan),
                           cmap="twilight", vmin=-12, vmax=12, interpolation="none")
            axes[1].set_title(f"Goldstein OFF cov={100*((cc_off>0)&mask).mean():.1f}%")
            axes[2].imshow(np.where(cc_on > 0, unw_on - ref_on, np.nan),
                           cmap="twilight", vmin=-12, vmax=12, interpolation="none")
            axes[2].set_title(f"Goldstein ON cov={100*((cc_on>0)&mask).mean():.1f}%")
            axes[3].imshow(np.where(em, diff, np.nan), cmap="RdBu_r",
                           vmin=-2 * np.pi, vmax=2 * np.pi, interpolation="none")
            axes[3].set_title("ON - OFF (aligned to nearest 2π)")
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(args.out / f"{pair}.png", dpi=120)
            plt.close(fig)
        except ImportError:
            pass

    print(f"\n[pv] figures in {args.out}/")


if __name__ == "__main__":
    main()
