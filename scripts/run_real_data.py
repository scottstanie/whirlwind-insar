"""Run whirlwind-rs on real interferograms and save PNG comparisons.

Two scenes are tested:
  1. Rosamond Capella (clean, short-baseline)
  2. Palos Verdes Sentinel-1 OPERA CSLC (noisier)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    print("rasterio not installed; pip install rasterio", file=sys.stderr)
    raise

import whirlwind as ww
from whirlwind.plot import save_wrapped_unwrapped_png


OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def read_tif(path: Path, dtype=None) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1)
    if dtype is not None:
        a = a.astype(dtype)
    return a


def unwrap_and_save(
    *,
    igram_path: Path,
    cor_path: Path,
    nlooks: float,
    name: str,
    crop: tuple[int, int, int, int] | None = None,
    mask_path: Path | None = None,
) -> None:
    print(f"\n=== {name} ===")
    igram = read_tif(igram_path)
    cor = read_tif(cor_path, dtype=np.float32)
    if crop:
        i0, i1, j0, j1 = crop
        igram = igram[i0:i1, j0:j1]
        cor = cor[i0:i1, j0:j1]

    # rasterio returns complex64 if it's a complex TIFF; otherwise it's the
    # wrapped phase, so map to exp(i·phase). NaN-fill before any math.
    if np.iscomplexobj(igram):
        igram = np.nan_to_num(igram, nan=0.0).astype(np.complex64)
    else:
        phase = np.nan_to_num(igram.astype(np.float32), nan=0.0)
        igram = np.exp(1j * phase).astype(np.complex64)

    cor = np.nan_to_num(cor, nan=0.0).clip(0.0, 0.999).astype(np.float32)

    print(f"  shape: {igram.shape}, dtype: {igram.dtype}")
    print(f"  coherence median: {np.median(cor):.3f}")

    mask = None
    if mask_path is not None and mask_path.exists():
        mask = read_tif(mask_path).astype(bool)
        if crop:
            mask = mask[i0:i1, j0:j1]
        print(f"  mask: {mask.sum()} / {mask.size} valid")

    t0 = time.perf_counter()
    unw = ww.unwrap(igram, cor, float(nlooks), mask)
    dt = time.perf_counter() - t0
    print(f"  unwrap took {dt:.2f}s ({igram.size / dt / 1e6:.2f} Mpx/s)")

    wrapped = np.angle(igram)
    save_wrapped_unwrapped_png(
        wrapped, unw, OUT_DIR / f"{name}.png", title=name, cor=cor
    )
    print(f"  wrote {OUT_DIR / (name + '.png')}")


def main() -> int:
    rosamond_dir = Path(
        "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/rosamond/"
        "Rosamond_C13_RO43_eSM/insar_output/network_output/20250807_20250810"
    )
    palos_dir = Path(
        "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes/"
        "opera-cslc-s1/Palos_Verdes_Landslides_D071/dolphin/interferograms"
    )

    do_rosamond = "--rosamond" in sys.argv
    if rosamond_dir.exists() and do_rosamond:
        # Capella scene is noisy (median coherence ~0.2-0.4) so the unwrap is
        # slow; opt in with --rosamond.
        phs = next(rosamond_dir.glob("*PHS*.tif"), None)
        coh = next(rosamond_dir.glob("*COH*.tif"), None)
        if phs and coh:
            unwrap_and_save(
                igram_path=phs,
                cor_path=coh,
                nlooks=4.0,
                name="rosamond_20250807_20250810",
                crop=(1000, 1000 + 256, 1000, 1000 + 256),
            )

    if palos_dir.exists():
        # Pick one Sentinel-1 pair (12-day).
        pairs = sorted(palos_dir.glob("*_*.int.tif"))[:2]
        for ig in pairs:
            stem = ig.stem.replace(".int", "")
            cor = palos_dir / f"{stem}.int.cor.tif"
            mask = palos_dir / f"{stem}.int.mask.tif"
            if not cor.exists():
                continue
            unwrap_and_save(
                igram_path=ig,
                cor_path=cor,
                nlooks=15.0,
                name=f"palos_{stem}",
                mask_path=mask if mask.exists() else None,
            )

    print(f"\nAll PNGs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
