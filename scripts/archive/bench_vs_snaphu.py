"""Benchmark whirlwind-rs against snaphu on synthetic + real interferograms.

Reports a runtime table and saves a 4-panel PNG per scene
(wrapped, whirlwind-rs unwrapped, snaphu unwrapped, coherence).
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import snaphu
except ImportError:
    snaphu = None

import whirlwind as ww

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def _bench(name: str, fn, igram, corr, nlooks, mask: Optional[np.ndarray] = None):
    t0 = time.perf_counter()
    out = fn(igram, corr, nlooks, mask)
    dt = time.perf_counter() - t0
    print(f"  {name:14s}: {dt:7.3f}s ({igram.size / dt / 1e6:5.2f} Mpx/s)")
    return out, dt


def _ww_unwrap(igram, corr, nlooks, mask):
    unw, _cc = ww.unwrap(igram, corr, float(nlooks), mask)
    return unw


def _snaphu_unwrap(igram, corr, nlooks, mask):
    unw, _ = snaphu.unwrap(
        igram, corr, nlooks=float(nlooks), cost="smooth", mask=mask
    )
    return unw.astype(np.float32)


def save_compare_png(
    wrapped: np.ndarray,
    ww_unw: np.ndarray,
    sn_unw: Optional[np.ndarray],
    cor: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    panels = [("wrapped", wrapped, "twilight_shifted", (-np.pi, np.pi))]
    panels.append(("whirlwind-unwrap", ww_unw, "viridis", (None, None)))
    if sn_unw is not None:
        panels.append(("snaphu", sn_unw, "viridis", (None, None)))
    panels.append(("coherence", cor, "gray", (0.0, 1.0)))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), constrained_layout=True)
    fig.suptitle(title)
    for ax, (label, arr, cmap, lim) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
        ax.set_title(label)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def bench_scene(name: str, igram: np.ndarray, corr: np.ndarray, *, nlooks: float, mask: Optional[np.ndarray] = None):
    print(f"\n=== {name} (shape={igram.shape}, cor_med={float(np.median(corr)):.3f}) ===")
    results = {}
    ww_unw, ww_t = _bench("whirlwind-unwrap", _ww_unwrap, igram, corr, nlooks, mask)
    results["whirlwind-unwrap"] = (ww_unw, ww_t)
    if snaphu is not None:
        try:
            sn_unw, sn_t = _bench("snaphu", _snaphu_unwrap, igram, corr, nlooks, mask)
            results["snaphu"] = (sn_unw, sn_t)
        except Exception as e:
            print(f"  snaphu failed: {e}")
            results["snaphu"] = (None, None)
    save_compare_png(
        np.angle(igram), ww_unw,
        results.get("snaphu", (None, None))[0],
        corr,
        OUT_DIR / f"bench_{name}.png",
        title=name,
    )
    return results


def synthetic_diagonal_ramp(size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.ogrid[-3:3:size * 1j, -3:3:size * 1j]
    truth = (np.pi * (x + y)).astype(np.float32)
    igram = np.exp(1j * truth).astype(np.complex64)
    corr = np.full(igram.shape, 0.99, dtype=np.float32)
    return igram, corr, truth


def synthetic_noisy_bump(size: int = 256, nlooks: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.zeros((size, size), dtype=np.float32)
    ci = cj = (size - 1) / 2
    sigma = size / 8.0
    for i in range(size):
        for j in range(size):
            truth[i, j] = 6.0 * np.exp(-((i - ci) ** 2 + (j - cj) ** 2) / (2 * sigma ** 2))
    gamma = np.full((size, size), 0.85, dtype=np.float32)
    igram, corr = ww.simulate_ifg(truth, gamma, nlooks=nlooks, seed=42)
    return igram, corr, truth


def load_capella(path_phs: Path, path_coh: Path, crop=None):
    with rasterio.open(path_phs) as src:
        phase = src.read(1)
    with rasterio.open(path_coh) as src:
        cor = src.read(1)
    if crop:
        i0, i1, j0, j1 = crop
        phase = phase[i0:i1, j0:j1]
        cor = cor[i0:i1, j0:j1]
    phase = np.nan_to_num(phase.astype(np.float32), nan=0.0)
    cor = np.nan_to_num(cor.astype(np.float32), nan=0.0).clip(0, 0.999)
    igram = np.exp(1j * phase).astype(np.complex64)
    return igram, cor


def load_opera(path_ig: Path, path_cor: Path, path_mask: Optional[Path] = None):
    with rasterio.open(path_ig) as src:
        igram = src.read(1)
    with rasterio.open(path_cor) as src:
        cor = src.read(1)
    igram = np.nan_to_num(igram, nan=0.0).astype(np.complex64)
    cor = np.nan_to_num(cor, nan=0.0).astype(np.float32).clip(0, 0.999)
    mask = None
    if path_mask and path_mask.exists():
        with rasterio.open(path_mask) as src:
            mask = src.read(1).astype(bool)
    return igram, cor, mask


def main():
    print("Whirlwind-rs vs SNAPHU benchmark")
    print(f"  snaphu installed: {snaphu is not None and getattr(snaphu, '__version__', '?')}")
    print(f"  rasterio installed: {rasterio is not None}")

    bench_table = []

    # --- Synthetic ---
    ig, co, _ = synthetic_diagonal_ramp(512)
    r = bench_scene("synth_diagonal_ramp_512", ig, co, nlooks=1.0)
    bench_table.append(("diagonal ramp 512x512", r))

    ig, co, _ = synthetic_noisy_bump(256, nlooks=10)
    r = bench_scene("synth_noisy_bump_256", ig, co, nlooks=10.0)
    bench_table.append(("noisy bump 256x256", r))

    # --- Real ---
    if rasterio is not None:
        rosamond_dir = Path(
            "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/rosamond/"
            "Rosamond_C13_RO43_eSM/insar_output/network_output/20250807_20250810"
        )
        phs = next(rosamond_dir.glob("*PHS*.tif"), None) if rosamond_dir.exists() else None
        coh = next(rosamond_dir.glob("*COH*.tif"), None) if rosamond_dir.exists() else None
        if phs and coh:
            ig, co = load_capella(phs, coh, crop=(1000, 1000 + 512, 1000, 1000 + 512))
            r = bench_scene("rosamond_512x512", ig, co, nlooks=4.0)
            bench_table.append(("Rosamond Capella 512x512", r))

        palos_dir = Path(
            "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes/"
            "opera-cslc-s1/Palos_Verdes_Landslides_D071/dolphin/interferograms"
        )
        if palos_dir.exists():
            pair = next(palos_dir.glob("*_*.int.tif"), None)
            if pair:
                stem = pair.stem.replace(".int", "")
                cor = palos_dir / f"{stem}.int.cor.tif"
                mask = palos_dir / f"{stem}.int.mask.tif"
                ig, co, m = load_opera(pair, cor, mask)
                r = bench_scene(f"palos_{stem}", ig, co, nlooks=15.0, mask=m)
                bench_table.append((f"Palos-Verdes S1 {stem}", r))

    # --- Summary table ---
    print("\n\n" + "=" * 72)
    print(f"{'Scene':40s} | {'whirlwind-rs':>14s} | {'snaphu':>10s} | speedup")
    print("-" * 72)
    for name, results in bench_table:
        ww_t = results.get("whirlwind-unwrap", (None, None))[1]
        sn_t = results.get("snaphu", (None, None))[1]
        ww_s = f"{ww_t:7.3f}s" if ww_t else "        n/a"
        sn_s = f"{sn_t:7.3f}s" if sn_t else "    n/a"
        if ww_t and sn_t:
            speed = f"{sn_t / ww_t:5.2f}x"
        else:
            speed = "    n/a"
        print(f"{name:40s} | {ww_s:>14s} | {sn_s:>10s} | {speed}")
    print("=" * 72)
    print(f"PNGs saved to {OUT_DIR}/bench_*.png")


if __name__ == "__main__":
    main()
