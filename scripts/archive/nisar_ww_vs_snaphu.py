"""NISAR 40 MHz HH 50 m posting (~6.8k x 6.9k) — ww vs SNAPHU comparison.

Three methods on the same interferogram (10 x 10 boxcar looks):

* ``ww``           — ``ww.unwrap_with_conncomp`` (whirlwind-rs MCF).
* ``snaphu_plain`` — ``snaphu.unwrap`` single-tile, ``cost='smooth'``.
* ``snaphu_tiled`` — ``snaphu.unwrap`` with ``ntiles=(3, 3)``.

Each runs in its own subprocess so we can measure peak RSS cleanly via
``resource.getrusage``. After all three finish, the orchestrator builds
comparison plots evaluated on the *common-conncomp* mask
(intersection of every method's conncomp > 0).

How to run::

    uv run --with snaphu --with rasterio --with matplotlib \\
        python scripts/nisar_ww_vs_snaphu.py \\
        --out /tmp/nisar-comparison \\
        --igram /Volumes/.../20251224_20260117.int.looked.tif \\
        --coh   /Volumes/.../20251224_20260117.int.coh.looked.tif \\
        --nlooks 100

Stages:

  1. ``--stage prep``    load inputs, save ``input/{ig,coh,mask}.npy``
  2. ``--stage run``     launch the 3 method subprocesses in sequence
  3. ``--stage plot``    build figures from the saved outputs

Default ``--stage all`` does everything end-to-end.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def macos_rss_to_gb(maxrss: int) -> float:
    """``ru_maxrss`` is bytes on macOS, kibibytes on Linux."""
    return (maxrss / (1024**3)) if sys.platform == "darwin" else (maxrss / (1024**2))


def peak_rss_gb() -> float:
    """Max of self RSS and largest single child RSS (in GB).

    SNAPHU spawns a subprocess for the actual C unwrapper, so its memory
    only shows up under ``RUSAGE_CHILDREN``. Whirlwind runs in-process,
    so its memory is under ``RUSAGE_SELF``. Take the max to cover both.
    """
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(macos_rss_to_gb(s), macos_rss_to_gb(c))


# ---------------------------------------------------------------------------
# Stage 1: prep
# ---------------------------------------------------------------------------


def stage_prep(igram_path: Path, coh_path: Path, out_dir: Path) -> None:
    import rasterio

    inp = out_dir / "input"
    inp.mkdir(parents=True, exist_ok=True)

    print(f"[prep] reading {igram_path.name}")
    with rasterio.open(igram_path) as src:
        ig = src.read(1).astype(np.complex64)
    print(f"[prep] reading {coh_path.name}")
    with rasterio.open(coh_path) as src:
        coh = src.read(1).astype(np.float32)

    assert ig.shape == coh.shape, (ig.shape, coh.shape)
    mask = np.isfinite(coh) & (coh > 0) & (coh < 1e10) & (np.abs(ig) > 0)
    # Zero out NoData consistently so neither method sees garbage.
    ig = ig.copy()
    ig[~mask] = 0
    coh = coh.copy()
    coh[~mask] = 0.0
    coh = np.clip(coh, 0.0, 1.0)

    n_valid = int(mask.sum())
    print(f"[prep] shape={ig.shape}  valid={n_valid} ({100 * mask.mean():.1f}%)")
    print(f"[prep] saving npy to {inp}")
    np.save(inp / "ig.npy", ig)
    np.save(inp / "coh.npy", coh)
    np.save(inp / "mask.npy", mask)
    with open(inp / "meta.json", "w") as f:
        json.dump(
            {
                "shape": list(ig.shape),
                "n_valid": n_valid,
                "valid_frac": float(mask.mean()),
                "igram_path": str(igram_path),
                "coh_path": str(coh_path),
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Stage 2: per-method runners (each invoked as a subprocess)
# ---------------------------------------------------------------------------


def run_ww(
    ig: np.ndarray,
    coh: np.ndarray,
    mask: np.ndarray,
    nlooks: float,
    cost_threshold: int = 50,
):
    import whirlwind as ww

    t0 = time.perf_counter()
    unw, cc = ww.unwrap(
        ig,
        coh,
        float(nlooks),
        mask=mask,
        cost_threshold=cost_threshold,
        goldstein_alpha=0.7,
    )
    return unw, cc, time.perf_counter() - t0


def run_snaphu(
    ig: np.ndarray,
    coh: np.ndarray,
    mask: np.ndarray,
    nlooks: float,
    ntiles: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    import snaphu

    # snaphu mask: 1 = valid, 0 = masked-out
    mask_u8 = mask.astype(np.uint8)
    t0 = time.perf_counter()
    # tile_cost_thresh + min_region_size eased per SNAPHU's
    # "Exceeded maximum number of secondary arcs" failure mode at default
    # values for this 6.8k x 6.9k NISAR scene.
    unw, cc = snaphu.unwrap(
        ig,
        coh,
        nlooks=float(nlooks),
        cost="smooth",
        mask=mask_u8,
        ntiles=ntiles,
        tile_overlap=500 if ntiles != (1, 1) else 0,
        nproc=os.cpu_count() or 1,
        tile_cost_thresh=200,
        min_region_size=200,
    )
    return (
        np.asarray(unw, dtype=np.float32),
        np.asarray(cc, dtype=np.uint32),
        time.perf_counter() - t0,
    )


METHOD_DISPATCH = {
    "ww": lambda ig, coh, mask, nlooks: run_ww(
        ig, coh, mask, nlooks, cost_threshold=50
    ),
    "ww_T10": lambda ig, coh, mask, nlooks: run_ww(
        ig, coh, mask, nlooks, cost_threshold=10
    ),
    "ww_T15": lambda ig, coh, mask, nlooks: run_ww(
        ig, coh, mask, nlooks, cost_threshold=15
    ),
    "ww_llr": lambda ig, coh, mask, nlooks: run_ww(
        ig, coh, mask, nlooks, cost_threshold=10
    ),
    "snaphu_plain": lambda ig, coh, mask, nlooks: run_snaphu(
        ig, coh, mask, nlooks, (1, 1)
    ),
    "snaphu_tiled": lambda ig, coh, mask, nlooks: run_snaphu(
        ig, coh, mask, nlooks, (3, 3)
    ),
}


def stage_run_one(method: str, out_dir: Path, nlooks: float) -> None:
    inp = out_dir / "input"
    ig = np.load(inp / "ig.npy")
    coh = np.load(inp / "coh.npy")
    mask = np.load(inp / "mask.npy")
    print(
        f"[{method}] inputs loaded  ig={ig.shape} {ig.dtype}  valid={int(mask.sum())}"
    )
    print(f"[{method}] starting...")
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    unw, cc, elapsed = METHOD_DISPATCH[method](ig, coh, mask, nlooks)

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak_rss_gb()
    print(
        f"[{method}] elapsed {elapsed:.1f}s  peak RSS {peak_gb:.2f} GB  (self+children max)"
    )

    out = out_dir / method
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "unw.npy", unw)
    np.save(out / "conncomp.npy", cc)
    with open(out / "timing.json", "w") as f:
        json.dump(
            {
                "method": method,
                "elapsed_sec": elapsed,
                "peak_rss_gb": peak_gb,
                "rss_before_gb": macos_rss_to_gb(rss_before),
                "n_components": int(cc.max()),
                "coverage": float((cc > 0).mean()),
            },
            f,
            indent=2,
        )


def stage_run_all(out_dir: Path, nlooks: float, methods: list[str]) -> None:
    script = Path(__file__).resolve()
    for m in methods:
        print(f"\n=== launching subprocess for {m} ===")
        env = {**os.environ}
        env.pop("CONDA_PREFIX", None)
        rc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--stage",
                "run-one",
                "--method",
                m,
                "--out",
                str(out_dir),
                "--nlooks",
                str(nlooks),
            ],
            env=env,
        ).returncode
        if rc != 0:
            print(f"[run] {m} FAILED rc={rc}")


# ---------------------------------------------------------------------------
# Stage 3: plots
# ---------------------------------------------------------------------------


def stage_plot(out_dir: Path, methods: list[str]) -> None:
    import matplotlib.pyplot as plt

    # Wait — the masks differ between methods because some pixels are not
    # covered by any conncomp. The fair eval mask is the *intersection* of
    # the per-method conncomp > 0 masks AND the input data mask.
    input_mask = np.load(out_dir / "input" / "mask.npy")
    ccs = {m: np.load(out_dir / m / "conncomp.npy") for m in methods}
    unws = {m: np.load(out_dir / m / "unw.npy") for m in methods}
    timings = {m: json.load(open(out_dir / m / "timing.json")) for m in methods}

    eval_mask = input_mask.copy()
    for m, cc in ccs.items():
        eval_mask &= cc > 0
    eval_frac = float(eval_mask.mean())
    print(f"[plot] eval mask: {int(eval_mask.sum())} px ({100 * eval_frac:.1f}%)")

    h, w = input_mask.shape
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    im_kw = dict(interpolation="none")

    # Wrapped phase (load from npy)
    ig = np.load(out_dir / "input" / "ig.npy")
    wrap = np.angle(ig)
    wrap_show = np.where(input_mask, wrap, np.nan)
    axes[0, 0].imshow(wrap_show, cmap="twilight", vmin=-np.pi, vmax=np.pi, **im_kw)
    axes[0, 0].set_title("Wrapped phase (input)")

    # Per-method unwrap visualisations on each method's own conncomp
    for ax, m in zip(axes[0, 1:], methods[:2]):
        u = np.where(ccs[m] > 0, unws[m], np.nan)
        # Reference each unw to the eval-mask median for color comparability
        if eval_mask.any():
            u = u - np.nanmedian(unws[m][eval_mask])
        ax.imshow(u, cmap="twilight", vmin=-12, vmax=12, **im_kw)
        cov = timings[m]["coverage"]
        ax.set_title(
            f"{m}\nt={timings[m]['elapsed_sec']:.1f}s  "
            f"peak={timings[m]['peak_rss_gb']:.1f}GB  cov={100 * cov:.1f}%"
        )

    # Third method on bottom-left
    if len(methods) >= 3:
        m = methods[2]
        u = np.where(ccs[m] > 0, unws[m], np.nan)
        if eval_mask.any():
            u = u - np.nanmedian(unws[m][eval_mask])
        axes[1, 0].imshow(u, cmap="twilight", vmin=-12, vmax=12, **im_kw)
        cov = timings[m]["coverage"]
        axes[1, 0].set_title(
            f"{m}\nt={timings[m]['elapsed_sec']:.1f}s  "
            f"peak={timings[m]['peak_rss_gb']:.1f}GB  cov={100 * cov:.1f}%"
        )

    # Diff: ww vs snaphu_plain on eval mask
    def aligned_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not eval_mask.any():
            return np.full_like(a, np.nan)
        d = b - a
        k = int(np.round(float(np.nanmedian(d[eval_mask])) / (2 * np.pi)))
        d = d - 2 * np.pi * k
        d[~eval_mask] = np.nan
        return d

    diffs: dict[str, np.ndarray] = {}
    ww_methods = [m for m in methods if m.startswith("ww")]
    ww_ref = ww_methods[0] if ww_methods else None
    if ww_ref is not None and "snaphu_plain" in methods:
        diff_wp = aligned_diff(unws[ww_ref], unws["snaphu_plain"])
        diffs[f"snaphu_plain − {ww_ref}"] = diff_wp
        rms = float(np.sqrt(np.nanmean(diff_wp**2)))
        axes[1, 1].imshow(
            diff_wp, cmap="RdBu_r", vmin=-2 * np.pi, vmax=2 * np.pi, **im_kw
        )
        axes[1, 1].set_title(f"snaphu_plain − {ww_ref}  (RMS={rms:.3f} rad)")

    if ww_ref is not None and "snaphu_tiled" in methods:
        diff_wt = aligned_diff(unws[ww_ref], unws["snaphu_tiled"])
        diffs[f"snaphu_tiled − {ww_ref}"] = diff_wt
        rms = float(np.sqrt(np.nanmean(diff_wt**2)))
        axes[1, 2].imshow(
            diff_wt, cmap="RdBu_r", vmin=-2 * np.pi, vmax=2 * np.pi, **im_kw
        )
        axes[1, 2].set_title(f"snaphu_tiled − {ww_ref}  (RMS={rms:.3f} rad)")

    if "snaphu_plain" in methods and "snaphu_tiled" in methods:
        diff_st = aligned_diff(unws["snaphu_plain"], unws["snaphu_tiled"])
        diffs["snaphu_tiled − snaphu_plain"] = diff_st

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    out_path = out_dir / "plots"
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / "overview.png", dpi=200)
    plt.close(fig)

    # Diff histograms — show whether method disagreement is structured at
    # integer multiples of 2π (true topology disagreement) or continuous noise.
    if diffs:
        fig, axes = plt.subplots(
            1,
            len(diffs),
            figsize=(6 * len(diffs), 4),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, (name, d) in zip(axes[0], diffs.items()):
            v = d[eval_mask & np.isfinite(d)]
            ax.hist(v, bins=200, range=(-4 * np.pi, 4 * np.pi), log=True)
            for k in (-3, -2, -1, 0, 1, 2, 3):
                ax.axvline(2 * np.pi * k, color="r", lw=0.5, alpha=0.4)
            frac_within = float((np.abs(v) < np.pi / 2).mean())
            rms = float(np.sqrt(np.mean(v**2)))
            ax.set_title(
                f"{name}\nRMS={rms:.3f} rad  {100 * frac_within:.1f}% within ±π/2"
            )
            ax.set_xlabel("rad")
        fig.savefig(out_path / "diff_histograms.png", dpi=120)
        plt.close(fig)

    # Per-method full-resolution unwrap panel (separate figure, high-DPI)
    # so the user can actually read pixel-level structure.
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(6 * len(methods), 6),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, m in zip(axes[0], methods):
        u = np.where(ccs[m] > 0, unws[m], np.nan)
        if eval_mask.any():
            u = u - np.nanmedian(unws[m][eval_mask])
        ax.imshow(u, cmap="twilight", vmin=-12, vmax=12, **im_kw)
        t = timings[m]
        ax.set_title(
            f"{m}\nt={t['elapsed_sec']:.0f}s  peak={t['peak_rss_gb']:.1f}GB  "
            f"cov={100 * t['coverage']:.1f}%  components={t['n_components']}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_path / "unwraps_hires.png", dpi=200)
    plt.close(fig)

    # Coverage mask figure
    fig, axes = plt.subplots(
        1,
        len(methods) + 1,
        figsize=(4 * (len(methods) + 1), 4),
        constrained_layout=True,
    )
    axes[0].imshow(input_mask, cmap="Greys_r", **im_kw)
    axes[0].set_title(f"input mask\n{100 * input_mask.mean():.1f}% valid")
    for ax, m in zip(axes[1:], methods):
        cov_mask = ccs[m] > 0
        ax.imshow(cov_mask, cmap="Greys_r", **im_kw)
        ax.set_title(
            f"{m}\n{100 * cov_mask.mean():.1f}% coverage\n{timings[m]['n_components']} components"
        )
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_path / "coverage.png", dpi=120)
    plt.close(fig)

    # Summary table
    table = {
        "shape": list(input_mask.shape),
        "input_valid_frac": float(input_mask.mean()),
        "eval_mask_frac": eval_frac,
        "methods": {m: timings[m] for m in methods},
    }
    with open(out_path / "timings.json", "w") as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2))
    print(f"[plot] figures in {out_path}/")

    del unws, ccs
    gc.collect()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/nisar-comparison"))
    ap.add_argument(
        "--igram",
        type=Path,
        default=Path(
            "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar/20251224_20260117.int.looked.tif"
        ),
    )
    ap.add_argument(
        "--coh",
        type=Path,
        default=Path(
            "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar/20251224_20260117.int.coh.looked.cleaned.tif"
        ),
    )
    ap.add_argument(
        "--nlooks",
        type=float,
        default=100.0,
        help="10 range x 10 az boxcar looks ⇒ nlooks=100",
    )
    ap.add_argument(
        "--stage",
        choices=("all", "prep", "run", "run-one", "plot"),
        default="all",
    )
    ap.add_argument(
        "--method",
        choices=tuple(METHOD_DISPATCH.keys()),
        help="for --stage run-one",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["ww", "snaphu_plain", "snaphu_tiled"],
        help="methods to run / include in plots",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "prep"):
        stage_prep(args.igram, args.coh, args.out)
    if args.stage in ("all", "run"):
        stage_run_all(args.out, args.nlooks, args.methods)
    if args.stage == "run-one":
        assert args.method is not None
        stage_run_one(args.method, args.out, args.nlooks)
    if args.stage in ("all", "plot"):
        stage_plot(args.out, args.methods)


if __name__ == "__main__":
    main()
