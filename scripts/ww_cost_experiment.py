"""Fast iteration harness for NISAR cost-variant experiments.

Loads inputs from ``/tmp/nisar-comparison/input/{ig,coh,mask}.npy`` (created
by ``scripts/nisar_ww_vs_snaphu.py --stage prep``) and a reference unwrap
from ``/tmp/nisar-comparison/snaphu_plain/{unw,conncomp}.npy``. Runs ww once,
compares against snaphu_plain on the common-conncomp mask, and saves:

* ``<out>/<label>/unw.npy``, ``conncomp.npy`` — ww outputs
* ``<out>/<label>/metrics.json`` — RMS, frac-within-π/2, frac-at-2π/4π/6π
* ``<out>/<label>/diff.png`` — error map + histogram
* appends a row to ``<out>/experiments_summary.csv``

Example::

    python scripts/ww_cost_experiment.py --label baseline --threshold 10
    python scripts/ww_cost_experiment.py --label exp1_scale10k --threshold 10
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np


def macos_rss_to_gb(maxrss: int) -> float:
    return (maxrss / (1024**3)) if sys.platform == "darwin" else (maxrss / (1024**2))


def peak_rss_gb() -> float:
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(macos_rss_to_gb(s), macos_rss_to_gb(c))


def aligned_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """b - a, snapped to the nearest 2π so they agree at the median."""
    d = b - a
    k = int(np.round(float(np.nanmedian(d[mask])) / (2 * np.pi)))
    return d - 2 * np.pi * k


def compute_metrics(diff: np.ndarray, mask: np.ndarray) -> dict:
    v = diff[mask & np.isfinite(diff)]
    abs_v = np.abs(v)
    return {
        "n_eval": int(v.size),
        "rms_rad": float(np.sqrt(np.mean(v**2))),
        "frac_within_pi_2": float((abs_v < np.pi / 2).mean()),
        "frac_within_pi": float((abs_v < np.pi).mean()),
        "frac_at_2pi": float(((abs_v >= np.pi) & (abs_v < 3 * np.pi)).mean()),
        "frac_at_4pi": float(((abs_v >= 3 * np.pi) & (abs_v < 5 * np.pi)).mean()),
        "frac_at_6pi_plus": float((abs_v >= 5 * np.pi).mean()),
    }


def run_ww(ig, coh, mask, nlooks, threshold):
    import whirlwind_rs as ww

    t0 = time.perf_counter()
    unw, cc = ww.unwrap_with_conncomp(
        ig, coh, float(nlooks), mask=mask, cost_threshold=threshold
    )
    return unw, cc, time.perf_counter() - t0


def make_diff_plot(
    diff: np.ndarray,
    eval_mask: np.ndarray,
    label: str,
    out_path: Path,
    metrics: dict,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    d_show = np.where(eval_mask, diff, np.nan)
    axes[0].imshow(d_show, cmap="RdBu_r", vmin=-2 * np.pi, vmax=2 * np.pi, interpolation="none")
    axes[0].set_title(
        f"{label}: snaphu_plain − ww  ({metrics['n_eval'] / 1e6:.1f}M px)\n"
        f"RMS={metrics['rms_rad']:.2f} rad  "
        f"within±π/2={100 * metrics['frac_within_pi_2']:.1f}%"
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    v = diff[eval_mask & np.isfinite(diff)]
    axes[1].hist(v, bins=200, range=(-4 * np.pi, 4 * np.pi), log=True)
    for k in (-2, -1, 0, 1, 2):
        axes[1].axvline(2 * np.pi * k, color="r", lw=0.5, alpha=0.4)
    axes[1].set_xlabel("rad")
    axes[1].set_title(
        f"diff histogram  "
        f"2π={100 * metrics['frac_at_2pi']:.1f}%  "
        f"4π={100 * metrics['frac_at_4pi']:.1f}%  "
        f"6π+={100 * metrics['frac_at_6pi_plus']:.1f}%"
    )

    fig.savefig(out_path / "diff.png", dpi=130)
    plt.close(fig)


def append_summary(summary_csv: Path, row: dict, fieldnames: list[str]) -> None:
    import csv

    new = not summary_csv.exists()
    with open(summary_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/nisar-comparison"))
    ap.add_argument("--label", required=True, help="experiment name (subdir)")
    ap.add_argument("--reference", default="snaphu_plain")
    ap.add_argument("--nlooks", type=float, default=100.0)
    ap.add_argument("--threshold", type=int, default=10, help="ww conncomp cost_threshold")
    ap.add_argument("--note", default="", help="free-text note saved into metrics.json")
    args = ap.parse_args()

    inp = args.out / "input"
    ref_dir = args.out / args.reference
    out_dir = args.out / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    ig = np.load(inp / "ig.npy")
    coh = np.load(inp / "coh.npy")
    mask = np.load(inp / "mask.npy")
    print(f"[{args.label}] inputs loaded  ig={ig.shape}  valid={int(mask.sum())}")
    print(f"[{args.label}] running ww (threshold={args.threshold})...")

    unw, cc, elapsed = run_ww(ig, coh, mask, args.nlooks, args.threshold)
    peak = peak_rss_gb()
    print(f"[{args.label}] elapsed {elapsed:.1f}s  peak {peak:.2f} GB")

    np.save(out_dir / "unw.npy", unw)
    np.save(out_dir / "conncomp.npy", cc)

    # Compare against reference.
    ref_unw = np.load(ref_dir / "unw.npy")
    ref_cc = np.load(ref_dir / "conncomp.npy")
    eval_mask = (cc > 0) & (ref_cc > 0) & mask

    diff = aligned_diff(unw, ref_unw, eval_mask)
    metrics = compute_metrics(diff, eval_mask)
    metrics["label"] = args.label
    metrics["reference"] = args.reference
    metrics["threshold"] = args.threshold
    metrics["elapsed_sec"] = elapsed
    metrics["peak_rss_gb"] = peak
    metrics["ww_coverage"] = float(((cc > 0) & mask).mean())
    metrics["eval_coverage"] = float(eval_mask.mean())
    metrics["note"] = args.note

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    make_diff_plot(diff, eval_mask, args.label, out_dir, metrics)

    summary = args.out / "experiments_summary.csv"
    cols = [
        "label", "reference", "threshold", "elapsed_sec", "peak_rss_gb",
        "ww_coverage", "eval_coverage",
        "rms_rad", "frac_within_pi_2", "frac_within_pi",
        "frac_at_2pi", "frac_at_4pi", "frac_at_6pi_plus", "note",
    ]
    append_summary(summary, metrics, cols)
    print(f"[{args.label}] wrote {out_dir} and appended {summary.name}")


if __name__ == "__main__":
    main()
