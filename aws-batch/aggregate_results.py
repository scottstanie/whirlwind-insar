#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "matplotlib", "numpy"]
# ///
"""Roll a whole campaign of per-granule GUNW comparisons into one table + plots.

Reads everything ``run_local.py`` produced under ``--root``:

* ``results/<granule>/<granule>/<crop>.json`` -- the comparison stats written by
  ``compare_gunw.py`` (agreement, components, coherence, ...).
* ``runs.jsonl`` -- the runner's wall time and true peak RSS per job.
* the manifest's ``.meta.csv`` sidecar, if given -- track/frame/bounding box for
  spatial analysis and grouping.

Writes ``campaign.csv`` (one row per granule x crop), ``campaign.md`` (headline
numbers and the worst frames to look at first), and a figure panel.

Example::

    uv run aggregate_results.py --root /data/ww-bench --meta manifest.meta.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter, ScalarFormatter  # noqa: E402

# The headline accuracy metric: agreement with the production unwrap on the 2*pi
# integer, after re-levelling within each production connected component.
SCORE = "ambiguity_match_frac_percomp"

# Keep the quality tiers identical anywhere agreement is encoded. Marker shape
# makes the categories readable in monochrome as well as in colour.
QUALITY_TIERS = (
    ("agreement >=99%", 0.99, 1.01, "#2a9d8f", "o", 15, 0.50),
    ("agreement 90-99%", 0.90, 0.99, "#e9a23b", "D", 22, 0.72),
    ("agreement <90%", -0.01, 0.90, "#d1495b", "X", 32, 0.90),
)


def norm_key(name: str) -> str:
    """Normalise a granule reference to the runner's job id.

    The three sources spell the same product differently -- ``compare_gunw.py``
    records it with the ``.h5`` suffix, the runner and the manifest without --
    so everything is funnelled through this before joining.
    """
    stem = re.sub(r"\.(h5|hdf5)$", "", str(name), flags=re.IGNORECASE)
    return stem.replace(".", "_")


def load_results(root: Path) -> pd.DataFrame:
    files = sorted((root / "results").glob("*/*/*.json"))
    assert files, f"No per-granule JSON under {root / 'results'}."
    rows = []
    for f in files:
        rec = json.loads(f.read_text())
        rec.setdefault("product", f.parent.name)
        rec.setdefault("crop", f.stem)
        rec["json_path"] = str(f)
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} comparison rows from {len(files)} JSON files", flush=True)
    return df


def load_runs(root: Path) -> pd.DataFrame | None:
    runs_path = root / "runs.jsonl"
    if not runs_path.exists():
        print(f"  (no {runs_path}; skipping runtime/peak-memory join)", flush=True)
        return None
    recs = [json.loads(l) for l in runs_path.read_text().splitlines() if l.strip()]
    runs = pd.DataFrame(recs)
    # A job may have been rerun; keep the last record for each.
    runs = runs.drop_duplicates("job_id", keep="last")
    return runs[
        [
            "job_id",
            "wall_s",
            "peak_rss_mb",
            "workers",
            "threads_per_worker",
            "returncode",
        ]
    ]


def merge_all(root: Path, meta_csv: Path | None) -> pd.DataFrame:
    df = load_results(root)
    df["key"] = df["product"].map(norm_key)
    runs = load_runs(root)
    if runs is not None:
        runs["key"] = runs["job_id"].map(norm_key)
        n_before = df["key"].isin(runs["key"]).sum()
        df = df.merge(runs.drop(columns=["job_id"]), on="key", how="left")
        assert n_before > 0, (
            "No comparison row matched a runs.jsonl record. The job-id and "
            "product-name conventions have diverged; check norm_key()."
        )
        if n_before < len(df):
            print(
                f"  note: {len(df) - n_before} rows have no runner record "
                "(runtime/peak memory will be blank for those)",
                flush=True,
            )
    if meta_csv is not None:
        meta = pd.read_csv(meta_csv)
        meta["key"] = meta["granule"].map(norm_key)
        keep = [
            "key",
            "track",
            "frame",
            "direction",
            "fraction_land",
            "min_lon",
            "min_lat",
            "max_lon",
            "max_lat",
            "temporal_baseline_days",
        ]
        df = df.merge(meta[[c for c in keep if c in meta]], on="key", how="left")
    df["megapixels"] = df["num_pixels"] / 1e6
    if "peak_rss_mb" in df:
        df["gb_per_mpx"] = (df["peak_rss_mb"] / 1e3) / df["megapixels"]
    return df


def scatter_quality_tiers(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: pd.Series,
    y: pd.Series,
    *,
    legend: bool = False,
) -> None:
    """Scatter x/y using the campaign's three agreement tiers."""
    finite = x.notna() & y.notna() & df[SCORE].notna()
    for label, low, high, color, marker, size, alpha in QUALITY_TIERS:
        use = finite & df[SCORE].ge(low) & df[SCORE].lt(high)
        if use.any():
            ax.scatter(
                x[use],
                y[use],
                s=size,
                alpha=alpha,
                c=color,
                marker=marker,
                linewidths=0.25,
                label=label,
                rasterized=True,
            )
    if legend:
        ax.legend(loc="best", frameon=False, fontsize=8)


def make_plots(df: pd.DataFrame, out_png: Path) -> None:
    has_mem = "peak_rss_mb" in df and df["peak_rss_mb"].notna().any()
    has_recall = all(
        c in df and df[c].notna().any()
        for c in ("prod_unwrapped_recall", "ww_unwrapped_recall")
    )
    has_coherence = "coh_mean_valid" in df and df["coh_mean_valid"].notna().any()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    ax = axes.ravel()
    score = df[SCORE].dropna().clip(0, 1)

    fig.suptitle(
        f"whirlwind vs NISAR GUNW — {len(df):,} comparisons",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    if len(score):
        fig.text(
            0.5,
            0.963,
            f"median agreement {score.median():.3%}   |   "
            f"{(score >= 0.99).mean():.1%} of comparisons at >=99%   |   "
            f"{(score >= 0.90).mean():.1%} at >=90%",
            ha="center",
            va="top",
            color="#444444",
        )

    # 1. ECDF of the *error* on a log x-axis. Start at 1% mismatch: that keeps
    # the headline 99%-agreement threshold while avoiding excessive visual
    # space for differences that are already operationally tiny.
    mismatch_pct = (100 * (1 - score)).sort_values()
    if len(mismatch_pct):
        plotted = mismatch_pct.clip(lower=1.0)
        cumulative = np.arange(1, len(plotted) + 1) / len(plotted)
        ax[0].step(plotted, cumulative, where="post", lw=2, c="#4477aa")
        for mismatch, threshold in ((1.0, 0.99), (10.0, 0.90)):
            frac = float((score >= threshold).mean())
            ax[0].axvline(mismatch, ls="--", c="#666666", lw=1)
            ax[0].scatter([mismatch], [frac], c="#222222", s=20, zorder=5)
            ax[0].annotate(
                f"{frac:.1%}",
                (mismatch, frac),
                xytext=(5, -12),
                textcoords="offset points",
                fontsize=9,
            )
        ax[0].set_xscale("log")
        ax[0].set_xlim(1, 100)
        ax[0].set_ylim(0, 1.01)
    ax[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    ax[0].set_xlabel("mismatched pixels (%) — lower is better")
    ax[0].set_ylabel("comparisons at or below mismatch")
    ax[0].set_title("Agreement success curve")

    # 2. Agreement against a useful scene-difficulty proxy.
    if has_coherence:
        scatter_quality_tiers(ax[1], df, df["coh_mean_valid"], df[SCORE], legend=True)
        for threshold in (0.90, 0.99):
            ax[1].axhline(threshold, ls="--", c="#777777", lw=0.8)
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1.01)
        ax[1].yaxis.set_major_formatter(PercentFormatter(1.0))
        ax[1].set_xlabel("mean coherence on valid pixels")
        ax[1].set_ylabel("per-component agreement")
        ax[1].set_title("Agreement vs scene coherence")
    else:
        ax[1].set_visible(False)

    # 3. Connected-component coverage/recall. Points above the identity line
    # are pixels whirlwind labels that production does not.
    if has_recall:
        prod_recall = df["prod_unwrapped_recall"]
        ww_recall = df["ww_unwrapped_recall"]
        scatter_quality_tiers(ax[2], df, prod_recall, ww_recall)
        ax[2].plot([0, 1], [0, 1], "--", c="#444444", lw=1, label="equal coverage")
        ax[2].fill_between(
            [0, 1], [0, 1], [1, 1], color="#2a9d8f", alpha=0.08, linewidth=0
        )
        delta_pp = 100 * (ww_recall - prod_recall).dropna().median()
        ax[2].set_xlim(0, 1)
        ax[2].set_ylim(0, 1)
        ax[2].xaxis.set_major_formatter(PercentFormatter(1.0))
        ax[2].yaxis.set_major_formatter(PercentFormatter(1.0))
        ax[2].set_xlabel("production labeled-pixel coverage")
        ax[2].set_ylabel("whirlwind labeled-pixel coverage")
        ax[2].set_title(
            f"Connected-component coverage (median delta {delta_pp:+.1f} pp)"
        )
        ax[2].legend(loc="lower right", frameon=False, fontsize=8)
    else:
        ax[2].set_visible(False)

    # 4. Runtime vs problem size. The median-throughput line is descriptive,
    # not a fit: parallel campaign timings include contention.
    runtime = df[["megapixels", "runtime_s"]].dropna()
    ax[3].scatter(
        runtime["megapixels"], runtime["runtime_s"], s=14, alpha=0.5, c="#4477aa"
    )
    seconds_per_mpx = (runtime["runtime_s"] / runtime["megapixels"]).median()
    if len(runtime) and np.isfinite(seconds_per_mpx):
        grid = np.array([runtime["megapixels"].min(), runtime["megapixels"].max()])
        ax[3].plot(
            grid,
            seconds_per_mpx * grid,
            "--",
            c="#333333",
            lw=1,
            label=f"median {seconds_per_mpx:.2f} s/Mpx",
        )
        ax[3].legend(frameon=False, fontsize=8)
    ax[3].set_xlabel("megapixels")
    ax[3].set_ylabel("unwrap runtime (s)")
    ax[3].set_title("Runtime scaling (in-process; excludes download)")

    # 5. Peak memory vs problem size.
    if has_mem:
        ax[4].scatter(
            df["megapixels"], df["peak_rss_mb"] / 1e3, s=14, alpha=0.6, c="darkorange"
        )
        med = df["gb_per_mpx"].median()
        xs = df["megapixels"].dropna()
        if len(xs):
            grid = [xs.min(), xs.max()]
            ax[4].plot(
                grid,
                [med * g for g in grid],
                "k--",
                lw=1,
                label=f"{med:.2f} GB/Mpx (median)",
            )
            ax[4].legend(frameon=False, fontsize=8)
        ax[4].set_xlabel("megapixels")
        ax[4].set_ylabel("peak RSS (GB, whole process tree)")
        ax[4].set_title("Peak-memory scaling")

    # 6. Connected components: whirlwind vs production.
    cc_ax = ax[5] if has_mem else ax[4]
    if not has_mem:
        ax[5].set_visible(False)
    if "ww_num_cc" in df and "prod_num_cc" in df:
        components = df[["prod_num_cc", "ww_num_cc"]].dropna()
        components = components[
            (components["prod_num_cc"] > 0) & (components["ww_num_cc"] > 0)
        ]
        if len(components):
            cc_ax.scatter(
                components["prod_num_cc"],
                components["ww_num_cc"],
                s=14,
                alpha=0.5,
                c="#2a9d8f",
                rasterized=True,
            )
            upper = max(components.max().max(), 2)
            cc_ax.plot([1, upper], [1, upper], "k--", lw=1)
            cc_ax.set_xscale("log", base=2)
            cc_ax.set_yscale("log", base=2)
            cc_ax.xaxis.set_major_formatter(ScalarFormatter())
            cc_ax.yaxis.set_major_formatter(ScalarFormatter())
            cc_ax.set_xlim(0.8, upper * 1.25)
            cc_ax.set_ylim(0.8, upper * 1.25)
            cc_ax.set_xlabel("production components")
            cc_ax.set_ylabel("whirlwind components")
            cc_ax.set_title("Connected-component count")
        else:
            cc_ax.set_visible(False)
    else:
        cc_ax.set_visible(False)

    for axis in ax:
        if axis.get_visible():
            axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(df: pd.DataFrame, out_md: Path, png: Path) -> str:
    n = len(df)
    score = df[SCORE].dropna()
    lines = [
        "# whirlwind vs NISAR GUNW -- campaign summary",
        "",
        f"- frames compared: **{n}**",
        f"- agreement (per-component ambiguity): median **{score.median():.4f}**, "
        f"mean {score.mean():.4f}, min {score.min():.4f}",
        f"- frames >= 0.99: **{(score >= 0.99).mean():.1%}**; >= 0.90: "
        f"**{(score >= 0.90).mean():.1%}**",
        f"- unwrap runtime: median {df['runtime_s'].median():.1f} s "
        f"(size median {df['megapixels'].median():.1f} Mpx)",
    ]
    if "peak_rss_mb" in df and df["peak_rss_mb"].notna().any():
        lines.append(
            f"- peak RSS: median {df['peak_rss_mb'].median() / 1e3:.2f} GB "
            f"({df['gb_per_mpx'].median():.2f} GB/Mpx), max "
            f"{df['peak_rss_mb'].max() / 1e3:.2f} GB"
        )
        w = df["workers"].dropna()
        if len(w):
            lines.append(
                f"- run with {int(w.mode().iloc[0])} concurrent workers; wall times "
                "include contention and are not single-frame benchmarks"
            )
    if "ww_num_cc" in df and "prod_num_cc" in df:
        lines.append(
            f"- components: whirlwind median {df['ww_num_cc'].median():.0f} vs "
            f"production {df['prod_num_cc'].median():.0f}"
        )
    recalls = ("prod_unwrapped_recall", "ww_unwrapped_recall")
    if all(c in df and df[c].notna().any() for c in recalls):
        prod_recall = df["prod_unwrapped_recall"]
        ww_recall = df["ww_unwrapped_recall"]
        lines.append(
            f"- labeled-pixel coverage: whirlwind median {ww_recall.median():.1%} "
            f"vs production {prod_recall.median():.1%} (median delta "
            f"{100 * (ww_recall - prod_recall).median():+.1f} pp)"
        )

    cols = [
        c
        for c in [
            "product",
            "track",
            "frame",
            SCORE,
            "runtime_s",
            "peak_rss_mb",
            "valid_frac",
            "coh_mean_valid",
        ]
        if c in df
    ]
    worst = df.nsmallest(15, SCORE)[cols]
    lines += [
        "",
        "## Worst 15 frames (investigate these first)",
        "",
        "```",
        worst.to_string(index=False),
        "```",
        "",
        f"![summary]({png.name})",
        "",
    ]
    text = "\n".join(lines)
    out_md.write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root", type=Path, required=True, help="run_local.py campaign dir."
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="manifest .meta.csv from discover_granules.py (adds track/frame + bounds).",
    )
    p.add_argument("--crop", default=None, help="Only this crop label, e.g. 'full'.")
    args = p.parse_args()

    df = merge_all(args.root, args.meta)
    if args.crop:
        df = df[df["crop"] == args.crop]
        assert len(df), f"No rows with crop={args.crop!r}."

    csv_path = args.root / "campaign.csv"
    png_path = args.root / "campaign_summary.png"
    md_path = args.root / "campaign.md"
    df.to_csv(csv_path, index=False)
    make_plots(df, png_path)
    text = write_report(df, md_path, png_path)

    print("\n" + text)
    print(
        f"\nWrote:\n  {csv_path.resolve()}\n  {md_path.resolve()}\n  {png_path.resolve()}"
    )


if __name__ == "__main__":
    main()
