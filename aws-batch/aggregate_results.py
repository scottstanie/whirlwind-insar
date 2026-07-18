#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "matplotlib"]
# ///
"""Roll a whole campaign of per-granule GUNW comparisons into one table + plots.

Reads everything ``run_local.py`` produced under ``--root``:

* ``results/<granule>/<granule>/<crop>.json`` -- the comparison stats written by
  ``compare_gunw.py`` (agreement, components, coherence, ...).
* ``runs.jsonl`` -- the runner's wall time and true peak RSS per job.
* the manifest's ``.meta.csv`` sidecar, if given -- track/frame/bounding box, so
  results can be mapped and grouped by track.

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
import pandas as pd  # noqa: E402

# The headline accuracy metric: agreement with the production unwrap on the 2*pi
# integer, after re-levelling within each production connected component.
SCORE = "ambiguity_match_frac_percomp"


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


def make_plots(df: pd.DataFrame, out_png: Path) -> None:
    has_mem = "peak_rss_mb" in df and df["peak_rss_mb"].notna().any()
    has_geo = "min_lon" in df and df["min_lon"].notna().any()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes.ravel()

    # 1. Distribution of agreement with the production unwrap.
    ax[0].hist(df[SCORE].dropna(), bins=40, range=(0, 1), color="steelblue")
    ax[0].set_xlabel("per-component ambiguity agreement")
    ax[0].set_ylabel("frames")
    ax[0].set_title(f"Agreement with NISAR GUNW (n={df[SCORE].notna().sum()})")

    # 2. Same, as an ECDF -- easier to read "how many frames are above 0.99".
    s = df[SCORE].dropna().sort_values()
    ax[1].plot(s.values, (pd.Series(range(1, len(s) + 1)) / len(s)).values, lw=2)
    for thr in (0.9, 0.99):
        frac = float((s >= thr).mean())
        ax[1].axvline(thr, ls="--", c="gray", lw=1)
        ax[1].text(thr, 0.05, f" >={thr}: {frac:.0%}", fontsize=9, rotation=90)
    ax[1].set_xlabel("agreement")
    ax[1].set_ylabel("cumulative fraction of frames")
    ax[1].set_title("ECDF of agreement")

    # 3. Runtime vs problem size.
    ax[2].scatter(df["megapixels"], df["runtime_s"], s=14, alpha=0.6)
    ax[2].set_xlabel("megapixels")
    ax[2].set_ylabel("unwrap runtime (s)")
    ax[2].set_title("Runtime vs size (in-process; excludes download)")

    # 4. Peak memory vs problem size.
    if has_mem:
        ax[3].scatter(
            df["megapixels"], df["peak_rss_mb"] / 1e3, s=14, alpha=0.6, c="darkorange"
        )
        med = df["gb_per_mpx"].median()
        xs = df["megapixels"].dropna()
        if len(xs):
            grid = [xs.min(), xs.max()]
            ax[3].plot(
                grid,
                [med * g for g in grid],
                "k--",
                lw=1,
                label=f"{med:.2f} GB/Mpx (median)",
            )
            ax[3].legend()
        ax[3].set_xlabel("megapixels")
        ax[3].set_ylabel("peak RSS (GB, whole process tree)")
        ax[3].set_title("Peak memory vs size")
    else:
        ax[3].set_visible(False)

    # 5. Connected components: whirlwind vs production.
    if "ww_num_cc" in df and "prod_num_cc" in df:
        ax[4].scatter(df["prod_num_cc"], df["ww_num_cc"], s=14, alpha=0.6, c="seagreen")
        lim = [1, max(df["prod_num_cc"].max(), df["ww_num_cc"].max(), 2)]
        ax[4].plot(lim, lim, "k--", lw=1)
        ax[4].set_xscale("log")
        ax[4].set_yscale("log")
        ax[4].set_xlabel("production components")
        ax[4].set_ylabel("whirlwind components")
        ax[4].set_title("Connected component count")
    else:
        ax[4].set_visible(False)

    # 6. Where the frames are, coloured by agreement -- shows both the spatial
    # spread of the campaign and whether failures cluster geographically.
    if has_geo:
        lon = (df["min_lon"] + df["max_lon"]) / 2
        lat = (df["min_lat"] + df["max_lat"]) / 2
        sc = ax[5].scatter(lon, lat, c=df[SCORE], cmap="RdYlGn", vmin=0, vmax=1, s=18)
        fig.colorbar(sc, ax=ax[5], label="agreement")
        ax[5].set_xlim(-180, 180)
        ax[5].set_ylim(-90, 90)
        ax[5].set_xlabel("longitude")
        ax[5].set_ylabel("latitude")
        ax[5].set_title("Campaign coverage")
    else:
        ax[5].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
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
        help="manifest .meta.csv from discover_granules.py (adds track/frame + map).",
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
