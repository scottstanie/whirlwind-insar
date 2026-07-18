#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["asf-search", "pandas"]
# ///
"""Build a spatially spread manifest of NISAR L2 GUNW products to benchmark on.

Two steps, either of which can be run alone:

1. **Inventory** -- page through the whole ASF catalog of NISAR GUNW products
   and write one row per product (track, frame, dates, URL, size). This is a
   large query, so the result is cached; later runs reuse the CSV unless
   ``--refresh`` is passed.
2. **Sample** -- join the inventory to a track/frame land table, then pick a
   handful of frames per track. Sampling by latitude *within* each track spreads
   the selection along the orbit, so a few hundred products still cover most of
   the land surface the mission images.

The output manifest is a plain list of download URLs, one per line, which
``run_local.py`` (and ``compare_gunw.py --inputs-file``) reads directly. A
sidecar CSV keeps the metadata (track, frame, bounding box, size) that the
aggregator uses for the coverage map.

Example::

    uv run discover_granules.py \
      --land-frames ~/repos/virtual-sar/src/virtual_sar/data/nisar_land_frames.csv \
      --per-track 2 --min-land 0.2 \
      --out manifest.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# NISAR_L2_PR_GUNW_<cycle>_<relative orbit>_<A|D>_<frame>_<...>_<mode>_<dates...>
# The relative orbit number is the "track" of the track/frame database, and the
# field after the orbit direction is the frame.
GRANULE_RE = re.compile(
    r"NISAR_L2_\w+?_GUNW_(?P<cycle>\d+)_(?P<track>\d+)_(?P<direction>[AD])_"
    r"(?P<frame>\d+)_"
)


def fetch_inventory(
    cache: Path, refresh: bool, max_results: int | None
) -> pd.DataFrame:
    """Page through the ASF catalog for NISAR GUNW products, with a CSV cache."""
    if cache.exists() and not refresh:
        df = pd.read_csv(cache)
        print(f"Inventory: {len(df)} products (cached {cache})", flush=True)
        return df

    import asf_search as asf

    print("Querying ASF for NISAR GUNW products (this takes a minute)...", flush=True)
    if max_results is None:
        results = asf.search(dataset=asf.DATASET.NISAR, processingLevel="GUNW")
    else:
        results = asf.search(
            dataset=asf.DATASET.NISAR, processingLevel="GUNW", maxResults=max_results
        )
    print(f"  {len(results)} products returned", flush=True)

    rows = []
    for r in results:
        p = r.properties
        name = p["sceneName"]
        m = GRANULE_RE.match(name)
        if m is None:
            # A product whose name we cannot parse cannot be joined to the land
            # table, so surface it instead of silently dropping it.
            raise ValueError(f"Unparseable NISAR GUNW granule name: {name!r}")
        size = p.get("bytes")
        if isinstance(size, dict):
            # Multi-file products report per-file sizes; the main .h5 is the one
            # we download.
            size = next(
                (v["bytes"] for k, v in size.items() if k.endswith(".h5")), None
            )
        rows.append(
            {
                "granule": name,
                "url": p["url"],
                "cycle": int(m.group("cycle")),
                "track": int(m.group("track")),
                "direction": m.group("direction"),
                "frame": int(m.group("frame")),
                "ref_time": p["startTime"],
                "sec_time": p["stopTime"],
                "size_bytes": size,
            }
        )

    df = pd.DataFrame(rows)
    df["ref_time"] = pd.to_datetime(df["ref_time"])
    df["sec_time"] = pd.to_datetime(df["sec_time"])
    df["temporal_baseline_days"] = (df["sec_time"] - df["ref_time"]).dt.days
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"Wrote inventory -> {cache}", flush=True)
    return df


def join_land_frames(
    inv: pd.DataFrame, land_csv: Path, min_land: float
) -> pd.DataFrame:
    """Keep only products whose (track, frame, direction) is a GUNW land frame."""
    land = pd.read_csv(land_csv)
    land = land[(land["produce_gunw"] == 1) & (land["fraction_land"] >= min_land)]
    keep = [
        "track",
        "frame",
        "direction",
        "fraction_land",
        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",
    ]
    merged = inv.merge(land[keep], on=["track", "frame", "direction"], how="inner")
    assert len(merged) > 0, (
        "No products joined to the land table. Check that the granule name fields "
        "really are (relative orbit, frame) -- a naming change would break this."
    )
    print(
        f"Land join: {len(merged)} of {len(inv)} products on "
        f"{merged.groupby(['track', 'direction']).ngroups} track/direction pairs",
        flush=True,
    )
    return merged


def pick_one_per_frame(df: pd.DataFrame, prefer: str) -> pd.DataFrame:
    """Collapse repeat passes so each (track, frame, direction) appears once."""
    sort_key = {
        "short-baseline": ["temporal_baseline_days", "ref_time"],
        "recent": ["ref_time"],
    }[prefer]
    ascending = prefer == "short-baseline"
    df = df.sort_values(sort_key, ascending=ascending)
    return df.groupby(["track", "frame", "direction"], as_index=False).first()


def sample_per_track(df: pd.DataFrame, per_track: int, seed: int) -> pd.DataFrame:
    """Take ``per_track`` frames from each track, spread along the orbit.

    Frames are ordered by latitude and sampled at evenly spaced positions, so a
    track contributing 2 frames gives one from roughly a third of the way up and
    one from two thirds -- not two neighbours that image the same ground.
    """
    picks = []
    for _, grp in df.groupby(["track", "direction"]):
        grp = grp.sort_values("min_lat").reset_index(drop=True)
        n = min(per_track, len(grp))
        # Evenly spaced interior positions: for n=2 -> 1/3, 2/3 of the range.
        idx = [round((i + 1) * (len(grp) - 1) / (n + 1)) for i in range(n)]
        picks.append(grp.iloc[sorted(set(idx))])
    out = pd.concat(picks, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--land-frames",
        type=Path,
        required=True,
        help="Track/frame land table CSV (track, frame, direction, fraction_land, ...).",
    )
    p.add_argument("--out", type=Path, default=Path("manifest.txt"))
    p.add_argument(
        "--inventory-csv",
        type=Path,
        default=Path("nisar_gunw_inventory.csv"),
        help="Cache for the full ASF catalog pull.",
    )
    p.add_argument(
        "--refresh", action="store_true", help="Re-query ASF even if the cache exists."
    )
    p.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Cap the ASF query (for a quick trial run). Default: pull everything.",
    )
    p.add_argument(
        "--per-track",
        type=int,
        default=2,
        help="Frames to keep per track/direction (1-4 gives good coverage).",
    )
    p.add_argument(
        "--min-land",
        type=float,
        default=0.2,
        help="Drop frames whose land fraction is below this.",
    )
    p.add_argument(
        "--prefer",
        choices=["short-baseline", "recent"],
        default="short-baseline",
        help="Which repeat pass to keep when a frame has several products.",
    )
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed for the manifest.")
    p.add_argument(
        "--limit", type=int, default=None, help="Truncate the manifest to N products."
    )
    args = p.parse_args()

    inv = fetch_inventory(args.inventory_csv, args.refresh, args.max_results)
    joined = join_land_frames(inv, args.land_frames, args.min_land)
    unique = pick_one_per_frame(joined, args.prefer)
    sampled = sample_per_track(unique, args.per_track, args.seed)
    if args.limit is not None:
        sampled = sampled.head(args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sampled["url"]) + "\n")
    meta_csv = args.out.with_suffix(".meta.csv")
    sampled.to_csv(meta_csv, index=False)

    total_gb = (
        sampled["size_bytes"].sum() / 1e9
        if sampled["size_bytes"].notna().any()
        else float("nan")
    )
    print(
        f"\nSelected {len(sampled)} products across "
        f"{sampled.groupby(['track', 'direction']).ngroups} track/direction pairs\n"
        f"  manifest -> {args.out.resolve()}\n"
        f"  metadata -> {meta_csv.resolve()}\n"
        f"  download volume ~{total_gb:.0f} GB "
        f"(run_local.py --delete-after keeps only the in-flight products on disk)",
        flush=True,
    )


if __name__ == "__main__":
    main()
