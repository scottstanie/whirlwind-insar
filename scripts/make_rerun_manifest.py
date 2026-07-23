"""Select the slowest frames from a campaign CSV and write a rerun manifest.

Given a ``compare_gunw`` campaign CSV (the per-frame ``runtime_s`` table) and
the server's frame manifest (the ``manifest_*.txt`` of GUNW URLs, one per
line), pick the frames whose runtime exceeds a threshold and/or the N slowest,
and write a new manifest of just those frames' URLs -- ready to feed back to
``run_local.py --manifest`` for a focused rerun.

The join is on the granule id: a campaign row's ``product`` is ``<granule>.h5``
and every manifest URL ends in ``<granule>/<granule>.h5``.

Usage::

    python scripts/make_rerun_manifest.py \\
        --campaign campaign.csv \\
        --manifest manifest_land_frames_provisional.txt \\
        --min-runtime 300 \\
        --out manifest_rerun_over5min.txt

    # Or take a fixed count of the slowest, regardless of threshold:
    python scripts/make_rerun_manifest.py --campaign campaign.csv \\
        --manifest manifest.txt --top 20 --out manifest_rerun_top20.txt

Without ``--manifest`` the URL is constructed from the granule id in the
provisional collection, which is correct for the current release but does not
verify the frame is actually in the server manifest.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ASF_DAAC_BASE = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR"
SHORT_NAME = "NISAR_L2_GUNW_PROVISIONAL_V1"


def granule_of(name: str) -> str:
    """Bare granule id from a ``<granule>.h5`` product name or a URL."""
    stem = name.rsplit("/", 1)[-1]
    for ext in (".h5", ".hdf5"):
        if stem.lower().endswith(ext):
            return stem[: -len(ext)]
    return stem


def load_manifest_urls(path: Path) -> dict[str, str]:
    """Map granule id -> full URL from a manifest of one URL per line."""
    urls: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            urls[granule_of(line)] = line
    return urls


def constructed_url(granule: str) -> str:
    return f"{ASF_DAAC_BASE}/{SHORT_NAME}/{granule}/{granule}.h5"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--campaign", type=Path, required=True, help="compare_gunw campaign CSV."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Server manifest of GUNW URLs (one per line). Provides the canonical "
        "URL per frame and restricts output to frames actually in the manifest.",
    )
    p.add_argument(
        "--min-runtime",
        type=float,
        default=300.0,
        help="Keep frames whose runtime_s exceeds this (default 300 = 5 min).",
    )
    p.add_argument(
        "--top",
        type=int,
        default=None,
        help="Keep only the N slowest of those (after the --min-runtime filter).",
    )
    p.add_argument(
        "--runtime-col",
        default="runtime_s",
        help="Runtime column in the campaign CSV (default runtime_s).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output manifest path.")
    args = p.parse_args()

    rows = list(csv.DictReader(args.campaign.open()))
    if not rows:
        raise SystemExit(f"No rows in {args.campaign}")
    if args.runtime_col not in rows[0]:
        raise SystemExit(
            f"Column {args.runtime_col!r} not in campaign CSV; columns: "
            f"{', '.join(rows[0])[:200]}..."
        )

    # One row per frame at its slowest crop, filtered by runtime, sorted slowest first.
    kept = [r for r in rows if float(r[args.runtime_col]) > args.min_runtime]
    kept.sort(key=lambda r: float(r[args.runtime_col]), reverse=True)
    # De-duplicate by granule (a campaign may have several crops per product),
    # keeping the slowest occurrence (first, since already sorted).
    seen: set[str] = set()
    unique = []
    for r in kept:
        g = granule_of(r["product"])
        if g not in seen:
            seen.add(g)
            unique.append(r)
    if args.top is not None:
        unique = unique[: args.top]

    manifest_urls = load_manifest_urls(args.manifest) if args.manifest else {}

    lines: list[str] = []
    missing: list[str] = []
    for r in unique:
        g = granule_of(r["product"])
        if args.manifest:
            url = manifest_urls.get(g)
            if url is None:
                missing.append(g)
                continue
        else:
            url = constructed_url(g)
        lines.append(url)

    args.out.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(
        f"{len(kept)} frame(s) over {args.min_runtime:.0f}s "
        f"({len(unique)} unique granules"
        + (f", capped to top {args.top}" if args.top else "")
        + f") -> {len(lines)} written to {args.out}"
    )
    if unique:
        slowest = unique[0]
        print(
            f"  slowest: {float(slowest[args.runtime_col]):.0f}s  "
            f"{granule_of(slowest['product'])}"
        )
    if missing:
        print(
            f"  WARNING: {len(missing)} frame(s) not found in the manifest "
            f"(first few: {', '.join(missing[:3])})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
