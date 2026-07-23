"""Estimate NISAR GUNW production runtimes from QA_STATS HDF5 files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from os import PathLike
from pathlib import Path
import re
from typing import Any, Iterable

import h5py

UTC = timezone.utc

PRODUCT_TIME_PATH = "/science/LSAR/identification/processingDateTime"
QA_TIME_PATH = "/science/LSAR/QA/processing/QAProcessingDateTime"
SOURCE_RUNCONFIG_PATH = "/science/LSAR/sourceData/runConfigurationContents"

# ASF DAAC layout: each granule directory holds ``<granule>.h5`` (the ~2 GB main
# product) next to ``<granule>_QA_STATS.h5`` (a few MB of metadata). The runtime
# estimate reads only the QA_STATS file, so we never need the main download.
ASF_DAAC_BASE = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR"
SHORT_NAME = "NISAR_L2_GUNW_PROVISIONAL_V1"
SHORT_NAME_BETA = "NISAR_L2_GUNW_BETA_V1"
QA_STATS_SUFFIX = "_QA_STATS.h5"

_JOB_PATH_RE = re.compile(
    r"/data/work/jobs/"
    r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/"
    r"(?P<hour>\d{2})/(?P<minute>\d{2})/"
)
_STATE_CONFIG_RE = re.compile(r"state-config-(?P<stamp>\d{8}T\d{6}(?:\.\d+)?Z)")


def _decode_h5_scalar(value: Any) -> str:
    """Decode a scalar fixed-width HDF5 string."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00")
    return str(value).rstrip("\x00")


def _parse_utc(text: str) -> datetime:
    """Parse an ISO timestamp, treating a missing timezone as UTC."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class GUNWRuntimeEstimate:
    """Runtime estimates derived from one NISAR GUNW QA_STATS file."""

    product_processing_time: datetime
    job_start_minute: datetime
    runtime_from_job_minute: timedelta
    runtime_lower_bound: timedelta
    runtime_upper_bound: timedelta
    qa_processing_time: datetime | None
    qa_lag_after_product: timedelta | None
    state_config_time: datetime | None
    state_config_to_product: timedelta | None
    job_start_candidates: tuple[datetime, ...]

    def to_record(self, source: str | None = None) -> dict[str, Any]:
        """Return a flat record suitable for pandas/Parquet."""

        def iso(value: datetime | None) -> str | None:
            return value.isoformat().replace("+00:00", "Z") if value else None

        def seconds(value: timedelta | None) -> float | None:
            return value.total_seconds() if value is not None else None

        return {
            "source": source,
            "product_processing_time": iso(self.product_processing_time),
            "job_start_minute": iso(self.job_start_minute),
            "runtime_seconds": seconds(self.runtime_from_job_minute),
            "runtime_minutes": seconds(self.runtime_from_job_minute) / 60.0,
            "runtime_lower_bound_seconds": seconds(self.runtime_lower_bound),
            "runtime_upper_bound_seconds": seconds(self.runtime_upper_bound),
            "qa_processing_time": iso(self.qa_processing_time),
            "qa_lag_after_product_seconds": seconds(self.qa_lag_after_product),
            "state_config_time": iso(self.state_config_time),
            "state_config_to_product_seconds": seconds(self.state_config_to_product),
            "job_start_candidate_count": len(self.job_start_candidates),
        }


def estimate_gunw_runtime_from_h5(h5: h5py.File) -> GUNWRuntimeEstimate:
    """Extract a GUNW runtime estimate from an already-open QA_STATS file."""
    product_time = _parse_utc(_decode_h5_scalar(h5[PRODUCT_TIME_PATH][()]))

    qa_time = None
    if QA_TIME_PATH in h5:
        qa_time = _parse_utc(_decode_h5_scalar(h5[QA_TIME_PATH][()]))

    if SOURCE_RUNCONFIG_PATH not in h5:
        raise KeyError(
            f"{SOURCE_RUNCONFIG_PATH!r} is missing; the QA-specific runconfig at "
            "'/science/LSAR/QA/processing/runConfigurationContents' does not "
            "contain the Product SAS job path."
        )

    source_runconfig = _decode_h5_scalar(h5[SOURCE_RUNCONFIG_PATH][()])

    starts = [
        datetime(
            int(match["year"]),
            int(match["month"]),
            int(match["day"]),
            int(match["hour"]),
            int(match["minute"]),
            tzinfo=UTC,
        )
        for match in _JOB_PATH_RE.finditer(source_runconfig)
    ]
    if not starts:
        raise ValueError(
            "No /data/work/jobs/YYYY/MM/DD/HH/MM path found in the source runconfig"
        )

    # Paths are normally repeated many times for inputs and ancillaries. Select
    # the most frequent candidate. If tied, choose the latest candidate that
    # does not occur after processingDateTime.
    counts = Counter(starts)
    max_count = max(counts.values())
    tied = [dt for dt, count in counts.items() if count == max_count]
    eligible = [dt for dt in tied if dt <= product_time]
    job_start = max(eligible or tied)

    nominal = product_time - job_start

    # The directory timestamp is only minute-resolution. If actual creation was
    # at any instant in that minute, the runtime is approximately within this
    # one-minute interval.
    lower = max(timedelta(0), nominal - timedelta(minutes=1))
    upper = nominal

    state_times = [
        _parse_utc(match["stamp"])
        for match in _STATE_CONFIG_RE.finditer(source_runconfig)
    ]
    state_time = min(state_times) if state_times else None

    return GUNWRuntimeEstimate(
        product_processing_time=product_time,
        job_start_minute=job_start,
        runtime_from_job_minute=nominal,
        runtime_lower_bound=lower,
        runtime_upper_bound=upper,
        qa_processing_time=qa_time,
        qa_lag_after_product=qa_time - product_time if qa_time else None,
        state_config_time=state_time,
        state_config_to_product=product_time - state_time if state_time else None,
        job_start_candidates=tuple(sorted(counts)),
    )


def estimate_gunw_runtime(
    source: str | PathLike[str],
    *,
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
) -> GUNWRuntimeEstimate:
    """Open a local/HTTPS/S3 QA_STATS file and estimate Product SAS runtime.

    Requires opera-utils for remote access. Earthdata credentials may be passed
    directly or supplied through ~/.netrc or EARTHDATA_USERNAME/PASSWORD.
    """
    from opera_utils._remote import open_h5

    # QA_STATS files are small and only metadata is read. Avoid open_h5's large
    # default raw-data chunk cache when processing many files.
    with open_h5(
        source,
        rdcc_nbytes=8 * 1024**2,
        earthdata_username=earthdata_username,
        earthdata_password=earthdata_password,
        fsspec_kwargs={"cache_type": "first"},
    ) as h5:
        return estimate_gunw_runtime_from_h5(h5)


# ---------------------------------------------------------------------------
# Input resolution: granule name / local .h5 path / URL  ->  QA_STATS source.
# ---------------------------------------------------------------------------


def _basename(token: str) -> str:
    """Basename of a URL path or filesystem path."""
    from urllib.parse import urlparse

    if "://" in token:
        return Path(urlparse(token).path).name
    return Path(token).name


def _granule_stem(name: str) -> str:
    """Bare granule id: drop a trailing ``.h5``/``.hdf5`` and a ``_QA_STATS``."""
    for ext in (".h5", ".hdf5"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    if name.endswith("_QA_STATS"):
        name = name[: -len("_QA_STATS")]
    return name


def is_qa_stats(token: str) -> bool:
    """True if the token already names a QA_STATS file (path, URL, or granule)."""
    return "_QA_STATS" in _basename(token)


def qa_stats_url(granule_stem: str, short_name: str = SHORT_NAME) -> str:
    """Remote QA_STATS URL for a granule id in the given ASF collection."""
    return (
        f"{ASF_DAAC_BASE}/{short_name}/{granule_stem}/{granule_stem}{QA_STATS_SUFFIX}"
    )


def resolve_qa_stats_sources(token: str | PathLike[str]) -> list[str]:
    """Ordered QA_STATS sources to try for one input token.

    The input may be a bare granule name, a local ``.h5`` path (the main
    product or a QA_STATS file), or an http(s)/s3 URL to either. Because the
    main product is ~2 GB and only the tiny QA_STATS file carries the runtime
    metadata, this switches a main-product reference to its QA_STATS sibling
    and prefers reading it remotely (no download):

    * already a QA_STATS reference -> use it as-is;
    * a URL to the main product    -> its QA_STATS sibling in the same dir;
    * a local main ``.h5``          -> a local ``*_QA_STATS.h5`` sibling if it
      exists, else the remote QA_STATS URL(s);
    * a bare granule name           -> the remote QA_STATS URL(s).

    Collection is unknown for bare names / local files, so both the
    provisional and beta URLs are returned to try in order.
    """
    token = str(token)

    # Already a QA_STATS reference (path or URL): use it directly.
    if is_qa_stats(token):
        return [token]

    name = _basename(token)
    stem = _granule_stem(name)
    qa_name = f"{stem}{QA_STATS_SUFFIX}"

    # URL to the main product: swap the basename for the QA_STATS sibling.
    if "://" in token:
        base, _, _ = token.rpartition("/")
        return [f"{base}/{qa_name}"]

    # Local main-product path: prefer a QA_STATS sibling on disk, else remote.
    p = Path(token)
    if p.exists():
        sibling = p.parent / qa_name
        if sibling.exists():
            return [str(sibling)]
        return [qa_stats_url(stem, SHORT_NAME), qa_stats_url(stem, SHORT_NAME_BETA)]

    # Otherwise treat the token as a bare granule name.
    return [qa_stats_url(stem, SHORT_NAME), qa_stats_url(stem, SHORT_NAME_BETA)]


def estimate_gunw_runtime_for(
    token: str | PathLike[str],
    *,
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
) -> tuple[GUNWRuntimeEstimate, str]:
    """Resolve any input token to a QA_STATS source and estimate its runtime.

    Accepts a granule name, a local ``.h5`` path, or a URL (see
    :func:`resolve_qa_stats_sources`). Returns the estimate and the QA_STATS
    source that produced it. Candidate sources are tried in order; the last
    error is raised only if every candidate fails.
    """
    candidates = resolve_qa_stats_sources(token)
    errors: list[str] = []
    for source in candidates:
        try:
            estimate = estimate_gunw_runtime(
                source,
                earthdata_username=earthdata_username,
                earthdata_password=earthdata_password,
            )
            return estimate, source
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "No QA_STATS source succeeded for "
        f"{token!r}. Tried:\n  " + "\n  ".join(errors)
    )


def estimate_many_gunw_runtimes(
    sources: Iterable[str | PathLike[str]],
    *,
    resolve: bool = False,
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
) -> list[dict[str, Any]]:
    """Estimate multiple products, recording errors instead of aborting.

    With ``resolve=False`` (default) each source is opened directly as a
    QA_STATS file, matching the original behavior. With ``resolve=True`` each
    source is first passed through :func:`resolve_qa_stats_sources`, so bare
    granule names, local main-product ``.h5`` paths, and main-product URLs are
    all accepted and switched to QA_STATS; the resolved source used is stored
    in ``qa_source``.
    """
    records: list[dict[str, Any]] = []
    for source in sources:
        source_str = str(source)
        try:
            if resolve:
                estimate, qa_source = estimate_gunw_runtime_for(
                    source,
                    earthdata_username=earthdata_username,
                    earthdata_password=earthdata_password,
                )
            else:
                estimate = estimate_gunw_runtime(
                    source,
                    earthdata_username=earthdata_username,
                    earthdata_password=earthdata_password,
                )
                qa_source = source_str
            record = estimate.to_record(source_str)
            record["qa_source"] = qa_source
            record["error"] = None
        except Exception as exc:
            record = {
                "source": source_str,
                "qa_source": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# CLI: like run_local.py, take positional tokens and/or a --manifest, but this
# only reads tiny QA_STATS metadata, so no download pool is needed.
# ---------------------------------------------------------------------------


def _read_manifest(path: Path) -> list[str]:
    """One token per line, ``#`` comments allowed (matches run_local.py)."""
    tokens = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tokens.append(line)
    return tokens


def _fmt_td(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return str(timedelta(seconds=round(seconds)))


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Estimate NISAR GUNW production runtimes from QA_STATS metadata. "
            "Inputs may be granule names, local .h5 paths (main product or "
            "QA_STATS), or URLs; a main-product reference is switched to its "
            "tiny QA_STATS sibling and read remotely without downloading."
        )
    )
    p.add_argument(
        "tokens",
        nargs="*",
        help="Granule names, local .h5 paths, or URLs (main product or QA_STATS).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="File of tokens, one per line (# comments ok); same format as run_local.py.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write a table to this path (.csv or .parquet; .parquet needs pandas).",
    )
    args = p.parse_args(argv)

    tokens = list(args.tokens)
    if args.manifest is not None:
        tokens += _read_manifest(args.manifest)
    if not tokens:
        p.error("No inputs: pass granule/path/URL tokens and/or --manifest.")

    records = estimate_many_gunw_runtimes(tokens, resolve=True)

    # Console table, keyed by granule so it reads at a glance.
    print(f"{'granule':<62} {'runtime':>10} {'range':>19}  {'state->prod':>12}")
    for rec in records:
        gran = _granule_stem(_basename(rec["source"]))
        if rec.get("error"):
            print(f"{gran:<62} {'ERROR':>10}  {rec['error']}")
            continue
        rng = f"{_fmt_td(rec['runtime_lower_bound_seconds'])}-{_fmt_td(rec['runtime_upper_bound_seconds'])}"
        print(
            f"{gran:<62} {_fmt_td(rec['runtime_seconds']):>10} {rng:>19}  "
            f"{_fmt_td(rec['state_config_to_product_seconds']):>12}"
        )

    n_ok = sum(1 for r in records if not r.get("error"))
    print(f"\n{n_ok}/{len(records)} estimated.")

    if args.out is not None:
        if args.out.suffix == ".parquet":
            import pandas as pd

            pd.DataFrame(records).to_parquet(args.out, index=False)
        else:
            import csv

            fields = sorted({k for r in records for k in r})
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(records)
        print(f"wrote {args.out}")

    return 0 if n_ok == len(records) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
