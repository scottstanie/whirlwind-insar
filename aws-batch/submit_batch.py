#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3"]
# ///
"""Submit one AWS Batch job per NISAR GUNW product.

Reads a manifest (or inputs on the command line) and submits a Batch job for
each, overriding the job-definition command so the container runs
``compare_gunw.py <input> --out-dir /work/out --upload-s3 <prefix>/<id>``. Each
job writes its results to its own S3 prefix.

Run one job per granule (rather than one big multi-granule job) so a single
crash or Spot reclamation only loses one product, and so they parallelize across
the Batch queue.

Example::

    uv run submit_batch.py \
      --inputs-file sample_granules.txt \
      --job-queue whirlwind-gunw-queue \
      --job-definition whirlwind-gunw \
      --s3-out s3://my-bucket/ww-gunw-bench \
      --region us-west-2 --profile my-profile
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import boto3


def granule_id(token: str) -> str:
    """Derive a short, Batch-safe job name from an input token."""
    stem = token.rstrip("/").split("/")[-1]
    stem = re.sub(r"\.h5$|\.hdf5$", "", stem, flags=re.IGNORECASE)
    # Use the orbit/track/frame core if it is a NISAR granule name, else the stem.
    m = re.search(r"GUNW_\d+_\d+_([AD]_\d+_\d+_\d+)", stem)
    label = m.group(1) if m else stem
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", f"ww-gunw-{label}")
    return safe[:128]


def read_inputs(args: argparse.Namespace) -> list[str]:
    tokens = list(args.inputs)
    if args.inputs_file:
        for line in Path(args.inputs_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                tokens.append(line)
    if not tokens:
        raise SystemExit("No inputs. Pass tokens or --inputs-file.")
    return tokens


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="*", help="GUNW URL / s3 URI / granule / path.")
    p.add_argument("--inputs-file")
    p.add_argument("--job-queue", required=True)
    p.add_argument("--job-definition", required=True)
    p.add_argument(
        "--s3-out",
        required=True,
        help="S3 prefix for results; each job uploads to <s3-out>/<id>/.",
    )
    p.add_argument("--nlooks", default="50")
    p.add_argument("--dump-flat", action="store_true")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--profile", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tokens = read_inputs(args)
    session = (
        boto3.Session(profile_name=args.profile, region_name=args.region)
        if args.profile
        else boto3.Session(region_name=args.region)
    )
    batch = session.client("batch")

    for tok in tokens:
        name = granule_id(tok)
        s3out = f"{args.s3_out.rstrip('/')}/{name}"
        command = [
            tok,
            "--out-dir",
            "/work/out",
            "--upload-s3",
            s3out,
            "--nlooks",
            str(args.nlooks),
        ]
        if args.dump_flat:
            command.append("--dump-flat")

        if args.dry_run:
            print(f"[dry-run] {name}\n    command: {command}\n    -> {s3out}")
            continue
        resp = batch.submit_job(
            jobName=name,
            jobQueue=args.job_queue,
            jobDefinition=args.job_definition,
            containerOverrides={"command": command},
        )
        print(f"submitted {name}: jobId={resp['jobId']} -> {s3out}")


if __name__ == "__main__":
    main()
