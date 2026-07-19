#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["psutil"]
# ///
"""Run the GUNW comparison over a manifest of granules on a local CPU server.

This is the no-cloud counterpart to ``submit_batch.py``: instead of one AWS
Batch job per granule, it runs one ``compare_gunw.py`` subprocess per granule
across a pool of worker slots on whatever machine you have.

What it adds on top of ``compare_gunw.py``:

* **Parallelism with honest CPU accounting.** Each worker gets its own slice of
  the machine via ``WHIRLWIND_NUM_THREADS``, so N workers do not each try to
  use every core. See the note on timing below.
* **True peak memory.** The runner samples the summed RSS of each job's process
  tree, which is the number to quote for "how much memory does a frame need" --
  ``compare_gunw.py``'s own ``rss_delta_mb`` is an instantaneous before/after
  difference, not a high-water mark.
* **Resumability.** Completed jobs are recorded in ``runs.jsonl`` and skipped on
  a rerun, so an interrupted campaign picks up where it left off.
* **Disk hygiene.** GUNW products are ~2 GB each; ``--delete-after`` removes each
  download once its job succeeds, so disk use stays at roughly
  (workers x 2 GB) instead of the whole manifest.

A note on timing: with several unwraps in flight the per-job wall time includes
contention and is *not* a clean single-frame benchmark. For headline runtime
numbers, rerun the interesting frames with ``--workers 1``; use the parallel
mode to get through the campaign and to compare accuracy, memory, and
components.

Run this with the interpreter whose whirlwind build you want to benchmark --
it launches ``compare_gunw.py`` with the same one, and checks the imports work
before starting. See ``LOCAL_BENCH.md`` for the full runbook.

Example::

    python run_local.py --manifest manifest.txt --root /data/ww-bench \\
      --workers 8 --delete-after
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import psutil

HERE = Path(__file__).resolve().parent
COMPARE = HERE / "compare_gunw.py"


def read_manifest(path: Path) -> list[str]:
    tokens = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tokens.append(line)
    assert tokens, f"Manifest {path} has no inputs."
    return tokens


def job_id(token: str) -> str:
    """Granule name for a token. Matches ``compare_gunw.py``'s product id, so
    the runner's records join to the comparison JSON on this key."""
    stem = Path(urlparse(token).path).name if "://" in token else Path(token).name
    for ext in (".h5", ".hdf5"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    return stem.replace(".", "_")


def sample_peak_rss(proc: subprocess.Popen, interval: float) -> tuple[int, float]:
    """Poll the job's process tree until it exits; return (peak bytes, wall s).

    Sums RSS across the whole tree rather than taking the root process, so any
    helper the job spawns is counted at the moment it is resident.
    """
    t0 = time.perf_counter()
    peak = 0
    try:
        root = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return 0, time.perf_counter() - t0

    while proc.poll() is None:
        total = 0
        try:
            for p in (root, *root.children(recursive=True)):
                try:
                    total += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except psutil.NoSuchProcess:
            break
        peak = max(peak, total)
        time.sleep(interval)
    proc.wait()
    return peak, time.perf_counter() - t0


def downloaded_files(data_dir: Path, jid: str) -> list[Path]:
    return [p for p in data_dir.glob(f"{jid}*") if p.is_file()]


class Campaign:
    """Shared state for a run: output serialisation, the record file, and the
    stop signal + live child processes used to shut down on Ctrl-C."""

    def __init__(self, runs_file):
        self.lock = threading.Lock()
        self.runs_file = runs_file
        self.stop = threading.Event()
        self.running: dict[str, subprocess.Popen] = {}

    def say(self, msg: str) -> None:
        with self.lock:
            print(msg, flush=True)

    def record(self, rec: dict) -> None:
        """Append a finished job immediately, so an interrupt never loses work
        that was actually done."""
        with self.lock:
            self.runs_file.write(json.dumps(rec) + "\n")
            self.runs_file.flush()

    def register(self, jid: str, proc: subprocess.Popen) -> None:
        with self.lock:
            self.running[jid] = proc

    def unregister(self, jid: str) -> None:
        with self.lock:
            self.running.pop(jid, None)

    def halt(self) -> int:
        """Signal queued jobs to stand down and terminate in-flight ones."""
        self.stop.set()
        with self.lock:
            procs = list(self.running.items())
            for _, proc in procs:
                proc.terminate()
        return len(procs)


def run_one(token: str, args: argparse.Namespace, camp: Campaign) -> dict | None:
    if camp.stop.is_set():
        return None
    jid = job_id(token)
    out_dir = args.root / "results" / jid
    log_path = args.root / "logs" / f"{jid}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        *args.launcher,
        str(COMPARE),
        token,
        "--out-dir",
        str(out_dir),
        "--data-dir",
        str(args.data_dir),
        "--nlooks",
        str(args.nlooks),
    ]
    cmd += args.compare_arg

    env = dict(os.environ)
    env["WHIRLWIND_NUM_THREADS"] = str(args.threads_per_worker)
    # Keep the numeric stack from oversubscribing on top of ww's own pool.
    env["OMP_NUM_THREADS"] = str(args.threads_per_worker)

    started = time.time()
    camp.say(f"[start] {jid}")

    with open(log_path, "w") as log:
        log.write(f"# {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        camp.register(jid, proc)
        timer = None
        if args.timeout:
            timer = threading.Timer(args.timeout, proc.kill)
            timer.start()
        try:
            peak_bytes, wall_s = sample_peak_rss(proc, args.sample_interval)
        finally:
            if timer is not None:
                timer.cancel()
            camp.unregister(jid)

    record = {
        "job_id": jid,
        "input": token,
        "returncode": proc.returncode,
        "wall_s": round(wall_s, 2),
        "peak_rss_mb": round(peak_bytes / 1e6, 1),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "log": str(log_path),
        "out_dir": str(out_dir),
    }

    if proc.returncode == 0 and args.delete_after:
        for f in downloaded_files(args.data_dir, jid):
            f.unlink()
            record["deleted_download"] = True

    # Record before returning: if the main loop is interrupted, a job that
    # actually finished is still marked done and will not be re-downloaded.
    camp.record(record)
    status = "ok" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
    camp.say(
        f"[{status}] {jid}  {wall_s / 60:.1f} min  "
        f"peak {peak_bytes / 1e9:.1f} GB  log={log_path}"
    )
    return record


def preflight(launcher: list[str]) -> str:
    """Check the worker interpreter can import what compare_gunw.py needs.

    Worth doing before a campaign: the failure mode otherwise is every job
    downloading a 2 GB product and then dying on an import.
    """
    probe = (
        "import whirlwind, h5py, matplotlib, pandas, numpy; "
        "print(getattr(whirlwind, '__version__', 'unknown'), whirlwind.__file__)"
    )
    r = subprocess.run([*launcher, "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"Worker interpreter cannot import compare_gunw.py's dependencies:\n"
        f"  launcher: {' '.join(launcher)}\n{r.stderr.strip()}\n"
        "Point --python at an environment with whirlwind installed, or pass --uv."
    )
    return r.stdout.strip()


def load_done(runs_path: Path, root: Path) -> set[str]:
    """Job ids to skip on a rerun.

    Two independent sources, so an interrupted campaign never redoes work (and
    never re-downloads a 2 GB product) it has already finished:

    * ``runs.jsonl`` records with a zero exit status;
    * a results directory that already holds comparison JSON, which covers jobs
      whose record was lost -- e.g. a hard ``kill -9`` of the runner.
    """
    done = set()
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("returncode") == 0:
                    done.add(rec["job_id"])
    # The comparison JSON is not written atomically, so a hard kill landing
    # mid-write can truncate one. Surface that instead of treating the job as
    # finished (which would also break the aggregator later).
    corrupt = []
    for js in (root / "results").glob("*/*/*.json"):
        try:
            json.loads(js.read_text())
        except json.JSONDecodeError:
            corrupt.append(js)
            continue
        done.add(js.parents[1].name)
    if corrupt:
        listing = "\n".join(f"  {c}" for c in corrupt)
        raise SystemExit(
            f"Truncated result JSON from an interrupted job:\n{listing}\n"
            "Delete these files and rerun to redo those granules."
        )
    return done


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Campaign directory: results/, logs/, runs.jsonl land here.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Where GUNW downloads are cached. Default: <root>/downloads.",
    )
    p.add_argument("--workers", type=int, default=8, help="Concurrent granules.")
    p.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="Threads each unwrap may use. Default: cores // workers (min 1).",
    )
    p.add_argument("--nlooks", type=float, default=16.0)
    p.add_argument(
        "--compare-arg",
        action="append",
        default=[],
        help="Extra flag passed through to compare_gunw.py. Repeatable, and "
        "needs the '=' form for values starting with a dash, e.g. "
        "--compare-arg=--plot-downsample --compare-arg=4",
    )
    p.add_argument(
        "--delete-after",
        action="store_true",
        help="Delete each product's download once its job succeeds.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Kill a job after this many seconds.",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter used to run compare_gunw.py. Must have whirlwind and "
        "compare_gunw.py's other dependencies importable. Default: the "
        "interpreter running this script.",
    )
    p.add_argument(
        "--uv",
        action="store_true",
        help="Launch compare_gunw.py with `uv run --script` instead, which "
        "resolves its declared dependencies -- note that installs "
        "whirlwind-insar from PyPI, so it does NOT test a local build.",
    )
    p.add_argument("--sample-interval", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=None, help="Only run the first N.")
    p.add_argument(
        "--force", action="store_true", help="Rerun jobs already recorded as ok."
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    assert COMPARE.exists(), f"compare_gunw.py not found next to this script: {COMPARE}"
    args.launcher = ["uv", "run", "--script"] if args.uv else [args.python]
    if args.data_dir is None:
        args.data_dir = args.root / "downloads"
    if args.threads_per_worker is None:
        args.threads_per_worker = max(1, (os.cpu_count() or 1) // args.workers)

    args.root.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.root / "runs.jsonl"

    tokens = read_manifest(args.manifest)
    if args.limit is not None:
        tokens = tokens[: args.limit]
    done = set() if args.force else load_done(runs_path, args.root)
    todo = [t for t in tokens if job_id(t) not in done]

    cores = os.cpu_count() or 1
    ww_info = (
        "not checked (--uv resolves its own env)"
        if args.uv
        else preflight(args.launcher)
    )
    print(
        f"{len(tokens)} in manifest, {len(done)} already done, {len(todo)} to run\n"
        f"  {args.workers} workers x {args.threads_per_worker} threads "
        f"(machine has {cores} cores)\n"
        f"  whirlwind: {ww_info}\n"
        f"  root={args.root.resolve()}  downloads={args.data_dir.resolve()}",
        flush=True,
    )
    if args.workers * args.threads_per_worker > cores:
        print(
            f"  NOTE: {args.workers * args.threads_per_worker} threads requested on "
            f"{cores} cores; runtimes will reflect contention.",
            flush=True,
        )
    if args.dry_run:
        for t in todo:
            print(f"  [dry-run] {job_id(t)}")
        return
    if not todo:
        print("Nothing to do.")
        return

    t0 = time.time()
    n_ok = 0
    n_done = 0
    interrupted = False
    with open(runs_path, "a") as runs:
        camp = Campaign(runs)
        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(run_one, t, args, camp) for t in todo]
        try:
            for fut in as_completed(futures):
                rec = fut.result()
                if rec is None:  # stood down after an interrupt
                    continue
                n_done += 1
                n_ok += rec["returncode"] == 0
                camp.say(f"  progress {n_done}/{len(todo)} ({n_ok} ok)")
            pool.shutdown(wait=True)
        except KeyboardInterrupt:
            # Without this, the pool would drain the entire remaining queue
            # before exiting -- hours of work the user asked to stop.
            interrupted = True
            n_live = camp.halt()
            print(
                f"\nInterrupted: cancelled queued jobs, stopping {n_live} in flight. "
                "Finished jobs are already recorded; rerun the same command to resume.",
                flush=True,
            )
            for f in futures:
                f.cancel()
            pool.shutdown(wait=True, cancel_futures=True)

    verb = "Stopped" if interrupted else "Done"
    print(
        f"\n{verb}: {n_ok}/{len(todo)} succeeded in {(time.time() - t0) / 3600:.2f} h\n"
        f"  records -> {runs_path.resolve()}\n"
        f"  next: python aggregate_results.py --root {args.root}",
        flush=True,
    )
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
