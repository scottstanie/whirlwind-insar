"""Measure TRUE peak memory of a command and ALL its descendants.

`/usr/bin/time -l` reports a single-process high-water mark. It does roll up a
*sequentially*-reaped grandchild's peak, but for a process that forks
*concurrent* workers (e.g. SNAPHU tiled with nproc>1) it reports the largest
single worker, NOT the simultaneous sum. It can also undercount a brief
end-of-run pass spawned as a separate subprocess.

This samples the SUMMED RSS across the whole process tree at high frequency, so
both the parallel tiled-phase aggregate and the end single-tile reoptimize spike
show up. Writes the peak, and optionally an RSS(t) trace + plot.

Usage:
  python scripts/peak_rss_tree.py [--interval 0.025] [--trace t.csv] [--plot t.png] \
      -- /usr/bin/env python scripts/snaphu_one.py <h5> 9
"""

import argparse
import subprocess
import sys
import threading
import time

import psutil


def _sample(root: psutil.Process, stop: threading.Event, interval: float, out: dict):
    t0 = time.perf_counter()
    peak = 0
    trace = []
    while not stop.is_set():
        total = 0
        nproc = 0
        try:
            procs = [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            break
        for p in procs:
            try:
                total += p.memory_info().rss
                nproc += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if total > peak:
            peak = total
            out["peak_nproc"] = nproc
        trace.append((time.perf_counter() - t0, total, nproc))
        time.sleep(interval)
    out["peak"] = peak
    out["trace"] = trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.025, help="sample period (s)")
    ap.add_argument("--trace", help="write RSS(t) CSV here")
    ap.add_argument("--plot", help="write RSS(t) PNG here")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("provide a command after `--`")

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd)
    root = psutil.Process(proc.pid)
    out: dict = {"peak": 0, "peak_nproc": 0, "trace": []}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample, args=(root, stop, args.interval, out))
    sampler.start()
    rc = proc.wait()
    stop.set()
    sampler.join()
    dt = time.perf_counter() - t0

    peak_gb = out["peak"] / 1e9
    print(
        f"tree peak RSS = {peak_gb:.2f} GB ({out['peak']} bytes) across up to "
        f"{out['peak_nproc']} process(es); wall {dt:.1f}s; exit {rc}"
    )

    if args.trace:
        with open(args.trace, "w") as f:
            f.write("t_s,rss_bytes,nproc\n")
            for t, rss, n in out["trace"]:
                f.write(f"{t:.3f},{rss},{n}\n")
        print(f"trace -> {args.trace}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [t for t, _, _ in out["trace"]]
        gb = [rss / 1e9 for _, rss, _ in out["trace"]]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, gb, lw=1)
        ax.axhline(peak_gb, color="r", ls=":", lw=0.8, label=f"peak {peak_gb:.2f} GB")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("summed tree RSS (GB)")
        ax.set_title(" ".join(cmd[-3:]))
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"plot -> {args.plot}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
