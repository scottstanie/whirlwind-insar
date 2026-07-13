#!/usr/bin/env python3
"""Extract a prepared NISAR GUNW frame to npz, or run unwrap_linear on it.

Profiling companion to bench_nisar_gunw_whirlwind.py: input preparation is
identical (re-wrap production unwrappedPhase, zero masked phase, sanitize
coherence), but split into an extract step (slow h5 read, cached npz) and a
run step (pure solver) so A/B builds time only the solver.

Usage:
    # one-time extract (writes <out>/<frame>.npz)
    .venv/bin/python scripts/perf_frame_runner.py extract <gunw.h5> <out_dir>
    # timed run (prints runtime, ru_maxrss, and checksum of the unwrap)
    .venv/bin/python scripts/perf_frame_runner.py run <frame.npz> [--nlooks 16]
"""

from __future__ import annotations

import argparse
import hashlib
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

POL = "HH"
UNW_BASE = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"


def extract(h5_path: Path, out_dir: Path) -> Path:
    import h5py

    with h5py.File(h5_path, "r") as h5:
        unw = h5[f"{UNW_BASE}/{POL}/unwrappedPhase"][()]
        coh = h5[f"{UNW_BASE}/{POL}/coherenceMagnitude"][()]
        mask_arr = h5[f"{UNW_BASE}/mask"][()]

    # mask policy "water-and-subswath" (bench default)
    water = (mask_arr // 100) % 10
    ref_sub = (mask_arr // 10) % 10
    sec_sub = mask_arr % 10
    mask = (mask_arr != 255) & (water == 0) & (ref_sub > 0) & (sec_sub > 0)
    mask &= np.isfinite(unw) & np.isfinite(coh) & (coh >= 0.0)

    ig = np.angle(np.exp(1j * unw)).astype(np.float32)  # re-wrapped phase
    ig_solver = np.where(mask, ig, 0.0).astype(np.float32)
    ig_complex = np.exp(1j * ig_solver).astype(np.complex64)
    coh_solver = np.where(mask, np.clip(np.nan_to_num(coh), 0.0, 1.0), 0.0).astype(
        np.float32
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (h5_path.stem.split("_20")[0] + ".npz")
    np.savez(out, ig_complex=ig_complex, coh=coh_solver, mask=mask)
    print(f"wrote {out}  shape={ig_complex.shape} valid={mask.mean():.3f}")
    return out


def run(npz_path: Path, nlooks: float) -> None:
    import whirlwind as ww

    d = np.load(npz_path)
    if "ig_complex" in d:
        ig_complex = np.ascontiguousarray(d["ig_complex"])
        coh = np.ascontiguousarray(d["coh"])
        mask = np.ascontiguousarray(d["mask"])
    else:
        # Ridgecrest-style rewrap_bench inputs (wrapped/corr/mask keys).
        mask = np.ascontiguousarray(d["mask"])
        ig_complex = np.where(mask, np.exp(1j * d["wrapped"]), 0).astype(np.complex64)
        coh = np.where(mask, d["corr"], 0).astype(np.float32)
    del d

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    t0 = time.perf_counter()
    unw = ww._native.unwrap_linear(ig_complex, coh, float(nlooks), mask)
    dt = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    unw = np.asarray(unw, dtype=np.float32)
    valid = unw[mask]
    csum = hashlib.sha1(valid.tobytes()).hexdigest()[:16]
    print(
        f"frame={npz_path.stem} shape={unw.shape} "
        f"runtime={dt:.2f}s peak_rss={rss1 / 2**30:.2f}GiB "
        f"(pre-solve rss={rss0 / 2**30:.2f}GiB) sha1(valid)={csum}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract")
    pe.add_argument("h5", type=Path)
    pe.add_argument("out_dir", type=Path)
    pr = sub.add_parser("run")
    pr.add_argument("npz", type=Path)
    pr.add_argument("--nlooks", type=float, default=16.0)
    args = p.parse_args()
    if args.cmd == "extract":
        extract(args.h5, args.out_dir)
    else:
        run(args.npz, args.nlooks)


if __name__ == "__main__":
    main()
