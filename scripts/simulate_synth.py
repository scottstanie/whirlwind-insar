#!/usr/bin/env python3
"""Generate a synthetic interferogram + coherence pair for the `whirlwind` CLI.

Replaces the removed `whirlwind simulate` subcommand (the generators live in
``whirlwind_core::simulate`` and stay exposed through the Python API as
``whirlwind.simulate_ifg`` / ``whirlwind.diagonal_ramp``). Writes flat float32
rasters (snaphu FLOAT_DATA) with ROI_PAC-style ``.rsc`` sidecars, so the CLI
reads the geometry without ``--cols``:

    python scripts/simulate_synth.py --out /tmp/sim
    whirlwind --phase /tmp/sim/wrapped.f32 --cor /tmp/sim/cor.f32 \\
        --nlooks 10 --out /tmp/sim/unw.f32
"""

import argparse
from pathlib import Path

import numpy as np

import whirlwind as ww


def gaussian_bump(m: int, n: int, amp: float, sigma: float) -> np.ndarray:
    """Centered Gaussian bump, mirroring whirlwind_core::simulate::gaussian_bump."""
    ii, jj = np.mgrid[0:m, 0:n].astype(np.float32)
    ci, cj = (m - 1) / 2.0, (n - 1) / 2.0
    return (amp * np.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2.0 * sigma**2))).astype(
        np.float32
    )


def write_flat(path: Path, arr: np.ndarray) -> None:
    """Flat float32 + a `.rsc` sidecar carrying the geometry."""
    arr.astype(np.float32).tofile(path)
    rows, cols = arr.shape
    Path(f"{path}.rsc").write_text(f"WIDTH         {cols}\nFILE_LENGTH   {rows}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shape", default="256x256", help="shape as MxN (e.g. 256x256)")
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument("--pattern", default="bump", choices=["ramp", "bump"])
    p.add_argument("--nlooks", type=int, default=10, help="looks for synthetic noise")
    p.add_argument("--coherence", type=float, default=0.85, help="uniform coherence")
    p.add_argument("--seed", type=int, default=42, help="rng seed")
    args = p.parse_args()

    m, n = map(int, args.shape.split("x"))
    if args.pattern == "ramp":
        truth = ww.diagonal_ramp(m, n)
    else:
        truth = gaussian_bump(m, n, amp=8.0, sigma=n / 8.0)
    gamma = np.full((m, n), args.coherence, dtype=np.float32)
    igram, cor = ww.simulate_ifg(truth, gamma, args.nlooks, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    write_flat(args.out / "wrapped.f32", np.angle(igram))
    # The sample coherence |acc| overshoots 1 where noise aligns with the
    # signal; coherence is [0, 1] by definition and the CLI validates its
    # --cor input, so clamp before writing.
    write_flat(args.out / "cor.f32", np.clip(cor, 0.0, 1.0))
    write_flat(args.out / "truth.f32", truth)
    print(f"wrote {args.out} (shape {m}x{n}, pattern {args.pattern}); unwrap with:")
    print(
        f"  whirlwind --phase {args.out}/wrapped.f32 --cor {args.out}/cor.f32"
        f" --nlooks {args.nlooks} --out {args.out}/unw.f32"
    )


if __name__ == "__main__":
    main()
