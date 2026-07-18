#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy"]
# ///
"""Write small synthetic GUNW products to smoke-test the comparison harness.

Real NISAR GUNW products are ~2 GB each, which makes them a slow way to find
out that a path is wrong or a worker slot is misconfigured. These files carry
the same group layout and dataset names the harness reads, at a few hundred
pixels a side, so the whole chain -- ``run_local.py`` -> ``compare_gunw.py`` ->
``aggregate_results.py`` -- can be exercised in seconds on any machine.

The scene is a deterministic multi-cycle phase ramp plus a Gaussian bump, with
a decorrelated band and a water strip, so the comparison produces a meaningful
(not degenerate) agreement score and more than one connected component.

Example::

    python make_synthetic_gunw.py --out-dir /tmp/synth --count 4
    python run_local.py --manifest /tmp/synth/manifest.txt --root /tmp/smoke --workers 2

Note the ``unwrappedPhase`` here is a synthetic truth field, not a production
unwrap, so the resulting agreement numbers say nothing about NISAR -- they only
prove the plumbing works.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

UNW_BASE = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
WRAP_BASE = "/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram"


def synth_scene(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n

    # Several fringe cycles plus a localised deformation-like bump.
    cycles = 3.0 + 2.0 * rng.random()
    unw = 2 * np.pi * cycles * (0.6 * x + 0.4 * y)
    unw += 6.0 * np.exp(-(((x - 0.35) ** 2 + (y - 0.6) ** 2) / 0.01))

    coh = np.full((n, n), 0.85, dtype=np.float32)
    # A low-coherence band across the scene, and a "water" strip.
    band = slice(int(0.45 * n), int(0.55 * n))
    coh[band, :] = 0.15
    unw[band, :] += rng.normal(0, 1.5, size=unw[band, :].shape)

    mask = np.full((n, n), 0, dtype=np.uint8)  # 0 = land, both subswaths valid
    water = slice(int(0.80 * n), int(0.88 * n))
    mask[water, :] = 100  # water digit set
    unw[water, :] = np.nan
    coh[water, :] = 0.0

    unw += rng.normal(0, 0.05, size=unw.shape)

    # Connected components: 1 above the low-coherence band, 2 below, 0 elsewhere.
    cc = np.zeros((n, n), dtype=np.uint16)
    cc[: band.start, :] = 1
    cc[band.stop : water.start, :] = 2
    cc[water, :] = 0

    return {
        "unw": unw.astype(np.float32),
        "coh": coh,
        "cc": cc,
        "mask": mask,
        "wrapped": np.exp(1j * unw).astype(np.complex64),
    }


def write_product(path: Path, scene: dict[str, np.ndarray], pol: str) -> None:
    with h5py.File(path, "w") as h5:
        g = h5.create_group(f"{UNW_BASE}/{pol}")
        d = g.create_dataset(
            "unwrappedPhase", data=np.nan_to_num(scene["unw"], nan=-1e30)
        )
        d.attrs["_FillValue"] = np.float32(-1e30)
        g.create_dataset("coherenceMagnitude", data=scene["coh"])
        g.create_dataset("connectedComponents", data=scene["cc"])
        h5[UNW_BASE].create_dataset("mask", data=scene["mask"])
        h5.create_group(f"{WRAP_BASE}/{pol}").create_dataset(
            "wrappedInterferogram", data=scene["wrapped"]
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=3, help="Number of products to write.")
    p.add_argument("--size", type=int, default=512, help="Pixels per side.")
    p.add_argument("--pol", default="HH")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(args.count):
        # Mirror the real granule naming so job ids and any name parsing behave
        # the same as they will on production data.
        name = (
            f"NISAR_L2_PR_GUNW_009_{100 + i:03d}_A_{10 + i:03d}_010_2000_SH_"
            f"20260108T004405_20260108T004440_20260120T004406_20260120T004440_"
            f"X05010_N_F_J_001.h5"
        )
        path = args.out_dir / name
        write_product(path, synth_scene(args.size, seed=i), args.pol)
        names.append(path)
        print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)

    manifest = args.out_dir / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in names) + "\n")
    print(f"\n{args.count} synthetic products -> {manifest.resolve()}", flush=True)


if __name__ == "__main__":
    main()
