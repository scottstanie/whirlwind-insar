#!/usr/bin/env python
"""Invert one engine's unwrapped Palos Verdes stack to a displacement time series.

Runs dolphin's network inversion (L2 + connected-component censoring, so an
engine's abstentions are treated as missing data rather than fake zeros) and a
velocity fit. The reference point is auto-selected per engine from temporal
coherence + that engine's conncomps (a fixed shared pixel does not exist: ICU
claims ~26% of pixels). Velocities are re-referenced to a common stable area in
the GPS comparison step, which is exact for linear fits.

Output rasters are in meters (Capella X-band wavelength), positive = motion
toward the satellite.

    /Users/staniewi/miniforge3/envs/mapping-312/bin/python scripts/pv_timeseries_invert.py --engine whirlwind
"""

import argparse
import time
from pathlib import Path

BASE = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes/"
    "Palos_Verdes_C13_RO23_SP/e2e_output_20260519"
)
TCOH = BASE / "dolphin/phase_linking/linked_phase/temporal_coherence_average_20251123203956_20260517062710.tif"
WAVELENGTH_M = 0.03106657595854922  # Capella X-band, from dolphin_config.yaml
FILE_DATE_FMT = "%Y%m%d%H%M%S"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True)
    p.add_argument("--out-base", type=Path, default=BASE / "unwrap_compare")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--method", choices=["L1", "L2"], default="L2")
    p.add_argument(
        "--label",
        default=None,
        help="output dir name (default: same as --engine); e.g. whirlwind_L1",
    )
    args = p.parse_args()
    label = args.label or args.engine

    from dolphin.timeseries import InversionMethod, run

    unw_dir = args.out_base / args.engine / "unwrapped"
    unw_paths = sorted(unw_dir.glob("2*.unw.tif"))
    assert len(unw_paths) == 150, f"{args.engine}: found {len(unw_paths)} unw files"
    if args.engine.startswith("spurt"):
        # spurt's conncomps mark only the solved high-coherence core (~16%);
        # its ambiguity-interpolated fill is part of the product claim, and its
        # nodata is nan -- censor by nodata masking instead of conncomps.
        conncomp_paths = None
    else:
        conncomp_paths = [
            p.parent / (p.name.replace(".unw.tif", ".unw.conncomp.tif"))
            for p in unw_paths
        ]
        for c in conncomp_paths:
            assert c.exists(), f"missing conncomp {c}"

    out_dir = args.out_base / label / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ts_paths, residual_paths, ref_point = run(
        unwrapped_paths=unw_paths,
        conncomp_paths=conncomp_paths,
        output_dir=out_dir,
        quality_file=TCOH,
        method=InversionMethod(args.method),
        run_velocity=True,
        wavelength=WAVELENGTH_M,
        num_threads=args.num_threads,
        file_date_fmt=FILE_DATE_FMT,
    )
    print(
        f"[{label}] inverted {len(unw_paths)} pairs ({args.method}) ->"
        f" {len(ts_paths)} dates in {time.perf_counter() - t0:.0f}s;"
        f" reference_point={ref_point}"
    )


if __name__ == "__main__":
    main()
