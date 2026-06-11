#!/usr/bin/env python
"""Palos Verdes Capella stack: re-unwrap the dolphin interferograms with one engine.

Re-runs ONLY the unwrapping step of the existing dolphin e2e output
(`e2e_output_20260519`), slotting in a chosen unwrap engine, so the resulting
unwrapped stacks can be inverted to time series and compared against the GPS
survey monuments on the Portuguese Bend landslide.

Engines: whirlwind | snaphu (2x2 tiles, nproc 4) | phass | icu | spurt.
ICU here is dolphin's tophu/isce3 route; the isce2 mroipac fallback lives in
pv_icu_isce2.py.

Matches the original run's settings: nlooks = 15 (strides 3x5), no mask,
no goldstein/interpolation preprocessing.

Run with the mapping-312 env:
    /Users/staniewi/miniforge3/envs/mapping-312/bin/python scripts/pv_unwrap_compare.py \
        --engine whirlwind [--limit 1] [--limit-dates 10] [--n-jobs 3]
"""

import argparse
import time
from pathlib import Path

BASE = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes/"
    "Palos_Verdes_C13_RO23_SP/e2e_output_20260519"
)
IFG_DIR = BASE / "dolphin/interferograms"
TCOH = BASE / "dolphin/phase_linking/linked_phase/temporal_coherence_average_20251123203956_20260517062710.tif"
SIMILARITY = BASE / "dolphin/phase_linking/linked_phase/similarity_full_20251123203956_20260517062710.tif"
OUT_BASE = BASE / "unwrap_compare"
NLOOKS = 15.0  # strides y=3, x=5 from the original dolphin_config.yaml
FILE_DATE_FMT = "%Y%m%d%H%M%S"


def get_pairs(limit: int | None, limit_dates: int | None) -> list[Path]:
    ifgs = sorted(IFG_DIR.glob("2*.int.tif"))
    assert len(ifgs) == 150, f"expected 150 ifgs, found {len(ifgs)}"
    if limit_dates is not None:
        dates = sorted({d for f in ifgs for d in f.name.split(".")[0].split("_")})
        keep = set(dates[:limit_dates])
        ifgs = [f for f in ifgs if set(f.name.split(".")[0].split("_")) <= keep]
    if limit is not None:
        ifgs = ifgs[:limit]
    return ifgs


def prep_spurt_inputs(out_base: Path, limit_dates: int | None) -> list[Path]:
    """Form single-reference interferograms from the phase-linked SLCs.

    spurt's SLCStackReader requires a directory of `<ref>_<sec>.int.tif` all
    sharing the first date as reference; it builds its own nearest-3 hop
    network internally (equal to this stack's bandwidth-3 network). dolphin's
    displacement workflow does the same single-reference conversion when
    `unwrap_method: spurt` is configured.
    """
    import numpy as np
    import rasterio

    slcs = sorted((BASE / "dolphin/phase_linking/linked_phase").glob("2*.slc.tif"))
    assert len(slcs) == 52, f"expected 52 linked SLCs, found {len(slcs)}"
    if limit_dates is not None:
        slcs = slcs[:limit_dates]
    ref_date = slcs[0].name.split(".")[0]

    in_dir = out_base / "spurt" / "input_ifgs"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [
        in_dir / f"{ref_date}_{sec.name.split('.')[0]}.int.tif" for sec in slcs[1:]
    ]
    missing = [
        (sec, path)
        for sec, path in zip(slcs[1:], out_paths)
        if not path.exists()
    ]
    if missing:
        with rasterio.open(slcs[0]) as s:
            slc0 = s.read(1)
            profile = dict(s.profile, dtype="complex64", count=1)
        for sec, path in missing:
            with rasterio.open(sec) as s:
                ifg = (slc0 * np.conj(s.read(1))).astype(np.complex64)
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(ifg, 1)
            print(f"  formed {path.name}", flush=True)
    return out_paths


def make_options(engine: str, n_jobs: int):
    from dolphin.workflows.config import UnwrapOptions

    opts = UnwrapOptions(
        unwrap_method="spurt" if engine.startswith("spurt") else engine,
        n_parallel_jobs=n_jobs,
        run_goldstein=False,
        run_interpolation=False,
        zero_where_masked=False,
    )
    if engine == "snaphu":
        opts.snaphu_options.ntiles = (2, 2)
        opts.snaphu_options.tile_overlap = (400, 400)
        opts.snaphu_options.n_parallel_tiles = 4
        opts.snaphu_options.init_method = "mcf"
        opts.snaphu_options.cost = "smooth"
    if engine == "spurt_singletile":
        opts.spurt_options.general_settings.use_tiles = False
        # whole-scene graph: parallelize the ~120 temporal link batches and the
        # 150 serial spatial MCFs; each spatial worker holds its own copy of
        # the 6.1M-point solver (~5-9 GB), so 3 is the RAM-safe ceiling
        opts.spurt_options.solver_settings.t_worker_count = 9
        opts.spurt_options.solver_settings.s_worker_count = 3
    return opts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--engine",
        required=True,
        choices=["whirlwind", "snaphu", "phass", "icu", "spurt", "spurt_singletile"],
    )
    p.add_argument("--limit", type=int, default=None, help="only first N pairs")
    p.add_argument(
        "--limit-dates",
        type=int,
        default=None,
        help="only pairs among the first N acquisition dates (for spurt smoke tests)",
    )
    p.add_argument("--n-jobs", type=int, default=-1, help="parallel unwrap jobs")
    p.add_argument("--out-base", type=Path, default=OUT_BASE)
    args = p.parse_args()

    from dolphin import unwrap

    out_dir = args.out_base / args.engine / "unwrapped"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"
    scratch.mkdir(exist_ok=True)

    opts = make_options(args.engine, args.n_jobs)

    t0 = time.perf_counter()
    if args.engine.startswith("spurt"):
        from dolphin.unwrap._unwrap_3d import unwrap_spurt

        assert TCOH.exists() and SIMILARITY.exists()
        ifgs = prep_spurt_inputs(args.out_base, args.limit_dates)
        print(f"[spurt] unwrapping {len(ifgs)} single-ref ifgs -> {out_dir}")
        unw_paths, cc_paths = unwrap_spurt(
            ifg_filenames=ifgs,
            output_path=out_dir,
            temporal_coherence_filename=TCOH,
            similarity_filename=SIMILARITY,
            options=opts.spurt_options,
            scratchdir=scratch,
            file_date_fmt=FILE_DATE_FMT,
        )
    else:
        ifgs = get_pairs(args.limit, args.limit_dates)
        cors = [f.parent / (f.name.split(".")[0] + ".int.cor.tif") for f in ifgs]
        for c in cors:
            assert c.exists(), f"missing correlation file {c}"
        print(f"[{args.engine}] unwrapping {len(ifgs)} pairs -> {out_dir}")
        unw_paths, cc_paths = unwrap.run(
            ifg_filenames=ifgs,
            cor_filenames=cors,
            output_path=out_dir,
            unwrap_options=opts,
            nlooks=NLOOKS,
            mask_filename=None,
            scratchdir=scratch,
            file_date_fmt=FILE_DATE_FMT,
        )
    dt = time.perf_counter() - t0
    print(
        f"[{args.engine}] DONE: {len(unw_paths)} unwrapped, {len(cc_paths)} conncomp"
        f" in {dt:.1f}s ({dt / max(len(ifgs), 1):.1f}s/pair)"
    )


if __name__ == "__main__":
    main()
