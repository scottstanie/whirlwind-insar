#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "whirlwind-insar>=0.3.1",
#   "numpy",
#   "h5py",
#   "matplotlib",
#   "pandas",
#   "earthaccess>=0.11",
#   "boto3",
#   "requests",
#   "psutil",
# ]
# ///
"""Compare whirlwind's unwrapping against a NISAR L2 GUNW product.

What this does
--------------
For each NISAR GUNW product (a local ``.h5``, an ASF download URL, an ``s3://``
URI, or a bare granule name), this script:

1. Reads the production 80 m ``unwrappedPhase`` and **re-wraps** it to
   ``[-pi, pi)``. That re-wrapped phase is the *only* input handed to whirlwind,
   so this is an apples-to-apples test of the unwrapping algorithm on the exact
   grid the production unwrapped product lives on. (The beta GUNW also ships a
   20 m ``wrappedInterferogram``, but its georeferencing has been flagged as
   unreliable in some beta products, so re-wrapping the 80 m unwrapped phase is
   the cleaner comparison. See ``--use-product-wrapped`` to override.)
2. Reads the production coherence and the GUNW ``mask`` (water / subswath).
3. Runs ``whirlwind.unwrap(...)`` -- the exact public API an external user would
   call -- producing an unwrapped phase **and** connected-component labels.
4. Compares whirlwind's output to the production ``unwrappedPhase`` and
   ``connectedComponents``: per-pixel 2*pi ambiguity agreement (overall and
   re-leveled within each production component), wrapped-residual RMSE,
   component counts, and label coverage / recall. Records runtime and peak-RSS.
5. Writes, per product: a ``<crop>.json`` of metrics, a ``<crop>.png`` six-panel
   figure, and a ``<crop>_arrays.npz`` of the rasters. Across all products it
   writes ``summary.csv`` / ``summary.md``.

Because the comparison only needs the production unwrapped phase, you do **not**
need to have pre-downloaded anything: point it at a URL or granule name and it
fetches the product first.

Authentication (for URL / granule inputs)
-----------------------------------------
NISAR GUNW products are behind NASA Earthdata Login (EDL). Provide credentials
by any one of:
  * ``EARTHDATA_TOKEN`` env var (an EDL bearer token), or
  * ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` env vars, or
  * a ``~/.netrc`` entry for ``urs.earthdata.nasa.gov``.
Running in **AWS us-west-2** keeps all transfers in-region (the ASF DAAC cloud
holdings live there), so downloads incur no egress cost.

Examples
--------
Local file::

    python compare_gunw.py NISAR_L2_PR_GUNW_..._001.h5 --out-dir out

ASF download URL (the format ASF publishes)::

    python compare_gunw.py \
      https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_BETA_V1/<ID>/<ID>.h5 \
      --out-dir out --upload-s3 s3://my-bucket/ww-gunw-bench/

Bare granule names (resolved via earthaccess/CMR)::

    python compare_gunw.py \
      NISAR_L2_PR_GUNW_009_163_A_140_010_7700_SH_..._001 \
      --out-dir out

A manifest file (one URL / granule / path per line, ``#`` comments allowed)::

    python compare_gunw.py --inputs-file sample_granules.txt --out-dir out
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TWOPI = 2.0 * np.pi
SHORT_NAME = "NISAR_L2_GUNW_BETA_V1"

# ---------------------------------------------------------------------------
# Comparison core.
#
# These functions are kept deliberately dependency-light (numpy + matplotlib)
# and mirror the metrics in ``scripts/bench_nisar_gunw_whirlwind.py``, the
# internal benchmark this tool is derived from. If you change the metric
# definitions here, change them there too (and vice-versa).
# ---------------------------------------------------------------------------


def wrap_phase(x: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi), preserving NaNs."""
    return (x + np.pi) % TWOPI - np.pi


def read_array(ds: h5py.Dataset, dtype: np.dtype | type = np.float32) -> np.ndarray:
    """Read an HDF5 dataset, turning fill values into NaN for float arrays."""
    arr = ds[()]
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    fill = ds.attrs.get("_FillValue")
    if fill is not None and np.issubdtype(arr.dtype, np.floating):
        fill_value = np.asarray(fill).reshape(-1)[0]
        arr = arr.copy()
        arr[arr == fill_value] = np.nan
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.where(arr < -1.0e20, np.nan, arr)
    return arr


def choose_pol(h5: h5py.File, base: str, requested: str | None) -> str:
    grp = h5[base]
    pols = [k for k, v in grp.items() if isinstance(v, h5py.Group)]
    pols = [p for p in pols if p.upper() not in {"MASK", "METADATA"}]
    if requested:
        if requested not in pols:
            raise KeyError(
                f"Requested pol {requested!r} not found under {base}; available={pols}"
            )
        return requested
    if not pols:
        raise KeyError(f"No polarization groups found under {base}")
    return sorted(pols)[0]


def gunw_paths(h5: h5py.File, pol: str | None) -> dict[str, str]:
    unw_base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
    wrap_base = "/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram"
    pol = choose_pol(h5, unw_base, pol)
    return {
        "pol": pol,
        "unw": f"{unw_base}/{pol}/unwrappedPhase",
        "coh_unw": f"{unw_base}/{pol}/coherenceMagnitude",
        "cc": f"{unw_base}/{pol}/connectedComponents",
        "mask": f"{unw_base}/mask",
        "wrapped": f"{wrap_base}/{pol}/wrappedInterferogram",
    }


def mask_to_bool(
    mask_arr: np.ndarray | None, policy: str, shape: tuple[int, int]
) -> np.ndarray:
    """Convert the GUNW ``mask`` code into a boolean valid-pixel mask.

    The GUNW ``mask`` is a 3-digit code ``[water][subswath_ref][subswath_sec]``
    with ``_FillValue`` 255 (water digit: 1 = water).
    """
    if mask_arr is None or policy == "ignore":
        return np.ones(shape, dtype=bool)
    if mask_arr.shape != shape:
        raise ValueError(f"Mask shape {mask_arr.shape} != data shape {shape}")
    if policy == "water_only":
        # Exclude only water; keep subswath-flagged pixels valid.
        return (mask_arr != 255) & ((mask_arr // 100) % 10 == 0)
    if policy == "nisar_land":
        # Keep non-water pixels that are valid samples in both RSLC subswaths.
        water = (mask_arr // 100) % 10
        ref_sub = (mask_arr // 10) % 10
        sec_sub = mask_arr % 10
        return (mask_arr != 255) & (water == 0) & (ref_sub > 0) & (sec_sub > 0)
    raise ValueError(f"Unknown mask policy {policy!r}")


def center_crop_slices(
    shape: tuple[int, int], size: int | str
) -> tuple[slice, slice, str]:
    ny, nx = shape
    if str(size).lower() == "full":
        return slice(0, ny), slice(0, nx), "full"
    n = int(size)
    if n > ny or n > nx:
        raise ValueError(f"Requested crop size {n} exceeds array shape {shape}")
    y0 = (ny - n) // 2
    x0 = (nx - n) // 2
    return slice(y0, y0 + n), slice(x0, x0 + n), f"{n}x{n}"


def component_summary(cc: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    vals = cc[valid]
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return {"num_cc": 0, "largest_cc_frac": 0.0, "nonzero_cc_frac": 0.0}
    labels, counts = np.unique(vals.astype(np.int64), return_counts=True)
    return {
        "num_cc": int(labels.size),
        "largest_cc_frac": float(counts.max() / max(1, valid.sum())),
        "nonzero_cc_frac": float(vals.size / max(1, valid.sum())),
    }


def safe_percentiles(x: np.ndarray, q: Iterable[float]) -> list[float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [math.nan for _ in q]
    return [float(v) for v in np.nanpercentile(x, list(q))]


def compute_compare_stats(
    ig: np.ndarray,
    coh: np.ndarray,
    mask: np.ndarray,
    prod_unw: np.ndarray,
    prod_cc: np.ndarray,
    ww_unw: np.ndarray,
    ww_cc: np.ndarray | None,
    runtime_s: float,
    rss_delta_mb: float | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Compare whirlwind output to the production unwrap + connected components.

    Returns ``(stats, ww_aligned, residual_wrapped, ambiguity_diff)``.
    """
    valid = (
        mask
        & np.isfinite(ig)
        & np.isfinite(coh)
        & np.isfinite(prod_unw)
        & np.isfinite(ww_unw)
    )
    if valid.sum() == 0:
        raise ValueError("No valid pixels for comparison after masking.")

    # Align a global 2*pi offset before measuring ambiguity differences.
    global_cycle_offset = int(
        np.rint(np.nanmedian((ww_unw[valid] - prod_unw[valid]) / TWOPI))
    )
    ww_aligned = ww_unw - global_cycle_offset * TWOPI

    # Per-pixel ambiguity integers relative to the exact wrapped input.
    prod_amb = np.rint((prod_unw - ig) / TWOPI).astype(np.float64)
    ww_amb = np.rint((ww_aligned - ig) / TWOPI).astype(np.float64)
    amb_diff = ww_amb - prod_amb

    residual = ww_aligned - prod_unw
    residual_wrapped = wrap_phase(residual)
    wrap_consistency = wrap_phase(ww_unw - ig)

    resid_valid = residual[valid]
    resid_wrap_valid = residual_wrapped[valid]
    amb_valid = amb_diff[valid]
    coh_valid = coh[valid]
    abs_amb = np.abs(amb_valid)
    nonzero_amb = abs_amb > 0

    # Per-(production)-component cycle alignment. The absolute cycle of a region
    # isolated by water / decorrelation is unobservable, so align ww to
    # production WITHIN each production connected component before scoring. This
    # separates "right shape within a region" from "guessed the same arbitrary
    # inter-region offset". Scored only over prod_cc > 0 pixels.
    in_comp = valid & (prod_cc > 0)
    if in_comp.any():
        off_map = np.zeros(amb_diff.shape, dtype=np.float64)
        for lab in np.unique(prod_cc[in_comp]):
            m = valid & (prod_cc == lab)
            off_map[m] = np.rint(np.median(amb_diff[m]))
        amb_pc = amb_diff - off_map
        match_percomp = float(np.mean(amb_pc[in_comp] == 0))
    else:
        match_percomp = float("nan")

    # Coverage / recall: of pixels that HAVE data, what fraction does each
    # unwrapper label (conncomp > 0)?
    data = mask
    ndata = int(data.sum())
    recall_prod = float((prod_cc[data] > 0).mean()) if ndata else float("nan")
    recall_ww = (
        float((np.asarray(ww_cc)[data] > 0).mean())
        if (ww_cc is not None and ndata)
        else None
    )

    stats: dict[str, Any] = {
        "runtime_s": float(runtime_s),
        "rss_delta_mb": None if rss_delta_mb is None else float(rss_delta_mb),
        "shape_y": int(ig.shape[0]),
        "shape_x": int(ig.shape[1]),
        "num_pixels": int(ig.size),
        "num_valid": int(valid.sum()),
        "valid_frac": float(valid.mean()),
        "coh_mean_valid": float(np.nanmean(coh_valid)),
        "coh_p50_valid": safe_percentiles(coh_valid, [50])[0],
        "global_cycle_offset_removed": global_cycle_offset,
        "ambiguity_match_frac": float(np.mean(amb_valid == 0)),
        "ambiguity_match_frac_percomp": match_percomp,
        "ambiguity_nonzero_frac": float(np.mean(nonzero_amb)),
        "data_frac": float(data.mean()),
        "num_data": ndata,
        "prod_unwrapped_recall": recall_prod,
        "ww_unwrapped_recall": recall_ww,
        "ambiguity_abs_mean_cycles": float(np.mean(abs_amb)),
        "ambiguity_abs_p95_cycles": safe_percentiles(abs_amb, [95])[0],
        "residual_mean_rad": float(np.nanmean(resid_valid)),
        "residual_std_rad": float(np.nanstd(resid_valid)),
        "residual_rmse_rad": float(np.sqrt(np.nanmean(resid_valid**2))),
        "residual_wrapped_rmse_rad": float(np.sqrt(np.nanmean(resid_wrap_valid**2))),
        "residual_wrapped_p95_abs_rad": safe_percentiles(
            np.abs(resid_wrap_valid), [95]
        )[0],
        "ww_wrap_consistency_p95_abs_rad": safe_percentiles(
            np.abs(wrap_consistency[valid]), [95]
        )[0],
    }
    stats |= {f"prod_{k}": v for k, v in component_summary(prod_cc, valid).items()}
    if ww_cc is not None:
        stats |= {
            f"ww_{k}": v for k, v in component_summary(np.asarray(ww_cc), valid).items()
        }
    return stats, ww_aligned, residual_wrapped, amb_diff


def plot_result(
    out_png: Path,
    ig: np.ndarray,
    coh: np.ndarray,
    prod_unw: np.ndarray,
    ww_aligned: np.ndarray,
    prod_cc: np.ndarray,
    ww_cc: np.ndarray | None,
    amb_diff: np.ndarray,
    valid: np.ndarray,
    title: str,
    stride: int = 1,
) -> None:
    """Eight-panel comparison figure.

    Row 1: wrapped input, coherence, NISAR GUNW unwrapped, whirlwind unwrapped.
    Row 2: NISAR GUNW (SNAPHU) conncomps, whirlwind conncomps, conncomp coverage
    difference (which unwrapper labels each pixel), and the 2*pi ambiguity diff.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    s = (slice(None, None, stride), slice(None, None, stride))
    shape = ig[s].shape
    valid_s = valid[s]

    def _cc(cc: np.ndarray | None) -> np.ndarray:
        a = np.asarray(cc) if cc is not None else None
        return a[s] if (a is not None and a.shape == ig.shape) else np.zeros(shape)

    prod_cc_s = _cc(prod_cc)
    ww_cc_s = _cc(ww_cc)

    # Connected-component coverage: which unwrapper assigned a component (cc > 0)
    # to each valid pixel. +1 = whirlwind only, -1 = production only, 0 = both,
    # NaN = neither. This is the meaningful conncomp comparison (the integer
    # label IDs themselves are arbitrary between the two algorithms).
    wl = ww_cc_s > 0
    pl = prod_cc_s > 0
    coverage = np.full(shape, np.nan)
    coverage[valid_s & wl & pl] = 0.0
    coverage[valid_s & wl & ~pl] = 1.0
    coverage[valid_s & ~wl & pl] = -1.0
    cov_cmap = ListedColormap(["#2c7bb6", "#d9d9d9", "#d7191c"])
    cov_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cov_cmap.N)

    panels = [
        ("wrapped input (rad)", ig[s], "twilight"),
        ("coherence", coh[s], "gray"),
        ("NISAR GUNW unwrapped", prod_unw[s], "viridis"),
        ("whirlwind aligned", ww_aligned[s], "viridis"),
        ("NISAR GUNW (SNAPHU) conncomps", prod_cc_s, "tab20"),
        ("whirlwind conncomps", ww_cc_s, "tab20"),
        ("conncomp coverage (ww-prod)", coverage, None),
        ("ambiguity diff (cycles)", amb_diff[s], "RdBu"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    fig.suptitle(title)
    for ax, (name, arr, cmap) in zip(axes.ravel(), panels, strict=True):
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        arrp = np.asarray(arr, dtype=float)
        if name == "conncomp coverage (ww-prod)":
            im = ax.imshow(arrp, cmap=cov_cmap, norm=cov_norm, interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, shrink=0.78, ticks=[-1, 0, 1])
            cb.ax.set_yticklabels(["prod only", "both", "ww only"])
            continue
        if name in {"NISAR GUNW (SNAPHU) conncomps", "whirlwind conncomps"}:
            arrp = np.where(arrp > 0, ((arrp - 1) % 20) + 1, np.nan)
            vmin, vmax = 0, 20
        else:
            arrp = np.where(valid_s, arrp, np.nan)
            if name == "wrapped input (rad)":
                vmin, vmax = -np.pi, np.pi
            elif name == "coherence":
                vmin, vmax = 0.0, 1.0
            elif name == "ambiguity diff (cycles)":
                vmax_abs = (
                    float(np.nanpercentile(np.abs(arrp), 99))
                    if np.isfinite(arrp).any()
                    else 1.0
                )
                vmax_abs = max(vmax_abs, 1.0)
                vmin, vmax = -vmax_abs, vmax_abs
            else:
                vmin, vmax = safe_percentiles(arrp, [2, 98])
        im = ax.imshow(arrp, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.78)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Input resolution + download.
# ---------------------------------------------------------------------------


def is_main_gunw_h5(path: Path) -> bool:
    name = path.name
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        return False
    if any(s in name for s in ["QA_STATS", "STATS", "_QA_", "REPORT"]):
        return False
    return "GUNW" in name


def product_id(path: Path) -> str:
    return path.stem.replace(".", "_")


def get_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def _edl_token() -> str | None:
    return os.environ.get("EARTHDATA_TOKEN") or os.environ.get("EDL_TOKEN")


def _ensure_netrc() -> None:
    """Write a ~/.netrc for urs.earthdata.nasa.gov from EARTHDATA_USERNAME /
    EARTHDATA_PASSWORD if those are set (e.g. injected from a Secrets Manager
    secret on AWS Batch) and no entry exists yet. This makes both the requests
    HTTPS path and earthaccess authenticate without a bearer token.
    """
    user = os.environ.get("EARTHDATA_USERNAME")
    pw = os.environ.get("EARTHDATA_PASSWORD")
    if not (user and pw):
        return
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists() and "urs.earthdata.nasa.gov" in netrc_path.read_text():
        return
    with open(netrc_path, "a") as f:
        f.write(f"\nmachine urs.earthdata.nasa.gov login {user} password {pw}\n")
    netrc_path.chmod(0o600)


def download_https(url: str, dest_dir: Path) -> Path:
    """Download an EDL-protected ASF URL over authenticated HTTPS.

    Uses ``EARTHDATA_TOKEN`` (bearer) if set, otherwise falls back to a
    ``~/.netrc`` entry for ``urs.earthdata.nasa.gov`` (requests' default).
    """
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urlparse(url).path).name
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest}", flush=True)
        return dest

    token = _edl_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    session = requests.Session()
    print(f"  downloading (https) {url}", flush=True)
    with session.get(
        url, headers=headers, stream=True, allow_redirects=True, timeout=(30, 1800)
    ) as r:
        if r.status_code in (401, 403):
            raise SystemExit(
                f"EDL auth failed ({r.status_code}) for {url}. Set EARTHDATA_TOKEN "
                "or add a ~/.netrc entry for urs.earthdata.nasa.gov."
            )
        r.raise_for_status()
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        tmp.rename(dest)
    return dest


def download_s3(uri: str, dest_dir: Path, profile: str | None = None) -> Path:
    """Download an ``s3://bucket/key`` object using boto3 (task-role creds)."""
    import boto3

    dest_dir.mkdir(parents=True, exist_ok=True)
    bucket, _, key = uri[len("s3://") :].partition("/")
    if not key:
        raise ValueError(f"Malformed S3 URI (need s3://bucket/key): {uri!r}")
    dest = dest_dir / Path(key).name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest}", flush=True)
        return dest
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    print(f"  downloading (s3) {uri}", flush=True)
    session.client("s3").download_file(bucket, key, str(dest))
    return dest


def download_granule(name: str, dest_dir: Path) -> list[Path]:
    """Resolve a bare granule name to local file(s) via earthaccess/CMR."""
    import earthaccess

    dest_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.login()
    results = earthaccess.search_data(
        short_name=SHORT_NAME, granule_name=f"{name}*", count=10
    )
    if not results:
        raise RuntimeError(f"No earthaccess results for granule_name={name!r}")
    paths = [Path(p) for p in earthaccess.download(results, local_path=str(dest_dir))]
    h5s = [p for p in paths if is_main_gunw_h5(p)]
    if not h5s:
        raise RuntimeError(f"Downloaded {name!r} but found no main GUNW .h5")
    return h5s


def resolve_inputs(
    tokens: list[str], data_dir: Path, s3_profile: str | None
) -> list[Path]:
    """Turn a mix of local paths / URLs / s3 URIs / granule names into local files."""
    out: list[Path] = []
    for tok in tokens:
        p = Path(tok)
        if p.exists() and p.is_file():
            out.append(p)
        elif tok.startswith("s3://"):
            out.append(download_s3(tok, data_dir, profile=s3_profile))
        elif tok.startswith(("http://", "https://")):
            out.append(download_https(tok, data_dir))
        else:
            out.extend(download_granule(tok, data_dir))
    # Deduplicate, preserving order.
    seen: set[Path] = set()
    uniq = [p for p in out if not (p.resolve() in seen or seen.add(p.resolve()))]
    bad = [p for p in uniq if not is_main_gunw_h5(p)]
    if bad:
        raise SystemExit(f"Not recognized as main GUNW .h5 files: {bad}")
    return uniq


def dump_flat_inputs(
    out_dir: Path,
    stem: str,
    phase: np.ndarray,
    cor: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Write the solver inputs as flat binary so the pure-Rust ``whirlwind`` CLI
    can be run on the exact same data (the CLI cannot read HDF5).

    Writes ``<stem>.phase`` (float32 wrapped radians, snaphu FLOAT_DATA),
    ``<stem>.cor`` (float32 coherence), ``<stem>.mask`` (uint8), and a
    ``<stem>.cli.txt`` with the equivalent CLI invocation. Masked pixels already
    have cor=0, so the CLI's default ``corr > 0`` masking reproduces the mask
    without a separate mask raster (the .mask file is kept for other tooling).
    """
    ny, nx = phase.shape
    phase.astype("<f4").tofile(out_dir / f"{stem}.phase")
    cor.astype("<f4").tofile(out_dir / f"{stem}.cor")
    mask.astype("u1").tofile(out_dir / f"{stem}.mask")
    (out_dir / f"{stem}.cli.txt").write_text(
        f"# {nx} cols x {ny} rows, flat binary little-endian float32\n"
        f"whirlwind --phase {stem}.phase --cor {stem}.cor --cols {nx} "
        f"--nlooks 16 --out {stem}.unw.tif --conncomp {stem}.conncomp.tif\n"
    )


def upload_dir_s3(local_dir: Path, s3_uri: str, profile: str | None = None) -> None:
    import boto3

    bucket, _, prefix = s3_uri[len("s3://") :].partition("/")
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    client = session.client("s3")
    n = 0
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            key = f"{prefix.rstrip('/')}/{rel}".lstrip("/")
            client.upload_file(str(p), bucket, key)
            n += 1
    print(f"Uploaded {n} files to s3://{bucket}/{prefix.rstrip('/')}/", flush=True)


# ---------------------------------------------------------------------------
# Run one product.
# ---------------------------------------------------------------------------


def compare_one(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    import whirlwind as ww

    print(f"\n=== {path.name} ===", flush=True)
    out_product = args.out_dir / product_id(path)
    out_product.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as h5:
        paths = gunw_paths(h5, args.pol)
        pol = paths["pol"]
        prod_unw_full = read_array(h5[paths["unw"]], np.float32)
        coh_full = read_array(h5[paths["coh_unw"]], np.float32)
        prod_cc_full = h5[paths["cc"]][()].astype(np.int64, copy=False)
        mask_arr_full = h5[paths["mask"]][()] if paths["mask"] in h5 else None
        if args.use_product_wrapped and paths["wrapped"] in h5:
            wrapped_complex = h5[paths["wrapped"]][()]
            if wrapped_complex.shape == prod_unw_full.shape:
                ig_full = np.angle(wrapped_complex).astype(np.float32)
            else:
                print(
                    f"  wrappedInterferogram shape {wrapped_complex.shape} != "
                    f"{prod_unw_full.shape}; re-wrapping unwrappedPhase instead.",
                    flush=True,
                )
                ig_full = wrap_phase(prod_unw_full).astype(np.float32)
        else:
            ig_full = wrap_phase(prod_unw_full).astype(np.float32)

    base_mask_full = mask_to_bool(mask_arr_full, args.mask_policy, prod_unw_full.shape)
    base_mask_full &= (
        np.isfinite(prod_unw_full)
        & np.isfinite(coh_full)
        & (coh_full >= args.coh_threshold)
    )

    crop_specs = [center_crop_slices(prod_unw_full.shape, s) for s in args.sizes]

    rows: list[dict[str, Any]] = []
    for ys, xs, label in crop_specs:
        result_json = out_product / f"{label}.json"
        if result_json.exists() and not args.force:
            print(f"  {label}: exists, skipping (--force to rerun)", flush=True)
            rows.append(json.loads(result_json.read_text()))
            continue

        ig = np.ascontiguousarray(ig_full[ys, xs])
        coh = np.ascontiguousarray(coh_full[ys, xs])
        prod_unw = np.ascontiguousarray(prod_unw_full[ys, xs])
        prod_cc = np.ascontiguousarray(prod_cc_full[ys, xs])
        mask = np.ascontiguousarray(base_mask_full[ys, xs])
        if mask.sum() == 0:
            print(f"  {label}: no valid pixels, skipping", flush=True)
            continue

        print(
            f"  {label}: running ww.unwrap on shape={ig.shape}, "
            f"valid={mask.mean():.3f}, nlooks={args.nlooks}, "
            f"downsample={args.downsample}, interpolate={args.interpolate}, "
            f"interp_across_mask={args.interp_across_mask}, "
            f"goldstein_alpha={args.goldstein_alpha}",
            flush=True,
        )
        # ZERO the phase outside the mask: the production unwrappedPhase is not
        # flat in masked regions, and feeding raw masked phase makes the solver
        # see spurious residues. Sanitize coherence for the solver too (clip to
        # [0,1], NaN->0, zero outside mask) so the cost LUT stays well-behaved.
        # The stats below still use the raw ``coh``.
        ig_solver = np.where(mask, ig, 0.0).astype(np.float32)
        ig_complex = np.where(mask, np.exp(1j * ig_solver), 0.0j).astype(np.complex64)
        coh_solver = np.where(mask, np.clip(np.nan_to_num(coh), 0.0, 1.0), 0.0).astype(
            np.float32
        )
        if args.dump_flat:
            dump_flat_inputs(out_product, label, ig_solver, coh_solver, mask)

        gc.collect()
        rss0 = get_rss_mb()
        t0 = time.perf_counter()
        # "auto" / "none" pass through; a number becomes a fixed coherence cutoff.
        mc = args.conncomp_min_coherence
        if isinstance(mc, str) and mc.lower() in ("none", "off"):
            mc = None
        elif isinstance(mc, str) and mc.lower() != "auto":
            mc = float(mc)
        ww_unw, ww_cc = ww.unwrap(
            ig_complex,
            coh_solver,
            float(args.nlooks),
            mask,
            bridge=args.bridge,
            downsample=args.downsample,
            interpolate=args.interpolate,
            interp_across_mask=args.interp_across_mask,
            interp_cutoff=args.interp_cutoff,
            conncomp_min_coherence=mc,
            goldstein_alpha=args.goldstein_alpha,
            goldstein_psize=args.goldstein_psize,
            phase_grad_window=tuple(args.phase_grad_window),
        )
        runtime_s = time.perf_counter() - t0
        rss1 = get_rss_mb()
        rss_delta = None if (rss0 is None or rss1 is None) else rss1 - rss0

        ww_unw = np.asarray(ww_unw, dtype=np.float32)
        ww_cc_arr = None if ww_cc is None else np.asarray(ww_cc)
        stats, ww_aligned, residual_wrapped, amb_diff = compute_compare_stats(
            ig=ig,
            coh=coh,
            mask=mask,
            prod_unw=prod_unw,
            prod_cc=prod_cc,
            ww_unw=ww_unw,
            ww_cc=ww_cc_arr,
            runtime_s=runtime_s,
            rss_delta_mb=rss_delta,
        )
        stats.update(
            {
                "product": path.name,
                "product_path": str(path),
                "crop": label,
                "pol": pol,
                "nlooks": args.nlooks,
                "bridge": args.bridge,
                "downsample": args.downsample,
                "interpolate": args.interpolate,
                "interp_across_mask": args.interp_across_mask,
                "interp_cutoff": args.interp_cutoff,
                "goldstein_alpha": args.goldstein_alpha,
                "goldstein_psize": args.goldstein_psize,
                "phase_grad_window": list(args.phase_grad_window),
                "conncomp_min_coherence": args.conncomp_min_coherence,
                "mask_policy": args.mask_policy,
                "whirlwind_version": getattr(ww, "__version__", "unknown"),
                "input_phase_source": "phase(wrappedInterferogram)"
                if args.use_product_wrapped
                else "wrap(unwrappedPhase)",
            }
        )

        result_json.write_text(json.dumps(stats, indent=2, sort_keys=True))
        np.savez_compressed(
            out_product / f"{label}_arrays.npz",
            ig=ig,
            coh=coh,
            mask=mask,
            prod_unw=prod_unw,
            prod_cc=prod_cc,
            ww_unw=ww_unw,
            ww_aligned=ww_aligned,
            ww_cc=np.asarray([]) if ww_cc_arr is None else ww_cc_arr,
            residual_wrapped=residual_wrapped,
            ambiguity_diff=amb_diff,
        )
        plot_result(
            out_product / f"{label}.png",
            ig=ig,
            coh=coh,
            prod_unw=prod_unw,
            ww_aligned=ww_aligned,
            prod_cc=prod_cc,
            ww_cc=ww_cc_arr,
            amb_diff=amb_diff,
            valid=mask & np.isfinite(ww_unw),
            title=(
                f"{path.name}\n{label}, pol={pol}, nlooks={args.nlooks}, "
                f"runtime={runtime_s:.1f}s"
            ),
            stride=max(1, args.plot_downsample),
        )
        rec_ww = stats.get("ww_unwrapped_recall")
        rec_ww_s = f"{rec_ww:.3f}" if rec_ww is not None else "n/a"
        print(
            f"  {label}: {runtime_s:.1f}s  match={stats['ambiguity_match_frac']:.3f} "
            f"per-comp={stats['ambiguity_match_frac_percomp']:.3f}  "
            f"data={stats['data_frac'] * 100:.0f}%  "
            f"recall ww={rec_ww_s}/prod={stats['prod_unwrapped_recall']:.3f}  "
            f"cc ww={stats.get('ww_num_cc')}/prod={stats.get('prod_num_cc')}",
            flush=True,
        )
        rows.append(stats)
        del ig, coh, prod_unw, prod_cc, mask, ww_unw, ww_aligned
        gc.collect()
    return rows


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    cols = [
        "product",
        "crop",
        "runtime_s",
        "shape_y",
        "shape_x",
        "valid_frac",
        "ambiguity_match_frac",
        "ambiguity_match_frac_percomp",
        "residual_wrapped_p95_abs_rad",
        "prod_num_cc",
        "ww_num_cc",
    ]
    cols = [c for c in cols if c in df.columns]
    table = df[cols].to_string(index=False)
    (out_dir / "summary.md").write_text(
        "# whirlwind vs NISAR GUNW comparison\n\n"
        "`ambiguity_match_frac` = per-pixel fraction where whirlwind and the\n"
        "production unwrap agree on the 2*pi integer (after a single global\n"
        "offset). `_percomp` re-levels within each production connected\n"
        "component first (the only fair comparison across water/decorrelation\n"
        "gaps). Higher is better; 1.0 = identical ambiguities.\n\n"
        "```\n" + table + "\n```\n"
    )
    print(f"\nWrote {csv_path}\n{table}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare whirlwind.unwrap against NISAR L2 GUNW products.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="*",
        help="GUNW inputs: local .h5 path, ASF https URL, s3:// URI, or granule name.",
    )
    p.add_argument(
        "--inputs-file",
        type=Path,
        help="Text file with one input per line (# comments allowed).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("ww_gunw_out"))
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("nisar_data"),
        help="Where downloads are cached.",
    )
    p.add_argument(
        "--upload-s3",
        default=None,
        help="If set (s3://bucket/prefix), upload the whole out-dir there when done.",
    )
    p.add_argument(
        "--s3-profile",
        default=None,
        help="AWS profile for s3:// inputs / --upload-s3 (omit to use the default chain).",
    )
    p.add_argument(
        "--nlooks",
        type=float,
        default=16.0,
        help="Effective looks for the cost model (NISAR GUNW unwrap looks ~13x16).",
    )
    p.add_argument(
        "--sizes",
        nargs="*",
        default=["full"],
        help="Square center-crop sizes plus optional 'full'. Default: full frame.",
    )
    p.add_argument(
        "--conncomp-min-coherence",
        default="auto",
        help="Coherence floor below which conncomp drops pixels to 0 (a reliability "
        "mask). Default 'auto' is whirlwind's gentle looks-aware floor "
        "(0.32/sqrt(nlooks), e.g. 0.08 at 16 looks). Pass a number for a fixed "
        "cutoff, or 'none' to label every unwrapped pixel.",
    )
    p.add_argument(
        "--no-bridge",
        dest="bridge",
        action="store_false",
        help="Disable the disconnected-mask component re-leveling post-pass.",
    )
    p.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Coarse-solve factor passed to whirlwind.unwrap.",
    )
    p.add_argument(
        "--interpolate",
        action="store_true",
        help="Interpolate valid low-coherence pixels before the solve.",
    )
    p.add_argument(
        "--interp-across-mask",
        action="store_true",
        help="Also interpolate across masked pixels (requires --interpolate); "
        "useful for narrow water bodies.",
    )
    p.add_argument(
        "--interp-cutoff",
        type=float,
        default=0.1,
        help="Coherence cutoff for --interpolate.",
    )
    p.add_argument(
        "--goldstein-alpha",
        type=float,
        default=0.0,
        help="Goldstein prefilter strength; 0 disables it.",
    )
    p.add_argument(
        "--goldstein-psize",
        type=int,
        default=64,
        help="Goldstein FFT patch size.",
    )
    p.add_argument(
        "--phase-grad-window",
        type=int,
        nargs=2,
        metavar=("PARALLEL", "PERPENDICULAR"),
        default=(7, 7),
        help="Local phase-gradient averaging window passed to whirlwind.unwrap.",
    )
    p.add_argument("--pol", default=None, help="Polarization, e.g. HH. Default: first.")
    p.add_argument(
        "--mask-policy",
        choices=["water_only", "nisar_land", "ignore"],
        default="water_only",
    )
    p.add_argument("--coh-threshold", type=float, default=0.0)
    p.add_argument(
        "--use-product-wrapped",
        action="store_true",
        help="Use phase(wrappedInterferogram) instead of re-wrapping unwrappedPhase.",
    )
    p.add_argument("--plot-downsample", type=int, default=1)
    p.add_argument(
        "--dump-flat",
        action="store_true",
        help="Also write the solver inputs as flat binary (phase/cor/mask) plus a "
        "CLI command, so the standalone Rust `whirlwind` binary can be run on the "
        "exact same data (it cannot read HDF5).",
    )
    p.add_argument("--force", action="store_true", help="Rerun even if JSON exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.interp_across_mask and not args.interpolate:
        raise SystemExit("--interp-across-mask requires --interpolate")
    tokens = list(args.inputs)
    if args.inputs_file:
        for line in args.inputs_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                tokens.append(line)
    if not tokens:
        raise SystemExit("No inputs. Pass paths/URLs/granules or --inputs-file.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_netrc()
    h5s = resolve_inputs(tokens, args.data_dir, args.s3_profile)
    print(f"Resolved {len(h5s)} GUNW product(s).", flush=True)

    all_rows: list[dict[str, Any]] = []
    # Sequential on purpose: one large unwrap at a time keeps peak memory bounded.
    for h5 in h5s:
        all_rows.extend(compare_one(h5, args))

    if all_rows:
        write_summary(all_rows, args.out_dir)
    if args.upload_s3:
        upload_dir_s3(args.out_dir, args.upload_s3, profile=args.s3_profile)


if __name__ == "__main__":
    sys.exit(main())
