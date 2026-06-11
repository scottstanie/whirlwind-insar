#!/usr/bin/env python
"""Compare Palos Verdes unwrap-engine velocity estimates against the survey
monuments on the Portuguese Bend landslide.

For each engine's inverted time series (pv_timeseries_invert.py output):
  1. map each monument lon/lat -> looked radar pixel via the full-res
     geometry x/y rasters (lon/lat per radar pixel),
  2. sample a 3x3 window time series at each monument, fit a LOS velocity over
     the GPS-overlap window,
  3. project the monument ENU displacements to LOS (ground->satellite unit
     vector) and fit the GPS LOS velocity over the same window,
  4. re-reference every engine to a common stable box (linear-fit exact),
  5. score r / RMSE / slope, count abstentions ("gives up" = too few valid
     epochs at the monument), and draw the scatter + map figures.

Conventions: positive LOS = motion toward the satellite. dolphin wrote the
time series in meters (wavelength passed at inversion); we plot mm/yr.

    /Users/staniewi/miniforge3/envs/mapping-312/bin/python scripts/pv_gps_compare_unwrappers.py \
        [--engines whirlwind snaphu phass icu spurt] [--stable-box r0 r1 c0 c1]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

BASE = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes/"
    "Palos_Verdes_C13_RO23_SP"
)
E2E = BASE / "e2e_output_20260519"
GEOM = E2E / "stack_output/geometry"
CMP = E2E / "unwrap_compare"
STRIDES_YX = (3, 5)  # full-res -> looked, from dolphin_config.yaml output strides
WINDOW = ("2025-11-20", "2026-03-10")  # GPS surveys span 2025-12-02..2026-03-04
MIN_EPOCHS = 5
MIN_SPAN_DAYS = 45
# Rank requirement for the censored L2 inversion: a pixel needs >= 51 of the
# 150 pairs claimed or the minimum-norm solution silently returns ~0 values.
MIN_CLAIM_FRAC = 51 / 150
# Urban block, ICU claim 0.82, |whirlwind velocity| < 10 mm/yr (see
# figures/stable_box_selection.png); all engines re-referenced to it.
DEFAULT_STABLE_BOX = (208, 400, 2688, 2880)


def station_pixels() -> pd.DataFrame:
    """Monument lon/lat -> full-res then looked radar pixel, via x/y rasters."""
    cache = CMP / "gps_compare/station_pixels.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0)

    st = pd.read_csv(BASE / "GPS_Stations_LatLon.csv")
    dec = 16
    with rasterio.open(GEOM / "x.tif") as sx, rasterio.open(GEOM / "y.tif") as sy:
        H, W = sx.height, sx.width
        Xc = sx.read(1, out_shape=(H // dec, W // dec))
        Yc = sy.read(1, out_shape=(H // dec, W // dec))
        coslat = np.cos(np.deg2rad(np.nanmean(Yc)))
        rows = []
        for _, s in st.iterrows():
            d2 = (coslat * (Xc - s.Longitude)) ** 2 + (Yc - s.Latitude) ** 2
            r0, c0 = np.unravel_index(np.nanargmin(d2), d2.shape)
            # refine at full resolution in a window around the coarse hit
            half = 2 * dec
            rlo = max(0, r0 * dec - half)
            clo = max(0, c0 * dec - half)
            win = Window(clo, rlo, min(4 * dec, W - clo), min(4 * dec, H - rlo))
            Xf = sx.read(1, window=win)
            Yf = sy.read(1, window=win)
            d2f = (coslat * (Xf - s.Longitude)) ** 2 + (Yf - s.Latitude) ** 2
            rf, cf = np.unravel_index(np.nanargmin(d2f), d2f.shape)
            dist_deg = np.sqrt(d2f[rf, cf])
            row_full, col_full = rlo + rf, clo + cf
            rows.append(
                {
                    "id": s.Point,
                    "lon": s.Longitude,
                    "lat": s.Latitude,
                    "row": row_full // STRIDES_YX[0],
                    "col": col_full // STRIDES_YX[1],
                    "dist_m": dist_deg * 111_320,
                }
            )
    df = pd.DataFrame(rows).set_index("id")
    n_off = (df.dist_m > 25).sum()
    if n_off:
        print(f"dropping {n_off} stations >25 m from any radar pixel (off-footprint)")
    df = df[df.dist_m <= 25]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def display_name(eng: str) -> str:
    """Panel title: strip the inversion suffix, give spurt its proper name."""
    base = eng[: -len("_L1")] if eng.endswith("_L1") else eng
    core = base.endswith("_core")
    if core:
        base = base[: -len("_core")]
    names = {"spurt_singletile": "Spurt (EMCF)", "spurt": "Spurt (EMCF, tiled)"}
    label = names.get(base, base)
    return f"{label} core" if core else label


def los_vector() -> np.ndarray:
    j = json.loads((E2E / "los_enu.json").read_text())
    c = j["los_enu_ground_to_satellite"]["center"]
    return np.array([c["east"], c["north"], c["up"]])


def gps_los_velocity(px: pd.DataFrame) -> pd.Series:
    """LOS velocity (mm/yr, positive toward satellite) per monument."""
    enu = los_vector()
    ts = pd.read_csv(BASE / "GPS_Timeseries.csv", parse_dates=["Date"])
    ts = ts[(ts.Date >= WINDOW[0]) & (ts.Date <= WINDOW[1])]
    out = {}
    for sid, g in ts.groupby("Point"):
        if sid not in px.index or len(g) < 2:
            continue
        span = (g.Date.max() - g.Date.min()).days
        if span < MIN_SPAN_DAYS:
            continue
        los = g.dE_m * enu[0] + g.dN_m * enu[1] + g.dH_m * enu[2]
        t_yr = (g.Date - g.Date.min()).dt.days / 365.25
        out[sid] = np.polyfit(t_yr, los, 1)[0] * 1000.0
    return pd.Series(out, name="gps_v")


def ts_files(engine: str) -> tuple[list[Path], list[datetime]]:
    files = sorted((CMP / engine / "timeseries").glob("2*_2*.tif"))
    files = [f for f in files if not f.name.startswith("residuals")]
    dates = [datetime.strptime(f.name.split("_")[1].split(".")[0], "%Y%m%d%H%M%S") for f in files]
    return files, dates


def sample_series(engine: str, px: pd.DataFrame) -> pd.DataFrame:
    """3x3-median LOS displacement series (m) at each monument; NaN = invalid."""
    cache = CMP / f"gps_compare/{engine}_station_series.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    files, dates = ts_files(engine)
    assert files, f"no timeseries for {engine}"
    data = {}
    for f, d in zip(files, dates):
        with rasterio.open(f) as src:
            vals = []
            for _, s in px.iterrows():
                win = src.read(
                    1, window=Window(int(s.col) - 1, int(s.row) - 1, 3, 3)
                ).astype(float)
                win[win == 0.0] = np.nan  # nodata/abstained
                vals.append(
                    np.nanmedian(win) if np.isfinite(win).sum() >= 5 else np.nan
                )
        data[d] = vals
    df = pd.DataFrame(data, index=px.index).T.sort_index()
    df.to_csv(cache)
    return df


def fit_velocity(series: pd.DataFrame, t0: str, t1: str) -> pd.Series:
    """Per-station LOS velocity (mm/yr) over [t0, t1]; NaN if too sparse."""
    s = series.loc[(series.index >= t0) & (series.index <= t1)]
    t_yr = (s.index - s.index.min()).days / 365.25
    out = {}
    for sid in s.columns:
        y = s[sid].to_numpy()
        ok = np.isfinite(y)
        if ok.sum() < MIN_EPOCHS or (ok.any() and (t_yr[ok].max() - t_yr[ok].min()) * 365.25 < MIN_SPAN_DAYS):
            out[sid] = np.nan
            continue
        out[sid] = np.polyfit(t_yr[ok], y[ok], 1)[0] * 1000.0
    return pd.Series(out)


def stable_box_velocity(engine: str, box: tuple[int, int, int, int]) -> float:
    """Velocity (mm/yr) of the stable-box mean series, for re-referencing."""
    files, dates = ts_files(engine)
    r0, r1, c0, c1 = box
    means, ts = [], []
    for f, d in zip(files, dates):
        with rasterio.open(f) as src:
            a = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(float)
        a[a == 0.0] = np.nan
        if np.isfinite(a).mean() > 0.3:
            means.append(np.nanmean(a))
            ts.append(d)
    assert len(means) >= MIN_EPOCHS, f"{engine}: stable box mostly empty"
    t_yr = np.array([(d - ts[0]).days / 365.25 for d in ts])
    dates_idx = pd.DatetimeIndex(ts)
    keep = (dates_idx >= WINDOW[0]) & (dates_idx <= WINDOW[1])
    return float(np.polyfit(t_yr[keep], np.array(means)[keep], 1)[0] * 1000.0)


TCOH_FILE = (
    E2E
    / "dolphin/phase_linking/linked_phase/temporal_coherence_average_20251123203956_20260517062710.tif"
)
CORE_TCOH = 0.7
CORE_HALF = 3  # 7x7 search window around the monument pixel
CORE_MIN_PX = 3


def water_mask_looked() -> np.ndarray:
    """Majority water mask on the looked grid (True = land), cached."""
    cache = CMP / "water_mask_looked.npy"
    if cache.exists():
        return np.load(cache)
    with rasterio.open(GEOM / "water_mask.rdr.tif") as src:
        full = src.read(1)  # 0 = water, 1 = land, full res 12197x19012
    ry, rx = STRIDES_YX
    H, W = full.shape[0] // ry, full.shape[1] // rx
    land = (
        full[: H * ry, : W * rx].reshape(H, ry, W, rx).mean(axis=(1, 3)) > 0.5
    )
    np.save(cache, land)
    return land


def core_pixels_at(px: pd.DataFrame) -> dict:
    """Per monument: offsets of tc>CORE_TCOH pixels in the 7x7 search window."""
    out = {}
    with rasterio.open(TCOH_FILE) as src:
        for sid, s in px.iterrows():
            n = 2 * CORE_HALF + 1
            win = src.read(
                1, window=Window(int(s.col) - CORE_HALF, int(s.row) - CORE_HALF, n, n)
            )
            out[sid] = win > CORE_TCOH
    return out


def sample_series_core(engine: str, eng_label: str, px: pd.DataFrame) -> pd.DataFrame:
    """Solved-core series: nanmean over tc>0.7 pixels within a 7x7 window
    around each monument (>= CORE_MIN_PX such pixels required, forgiving of
    GPS-to-pixel misalignment); NaN where the core is absent."""
    cache = CMP / f"gps_compare/{eng_label}_station_series.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    core = core_pixels_at(px)
    files, dates = ts_files(engine)
    assert files, f"no timeseries for {engine}"
    n = 2 * CORE_HALF + 1
    data = {}
    for f, d in zip(files, dates):
        with rasterio.open(f) as src:
            vals = []
            for sid, s in px.iterrows():
                m = core[sid]
                if m.sum() < CORE_MIN_PX:
                    vals.append(np.nan)
                    continue
                win = src.read(
                    1,
                    window=Window(
                        int(s.col) - CORE_HALF, int(s.row) - CORE_HALF, n, n
                    ),
                ).astype(float)
                win[win == 0.0] = np.nan
                sel = win[m]
                vals.append(
                    np.nanmean(sel) if np.isfinite(sel).sum() >= CORE_MIN_PX else np.nan
                )
        data[d] = vals
    df = pd.DataFrame(data, index=px.index).T.sort_index()
    df.to_csv(cache)
    return df


def claim_fraction(engine: str, px: pd.DataFrame) -> pd.Series:
    """Fraction of the 150 pairs whose conncomp claims the monument pixel."""
    cache = CMP / f"gps_compare/{engine}_claim_fraction.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0).iloc[:, 0]
    ccs = sorted((CMP / engine / "unwrapped").glob("2*.unw.conncomp.tif"))
    counts = np.zeros(len(px))
    for f in ccs:
        with rasterio.open(f) as src:
            for i, (_, s) in enumerate(px.iterrows()):
                win = src.read(1, window=Window(int(s.col) - 1, int(s.row) - 1, 3, 3))
                counts[i] += (win > 0).mean() >= 0.5
    frac = pd.Series(counts / len(ccs), index=px.index, name="claim_frac")
    frac.to_csv(cache)
    return frac


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--engines", nargs="+", default=["whirlwind", "snaphu", "phass", "icu", "spurt"]
    )
    p.add_argument(
        "--stable-box",
        nargs=4,
        type=int,
        default=DEFAULT_STABLE_BOX,
        metavar=("R0", "R1", "C0", "C1"),
        help="looked-coords box assumed stable; all engines re-referenced to it",
    )
    p.add_argument(
        "--out-suffix",
        default="",
        help="appended to figure/scoreboard filenames (e.g. _nospurt)",
    )
    p.add_argument(
        "--no-suptitle",
        action="store_true",
        help="omit figure suptitles (for paper figures with LaTeX captions)",
    )
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = CMP / "gps_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    px = station_pixels()
    gps_v = gps_los_velocity(px)
    print(f"{len(gps_v)} monuments with usable GPS in window; range "
          f"[{gps_v.min():.0f}, {gps_v.max():.0f}] mm/yr")

    rows = []
    results = {}
    for eng in args.engines:
        # "<engine>_core" = series restricted to the tc>0.7 solved core
        # (7x7 nanmean), separating solver quality from interpolated fill
        is_core = eng.endswith("_core")
        src_eng = eng[: -len("_core")] if is_core else eng
        if is_core:
            series = sample_series_core(src_eng, eng, px)
        else:
            series = sample_series(src_eng, px)
        v = fit_velocity(series, *WINDOW)
        shift = 0.0
        if args.stable_box:
            shift = stable_box_velocity(src_eng, tuple(args.stable_box))
            v = v - shift
        claims = claim_fraction(src_eng, px)
        if not is_core and not src_eng.startswith("spurt"):
            # censored-LSQ rank gate: too few claimed pairs -> honest abstention
            # (spurt's conncomps mark only its solved core; its nan nodata
            # already encodes abstention in the sampled series)
            v[claims < MIN_CLAIM_FRAC] = np.nan
        df = pd.concat([gps_v, v.rename("insar_v"), claims], axis=1).dropna(
            subset=["gps_v"]
        )
        ok = df.dropna(subset=["insar_v"])
        gave_up = df[df.insar_v.isna()]
        r = ok.gps_v.corr(ok.insar_v)
        rmse = float(np.sqrt(((ok.insar_v - ok.gps_v) ** 2).mean()))
        slope = (
            float(np.polyfit(ok.gps_v, ok.insar_v, 1)[0]) if len(ok) > 2 else np.nan
        )
        results[eng] = df
        rows.append(
            {
                "engine": eng,
                "n": len(ok),
                "n_gave_up": len(gave_up),
                "r": round(r, 3),
                "rmse_mm_yr": round(rmse, 1),
                "slope": round(slope, 3),
                "ref_shift_mm_yr": round(shift, 1),
            }
        )
        df.to_csv(out_dir / f"{eng}_station_velocities.csv")

    board = pd.DataFrame(rows).set_index("engine")
    board.to_csv(out_dir / f"scoreboard{args.out_suffix}.csv")
    print(board.to_string())

    # --- scatter grid -------------------------------------------------------
    n = len(args.engines)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2), sharex=True, sharey=True)
    lims = (
        min(gps_v.min(), -50) * 1.1,
        max(gps_v.max(), 50) * 1.1,
    )
    for ax, eng in zip(np.atleast_1d(axes), args.engines):
        df = results[eng]
        ok = df.dropna(subset=["insar_v"])
        gu = df[df.insar_v.isna()]
        ax.plot(lims, lims, "k-", lw=0.5, alpha=0.5)
        ax.scatter(ok.gps_v, ok.insar_v, s=18, c="C0", alpha=0.8, zorder=3)
        ax.scatter(
            gu.gps_v,
            np.full(len(gu), lims[0] * 0.97),
            s=40,
            facecolors="none",
            edgecolors="C3",
            linestyle="--",
            zorder=3,
            label=f"gave up (n={len(gu)})",
        )
        b = board.loc[eng]
        ax.set_title(
            f"{display_name(eng)}\nr={b.r:.2f} rmse={b.rmse_mm_yr:.0f} mm/yr"
            f" slope={b.slope:.2f} n={b.n:.0f}"
        )
        ax.set_xlabel("GPS LOS velocity (mm/yr)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("InSAR LOS velocity (mm/yr)")
    if not args.no_suptitle:
        fig.suptitle(
            "Palos Verdes (Capella spotlight, 52 dates): unwrap engines vs survey"
            f" monuments, window {WINDOW[0]}..{WINDOW[1]}",
            y=1.02,
        )
    fig.tight_layout()
    f1 = out_dir / f"scatter_engines_vs_gps{args.out_suffix}.png"
    fig.savefig(f1, dpi=130, bbox_inches="tight")
    fig.savefig(f1.with_suffix(".pdf"), bbox_inches="tight")

    # --- velocity maps with monuments --------------------------------------
    land = water_mask_looked()
    fig2, axes2 = plt.subplots(1, n, figsize=(4 * n, 5))
    for ax, eng in zip(np.atleast_1d(axes2), args.engines):
        src_eng = eng[: -len("_core")] if eng.endswith("_core") else eng
        vel_file = CMP / src_eng / "timeseries/velocity.tif"
        with rasterio.open(vel_file) as src:
            v = src.read(1, out_shape=(src.height // 4, src.width // 4)).astype(float)
        v[v == 0.0] = np.nan
        v *= 1000.0
        v -= board.loc[eng].ref_shift_mm_yr  # common stable-box reference
        land4 = land[::4, ::4][: v.shape[0], : v.shape[1]]
        v[~land4] = np.nan
        im = ax.imshow(v, cmap="RdBu_r", vmin=-100, vmax=100, interpolation="nearest")
        df = results[eng]
        ok = df.dropna(subset=["insar_v"])
        gu = df[df.insar_v.isna()]
        ax.scatter(px.loc[ok.index].col / 4, px.loc[ok.index].row / 4, s=12,
                   c=ok.gps_v, cmap="RdBu_r", vmin=-100, vmax=100,
                   edgecolors="k", linewidths=0.4)
        ax.scatter(px.loc[gu.index].col / 4, px.loc[gu.index].row / 4, s=25,
                   facecolors="none", edgecolors="lime", linestyle="--", linewidths=1.0)
        ax.set_title(f"{display_name(eng)} velocity (mm/yr)")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.6)
    if not args.no_suptitle:
        fig2.suptitle(
            "Velocity maps (full window); dots = monuments (GPS color),"
            " dashed = engine gave up"
        )
    fig2.tight_layout()
    f2 = out_dir / f"velocity_maps_engines{args.out_suffix}.png"
    fig2.savefig(f2, dpi=130, bbox_inches="tight")
    fig2.savefig(f2.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    print(f1)
    print(f2)


if __name__ == "__main__":
    main()
