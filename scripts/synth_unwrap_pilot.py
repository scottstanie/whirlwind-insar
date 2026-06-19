"""Pilot for the synthetic-truth unwrapping benchmark (metrics_plan.md leg 3).

Reads a `synth-run` output directory, forms multilooked wrapped interferograms
(single-reference pairs for direct truth comparison + adjacent-date pairs to
build closure triplets), unwraps each with whirlwind, and reports:

  * per-pair integer-cycle accuracy vs truth (% pixels with K error == 0,
    after removing the single unobservable 2pi offset) and RMSE,
  * per-triplet integer closure-error rate (the truth-free metric of
    metrics_plan.md leg 2), alongside the wrapped-closure floor from
    multilooking,
  * the calibration scatter: closure-K rate vs true error rate per triplet.

Usage:
    python scripts/synth_unwrap_pilot.py <simdir> <outdir> [looks]

<simdir> is the synth output dir (contains slcs/ and
input_layers/truth_unwrapped_diffs/). Default looks = 5 (5x5 box).
"""

import sys
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio

import whirlwind as ww

TAU = 2 * np.pi


def read_band(path):
    with rio.open(path) as src:
        return src.read(1)


def multilook(arr, looks):
    m, n = arr.shape
    m2, n2 = m // looks * looks, n // looks * looks
    return (
        arr[:m2, :n2].reshape(m2 // looks, looks, n2 // looks, looks).mean(axis=(1, 3))
    )


def form_igram(slc_a, slc_b, looks):
    """Multilooked interferogram + coherence for the pair (a, b).

    Convention: phase(ig) ~ phi_b - phi_a (matches the truth diff files
    truth(d0_dN) = phase accumulated from d0 to dN).
    """
    cross = multilook(slc_b * slc_a.conj(), looks)
    pow_a = multilook((slc_a * slc_a.conj()).real, looks)
    pow_b = multilook((slc_b * slc_b.conj()).real, looks)
    coh = np.abs(cross) / np.sqrt(pow_a * pow_b)
    return cross.astype(np.complex64), np.clip(coh, 0, 1).astype(np.float32)


def k_error(unw, truth, valid):
    """Integer-cycle error map after removing the single 2pi gauge offset."""
    d = unw - truth
    off = TAU * np.round(np.median(d[valid]) / TAU)
    return np.round((d - off) / TAU).astype(int)


def main():
    simdir = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    looks = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    outdir.mkdir(parents=True, exist_ok=True)

    slc_files = sorted((simdir / "slcs").glob("2*.slc.tif"))
    dates = [f.name.split(".")[0] for f in slc_files]
    slcs = {d: read_band(f) for d, f in zip(dates, slc_files)}
    truth_dir = simdir / "input_layers" / "truth_unwrapped_diffs"
    # truth(d0_dN), multilooked with the same box as the igrams.
    truth = {
        d: multilook(read_band(truth_dir / f"{dates[0]}_{d}.int.tif"), looks)
        for d in dates[1:]
    }
    truth[dates[0]] = np.zeros_like(truth[dates[1]])

    nlooks = float(looks * looks)
    # Single-reference pairs (truth-difficulty ladder), adjacent pairs, and
    # 2-skip pairs (so we can form BOTH triplet families: the long-pair
    # triplets (0,i)(i,i+1)(0,i+1) and the production-like short-pair
    # triplets (i,i+1)(i+1,i+2)(i,i+2)).
    pairs = (
        [(dates[0], d) for d in dates[1:]]
        + list(pairwise(dates))
        + [(dates[i], dates[i + 2]) for i in range(len(dates) - 2)]
    )
    pairs = sorted(set(pairs))

    results = {}
    print(f"{'pair':23s} {'coh':>5s} {'%K=0':>7s} {'RMSE(rad)':>10s}")
    for a, b in pairs:
        ig, coh = form_igram(slcs[a], slcs[b], looks)
        unw, _cc = ww.unwrap(ig, coh, nlooks)
        unw = np.asarray(unw, np.float32)
        t = truth[b] - truth[a]
        valid = np.isfinite(unw)
        k = k_error(unw, t, valid)
        pct_ok = 100.0 * np.mean(k[valid] == 0)
        resid = (unw - t)[valid]
        rmse = float(np.sqrt(np.mean((resid - resid.mean()) ** 2)))
        results[(a, b)] = dict(
            unw=unw, coh=coh, ig=ig, truth=t, k=k, pct_ok=pct_ok, rmse=rmse
        )
        print(f"{a}_{b}  {coh.mean():5.2f} {pct_ok:7.2f} {rmse:10.3f}")

    # --- closure triplets: long-pair family + production-like short family --
    def closure_row(ta, tb, tc):
        """Triplet (ta,tb), (tb,tc), (ta,tc): closure rate vs true error."""
        r1, r2, r3 = results[(ta, tb)], results[(tb, tc)], results[(ta, tc)]
        mis = r1["unw"] + r2["unw"] - r3["unw"]
        # Wrapped-closure floor: multilooking breaks exact closure; the
        # integer rounding below is only trustworthy where this is << pi.
        wrapped_mis = np.angle(
            np.exp(1j * (np.angle(r1["ig"]) + np.angle(r2["ig"]) - np.angle(r3["ig"])))
        )
        k_clo = np.round((mis - wrapped_mis) / TAU).astype(int)
        clo_rate = 100.0 * np.mean(k_clo != 0)
        # Truth: a (pixel, triplet) is truly wrong if ANY of its 3 IGs is wrong.
        true_any = 100.0 * np.mean((r1["k"] != 0) | (r2["k"] != 0) | (r3["k"] != 0))
        p95 = float(np.percentile(np.abs(wrapped_mis), 95))
        print(f"{ta}_{tb}_{tc}    {clo_rate:14.2f} {true_any:11.2f} {p95:26.2f}")
        return clo_rate, true_any

    print(
        f"\nLONG-pair triplets (d0,di)(di,di+1)(d0,di+1) - systematic errors "
        f"shared by the two long pairs cancel:"
    )
    print(
        f"{'triplet':32s} {'%K_closure!=0':>14s} {'true-err %':>11s} "
        f"{'wrapped-closure rad (p95)':>26s}"
    )
    cal = []
    for i in range(1, len(dates) - 1):
        cal.append(closure_row(dates[0], dates[i], dates[i + 1]))

    print(
        f"\nSHORT-pair triplets (di,di+1)(di+1,di+2)(di,di+2) - the "
        f"production (nearest-k) graph:"
    )
    print(
        f"{'triplet':32s} {'%K_closure!=0':>14s} {'true-err %':>11s} "
        f"{'wrapped-closure rad (p95)':>26s}"
    )
    cal_short = []
    for i in range(len(dates) - 2):
        cal_short.append(closure_row(dates[i], dates[i + 1], dates[i + 2]))

    # --- figure: hardest single-reference pair + calibration scatter --------
    hard = min(
        ((a, b) for (a, b) in results if a == dates[0]),
        key=lambda p: results[p]["coh"].mean(),
    )
    r = results[hard]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (arr, title, kw) in zip(
        axes,
        [
            (
                np.angle(r["ig"]),
                f"wrapped {hard[0]}_{hard[1]}",
                dict(cmap="twilight", vmin=-np.pi, vmax=np.pi),
            ),
            (r["truth"], "truth", dict(cmap="viridis")),
            (r["unw"], f"whirlwind ({r['pct_ok']:.1f}% K=0)", dict(cmap="viridis")),
            (
                r["k"].astype(float),
                "integer-cycle error",
                dict(cmap="coolwarm", vmin=-2, vmax=2),
            ),
        ],
    ):
        im = ax.imshow(arr, **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    f1 = outdir / "pilot_hardest_pair.png"
    fig.savefig(f1, dpi=120)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    clo, tru = zip(*cal)
    clo_s, tru_s = zip(*cal_short)
    ax.scatter(tru, clo, label="long-pair triplets")
    ax.scatter(tru_s, clo_s, marker="x", label="short-pair (production) triplets")
    lim = max(max(clo), max(tru), max(clo_s), max(tru_s), 1) * 1.1
    ax.plot([0, lim], [0, lim], "k:", lw=1, label="closure = true")
    ax.set_xlabel("true error rate, any of 3 IGs (%)")
    ax.set_ylabel("integer closure-error rate (%)")
    ax.set_title("Calibration: truth-free closure vs true error")
    ax.legend()
    fig.tight_layout()
    f2 = outdir / "pilot_closure_calibration.png"
    fig.savefig(f2, dpi=120)
    print(f"\nfigures:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
