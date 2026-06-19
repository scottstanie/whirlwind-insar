"""Multi-engine synthetic-truth unwrapping benchmark (metrics_plan.md leg 3).

Generalizes synth_unwrap_pilot.py to several engines. Reads a `synth-run`
output dir, forms multilooked wrapped interferograms (single-reference
difficulty ladder + adjacent + 2-skip pairs for both closure-triplet
families), unwraps every pair with every requested engine, and writes:

  * results.csv - per (engine, pair): mean coherence, %K=0 vs truth, RMSE,
    runtime seconds,
  * closure.csv - per (engine, triplet family, triplet): integer
    closure-error rate vs the true any-of-3 error rate,
  * per-pair npz files with the unwrapped arrays (reused on re-runs:
    the bench is resume-friendly per engine+pair),
  * a per-engine figure for the hardest single-reference pair.

Run in an env with tophu + isce3 + whirlwind (mapping-312):
    ~/miniforge3/envs/mapping-312/bin/python scripts/synth_unwrap_bench.py \
        <simdir> <outdir> --looks 3 --engines whirlwind snaphu phass icu

SNAPHU runs via tophu's SnaphuUnwrap(cost='smooth', init_method='mcf') -
the NISAR GUNW production cost/init, single tile. ONE heavy unwrap at a
time (laptop concurrency limit).
"""

import argparse
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio

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
    """Multilooked interferogram + coherence; phase(ig) ~ phi_b - phi_a."""
    cross = multilook(slc_b * slc_a.conj(), looks)
    pow_a = multilook((slc_a * slc_a.conj()).real, looks)
    pow_b = multilook((slc_b * slc_b.conj()).real, looks)
    coh = np.abs(cross) / np.sqrt(pow_a * pow_b)
    return cross.astype(np.complex64), np.clip(coh, 0, 1).astype(np.float32)


def k_error(unw, truth, valid):
    d = unw - truth
    off = TAU * np.round(np.median(d[valid]) / TAU)
    return np.round((d - off) / TAU).astype(int)


def make_engine(name):
    """Return fn(ig, coh, nlooks) -> (unw, runtime_s)."""
    if name == "whirlwind":
        import whirlwind as ww

        def run(ig, coh, nlooks):
            t0 = time.perf_counter()
            unw, cc = ww.unwrap(ig, coh, nlooks)
            return (
                np.asarray(unw, np.float32),
                np.asarray(cc, np.int32),
                time.perf_counter() - t0,
            )

        return run

    if name == "snaphu":
        # snaphu-py direct (the GUNW production settings: smooth cost, MCF
        # init, single tile + reoptimize) - this isce3 build dropped the
        # tophu SnaphuUnwrap backend.
        import snaphu

        def run(ig, coh, nlooks):
            t0 = time.perf_counter()
            unw, cc = snaphu.unwrap(ig, coh, nlooks=nlooks, cost="smooth", init="mcf")
            return (
                np.asarray(unw, np.float32),
                np.asarray(cc, np.int32),
                time.perf_counter() - t0,
            )

        return run

    import tophu

    if name == "phass":
        cb = tophu.PhassUnwrap(good_coherence=0.7, min_region_size=200)
    elif name == "icu":
        cb = tophu.ICUUnwrap()
    else:
        raise ValueError(name)

    def run(ig, coh, nlooks):
        with tempfile.TemporaryDirectory() as sd:
            t0 = time.perf_counter()
            unw, cc = cb(ig, coh, nlooks, Path(sd))
            return (
                np.asarray(unw, np.float32),
                np.asarray(cc, np.int32),
                time.perf_counter() - t0,
            )

    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("simdir", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("--looks", type=int, default=3)
    p.add_argument(
        "--engines", nargs="+", default=["whirlwind", "snaphu", "phass", "icu"]
    )
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    slc_files = sorted((args.simdir / "slcs").glob("2*.slc.tif"))
    dates = [f.name.split(".")[0] for f in slc_files]
    truth_dir = args.simdir / "input_layers" / "truth_unwrapped_diffs"

    pairs = sorted(
        set(
            [(dates[0], d) for d in dates[1:]]
            + list(pairwise(dates))
            + [(dates[i], dates[i + 2]) for i in range(len(dates) - 2)]
        )
    )

    # Pre-form igrams once (shared across engines); truth multilooked to match.
    print(
        f"forming {len(pairs)} igrams at {args.looks}x{args.looks} looks ...",
        flush=True,
    )
    slcs = {d: read_band(f) for d, f in zip(dates, slc_files)}
    truth0 = {
        d: multilook(read_band(truth_dir / f"{dates[0]}_{d}.int.tif"), args.looks)
        for d in dates[1:]
    }
    truth0[dates[0]] = np.zeros_like(truth0[dates[1]])
    igs, cohs, truths = {}, {}, {}
    for a, b in pairs:
        igs[(a, b)], cohs[(a, b)] = form_igram(slcs[a], slcs[b], args.looks)
        truths[(a, b)] = truth0[b] - truth0[a]
    del slcs  # free the SLC stack before the solves
    nlooks = float(args.looks**2)

    res_csv = args.outdir / "results.csv"
    if not res_csv.exists():
        res_csv.write_text("engine,pair,mean_coh,pct_k0,rmse_rad,runtime_s\n")

    unws = {}  # (engine, pair) -> unw, for the closure pass
    for eng in args.engines:
        run = make_engine(eng)
        for a, b in pairs:
            npz = args.outdir / f"{eng}_{a}_{b}.npz"
            if npz.exists():
                unws[(eng, (a, b))] = np.load(npz)["unw"]
                continue
            unw, cc, dt = run(igs[(a, b)], cohs[(a, b)], nlooks)
            unws[(eng, (a, b))] = unw
            valid = np.isfinite(unw)
            k = k_error(unw, truths[(a, b)], valid)
            pct_ok = 100.0 * np.mean(k[valid] == 0)
            resid = (unw - truths[(a, b)])[valid]
            rmse = float(np.sqrt(np.mean((resid - resid.mean()) ** 2)))
            np.savez_compressed(npz, unw=unw, cc=cc)
            with res_csv.open("a") as f:
                f.write(
                    f"{eng},{a}_{b},{cohs[(a, b)].mean():.3f},"
                    f"{pct_ok:.2f},{rmse:.3f},{dt:.1f}\n"
                )
            print(
                f"{eng:10s} {a}_{b}  coh={cohs[(a, b)].mean():.2f} "
                f"%K0={pct_ok:6.2f}  rmse={rmse:5.2f}  {dt:6.1f}s",
                flush=True,
            )

    # --- closure, both triplet families, per engine -------------------------
    clo_csv = args.outdir / "closure.csv"
    with clo_csv.open("w") as f:
        f.write(
            "engine,family,triplet,pct_k_closure,pct_true_any," "wrapped_closure_p95\n"
        )
        for eng in args.engines:
            fams = {
                "long": [
                    (dates[0], dates[i], dates[i + 1]) for i in range(1, len(dates) - 1)
                ],
                "short": [
                    (dates[i], dates[i + 1], dates[i + 2])
                    for i in range(len(dates) - 2)
                ],
            }
            for fam, triplets in fams.items():
                for ta, tb, tc in triplets:
                    u1 = unws[(eng, (ta, tb))]
                    u2 = unws[(eng, (tb, tc))]
                    u3 = unws[(eng, (ta, tc))]
                    valid = np.isfinite(u1) & np.isfinite(u2) & np.isfinite(u3)
                    wrapped_mis = np.angle(
                        np.exp(
                            1j
                            * (
                                np.angle(igs[(ta, tb)])
                                + np.angle(igs[(tb, tc)])
                                - np.angle(igs[(ta, tc)])
                            )
                        )
                    )
                    mis = u1 + u2 - u3
                    k_clo = np.round((mis - wrapped_mis) / TAU)
                    clo = 100.0 * np.mean(k_clo[valid] != 0)
                    k1 = k_error(u1, truths[(ta, tb)], valid)
                    k2 = k_error(u2, truths[(tb, tc)], valid)
                    k3 = k_error(u3, truths[(ta, tc)], valid)
                    true_any = 100.0 * np.mean(
                        ((k1 != 0) | (k2 != 0) | (k3 != 0))[valid]
                    )
                    p95 = float(np.percentile(np.abs(wrapped_mis[valid]), 95))
                    f.write(
                        f"{eng},{fam},{ta}_{tb}_{tc},{clo:.2f},"
                        f"{true_any:.2f},{p95:.2f}\n"
                    )
    print(f"\nwrote {res_csv}\nwrote {clo_csv}")

    # --- per-engine figure on the hardest single-reference pair -------------
    hard = min(
        ((a, b) for (a, b) in pairs if a == dates[0]), key=lambda pr: cohs[pr].mean()
    )
    n_eng = len(args.engines)
    fig, axes = plt.subplots(1, n_eng + 2, figsize=(4 * (n_eng + 2), 4.2))
    im = axes[0].imshow(np.angle(igs[hard]), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title(
        f"wrapped {hard[0]}_{hard[1]}\n" f"(coh {cohs[hard].mean():.2f})", fontsize=9
    )
    fig.colorbar(im, ax=axes[0], shrink=0.75)
    t = truths[hard]
    im = axes[1].imshow(t, cmap="viridis")
    axes[1].set_title("truth", fontsize=9)
    fig.colorbar(im, ax=axes[1], shrink=0.75)
    for ax, eng in zip(axes[2:], args.engines):
        unw = unws[(eng, hard)]
        valid = np.isfinite(unw)
        k = k_error(unw, t, valid)
        pct = 100.0 * np.mean(k[valid] == 0)
        im = ax.imshow(
            np.where(valid, k, np.nan).astype(float), cmap="coolwarm", vmin=-2, vmax=2
        )
        ax.set_title(f"{eng} K-error ({pct:.1f}% K=0)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.75)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fpath = args.outdir / "bench_hardest_pair.png"
    fig.savefig(fpath, dpi=110)
    print(f"figure: {fpath}")


if __name__ == "__main__":
    main()
