"""Report for synth_unwrap_bench.py runs: coherence-masked metrics + plots.

For every pair and engine, recomputes the integer-cycle error map and scores
it three ways: over all pixels, over coherent pixels (gamma > threshold),
and as the blob-filtered "region error" fraction. Writes:

  * summary.csv - per (engine, pair): pct_k0_all, pct_k0_coh, pct_region_err
  * pair_<a>_<b>.png - per-pair panel: wrapped / coherence / truth + one
    K-error map per engine (the failure-inspection plots)
  * summary.png - per-engine curves across pairs, all-pixel vs coh-masked

Usage:
    python scripts/synth_bench_report.py <simdir> <benchdir> \
        [--looks 3] [--coh-threshold 0.4] [--engines ...]
"""

import argparse
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
    cross = multilook(slc_b * slc_a.conj(), looks)
    pow_a = multilook((slc_a * slc_a.conj()).real, looks)
    pow_b = multilook((slc_b * slc_b.conj()).real, looks)
    coh = np.abs(cross) / np.sqrt(pow_a * pow_b)
    return cross.astype(np.complex64), np.clip(coh, 0, 1).astype(np.float32)


def k_error(unw, truth, wrapped, valid):
    """Integer AMBIGUITY error and truth ambiguity field.

    Returns (k_err, k_true): k_err = K_est - K_true, both relative to the
    wrapped input. Noise-immune: a congruent engine (output == input mod 2pi)
    is not penalized for pixels whose phase NOISE exceeds pi, and a smoothing
    engine (e.g. ICU's filtered output) is quantized back onto the same
    integer lattice instead of being rewarded for suppressing noise. Matches
    the NISAR bench's percomp ambiguity convention. k_true supports the
    hit/false-alarm split: a do-nothing engine scores 0% hits where
    k_true != 0, regardless of how flat the truth is."""
    off = TAU * np.round(np.median((unw - truth)[valid]) / TAU)
    k_est = np.round((unw - wrapped) / TAU)
    k_true = np.round((truth + off - wrapped) / TAU).astype(int)
    return (k_est - k_true).astype(int), k_true


def main():
    p = argparse.ArgumentParser()
    p.add_argument("simdir", type=Path)
    p.add_argument("benchdir", type=Path)
    p.add_argument("--looks", type=int, default=3)
    p.add_argument("--coh-threshold", type=float, default=0.4)
    p.add_argument(
        "--engines", nargs="+", default=["whirlwind", "snaphu", "phass", "icu"]
    )
    args = p.parse_args()

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

    slcs = {d: read_band(f) for d, f in zip(dates, slc_files)}
    truth0 = {
        d: multilook(read_band(truth_dir / f"{dates[0]}_{d}.int.tif"), args.looks)
        for d in dates[1:]
    }
    truth0[dates[0]] = np.zeros_like(truth0[dates[1]])

    import whirlwind  # label_components for the blob filter

    sum_csv = args.benchdir / "summary.csv"
    rows = []
    for a, b in pairs:
        ig, coh = form_igram(slcs[a], slcs[b], args.looks)
        truth = truth0[b] - truth0[a]
        coh_ok = coh > args.coh_threshold
        # Recoverable set (engine-independent): synth's wrapped-minus-truth is
        # pure noise mod 2pi, so pixels with |noise| >= pi/2 have no
        # well-defined correct integer - exclude them for every engine.
        noise = np.angle(np.exp(1j * (np.angle(ig) - truth)))
        recoverable = np.abs(noise) < (np.pi / 2)

        engines_done = [
            e for e in args.engines if (args.benchdir / f"{e}_{a}_{b}.npz").exists()
        ]
        # 'zero' = the do-nothing baseline (output == wrapped input, K_est = 0
        # everywhere). Any engine scoring near it on a metric did NOT earn its
        # number - guards against trivial/degenerate winners.
        ks = {}
        for eng in engines_done + ["zero"]:
            if eng == "zero":
                unw = np.angle(ig).astype(np.float32)
                cc = np.zeros(unw.shape, np.int32)  # claims nothing
            else:
                npz_data = np.load(args.benchdir / f"{eng}_{a}_{b}.npz")
                unw = npz_data["unw"]
                cc = npz_data["cc"] if "cc" in npz_data.files else None
            valid = np.isfinite(unw)
            k, k_true = k_error(unw, truth, np.angle(ig), valid)
            ks[eng] = (k, valid)
            pct_all = 100.0 * np.mean(k[valid] == 0)
            sel = valid & coh_ok
            pct_coh = 100.0 * np.mean(k[sel] == 0) if sel.any() else np.nan
            sel_r = valid & recoverable
            pct_rec = 100.0 * np.mean(k[sel_r] == 0) if sel_r.any() else np.nan
            # Hit / false-alarm split over the recoverable set: hits = correct
            # where the truth actually has cycles to recover; false alarms =
            # invented cycles where the truth has none. The zero baseline is
            # 0% hits / 0% FA by construction.
            pos = sel_r & (k_true != 0)
            neg = sel_r & (k_true == 0)
            pct_hit = 100.0 * np.mean(k[pos] == 0) if pos.any() else np.nan
            pct_fa = 100.0 * np.mean(k[neg] != 0) if neg.any() else np.nan
            # PRECISION / COVERAGE from the engine's OWN conncomp claims:
            # coverage = fraction of the grid the engine claims (cc > 0);
            # precision = K-correct fraction over claimed pixels, with the
            # 2pi offset aligned PER COMPONENT (each component's offset is
            # unobservable - gauge-free, mirroring the NISAR percomp metric).
            # The zero baseline claims nothing: coverage 0, precision nan.
            if cc is None:
                pct_cov = pct_prec = np.nan
            else:
                claimed = valid & (cc > 0)
                pct_cov = 100.0 * claimed.mean()
                n_ok, n_tot = 0, 0
                for lbl in np.unique(cc[claimed]) if claimed.any() else []:
                    sel_c = claimed & (cc == lbl)
                    kk = k[sel_c] - int(np.round(np.median(k[sel_c])))
                    n_ok += int((kk == 0).sum())
                    n_tot += int(sel_c.sum())
                pct_prec = 100.0 * n_ok / n_tot if n_tot else np.nan
            # Region errors: fraction of valid px inside K!=0 blobs >= 100 px
            # (whole-region 2pi offsets - the failures that matter downstream).
            bad = (k != 0) & valid
            labels, _n = whirlwind.label_components(bad)
            labels = np.asarray(labels)
            sizes = np.bincount(labels.ravel())
            big = np.isin(labels, np.nonzero(sizes >= 100)[0][1:])
            pct_region = 100.0 * big[valid].mean()
            rows.append(
                (
                    eng,
                    f"{a}_{b}",
                    float(coh.mean()),
                    pct_all,
                    pct_coh,
                    pct_rec,
                    pct_region,
                    pct_hit,
                    pct_fa,
                    pct_prec,
                    pct_cov,
                )
            )

        # ---- per-pair failure-inspection panel -----------------------------
        n_eng = len(engines_done)
        fig, axes = plt.subplots(1, n_eng + 3, figsize=(3.4 * (n_eng + 3), 3.7))
        panels = [
            (
                np.angle(ig),
                dict(cmap="twilight", vmin=-np.pi, vmax=np.pi),
                f"wrapped {a}_{b}",
            ),
            (
                coh,
                dict(cmap="gray", vmin=0, vmax=1),
                f"coherence (mean {coh.mean():.2f})",
            ),
            (truth, dict(cmap="viridis"), "truth"),
        ]
        for ax, (arr, kw, title) in zip(axes, panels):
            im = ax.imshow(arr, **kw)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.7)
        for ax, eng in zip(axes[3:], engines_done):
            k, valid = ks[eng]
            pct_all = 100.0 * np.mean(k[valid] == 0)
            sel = valid & coh_ok
            pct_coh = 100.0 * np.mean(k[sel] == 0) if sel.any() else np.nan
            im = ax.imshow(
                np.where(valid, k, np.nan).astype(float),
                cmap="coolwarm",
                vmin=-2,
                vmax=2,
            )
            ax.set_title(
                f"{eng}\nK=0: {pct_all:.0f}% all | {pct_coh:.0f}% coh", fontsize=9
            )
            fig.colorbar(im, ax=ax, shrink=0.7)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(args.benchdir / f"pair_{a}_{b}.png", dpi=100)
        plt.close(fig)

    with sum_csv.open("w") as f:
        f.write(
            "engine,pair,mean_coh,pct_k0_all,pct_k0_coh,pct_k0_recoverable,"
            "pct_region_err,pct_hit,pct_false_alarm,pct_precision,"
            "pct_coverage\n"
        )
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # ---- summary figure: per-engine curves across pairs ---------------------
    engs = sorted(
        {r[0] for r in rows}, key=lambda e: (args.engines + ["zero"]).index(e)
    )
    pair_names = sorted({r[1] for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for metric_i, (col, label) in enumerate(
        [(3, "% K=0 (all px)"), (5, "% K=0 (recoverable px)")]
    ):
        ax = axes[metric_i]
        for eng in engs:
            vals = {r[1]: r[col] for r in rows if r[0] == eng}
            ax.plot(
                pair_names,
                [vals.get(p, np.nan) for p in pair_names],
                marker="o",
                ms=4,
                label=eng,
            )
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1].set_xticklabels(pair_names, rotation=60, ha="right", fontsize=7)
    fig.suptitle("Synthetic-truth bench: integer-cycle accuracy per pair")
    fig.tight_layout()
    fig.savefig(args.benchdir / "summary.png", dpi=120)

    # ---- console table -------------------------------------------------------
    print(
        f"{'engine':10s} {'%K0 all':>8s} {'%K0 rec':>8s} "
        f"{'%region-err':>12s} {'%HIT':>7s} {'%FA':>7s} "
        f"{'%PRECISION':>11s} {'%COVERAGE':>10s}"
    )
    for eng in engs:
        sel = [r for r in rows if r[0] == eng]
        print(
            f"{eng:10s} {np.mean([r[3] for r in sel]):8.1f} "
            f"{np.nanmean([r[5] for r in sel]):8.1f} "
            f"{np.mean([r[6] for r in sel]):12.1f} "
            f"{np.nanmean([r[7] for r in sel]):7.1f} "
            f"{np.nanmean([r[8] for r in sel]):7.1f} "
            f"{np.nanmean([r[9] for r in sel]):11.1f} "
            f"{np.nanmean([r[10] for r in sel]):10.1f}"
        )
    print(
        f"\nwrote {sum_csv}\nfigures: {args.benchdir}/pair_*.png, "
        f"{args.benchdir}/summary.png"
    )


if __name__ == "__main__":
    main()
