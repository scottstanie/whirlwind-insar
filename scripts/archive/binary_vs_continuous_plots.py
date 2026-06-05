"""Plot the binary-vs-continuous Palos Verdes comparison.

Reads the per-variant outputs written by ``binary_vs_continuous_pv.py`` and
produces three artifacts in the same output directory:

1. ``coverage.png``    — temp_coh map + per-variant "where did we get a
                          finite unwrapped value" coverage masks. Frames
                          the data-loss story.
2. ``per_ig_triptych.png`` — a representative IG: continuous | binary
                              variant unwrapped phases side by side, with
                              the temp_coh map and chosen IG metadata.
3. ``timeseries.png``  — hand-picked pixels (high-coh, low-coh, near-
                          threshold-boundary, near-reference): per-variant
                          date-phase time series.
4. ``aggregate.json``  — % coverage, RMS cycle differences between
                          variants over the common-finite mask, runtimes
                          (where available).

How to rerun
------------
::

    uv run python scripts/binary_vs_continuous_plots.py \\
        --pv-out /tmp/binary-vs-continuous/pv

Optionally pass ``--spurt-out /path/to/spurt/output`` to overlay spurt
date-phase outputs at the same pixels (NaN-aware comparison).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_variants(
    pv_out: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    with (pv_out / "summary.json").open() as f:
        summary = json.load(f)
    variants: dict[str, dict[str, np.ndarray]] = {}
    for v in summary["variants"]:
        label = v["label"]
        vdir = pv_out / label
        variants[label] = {
            "date_phases": np.load(vdir / "date_phases.npy"),
            "unw_stack": np.load(vdir / "unw_stack.npy"),
            "meta": v,
        }
        mask_path = vdir / "variant_mask.npy"
        if mask_path.exists():
            variants[label]["variant_mask"] = np.load(mask_path)
    return summary, variants


# ---------------------------------------------------------------------------
# Pixel selection
# ---------------------------------------------------------------------------


def pick_test_pixels(
    temp_coh: np.ndarray,
    ref: tuple[int, int],
    rng_seed: int = 0,
    binary_finite: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Pick four illustrative pixels in the window.

    - ``high_coh``: γ near max, away from reference
    - ``low_coh``: γ near 0.3, mid-scene
    - ``near_threshold``: γ within ±0.05 of 0.7 (the "fragile" band)
    - ``near_reference``: a few pixels from the reference (sanity check)
    """
    rng = np.random.default_rng(rng_seed)
    m, n = temp_coh.shape
    ref_i, ref_j = ref
    picks = []

    # high-coh: 99th percentile, but at least 100 pixels away from ref
    flat = temp_coh.ravel()
    high_thresh = np.nanpercentile(flat[np.isfinite(flat)], 99)
    yy, xx = np.mgrid[0:m, 0:n]
    far = (yy - ref_i) ** 2 + (xx - ref_j) ** 2 > 100**2
    candidates = np.argwhere((temp_coh >= high_thresh) & far)
    if len(candidates):
        c = candidates[rng.integers(len(candidates))]
        picks.append(
            {
                "name": "high_coh",
                "i": int(c[0]),
                "j": int(c[1]),
                "temp_coh": float(temp_coh[c[0], c[1]]),
            }
        )

    # low-coh
    candidates = np.argwhere((temp_coh > 0.25) & (temp_coh < 0.35) & far)
    if len(candidates):
        c = candidates[rng.integers(len(candidates))]
        picks.append(
            {
                "name": "low_coh",
                "i": int(c[0]),
                "j": int(c[1]),
                "temp_coh": float(temp_coh[c[0], c[1]]),
            }
        )

    # near-threshold (~0.7)
    candidates = np.argwhere((temp_coh > 0.65) & (temp_coh < 0.75) & far)
    if len(candidates):
        c = candidates[rng.integers(len(candidates))]
        picks.append(
            {
                "name": "near_threshold",
                "i": int(c[0]),
                "j": int(c[1]),
                "temp_coh": float(temp_coh[c[0], c[1]]),
            }
        )

    # near-reference: 20 pixels in each direction
    pi, pj = ref_i + 20, ref_j + 20
    if 0 <= pi < m and 0 <= pj < n:
        picks.append(
            {
                "name": "near_reference",
                "i": pi,
                "j": pj,
                "temp_coh": float(temp_coh[pi, pj]),
            }
        )

    # If binary has any survivors, pick one inside that region so the
    # time-series panel can show a binary-vs-continuous comparison on
    # pixels where binary actually has data.
    if binary_finite is not None and binary_finite.any():
        candidates = np.argwhere(binary_finite)
        c = candidates[rng.integers(len(candidates))]
        picks.append(
            {
                "name": "binary_survives",
                "i": int(c[0]),
                "j": int(c[1]),
                "temp_coh": float(temp_coh[c[0], c[1]]),
            }
        )

    return picks


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_coverage(
    summary: dict[str, Any], variants: dict, temp_coh: np.ndarray, out: Path
) -> None:
    labels = list(variants.keys())
    ncols = 1 + len(labels)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.4))

    ax = axes[0]
    im = ax.imshow(temp_coh, cmap="cividis", vmin=0, vmax=1, interpolation="none")
    ax.set_title("temporal_coherence_average", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i, label in enumerate(labels, start=1):
        unw = variants[label]["unw_stack"]
        coverage = np.isfinite(unw).mean(axis=0)
        ax = axes[i]
        # Two-layer view: gray = "in variant_mask but not finite" (kept-by-
        # threshold but disconnected from seed); orange = finite-anywhere
        # (BFS-reached pixels). Continuous has no variant_mask so just
        # show coverage.
        vm = variants[label].get("variant_mask")
        finite_any = coverage > 0
        if vm is None:
            im = ax.imshow(coverage, cmap="magma", vmin=0, vmax=1, interpolation="none")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            disp = np.zeros(coverage.shape, dtype=np.float32)
            disp[vm] = 0.4  # gray: in mask
            disp[finite_any] = 1.0  # bright: finite anywhere
            ax.imshow(disp, cmap="magma", vmin=0, vmax=1, interpolation="none")
        kept_pct = 100 * float(vm.mean()) if vm is not None else 100.0
        finite_pct = 100 * float(finite_any.mean())
        ax.set_title(
            f"{label}\nin mask: {kept_pct:.1f}% • finite: {finite_pct:.2f}%", fontsize=9
        )
        ax.set_xticks([])
        ax.set_yticks([])

    ref_i, ref_j = summary["reference_pixel"]
    for ax in axes:
        ax.plot(ref_j, ref_i, "rx", markersize=10, markeredgewidth=2)

    fig.suptitle(
        "Coverage: temp_coh map + per-variant fraction of IGs with a finite unwrap value.\n"
        "Red x marks the reference pixel. Binary thresholding sparsifies the kept set; "
        "the 4-connected grid then disconnects most kept pixels from the seed.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_per_ig(variants: dict, ig_index: int, out: Path) -> None:
    labels = list(variants.keys())
    ncols = len(labels)
    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 6.5))

    # Row 0: raw unwrapped value.
    # Row 1: difference from continuous (the reference, NaN-safe).
    cont = variants["continuous"]["unw_stack"][ig_index]
    vmax_unw = float(np.nanpercentile(np.abs(cont), 99) or 1.0)

    for j, label in enumerate(labels):
        unw = variants[label]["unw_stack"][ig_index]
        ax = axes[0, j]
        im = ax.imshow(
            unw, cmap="RdBu_r", vmin=-vmax_unw, vmax=vmax_unw, interpolation="none"
        )
        finite_frac = float(np.isfinite(unw).mean())
        ax.set_title(f"{label}\nfinite: {100*finite_frac:.1f}%", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, j]
        diff = unw - cont
        if label == "continuous":
            ax.axis("off")
            ax.set_title("(reference)", fontsize=10)
            continue
        # Show diff modulo 2π to highlight cycle disagreements.
        # finite-finite only.
        both = np.isfinite(unw) & np.isfinite(cont)
        diff_plot = np.where(both, diff, np.nan)
        vmax = max(np.pi, float(np.nanpercentile(np.abs(diff_plot), 99) or np.pi))
        im = ax.imshow(
            diff_plot, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="none"
        )
        ax.set_title(
            f"{label} − continuous\n(NaN-safe; finite both: {100*float(both.mean()):.1f}%)",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"IG index {ig_index}: per-variant unwrap and difference from continuous",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_timeseries(
    summary: dict[str, Any],
    variants: dict,
    picks: list[dict[str, Any]],
    out: Path,
) -> None:
    dates = summary["dates"]
    t = np.arange(len(dates))  # x axis = date index (chronological)

    nplots = len(picks)
    fig, axes = plt.subplots(nplots, 1, figsize=(8, 2.6 * nplots), sharex=True)
    if nplots == 1:
        axes = [axes]

    colors = {"continuous": "black", "binary_T0.60": "C0", "binary_T0.90": "C3"}

    for ax, p in zip(axes, picks):
        i, j = p["i"], p["j"]
        title = f"{p['name']} @ ({i}, {j})  temp_coh={p['temp_coh']:.2f}"
        for label, payload in variants.items():
            phases = payload["date_phases"][:, i, j]
            color = colors.get(label, None)
            finite_n = int(np.isfinite(phases).sum())
            ax.plot(
                t,
                phases,
                "o-",
                color=color,
                label=f"{label} ({finite_n}/{len(dates)} finite)",
                alpha=0.85,
            )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("date phase [rad]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("date index")
    fig.suptitle("Time series comparison at picked pixels", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_aggregate(variants: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    labels = list(variants.keys())
    cont = variants["continuous"]["unw_stack"]
    out["per_variant"] = {}
    for label in labels:
        unw = variants[label]["unw_stack"]
        finite = np.isfinite(unw)
        out["per_variant"][label] = {
            "finite_fraction": float(finite.mean()),
            "finite_pixels_with_any_data": int((finite.any(axis=0)).sum()),
            "n_pixels": int(unw.shape[1] * unw.shape[2]),
            "n_igs": int(unw.shape[0]),
        }
        if label == "continuous":
            continue
        # Diff vs continuous on the intersection.
        both = finite & np.isfinite(cont)
        if not both.any():
            continue
        diff = unw[both] - cont[both]
        # Cycle-fold the diff so we measure cycle disagreement (∈ (-π, π]).
        diff_folded = ((diff + np.pi) % (2 * np.pi)) - np.pi
        out["per_variant"][label].update(
            {
                "rms_diff_vs_continuous_rad": float(np.sqrt(np.mean(diff**2))),
                "rms_cycle_diff_vs_continuous_rad": float(
                    np.sqrt(np.mean(diff_folded**2))
                ),
                "n_cycle_disagreements": int((np.abs(diff) > np.pi).sum()),
                "common_finite_pixels": int(both.sum()),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pv-out", type=Path, required=True)
    ap.add_argument(
        "--ig-index",
        type=int,
        default=10,
        help="which IG to use for the per-IG triptych",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for pixel selection")
    args = ap.parse_args()

    summary, variants = load_variants(args.pv_out)
    temp_coh = np.load(args.pv_out / "temp_coh.npy")

    print(f"[plot] coverage maps → {args.pv_out}/coverage.png")
    plot_coverage(summary, variants, temp_coh, args.pv_out / "coverage.png")

    print(
        f"[plot] per-IG triptych for IG {args.ig_index} → {args.pv_out}/per_ig_triptych.png"
    )
    plot_per_ig(variants, args.ig_index, args.pv_out / "per_ig_triptych.png")

    ref_list = summary["reference_pixel"]
    ref = (int(ref_list[0]), int(ref_list[1]))
    # Use the most-restrictive binary variant's finite-anywhere mask as the
    # "binary_survives" source so the picked pixel works in the strictest run.
    binary_labels = [lbl for lbl in variants if lbl != "continuous"]
    binary_finite: np.ndarray | None = None
    if binary_labels:
        last_binary = binary_labels[-1]
        unw_b = variants[last_binary]["unw_stack"]
        bf = np.asarray(np.isfinite(unw_b).any(axis=0))
        if not bf.any() and len(binary_labels) > 1:
            unw_b = variants[binary_labels[0]]["unw_stack"]
            bf = np.asarray(np.isfinite(unw_b).any(axis=0))
        binary_finite = bf
    picks = pick_test_pixels(
        temp_coh, ref, rng_seed=args.seed, binary_finite=binary_finite
    )
    print(f"[plot] picked pixels:")
    for p in picks:
        print(
            f"        {p['name']:18s} ({p['i']}, {p['j']})  temp_coh={p['temp_coh']:.3f}"
        )
    plot_timeseries(summary, variants, picks, args.pv_out / "timeseries.png")

    print(f"[plot] aggregate metrics → {args.pv_out}/aggregate.json")
    agg = compute_aggregate(variants)
    agg["picked_pixels"] = picks
    with (args.pv_out / "aggregate.json").open("w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg["per_variant"], indent=2))


if __name__ == "__main__":
    main()
