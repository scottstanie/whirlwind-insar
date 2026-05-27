"""Synthetic test: continuous-cost vs binary-mask unwrapping in whirlwind-rs.

Question
--------
Spurt builds its spatial graph over pixels with ``temp_coh > threshold`` —
a hard good/bad partition. We use a continuous coherence- (or CRLB-) weighted
cost over all pixels. Does binarizing our costs by thresholding the same
quality map recover spurt's behavior? And is binarization fragile when the
threshold lands in the middle of the coherence distribution?

In whirlwind-rs the closest analog to spurt's "exclude bad pixels" is the
existing ``mask`` argument. ``mask[i,j] = False`` sets every arc touching
``(i,j)`` to integer cost 0 — i.e. "free to cut". (Strict exclusion would
need a Rust-side change to set those arcs to large cost; not done here.)

This script runs three controlled scenarios and writes PNGs + a JSON summary.

Scenarios
---------
1. ``bridge_between_blobs`` — two high-coherence blobs joined by a thin
   strip of moderate coherence (γ≈0.7), embedded in a low-coherence
   background (γ≈0.3). A phase ramp runs across the bridge. Tests whether
   threshold position around 0.7 changes the *topology* of the graph:
   T=0.6 keeps the bridge → blobs are correctly relatively unwrapped;
   T=0.8 cuts the bridge → blobs are disconnected and the integer cycle
   between them is unconstrained. Continuous costs should not have this
   knife-edge.

2. ``noise_spike_at_boundary`` — flat truth phase. A single 2π noise spike
   sits one pixel inside the high-coh side of a γ≈0.4 / γ≈0.9 boundary.
   Tests whether moving the threshold across the boundary changes how
   the spike's residue dipole is routed.

3. ``threshold_sweep`` — a noisy-bump scene; sweep the binary threshold
   from 0.5 to 0.95 and plot RMSE-vs-threshold against the continuous
   baseline, measured on a common evaluation mask.

Comparing fairly
----------------
Each method outputs unwrapped phase on the full grid; binary methods have
unreliable values where the mask=False (cost=0 lets flow stream freely
through, but the integrated phase is not meaningful). To avoid making
binary look artificially better by only evaluating on its own (smaller,
easier) mask, the metrics are computed on a single *common evaluation
mask* per scenario — the intersection of all per-method masks, or the
strictest binary mask in the sweep.

How to rerun
------------
::

    uv run python scripts/binary_vs_continuous_synth.py \\
        --out /tmp/binary-vs-continuous/synth

Inputs are deterministic (seed=0). Total runtime is a few seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import whirlwind_rs as ww


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------
#
# All return unwrapped phase on the same (m, n) grid as the input igram.
# They differ in how per-edge cost is computed inside whirlwind-rs:
#
#   continuous : pass coherence directly; cost = γ_edge · (π − |α_smooth|).
#                Bad-but-not-excluded pixels (γ low) get low cost → flow
#                cheap to route through them, but the gradient term still
#                pulls toward smoothness.
#
#   binary     : pass coherence as a constant on "good" pixels (γ_uniform)
#                + a mask = (coh > threshold). Mask=False sets arc cost to
#                0 (cheap to cut), the spurt-style "this pixel doesn't pay
#                penalty to cut around" behavior. Good edges get a uniform
#                cost regardless of their actual coherence — that's the
#                "binary" part.
#
# Both use the Carballo cost (``ww.unwrap``). nlooks fixed at 5.


def unwrap_continuous(igram: np.ndarray, coh: np.ndarray, nlooks: float) -> np.ndarray:
    return ww.unwrap(igram.astype(np.complex64), coh.astype(np.float32), float(nlooks))


def unwrap_binary(
    igram: np.ndarray,
    coh: np.ndarray,
    threshold: float,
    nlooks: float,
    gamma_uniform: float = 0.95,
) -> np.ndarray:
    mask = (coh > threshold).astype(bool)
    coh_uniform = np.where(mask, gamma_uniform, 0.0).astype(np.float32)
    return ww.unwrap(igram.astype(np.complex64), coh_uniform, float(nlooks), mask)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def integer_cycle_offset(unw: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> int:
    """Pick the integer k that minimizes |unw + 2πk − truth| (median) on mask.

    NaN values in `unw` (from disconnected components of a binary mask) are
    excluded from the median.
    """
    if not mask.any():
        return 0
    diff = (truth - unw)[mask]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return 0
    return int(np.round(float(np.median(diff)) / (2 * np.pi)))


def metrics(unw: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray) -> dict[str, Any]:
    """RMSE + cycle-error count on `eval_mask`, NaN-skipping.

    Reports four things:
      - ``rmse_rad`` over the eval-mask ∩ non-NaN region (apples-to-apples
        when comparison is possible)
      - ``n_cycle_errors`` (|err| > π) over the same
      - ``n_pixels_compared`` (where we could compute)
      - ``n_pixels_nan`` (where this method had no answer in the eval mask)
    """
    if not eval_mask.any():
        return {"rmse_rad": float("nan"), "n_cycle_errors": 0, "n_pixels_compared": 0, "n_pixels_nan": 0, "k": 0}
    valid = eval_mask & np.isfinite(unw)
    if not valid.any():
        return {"rmse_rad": float("nan"), "n_cycle_errors": 0, "n_pixels_compared": 0,
                "n_pixels_nan": int(eval_mask.sum()), "k": 0}
    k = integer_cycle_offset(unw, truth, valid)
    err = (unw + 2 * np.pi * k - truth)[valid]
    return {
        "rmse_rad": float(np.sqrt(np.mean(err ** 2))),
        "n_cycle_errors": int(np.sum(np.abs(err) > np.pi)),
        "n_pixels_compared": int(valid.sum()),
        "n_pixels_nan": int(eval_mask.sum() - valid.sum()),
        "k": k,
    }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _add_phase_noise(
    truth: np.ndarray, coh: np.ndarray, nlooks: float, rng: np.random.Generator
) -> np.ndarray:
    """Gaussian-approximate Lee phase noise; std = sqrt((1-γ²)/(2 L γ²))."""
    sigma = np.sqrt((1 - coh ** 2) / (2 * nlooks * np.maximum(coh ** 2, 1e-3)))
    return (truth + sigma * rng.standard_normal(truth.shape)).astype(np.float32)


def scene_bridge_between_blobs(
    shape: tuple[int, int] = (256, 256), seed: int = 0
) -> dict[str, Any]:
    """Two γ=0.9 blobs + a γ=0.7 bridge embedded in γ=0.3 background."""
    rng = np.random.default_rng(seed)
    m, n = shape
    # Truth: a ramp in x. Multiple 2π wrap lines so unwrapping is non-trivial.
    truth = (np.arange(n, dtype=np.float32) * (6 * np.pi / n))[None, :] * np.ones((m, 1), dtype=np.float32)

    # Coherence layout: two disk blobs joined by a thin horizontal bridge.
    yy, xx = np.mgrid[0:m, 0:n].astype(np.float32)
    blob_radius = m // 6
    cy = m // 2
    cx1, cx2 = n // 4, 3 * n // 4
    in_blob1 = (yy - cy) ** 2 + (xx - cx1) ** 2 < blob_radius ** 2
    in_blob2 = (yy - cy) ** 2 + (xx - cx2) ** 2 < blob_radius ** 2
    in_bridge = (np.abs(yy - cy) < 3) & (xx > cx1) & (xx < cx2)
    coh = np.full(shape, 0.3, dtype=np.float32)
    coh[in_bridge] = 0.7
    coh[in_blob1 | in_blob2] = 0.9

    nlooks = 5.0
    noisy_phase = _add_phase_noise(truth, coh, nlooks, rng)
    igram = np.exp(1j * noisy_phase).astype(np.complex64)
    # Evaluation mask: the two blobs only (where truth comparison is meaningful
    # AND every method has reliable data — we DON'T evaluate on the bridge or
    # background, only on the blobs).
    eval_mask = in_blob1 | in_blob2
    return {
        "truth": truth,
        "igram": igram,
        "coh": coh,
        "nlooks": nlooks,
        "eval_mask": eval_mask,
        "blob1": in_blob1,
        "blob2": in_blob2,
        "bridge": in_bridge,
    }


def scene_noise_spike_at_boundary(
    shape: tuple[int, int] = (256, 256), seed: int = 0
) -> dict[str, Any]:
    """Flat truth; a 2π noise spike sits one pixel inside the good (γ=0.9) side
    of a sharp γ=0.4 | γ=0.9 boundary at x=n/2."""
    rng = np.random.default_rng(seed)
    m, n = shape
    truth = np.zeros(shape, dtype=np.float32)
    coh = np.where(np.arange(n) < n // 2, 0.4, 0.9).astype(np.float32)
    coh = np.broadcast_to(coh[None, :], shape).copy()
    nlooks = 5.0
    noisy_phase = _add_phase_noise(truth, coh, nlooks, rng)
    # Inject a single 2π upward shift on the good side, two pixels inside.
    si, sj = m // 2, n // 2 + 2
    noisy_phase[si, sj] += 2 * np.pi
    igram = np.exp(1j * noisy_phase).astype(np.complex64)
    # Evaluation mask: the good side, away from the spike.
    eval_mask = np.zeros(shape, dtype=bool)
    eval_mask[:, n // 2 + 8 :] = True
    return {
        "truth": truth,
        "igram": igram,
        "coh": coh,
        "nlooks": nlooks,
        "eval_mask": eval_mask,
        "spike": (si, sj),
    }


def scene_noisy_bump(
    shape: tuple[int, int] = (256, 256), seed: int = 0
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    m, n = shape
    yy, xx = np.mgrid[0:m, 0:n].astype(np.float32)
    cy, cx = m / 2, n / 2
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    truth = (5.0 * np.pi * np.exp(-r2 / (2 * (m / 6) ** 2))).astype(np.float32)
    coh = (0.95 - 0.55 * (r2 / r2.max())).astype(np.float32)
    nlooks = 5.0
    noisy_phase = _add_phase_noise(truth, coh, nlooks, rng)
    igram = np.exp(1j * noisy_phase).astype(np.complex64)
    # Evaluation mask: a moderate band (γ ∈ [0.6, 0.8]) where threshold choice
    # actually affects whether these pixels are in the binary mask. Pixels
    # always-kept (γ > 0.9) would give a flat curve regardless of threshold.
    eval_mask = (coh > 0.6) & (coh < 0.8)
    return {
        "truth": truth,
        "igram": igram,
        "coh": coh,
        "nlooks": nlooks,
        "eval_mask": eval_mask,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _residual_panel(ax, unw: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray, title: str) -> None:
    valid = eval_mask & np.isfinite(unw)
    k = integer_cycle_offset(unw, truth, valid)
    resid = np.where(np.isfinite(unw), unw + 2 * np.pi * k - truth, np.nan)
    if np.isfinite(resid).any():
        vmax = max(2 * np.pi, float(np.nanpercentile(np.abs(resid), 99)))
    else:
        vmax = float(2 * np.pi)
    im = ax.imshow(resid, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_scenario(
    name: str, scene: dict[str, Any], results: dict[str, dict[str, Any]], out: Path
) -> None:
    truth = scene["truth"]
    coh = scene["coh"]
    wrapped = np.angle(scene["igram"])

    methods = list(results.keys())
    ncols = 3 + len(methods)
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3.2))

    ax = axes[0]
    im = ax.imshow(wrapped, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
    ax.set_title("wrapped", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(truth, cmap="viridis")
    ax.set_title("truth (unwrapped)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    im = ax.imshow(coh, cmap="cividis", vmin=0, vmax=1)
    ax.set_title("coherence", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i, (label, payload) in enumerate(results.items(), start=3):
        _residual_panel(axes[i], payload["unw"], truth, scene["eval_mask"], label)

    fig.suptitle(f"{name}: residual = unw − truth, evaluated on common mask", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_threshold_sweep(
    sweep: list[tuple[float, dict[str, Any]]],
    cont: dict[str, Any],
    eval_n: int,
    out: Path,
) -> None:
    thresholds = [t for t, _ in sweep]
    rmses = [m["rmse_rad"] for _, m in sweep]
    kept_frac = [m["n_pixels_compared"] / eval_n for _, m in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(thresholds, rmses, "o-", label="binary mask")
    ax1.axhline(cont["rmse_rad"], color="black", linestyle="--", label="continuous (all eval pixels)")
    ax1.set_xlabel("threshold")
    ax1.set_ylabel("RMSE vs truth [rad]")
    ax1.set_title("RMSE on surviving eval pixels", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(thresholds, kept_frac, "o-", color="C1")
    ax2.set_xlabel("threshold")
    ax2.set_ylabel("fraction of eval-mask pixels kept")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("data loss in eval band (γ ∈ [0.6, 0.8])", fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle("threshold sweep (noisy_bump): binary loses pixels as T climbs;\ncontinuous keeps the whole eval band", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/tmp/binary-vs-continuous/synth"))
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, Any]] = {}

    # --- scenario 1: bridge between blobs -------------------------------------
    name = "bridge_between_blobs"
    print(f"[{name}]")
    scene = scene_bridge_between_blobs()
    runs: dict[str, dict[str, Any]] = {}
    runs["continuous"] = {
        "unw": unwrap_continuous(scene["igram"], scene["coh"], scene["nlooks"]),
    }
    for t in (0.6, 0.8):
        runs[f"binary T={t}"] = {
            "unw": unwrap_binary(scene["igram"], scene["coh"], t, scene["nlooks"]),
        }
    # Metric on common eval mask (the two blobs only).
    scn_metrics = {label: metrics(p["unw"], scene["truth"], scene["eval_mask"]) for label, p in runs.items()}
    # Bonus: per-blob integer offset to expose disconnect.
    for label, p in runs.items():
        k1 = integer_cycle_offset(p["unw"], scene["truth"], scene["blob1"])
        k2 = integer_cycle_offset(p["unw"], scene["truth"], scene["blob2"])
        scn_metrics[label]["blob1_offset_cycles"] = k1
        scn_metrics[label]["blob2_offset_cycles"] = k2
    summary[name] = scn_metrics
    plot_scenario(name, scene, runs, out / f"{name}.png")
    for label, mt in scn_metrics.items():
        print(
            f"    {label:14s}  RMSE={mt['rmse_rad']:.3f}  errs={mt['n_cycle_errors']:>5d}/{mt['n_pixels_compared']}  "
            f"nan={mt['n_pixels_nan']:>5d}  blob1_k={mt['blob1_offset_cycles']}  blob2_k={mt['blob2_offset_cycles']}"
        )

    # --- scenario 2: noise spike at boundary ----------------------------------
    name = "noise_spike_at_boundary"
    print(f"[{name}]")
    scene = scene_noise_spike_at_boundary()
    runs = {
        "continuous": {"unw": unwrap_continuous(scene["igram"], scene["coh"], scene["nlooks"])},
    }
    for t in (0.5, 0.7, 0.95):
        runs[f"binary T={t}"] = {"unw": unwrap_binary(scene["igram"], scene["coh"], t, scene["nlooks"])}
    scn_metrics = {label: metrics(p["unw"], scene["truth"], scene["eval_mask"]) for label, p in runs.items()}
    summary[name] = scn_metrics
    plot_scenario(name, scene, runs, out / f"{name}.png")
    for label, mt in scn_metrics.items():
        print(
            f"    {label:14s}  RMSE={mt['rmse_rad']:.3f}  errs={mt['n_cycle_errors']:>5d}/{mt['n_pixels_compared']}  "
            f"nan={mt['n_pixels_nan']:>5d}  k={mt['k']}"
        )

    # --- scenario 3: threshold sweep ------------------------------------------
    name = "threshold_sweep"
    print(f"[{name}]")
    scene = scene_noisy_bump()
    cont = metrics(
        unwrap_continuous(scene["igram"], scene["coh"], scene["nlooks"]),
        scene["truth"],
        scene["eval_mask"],
    )
    sweep: list[tuple[float, dict[str, Any]]] = []
    for t_val in np.linspace(0.5, 0.93, 10):
        t = float(round(float(t_val), 3))
        unw = unwrap_binary(scene["igram"], scene["coh"], t, scene["nlooks"])
        sweep.append((t, metrics(unw, scene["truth"], scene["eval_mask"])))
    summary[name] = {"continuous": cont, "sweep": [{"threshold": t, **m} for t, m in sweep]}
    plot_threshold_sweep(sweep, cont, int(scene["eval_mask"].sum()), out / f"{name}.png")
    print(f"    continuous     RMSE={cont['rmse_rad']:.3f}  errs={cont['n_cycle_errors']:>5d}/{cont['n_pixels_compared']}")
    for t, mt in sweep:
        print(
            f"    binary T={t:.2f}  RMSE={mt['rmse_rad']:.3f}  errs={mt['n_cycle_errors']:>5d}/{mt['n_pixels_compared']}  "
            f"nan={mt['n_pixels_nan']:>5d}"
        )

    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
