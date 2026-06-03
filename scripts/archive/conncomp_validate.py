"""Validate SNAPHU-style connected components from the whirlwind MCF solve.

Compares three component sources on the same scene:

1. ``skimage.label`` on ``coh > threshold`` — what spurt would do upstream.
2. ``ww.unwrap_with_conncomp`` — components grown from the MCF cost graph.
3. The same MCF components but at multiple ``cost_threshold`` values, to
   show the gradient instead of a single number.

Two synthetic scenes:

* **bridge_between_blobs**: two γ≈0.9 disks joined by a thin γ≈0.7 bridge
  in γ≈0.3 background. Tests whether the conncomp output separates the
  blobs from the background and how the bridge survives at different
  cost thresholds.

* **noisy_ramp_with_hole**: smooth ramp with a low-coherence hole. The
  unwrap should be fine on the bulk but the hole should be a cut.

Outputs PNGs under ``--out`` plus a small JSON summary.

How to rerun::

    uv run python scripts/conncomp_validate.py \\
        --out /tmp/whirlwind-conncomp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label as sk_label

import whirlwind as ww


def _add_phase_noise(truth, coh, nlooks, rng):
    sigma = np.sqrt((1 - coh ** 2) / (2 * nlooks * np.maximum(coh ** 2, 1e-3)))
    return (truth + sigma * rng.standard_normal(truth.shape)).astype(np.float32)


def scene_bridge():
    rng = np.random.default_rng(0)
    m = n = 256
    truth = (np.arange(n, dtype=np.float32) * (6 * np.pi / n))[None, :] * np.ones((m, 1), dtype=np.float32)
    yy, xx = np.mgrid[0:m, 0:n].astype(np.float32)
    r = m // 6
    cy, cx1, cx2 = m // 2, n // 4, 3 * n // 4
    in_b1 = (yy - cy) ** 2 + (xx - cx1) ** 2 < r ** 2
    in_b2 = (yy - cy) ** 2 + (xx - cx2) ** 2 < r ** 2
    in_br = (np.abs(yy - cy) < 3) & (xx > cx1) & (xx < cx2)
    coh = np.full((m, n), 0.3, dtype=np.float32)
    coh[in_br] = 0.7
    coh[in_b1 | in_b2] = 0.9
    nlooks = 5.0
    igram = np.exp(1j * _add_phase_noise(truth, coh, nlooks, rng)).astype(np.complex64)
    return {"name": "bridge", "truth": truth, "igram": igram, "coh": coh, "nlooks": nlooks}


def scene_hole():
    rng = np.random.default_rng(1)
    m = n = 256
    truth = (0.05 * np.pi * (np.arange(m)[:, None] + np.arange(n)[None, :])).astype(np.float32)
    yy, xx = np.mgrid[0:m, 0:n].astype(np.float32)
    cy, cx, r = m // 2, n // 2, 32
    in_hole = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    coh = np.full((m, n), 0.9, dtype=np.float32)
    coh[in_hole] = 0.2
    nlooks = 5.0
    igram = np.exp(1j * _add_phase_noise(truth, coh, nlooks, rng)).astype(np.complex64)
    return {"name": "hole", "truth": truth, "igram": igram, "coh": coh, "nlooks": nlooks}


def spurt_style_components(coh: np.ndarray, threshold: float) -> np.ndarray:
    """``skimage.label`` on a hard temp-coh mask; mimics spurt's component finder."""
    mask = (coh > threshold).astype(np.uint8)
    return sk_label(mask, connectivity=1)


def render(scene, out_dir: Path):
    name = scene["name"]
    coh = scene["coh"]
    ig = scene["igram"]
    nlooks = scene["nlooks"]
    truth = scene["truth"]

    # --- spurt-style baselines ---
    sp_low = spurt_style_components(coh, 0.6)
    sp_high = spurt_style_components(coh, 0.85)

    # --- MCF-derived components at three thresholds ---
    results = {}
    for thresh in (50, 150, 250):
        unw, cc = ww.unwrap(ig, coh, nlooks, cost_threshold=thresh, goldstein_alpha=0.7)
        results[thresh] = {"unw": unw, "cc": cc}

    # Continuous unwrap (no components) for reference.
    unw_ref, _cc = ww.unwrap(ig.astype(np.complex64), coh.astype(np.float32), float(nlooks))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    im_kw = dict(interpolation="none")

    axes[0, 0].imshow(coh, vmin=0, vmax=1, cmap="magma", **im_kw)
    axes[0, 0].set_title(f"{name}: input γ̂")
    axes[0, 1].imshow(unw_ref, cmap="twilight", **im_kw)
    axes[0, 1].set_title("ww.unwrap (continuous)")

    axes[0, 2].imshow(sp_low, cmap="tab20", **im_kw)
    axes[0, 2].set_title(f"spurt-style label(γ̂ > 0.60)\n{int(sp_low.max())} components")
    axes[0, 3].imshow(sp_high, cmap="tab20", **im_kw)
    axes[0, 3].set_title(f"spurt-style label(γ̂ > 0.85)\n{int(sp_high.max())} components")

    for i, thresh in enumerate((50, 150, 250)):
        ax = axes[1, i]
        cc = results[thresh]["cc"]
        ax.imshow(cc, cmap="tab20", **im_kw)
        ax.set_title(
            f"MCF conncomp (cost_threshold={thresh})\n"
            f"{int(cc.max())} components, coverage={(cc > 0).mean():.2%}"
        )
    # Overlay: difference between MCF@150 and spurt@0.6 in coverage
    overlay = np.zeros_like(coh)
    cc_mid = (results[150]["cc"] > 0).astype(int)
    sp_cov = (sp_low > 0).astype(int)
    overlay = cc_mid - sp_cov
    axes[1, 3].imshow(overlay, vmin=-1, vmax=1, cmap="bwr", **im_kw)
    axes[1, 3].set_title("MCF@150 minus spurt@0.6 coverage\n(red: only MCF, blue: only spurt)")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    out_path = out_dir / f"conncomp_{name}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {
        "scene": name,
        "spurt_0.60_n": int(sp_low.max()),
        "spurt_0.85_n": int(sp_high.max()),
        "spurt_0.60_coverage": float((sp_low > 0).mean()),
        "spurt_0.85_coverage": float((sp_high > 0).mean()),
        "mcf": {
            str(t): {
                "n_components": int(r["cc"].max()),
                "coverage": float((r["cc"] > 0).mean()),
            }
            for t, r in results.items()
        },
    }


def threshold_sweep(scene, out_dir: Path):
    """Sweep ``cost_threshold`` over a wide range; plot coverage of the
    largest component and #components vs threshold. The point: MCF cost
    threshold gives a smooth gradient, not a knife-edge — small changes
    don't catastrophically partition the scene.
    """
    name = scene["name"]
    ig = scene["igram"]
    coh = scene["coh"]
    nlooks = scene["nlooks"]

    thresholds = np.arange(0, 320, 10)
    coverages = []
    n_comps = []
    largest_frac = []
    for t in thresholds:
        unw, cc = ww.unwrap(ig, coh, nlooks, cost_threshold=int(t), goldstein_alpha=0.7)
        sizes = np.bincount(cc.ravel())
        n_comps.append(int(cc.max()))
        coverages.append(float((cc > 0).mean()))
        if len(sizes) > 1 and sizes[1:].size > 0:
            largest_frac.append(float(sizes[1:].max()) / cc.size)
        else:
            largest_frac.append(0.0)

    # Compare to a spurt-style γ̂ threshold sweep on the same scene.
    coh_thresholds = np.linspace(0.0, 1.0, 21)
    sp_n = []
    sp_cov = []
    sp_largest = []
    for t in coh_thresholds:
        labels = spurt_style_components(coh, float(t))
        sizes = np.bincount(labels.ravel())
        sp_n.append(int(labels.max()))
        sp_cov.append(float((labels > 0).mean()))
        if len(sizes) > 1 and sizes[1:].size > 0:
            sp_largest.append(float(sizes[1:].max()) / labels.size)
        else:
            sp_largest.append(0.0)

    fig, (axc, axn) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axc.plot(thresholds, coverages, "-o", label="MCF: total coverage", color="C0")
    axc.plot(thresholds, largest_frac, "--", label="MCF: largest component", color="C0", alpha=0.6)
    ax2 = axc.twiny()
    ax2.plot(coh_thresholds, sp_cov, "-s", label="spurt-style: total coverage", color="C3", markersize=4)
    ax2.plot(coh_thresholds, sp_largest, "--", label="spurt-style: largest", color="C3", alpha=0.6)
    axc.set_xlabel("MCF cost_threshold (integer Carballo units, COST_SCALE=100)")
    ax2.set_xlabel("spurt-style coherence threshold")
    axc.set_ylabel("fraction of pixels")
    axc.set_title(f"{name}: coverage vs threshold")
    axc.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    axc.grid(alpha=0.3)

    axn.plot(thresholds, n_comps, "-o", label="MCF", color="C0")
    ax2b = axn.twiny()
    ax2b.plot(coh_thresholds, sp_n, "-s", label="spurt-style", color="C3", markersize=4)
    axn.set_xlabel("MCF cost_threshold")
    ax2b.set_xlabel("spurt-style coherence threshold")
    axn.set_ylabel("number of components")
    axn.set_title(f"{name}: component count vs threshold")
    axn.legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="lower right", fontsize=8)
    axn.grid(alpha=0.3)

    fig.savefig(out_dir / f"conncomp_sweep_{name}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/whirlwind-conncomp")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for scene_fn in (scene_bridge, scene_hole):
        scene = scene_fn()
        summary.append(render(scene, out_dir))
        threshold_sweep(scene, out_dir)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nFigures in {out_dir}/")


if __name__ == "__main__":
    main()
