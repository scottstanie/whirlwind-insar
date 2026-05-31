"""Synthetic dense-fringe stress test: multilook-first vs. pyramidal unwrap.

Why this script exists
----------------------
We optionally multilook the complex interferogram by a big factor as a first
step (noise suppression), unwrap the coarse grid, then block-replicate the
result back to full resolution (``ww.unwrap(..., multilook=L)``). That is a
fine noise filter but a *destructive* one for steep signals. Coherently
down-looking by ``L`` multiplies the per-pixel fringe rate by ``L``: a full-res
gradient ``g`` (rad/pixel) becomes ``L·g`` on the coarse grid. As soon as
``L·g > π`` the coarse grid is **aliased**, the coarse unwrap locks onto the
wrong (too few) integer cycle count, and block-replicating that wrong ``K`` can
never recover the true surface. A dense-fringe deformation that full-res
unwrapping would get right (a volcano eruption bowl, an earthquake near-field)
is silently destroyed.

This script builds synthetic very-dense-fringe scenes (a steep bowl and a
constant-rate cone), sweeps the fringe rate up toward the full-res Nyquist
limit (``g → π``), and compares strategies on the same data:

  * ``full``       — full-resolution ``ww.unwrap`` (no multilook).
  * ``ml4``/``ml8``— single-shot multilook-first, ``ww.unwrap(multilook=L)``.
  * ``pyr2``/``pyr4`` — pyramidal coarse-to-fine, fixed ``base_factor``.
  * ``pyrA``       — pyramidal with automatic ``base_factor`` (``=0``).

The pyramid uses its default *reuse* base solver (the linear coherence cost
mis-routes the corners of smooth steep signals — see ``make_corner_panels`` and
``paper/pyramid_aliasing.md``). Each finer level unwraps only the *residual*
against the upsampled coarser solution (the previous level's ``K`` as a prior),
recovering full resolution without the single big multilook jump — *provided its
coarsest level is itself unaliased* (``base·g < π``).

How to rerun
------------
::

    uv run python scripts/dense_fringe_pyramid.py --out /tmp/dense-fringe

Deterministic (seed=0). A couple of minutes on a 384² grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import whirlwind as ww

PI = np.pi
METHODS = ["full", "ml4", "ml8", "pyr2", "pyr4", "pyrA"]
COLORS = {
    "full": "k",
    "ml4": "tab:orange",
    "ml8": "tab:red",
    "pyr2": "tab:green",
    "pyr4": "tab:blue",
    "pyrA": "tab:purple",
}


# ---------------------------------------------------------------------------
# Synthetic truth fields
# ---------------------------------------------------------------------------
def cone(shape: tuple[int, int], g: float) -> np.ndarray:
    """Constant radial fringe rate ``g`` rad/pixel: φ(r) = g·r."""
    m, n = shape
    ci, cj = (m - 1) / 2, (n - 1) / 2
    i, j = np.ogrid[:m, :n]
    r = np.sqrt((i - ci) ** 2 + (j - cj) ** 2)
    return (g * r).astype(np.float32)


def bowl(shape: tuple[int, int], g_edge: float) -> np.ndarray:
    """Paraboloid φ(r) = a·r² with edge fringe rate ``g_edge`` rad/pixel.

    Gradient grows linearly from 0 at the centre to ``g_edge`` at the corner,
    so a single scene contains the whole rate spectrum up to ``g_edge``.
    """
    m, n = shape
    ci, cj = (m - 1) / 2, (n - 1) / 2
    i, j = np.ogrid[:m, :n]
    r = np.sqrt((i - ci) ** 2 + (j - cj) ** 2)
    r_max = float(np.sqrt(ci**2 + cj**2))
    a = g_edge / (2.0 * r_max)
    return (a * r * r).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def k_correct(unw: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of pixels whose integer 2π cycle matches truth (offset-aligned)."""
    valid = np.isfinite(unw)
    if not valid.any():
        return float("nan")
    d = unw[valid] - truth[valid]
    d = d - 2 * PI * round(float(np.median(d)) / (2 * PI))
    return float(np.mean(np.round(d / (2 * PI)) == 0))


def rmse(unw: np.ndarray, truth: np.ndarray) -> float:
    valid = np.isfinite(unw)
    if not valid.any():
        return float("nan")
    d = unw[valid] - truth[valid]
    d = d - 2 * PI * round(float(np.median(d)) / (2 * PI))
    return float(np.sqrt(np.mean(d**2)))


# ---------------------------------------------------------------------------
# Method runners — all return full-res unwrapped phase on the input grid.
# ---------------------------------------------------------------------------
def run_methods(ig: np.ndarray, corr: np.ndarray, nlooks: float) -> dict[str, np.ndarray]:
    ig = ig.astype(np.complex64)
    corr = corr.astype(np.float32)
    return {
        "full": ww.unwrap(ig, corr, nlooks=nlooks),
        "ml4": ww.unwrap(ig, corr, nlooks=nlooks, multilook=4),
        "ml8": ww.unwrap(ig, corr, nlooks=nlooks, multilook=8),
        "pyr2": ww.unwrap_pyramid(ig, corr, nlooks=nlooks, base_factor=2),
        "pyr4": ww.unwrap_pyramid(ig, corr, nlooks=nlooks, base_factor=4),
        "pyrA": ww.unwrap_pyramid(ig, corr, nlooks=nlooks, base_factor=0),
    }


def simulate(truth: np.ndarray, gamma: float, nlooks: int, seed: int):
    g = np.full(truth.shape, gamma, np.float32)
    ig, corr = ww.simulate_ifg(truth, g, nlooks, seed)
    return ig.astype(np.complex64), corr.astype(np.float32)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def sweep_rate(shape, shape_fn, gammas, g_fracs, nlooks, seed):
    """K-correct vs fringe rate, for each (coherence, method)."""
    rows = []
    for gamma in gammas:
        for gf in g_fracs:
            truth = shape_fn(shape, gf * PI)
            ig, corr = simulate(truth, gamma, nlooks, seed)
            res = run_methods(ig, corr, float(nlooks))
            for name, unw in res.items():
                rows.append(
                    {
                        "gamma": float(gamma),
                        "g_frac_pi": float(gf),
                        "method": name,
                        "k_correct": k_correct(unw, truth),
                        "rmse": rmse(unw, truth),
                    }
                )
            print(
                f"  gamma={gamma:.2f} g={gf:.2f}π  "
                + "  ".join(f"{n}={k_correct(res[n], truth) * 100:4.0f}%" for n in METHODS)
            )
    return rows


def sweep_noise(shape, shape_fn, g_frac, nlooks, gammas, seed):
    """K-correct vs coherence at a fixed mild fringe rate (the regime that
    justifies multilooking: coarse grids stay unaliased while full-res drowns)."""
    rows = []
    truth = shape_fn(shape, g_frac * PI)
    for gamma in gammas:
        ig, corr = simulate(truth, gamma, nlooks, seed)
        res = run_methods(ig, corr, float(nlooks))
        for name, unw in res.items():
            rows.append(
                {
                    "gamma": float(gamma),
                    "g_frac_pi": float(g_frac),
                    "method": name,
                    "k_correct": k_correct(unw, truth),
                    "rmse": rmse(unw, truth),
                }
            )
        print(
            f"  γ={gamma:.2f} (g={g_frac:.2f}π, {nlooks} looks)  "
            + "  ".join(f"{n}={k_correct(res[n], truth) * 100:4.0f}%" for n in METHODS)
        )
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_curves(out: Path, rate_rows, title, fname):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gammas = sorted({r["gamma"] for r in rate_rows})
    fig, axes = plt.subplots(1, len(gammas), figsize=(4.6 * len(gammas), 4.0), squeeze=False)
    for ax, gamma in zip(axes[0], gammas):
        for name in METHODS:
            xs = sorted({r["g_frac_pi"] for r in rate_rows if r["method"] == name})
            ys = [
                next(
                    r["k_correct"]
                    for r in rate_rows
                    if r["method"] == name and r["gamma"] == gamma and r["g_frac_pi"] == x
                )
                for x in xs
            ]
            ax.plot(xs, np.array(ys) * 100, "o-", color=COLORS[name], label=name, ms=4)
        ax.axvline(1 / 8, color="tab:red", ls=":", lw=1, alpha=0.7)
        ax.axvline(1 / 2, color="tab:green", ls=":", lw=1, alpha=0.7)
        ax.set_title(f"γ = {gamma}")
        ax.set_xlabel("fringe rate g  (× π rad/pixel)")
        ax.set_ylabel("K-correct (%)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / fname
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def make_noise_curve(out: Path, rows, title, fname):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name in METHODS:
        xs = sorted({r["gamma"] for r in rows if r["method"] == name})
        ys = [next(r["k_correct"] for r in rows if r["method"] == name and r["gamma"] == x) for x in xs]
        ax.plot(xs, np.array(ys) * 100, "o-", color=COLORS[name], label=name, ms=4)
    ax.set_xlabel("coherence γ")
    ax.set_ylabel("K-correct (%)")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    p = out / fname
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def make_panels(out: Path, shape, seed):
    """Phase-map panels for one representative steep, noisy bowl."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g_edge = 0.7 * PI  # steep: full-res still unaliased, but ml8 aliases hard
    gamma = 0.8
    truth = bowl(shape, g_edge)
    ig, corr = simulate(truth, gamma, 8, seed)
    res = run_methods(ig, corr, 8.0)
    wrapped = np.angle(ig)

    ncol = 2 + len(METHODS)
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.2))

    def show(ax, img, title, **kw):
        im = ax.imshow(img, **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    show(axes[0, 0], wrapped, "wrapped φ", cmap="twilight", vmin=-PI, vmax=PI)
    vmax = float(np.nanmax(truth))
    show(axes[0, 1], truth, "truth (unwrapped)", cmap="viridis", vmin=0, vmax=vmax)
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    for c, name in enumerate(METHODS, start=2):
        unw = res[name]
        off = 2 * PI * round(float(np.nanmedian(unw - truth)) / (2 * PI))
        kk = k_correct(unw, truth)
        show(axes[0, c], unw - off, f"{name}\nK={kk * 100:.0f}%", cmap="viridis", vmin=0, vmax=vmax)
        err = (unw - off) - truth
        show(axes[1, c], err, "error", cmap="RdBu_r", vmin=-3 * PI, vmax=3 * PI)
    fig.suptitle(
        f"Steep noisy bowl: g_edge={g_edge / PI:.1f}π, γ={gamma}, 8 looks "
        f"(ml8 aliases at g>π/8; full-res still unaliased at g<π)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / "panels_steep_bowl.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def make_corner_panels(out: Path, shape):
    """Clean steep bowl: the linear cost mis-routes the corners (the steepest
    part), the reuse/convex solvers do not. This is why the pyramid does NOT
    default to the linear coherence cost."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = bowl(shape, 0.7 * PI)
    ig = np.exp(1j * truth).astype(np.complex64)  # perfectly clean
    corr = np.full(shape, 0.999, np.float32)
    solvers = ["linear", "convex", "reuse"]
    outs = {s: ww.unwrap_pyramid(ig, corr, nlooks=1.0, base_factor=1, solver=s) for s in solvers}

    fig, axes = plt.subplots(1, len(solvers), figsize=(4.0 * len(solvers), 4.2))
    im = None
    for ax, s in zip(axes, solvers):
        unw = outs[s]
        off = 2 * PI * round(float(np.nanmedian(unw - truth)) / (2 * PI))
        err = (unw - off) - truth
        kk = k_correct(unw, truth)
        im = ax.imshow(err, cmap="RdBu_r", vmin=-3 * PI, vmax=3 * PI)
        ax.set_title(f"solver={s}\nK={kk * 100:.0f}%  (error)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Clean 0.7π bowl, base=1: linear cost mis-routes the corners "
        "(capacity-1 boundary stacking); reuse/convex do not",
        fontsize=11,
    )
    fig.colorbar(im, ax=list(axes), fraction=0.025)
    p = out / "panels_corner_solver.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("/tmp/dense-fringe"))
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    shape = (args.size, args.size)

    g_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("== cone, clean (γ=0.95) and noisy (γ=0.6) ==")
    cone_rows = sweep_rate(shape, cone, [0.95, 0.6], g_fracs, nlooks=8, seed=args.seed)
    print("== bowl, clean (γ=0.95) and noisy (γ=0.6) ==")
    bowl_rows = sweep_rate(shape, bowl, [0.95, 0.6], g_fracs, nlooks=8, seed=args.seed)
    print("== noise sweep: mild rate g=0.2π, 4 looks, falling coherence ==")
    noise_rows = sweep_noise(
        shape, cone, g_frac=0.2, nlooks=4,
        gammas=[0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2], seed=args.seed,
    )

    summary = {
        "cone": cone_rows, "bowl": bowl_rows, "noise": noise_rows,
        "size": args.size, "seed": args.seed,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    figs = [
        make_curves(args.out, cone_rows, "Constant-rate cone: K-correct vs fringe rate", "curves_cone.png"),
        make_curves(args.out, bowl_rows, "Steep bowl (paraboloid): K-correct vs edge fringe rate", "curves_bowl.png"),
        make_noise_curve(
            args.out, noise_rows,
            "Mild rate (g=0.2π), 4 looks: K-correct vs coherence\n"
            "(full drowns in noise; ml8 aliases; pyramid holds both)",
            "curves_noise.png",
        ),
        make_panels(args.out, shape, args.seed),
        make_corner_panels(args.out, shape),
    ]
    print("\nWrote:")
    for f in figs:
        print(f"  {f}")
    print(f"  {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
