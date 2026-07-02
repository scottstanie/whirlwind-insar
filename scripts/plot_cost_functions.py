#!/usr/bin/env python3
"""Cost vs. phase-gradient figure for SNAPHU and whirlwind.

Analogous to Fig. 11 of Chen & Zebker (2001), which plots the arc cost as a
function of the unwrapped phase gradient. We draw three panels:

  A. SNAPHU DEFO, reproducing the Chen & Zebker figure with the real snaphu
     2.0.7 default parameters: a parabola near zero, capped at a discontinuity
     "shelf" of height g_d out to +/- dphi_max, then rising again. High
     coherence collapses it to a pure parabola (the dashed curve).
  B. SNAPHU SMOOTH vs DEFO across coherence (SMOOTH is what NISAR GUNW
     production uses): both are coherence-scaled parabolas; DEFO adds the shelf
     only where the coherence falls below snaphu's deformation threshold.
  C. whirlwind's Carballo/Touzi cost: the negative log-likelihood of the
     phase-gradient noise PDF that the embedded LUTs are built from. No
     engineered shelf - the well simply widens and flattens as coherence drops,
     straight out of the interferometric phase statistics.

The SNAPHU curves are computed directly from the cost formulas in the bundled
snaphu 2.0.7 C source (`snaphu_cost.c`: BuildStatCosts{Defo,Smooth} /
CalcCost{Defo,Smooth}) with the defaults from `snaphu.h`; snaphu-py exposes no
cost-dump, so there is no runtime SNAPHU call. The whirlwind curve reuses the
exact PDF code in `generate_carballo_tables.py`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_carballo_tables as gct  # noqa: E402

TAU = 2.0 * math.pi

# --------------------------------------------------------------------------
# SNAPHU 2.0.7 statistical costs (from snaphu.h defaults + snaphu_cost.c)
# --------------------------------------------------------------------------
NSHORTCYCLE = 200.0  # DEF_NSHORTCYCLE: short-cycle units per 2*pi
COSTSCALE = 100.0  # DEF_COSTSCALE
RHOSCONST1, RHOSCONST2 = 1.3, 0.14  # DEF_RHOSCONST1/2
CSTD1, CSTD2, CSTD3 = 0.4, 0.35, 0.06  # DEF_CSTD1/2/3
DEFOTHRESHFACTOR = 1.2  # DEF_DEFOTHRESHFACTOR
DEFOMAX = 1.2  # DEF_DEFOMAX (cycles)
SIGSQCORR = 0.05  # DEF_SIGSQCORR
DEFOLAYCONST = 0.9  # DEF_DEFOLAYCONST
LAYFALLOFFCONST = 2.0  # DEF_LAYFALLOFFCONST
SIGSQSHORTMIN = 1.0  # DEF_SIGSQSHORTMIN
SIGSQRHOCONST = 2.0 / 12.0  # hard-coded in BuildStatCosts*


def snaphu_params(rho: float, nlooks: float, mode: str):
    """Return (sigsq, laycost, dzmax, thresh) for one arc; laycost/dzmax None = no shelf."""
    rho0 = RHOSCONST1 / nlooks + RHOSCONST2
    thresh = DEFOTHRESHFACTOR * rho0
    rhopow = 2 * CSTD1 + CSTD2 * math.log(nlooks) + CSTD3 * nlooks
    below = rho < thresh
    rho_eff = 0.0 if below else rho  # snaphu clips sub-threshold rho to 0
    sigsqrho = (SIGSQRHOCONST * (1.0 - rho_eff) ** rhopow + SIGSQCORR) * NSHORTCYCLE**2
    sigsq = max(sigsqrho / COSTSCALE, SIGSQSHORTMIN)
    if mode == "smooth":
        return sigsq, None, None, thresh
    # defo: add a discontinuity shelf only where rho is below threshold
    laycost: float | None
    dzmax: float | None
    if below:
        laycost = -COSTSCALE * math.log(DEFOLAYCONST)  # = weight * glay, weight=1
        dzmax = math.ceil(DEFOMAX * NSHORTCYCLE)
        if dzmax < math.floor(math.sqrt(laycost * sigsq)):  # NOCOSTSHELF guard
            laycost, dzmax = None, None
    else:
        laycost, dzmax = None, None
    return sigsq, laycost, dzmax, thresh


def snaphu_cost(
    dphi_rad: np.ndarray, rho: float, nlooks: float, mode: str
) -> np.ndarray:
    """Total arc cost g(dphi) for the unwrapped gradient dphi (radians)."""
    sigsq, laycost, dzmax, _ = snaphu_params(rho, nlooks, mode)
    idz = np.abs(dphi_rad) / TAU * NSHORTCYCLE  # short-cycle units
    if laycost is None or dzmax is None:
        return idz**2 / sigsq
    return np.where(
        idz <= dzmax,
        np.minimum(idz**2 / sigsq, laycost),
        (idz - dzmax) ** 2 / (LAYFALLOFFCONST * sigsq) + laycost,
    )


# --------------------------------------------------------------------------
# whirlwind Carballo/Touzi cost = -log(phase-gradient noise PDF)
# --------------------------------------------------------------------------
def whirlwind_gradient_pdf(
    gamma_hat: float,
    nlooks: float,
    *,
    phase_samples: int = 8192,
    gamma_quad: int = 257,
    slope_window: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """PDF of the phase-gradient (arc phase-difference) noise, incl. slope + coherence marg.

    Mirrors ``generate_carballo_tables.gradient_noise_cdf`` (self-convolved Lee
    PDF with Touzi coherence marginalization), then applies the Carballo slope
    marginalization as a convolution with the slope-error Gaussian - the same
    two marginalizations the shipping LUTs use.
    """
    phi = np.linspace(-math.pi, math.pi, phase_samples, endpoint=False)
    phi += math.pi / phase_samples
    dphi = TAU / phase_samples

    pdf = gct.phase_pdf_given_sample_coherence(
        phi,
        gamma_hat,
        nlooks,
        gamma_eps=1e-6,
        gamma_quad=gamma_quad,
        marginalize_coherence=True,
    )
    pdf = pdf / max(pdf.sum() * dphi, 1e-300)

    # difference PDF on [-2pi, 2pi] (Lee PDF is symmetric => self-convolution)
    grad = fftconvolve(pdf, pdf[::-1], mode="full") * dphi
    x = np.arange(-(phase_samples - 1), phase_samples) * dphi
    grad = np.maximum(np.nan_to_num(grad), 0.0)
    grad /= max(grad.sum() * dphi, 1e-300)

    # slope marginalization: convolve with N(0, sigma), sigma from Carballo eq. 15
    sigma = gct.slope_error_sigma(gamma_hat, slope_window)
    if sigma > 1e-6:
        kern = np.exp(-0.5 * (x / sigma) ** 2)
        kern /= max(kern.sum() * dphi, 1e-300)
        grad = fftconvolve(grad, kern, mode="same") * dphi
        grad = np.maximum(grad, 0.0)
        grad /= max(grad.sum() * dphi, 1e-300)
    return x, grad


def whirlwind_cost(dphi_rad: np.ndarray, gamma_hat: float, nlooks: float) -> np.ndarray:
    x, pdf = whirlwind_gradient_pdf(gamma_hat, nlooks)
    p = np.interp(dphi_rad, x, pdf, left=np.nan, right=np.nan)
    cost = -np.log(p)
    return cost - np.nanmin(cost)


def whirlwind_weight(gamma_hat: float, nlooks: float) -> float:
    """Fixed per-arc MCF weight round(100*max(-ln(p1/p0), 0)) at wrapped gradient 0.

    p0/p1 are the residual-0 and residual-(+/-1) probabilities the LUT stores. The
    solver applies this single weight linearly to the arc flow (cost = w*|k|), so
    the cost never depends on the current flow - that is what keeps the problem a
    fast fixed-integer min-cost-flow.
    """
    x, cdf = gct.gradient_noise_cdf(
        gamma_hat,
        nlooks,
        phase_samples=8192,
        gamma_eps=1e-6,
        gamma_quad=257,
        marginalize_coherence=True,
    )

    def prob(residual: int) -> float:
        return gct.marginalized_probability(
            x,
            cdf,
            0.0,
            gamma_hat,
            residual,
            alpha_quad=1001,
            slope_tail_sigma=8.0,
            slope_window_pixels=64,
            use_slope_marginalization=True,
        )

    return round(100.0 * max(-math.log(prob(1) / prob(0)), 0.0))


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/figures/cost_functions_snaphu_vs_whirlwind.png"),
    )
    ap.add_argument("--nlooks", type=float, default=16.0)
    args = ap.parse_args()
    L = args.nlooks

    cohs = [0.2, 0.5, 0.8]
    colors = ["#c0392b", "#2e86c1", "#1e8449"]

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- Panel A: reproduce Chen & Zebker Fig. 11 with real snaphu defaults ---
    xr = np.linspace(-1.5 * TAU, 1.5 * TAU, 2001)
    lo = 0.1  # a low-coherence arc (below the defo threshold) -> gets the shelf
    hi = 0.5  # a higher-coherence arc (above threshold) -> pure parabola
    axA.plot(
        xr,
        snaphu_cost(xr, lo, L, "defo"),
        "k-",
        lw=2,
        label=f"discontinuity expected (rho={lo})",
    )
    axA.plot(
        xr,
        snaphu_cost(xr, hi, L, "defo"),
        "k--",
        lw=1.5,
        label=f"not expected (rho={hi})",
    )
    _, glay, dzmax, _ = snaphu_params(lo, L, "defo")
    assert glay is not None and dzmax is not None  # lo is below the defo threshold
    dphimax = dzmax / NSHORTCYCLE * TAU
    axA.axhline(glay, color="grey", ls=":", lw=0.8)
    axA.annotate(
        "$g_d$",
        xy=(0.15 * TAU, glay),
        xytext=(0.15 * TAU, glay + 3),
        ha="center",
        fontsize=11,
    )
    for s in (-1, 1):
        axA.axvline(s * dphimax, color="grey", ls=":", lw=0.8)
    axA.annotate(
        r"$\Delta\phi_{max}$",
        xy=(dphimax, 2),
        xytext=(dphimax, 20),
        ha="center",
        fontsize=10,
    )
    axA.set_ylim(0, 32)
    axA.set_title(f"A. SNAPHU DEFO  (Chen & Zebker Fig. 11, L={L:g})")
    axA.legend(fontsize=8, loc="upper center")

    # --- Panel B: SMOOTH (solid) vs DEFO (dashed) across coherence ---
    xr2 = np.linspace(-4.0, 4.0, 2001)
    for c, col in zip(cohs, colors):
        axB.plot(
            xr2,
            snaphu_cost(xr2, c, L, "smooth"),
            "-",
            color=col,
            lw=1.8,
            label=rf"$\rho$={c}",
        )
        axB.plot(xr2, snaphu_cost(xr2, c, L, "defo"), "--", color=col, lw=1.4)
    _, glayB, _, _ = snaphu_params(cohs[0], L, "defo")
    if glayB is not None:  # low-coherence arc: DEFO caps the parabola at the shelf
        axB.annotate(
            f"$g_d$ (DEFO shelf, $\\rho$={cohs[0]})",
            xy=(2.6, glayB),
            xytext=(0.3, glayB + 26),
            fontsize=8,
            ha="left",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"),
        )
    axB.set_ylim(0, 80)
    axB.set_title(f"B. SNAPHU SMOOTH (solid) vs DEFO (dashed), L={L:g}")
    axB.legend(fontsize=8, loc="upper center", title="smooth")

    # --- Panel C: whirlwind's actual arc cost - a fixed weight, LINEAR in flow ---
    ks = np.arange(-2, 3)
    for c, col in zip(cohs, colors):
        w = whirlwind_weight(c, L)
        axC.plot(
            ks,
            np.abs(ks) * w,
            "o-",
            color=col,
            ms=6,
            lw=1.6,
            label=rf"$\hat\gamma$={c}  ($w$={w:.0f})",
        )
    axC.set_xlabel(r"arc flow $k$ (2$\pi$ cycles)")
    axC.set_ylabel("whirlwind arc cost (shipping units)")
    axC.set_xticks(ks)
    axC.axvline(0, color="0.85", lw=0.8, zorder=0)
    axC.set_ylim(bottom=0)
    axC.set_title(f"C. whirlwind arc cost: L={L:g}")
    axC.legend(fontsize=8, loc="upper center")

    for ax in (axA, axB):
        ax.set_xlabel("unwrapped phase gradient (rad)")
        ax.set_ylabel("cost (snaphu units)")
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        secx = ax.secondary_xaxis(
            "top", functions=(lambda r: r / TAU, lambda c: c * TAU)
        )
        secx.set_xlabel("(cycles)", fontsize=8)

    fig.suptitle(
        "Arc cost functions: SNAPHU's convex, flow-dependent statistical cost "
        "vs. whirlwind linear costs",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
