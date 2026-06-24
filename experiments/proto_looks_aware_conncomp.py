#!/usr/bin/env python3
"""Plot a looks-aware conncomp coherence floor against the noise floor.

A coherence value means different things at different looks: the sample coherence
is biased high, and the noise floor -- the coherence estimated from uncorrelated
data -- falls as 1/sqrt(L). A fixed cutoff does not track it.

Plots the candidate floor 0.32/sqrt(L) (= 0.08 at 16 looks) against the simulated
noise floor and against SNAPHU's 1.25*(1.3/L + 0.14), and prints a table. Writes
gunw_results/proto_looks_aware/min_coh_vs_looks.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("gunw_results/proto_looks_aware")

# Calibrated so min_coh(16) == 0.08 (the validated default) and it scales like
# the coherence noise floor (~1/sqrt(L)). 0.32 = 0.08 * sqrt(16).
GENTLE_K = 0.32


def min_coh_for_looks(nlooks: float) -> float:
    """Gentle, looks-aware conncomp coherence floor. ~0.08 at L=16."""
    return float(np.clip(GENTLE_K / np.sqrt(nlooks), 0.02, 0.30))


def snaphu_rho0(nlooks: float) -> float:
    """SNAPHU's zero-correlation cutoff: 1.25 * (1.3/L + 0.14)."""
    return 1.25 * (1.3 / nlooks + 0.14)


def simulate_noise_floor(nlooks: int, ntrials: int = 20000, rng_seed: int = 0) -> float:
    """Mean |gamma_hat| from uncorrelated data (true coherence 0) at `nlooks`.

    Two independent complex-Gaussian series; mean sample-coherence magnitude.
    """
    rng = np.random.default_rng(rng_seed + nlooks)
    a = rng.standard_normal((ntrials, nlooks)) + 1j * rng.standard_normal(
        (ntrials, nlooks)
    )
    b = rng.standard_normal((ntrials, nlooks)) + 1j * rng.standard_normal(
        (ntrials, nlooks)
    )
    num = np.abs((a * np.conj(b)).sum(axis=1))
    den = np.sqrt((np.abs(a) ** 2).sum(axis=1) * (np.abs(b) ** 2).sum(axis=1))
    return float((num / den).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    looks = [2, 4, 8, 16, 32, 64, 128, 256]
    print(
        f"{'looks':>6} {'noise_floor':>12} {'gentle(ours)':>13} {'frac_of_floor':>14} {'SNAPHU':>8}"
    )
    floors, gentle, snaphu = [], [], []
    for L in looks:
        nf = simulate_noise_floor(L)
        g = min_coh_for_looks(L)
        s = snaphu_rho0(L)
        floors.append(nf)
        gentle.append(g)
        snaphu.append(s)
        print(f"{L:6d} {nf:12.3f} {g:13.3f} {g / nf:14.2f} {s:8.3f}")

    Lc = np.logspace(np.log10(2), np.log10(256), 100)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.plot(
        looks, floors, "ko", label="coherence noise floor  E[|γ̂| | γ=0]  (simulated)"
    )
    ax.plot(
        Lc,
        [min_coh_for_looks(L) for L in Lc],
        "-",
        lw=2,
        label=f"gentle looks-aware floor  {GENTLE_K}/√L  (ours, prototype)",
    )
    ax.plot(
        Lc,
        [snaphu_rho0(L) for L in Lc],
        "--",
        lw=2,
        label="SNAPHU  1.25·(1.3/L + 0.14)",
    )
    ax.axhline(0.08, color="gray", ls=":", lw=1)
    ax.axvline(16, color="gray", ls=":", lw=1)
    ax.plot([16], [0.08], "r*", ms=14, label="validated default (0.08 @ L=16)")
    ax.set_xscale("log")
    ax.set_xlabel("effective number of looks (L)")
    ax.set_ylabel("conncomp coherence floor")
    ax.set_title(
        "Conncomp coherence floor vs looks\n"
        "(0.32/sqrt(L) is a fraction of the noise floor; SNAPHU's cutoff is higher)"
    )
    ax.set_xticks(looks)
    ax.set_xticklabels(looks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    out = OUT / "min_coh_vs_looks.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out.resolve()}")
    print(
        f"\nSanity: min_coh_for_looks(16) = {min_coh_for_looks(16):.4f}  (== validated 0.08)"
    )


if __name__ == "__main__":
    main()
