"""A/B: ``ww.unwrap`` accuracy with and without ``WHIRLWIND_COH_BIAS_CORRECT``.

Runs the same uniform-coherence noisy ramps that ``bench.py`` uses (γ ∈
{0.3, 0.5, 0.7, 0.9} × L ∈ {5, 20}) but reports RMSE-vs-truth and integer
cycle errors instead of wall-time. Invocation:

    uv run python scripts/coh_bias_ab.py        # raw sample coherence
    WHIRLWIND_COH_BIAS_CORRECT=1 \\
        uv run python scripts/coh_bias_ab.py    # Touzi-style bias correction

The point: bias correction is a *real* concept (sample γ̂ is biased upward
on noisy pixels, the LUT-fed Lee PDF wants true γ) but the closed-form
correction we implemented `γ_corr² = max(0, (Lγ̂² − 1)/(L − 1))` floors γ
to 0 whenever `γ̂ < √(1/L)`. On uniform-low-coherence scenes that wipes
out the cost gradient MCF needs for routing, making unwrap noticeably
worse. See the per-scene numbers below and the discussion in
``paper/binary_vs_continuous.md``.

This script is reproducible evidence supporting the "leave it default-off"
decision recorded in ``cost/mod.rs::coh_bias_correct_enabled``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

import whirlwind_rs as ww


def make_scenes(seed: int = 42) -> list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray]]:
    scenes = []
    for gamma in (0.3, 0.5, 0.7, 0.9):
        for nlooks in (5, 20):
            n = 256
            x, y = np.meshgrid(np.arange(n), np.arange(n))
            # Several wrap lines crossing the scene.
            truth = (np.pi * 0.05 * (x + y)).astype(np.float32)
            g = np.full(truth.shape, gamma, dtype=np.float32)
            igram, corr = ww.simulate_ifg(truth, g, nlooks=nlooks, seed=seed)
            scenes.append((gamma, nlooks, igram, corr, truth))
    return scenes


def metric(unw: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    """RMSE + cycle-error count, after global integer-cycle alignment."""
    diff = (truth - unw)[np.isfinite(unw)]
    if diff.size == 0:
        return float("nan"), 0
    k = int(np.round(float(np.median(diff)) / (2 * np.pi)))
    err = unw + 2 * np.pi * k - truth
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err ** 2))), int(np.sum(np.abs(err) > np.pi))


def main() -> None:
    flag = os.environ.get("WHIRLWIND_COH_BIAS_CORRECT", "")
    print(f"# WHIRLWIND_COH_BIAS_CORRECT={flag!r}")
    print(f"  {'γ':>4}  {'L':>3}  {'RMSE':>8}  {'cycle_err':>9}")
    for gamma, nlooks, ig, cor, truth in make_scenes():
        unw = ww.unwrap(ig, cor, float(nlooks))
        rmse, errs = metric(unw, truth)
        print(f"  {gamma:>4.2f} {nlooks:>3}  {rmse:>8.3f}  {errs:>9d}")


if __name__ == "__main__":
    main()
    sys.exit(0)
