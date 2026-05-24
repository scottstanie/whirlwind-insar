"""Plotting helpers for whirlwind-rs results."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_wrapped_unwrapped_png(
    wrapped: np.ndarray,
    unwrapped: np.ndarray,
    path: str | Path,
    *,
    title: str | None = None,
    cor: np.ndarray | None = None,
) -> None:
    """Save a 2- or 3-panel PNG: wrapped phase, unwrapped phase, (optional) coherence."""
    import matplotlib.pyplot as plt

    n_panels = 2 + (cor is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), constrained_layout=True)
    if title:
        fig.suptitle(title)

    ax = axes[0]
    im = ax.imshow(wrapped, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    ax.set_title("wrapped phase")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(unwrapped, cmap="viridis", interpolation="nearest")
    ax.set_title("unwrapped phase")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if cor is not None:
        ax = axes[2]
        im = ax.imshow(cor, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title("coherence")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
