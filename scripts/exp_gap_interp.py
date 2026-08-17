#!/usr/bin/env python3
"""Experiment: connect the NISAR subswath gaps before unwrapping.

ISCE3 production never bridges -- it distance-fills the invalid pixels (Chen
2015) so SNAPHU sees a connected frame -- and OPERA reportedly got decent
results from filling too. This re-solves cached ``compare_gunw`` frames with the
gaps crossed, so the gap treatment is the only thing that changes.

The arms, which are *not* equivalent:

``nearest``
    each gap pixel takes its nearest valid phasor (the ISCE3 / Chen 2015 fill).
``linear``
    complex-linear interpolation between the two edges. Integrating through this
    recovers the SHORT-ARC phase difference -- the same integer minimum-jump
    bridging already picks -- so it is the null hypothesis, expected to change
    little.
``connect_gaps``
    the shipped ``whirlwind.unwrap(..., connect_gaps=True)``: continue the fringe
    rate from both sides into the gap and blend, so the number of fringes
    crossing the gap is preserved rather than minimised.

``nearest`` and ``linear`` are implemented locally, on purpose -- they are
deliberately-inferior baselines that exist to show why the shipped rule is
shaped the way it is, and they are not part of the library. ``connect_gaps``
calls the public API with no local reimplementation, so what this figure shows
is what a caller actually gets.

Inputs come from the cached ``full_arrays.npz``, so no re-download and no HDF5:
the wrapped phase, coherence and mask are exactly what the original run solved.
Note ``npz["ig"]`` is the wrapped PHASE (float32), not a complex interferogram.

Usage::

    python scripts/exp_gap_interp.py ww_gunw_out --out nisar-pngs/gap_interp.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from exp_gap_relevel import _load_compare_gunw, _runs, short  # noqa: E402

TWOPI = 2.0 * np.pi


def interior_gaps(mask: np.ndarray, max_gap: int) -> list[tuple[int, int, int]]:
    """(row, start, stop) of every interior gap narrow enough to fill."""
    out = []
    for i in range(mask.shape[0]):
        runs = _runs(mask[i])
        for (_a1, b1), (a2, _b2) in zip(runs, runs[1:]):
            if 2 <= a2 - b1 <= max_gap:
                out.append((i, b1, a2))
    return out


def baseline_fill(
    wrapped: np.ndarray,
    mask: np.ndarray,
    method: str,
    max_gap: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """A deliberately-inferior gap fill, for contrast with ``connect_gaps``.

    Local to this experiment; see the module docstring. Takes wrapped PHASE and
    returns ``(complex igram with gaps filled, bool map of filled pixels)``.
    """
    z = np.exp(1j * np.where(mask, np.nan_to_num(wrapped), 0.0)).astype(np.complex64)
    z[~mask] = 0
    filled = np.zeros(mask.shape, dtype=bool)
    gaps = interior_gaps(mask, max_gap)

    if method == "nearest":
        _d, idx = ndimage.distance_transform_edt(~mask, return_indices=True)
        for i, b1, a2 in gaps:
            filled[i, b1:a2] = True
        z[filled] = z[idx[0][filled], idx[1][filled]]
        return z, filled

    if method != "linear":
        raise ValueError(f"unknown baseline fill {method!r}")

    for i, b1, a2 in gaps:
        gap = a2 - b1
        xs = np.arange(1, gap + 1, dtype=np.float64)
        zl, zr = z[i, b1 - 1], z[i, a2]
        t = xs / (gap + 1)
        vals = (1 - t) * zl + t * zr
        nz = np.abs(vals) > 1e-6
        vals = np.where(nz, vals / np.where(nz, np.abs(vals), 1.0), 1.0 + 0j)
        z[i, b1:a2] = vals.astype(np.complex64)
        filled[i, b1:a2] = True
    return z, filled


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("nisar-pngs/gap_interp.png"))
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--max-gap", type=int, default=300)
    p.add_argument(
        "--methods", nargs="*", default=["nearest", "linear", "connect_gaps"]
    )
    p.add_argument("--stride", type=int, default=4)
    args = p.parse_args()

    import whirlwind as ww
    from whirlwind import _SYNTHETIC_COHERENCE

    cg = _load_compare_gunw()
    hdr = (
        f"{'frame':<16} {'baseline':>9} "
        + " ".join(f"{m:>13}" for m in args.methods)
        + "   (per-comp vs production)"
    )
    print(hdr)
    print("-" * len(hdr))
    rows, panels = [], []

    for d in args.dirs:
        for sub in sorted(d.iterdir()):
            if not (sub / "full_arrays.npz").exists() or any(
                s in sub.name for s in args.skip
            ):
                continue
            z = np.load(sub / "full_arrays.npz")
            meta = json.loads((sub / "full.json").read_text())
            mask, coh, phase = z["mask"], z["coh"], z["ig"]
            nlooks = float(meta["nlooks"])
            coh_s = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0).astype(
                np.float32
            )
            igram = np.exp(1j * np.where(mask, np.nan_to_num(phase), 0.0)).astype(
                np.complex64
            )
            igram[~mask] = 0

            def score(unw):
                st, *_rest = cg.compute_compare_stats(
                    ig=phase,
                    coh=coh,
                    mask=mask,
                    prod_unw=z["prod_unw"],
                    prod_cc=z["prod_cc"],
                    ww_unw=np.where(mask, unw, 0.0).astype(np.float32),
                    ww_cc=None,
                    runtime_s=0.0,
                    rss_delta_mb=None,
                )
                return st

            base = meta["ambiguity_match_frac_percomp"]
            line, amb_maps = [base], []
            for method in args.methods:
                t0 = time.perf_counter()
                if method == "connect_gaps":
                    # The shipped path, called exactly as a user would.
                    unw, _cc = ww.unwrap(
                        igram,
                        coh_s,
                        nlooks,
                        mask,
                        connect_gaps=True,
                        connect_gaps_max_px=args.max_gap,
                    )
                else:
                    zf, filled = baseline_fill(
                        phase, mask, method, max_gap=args.max_gap
                    )
                    c2 = coh_s.copy()
                    c2[filled] = _SYNTHETIC_COHERENCE
                    unw, _cc = ww.unwrap(zf, c2, nlooks, mask | filled)
                dt = time.perf_counter() - t0
                st = score(np.asarray(unw, np.float32))
                line.append(st["ambiguity_match_frac_percomp"])
                amb_maps.append(
                    (method, st["ambiguity_match_frac_percomp"], _amb(z, mask, unw))
                )
                del unw
            rows.append((short(sub.name), line))
            panels.append((short(sub.name), base, z["ambiguity_diff"], amb_maps, mask))
            print(
                f"{short(sub.name):<16} {base:>9.4f} "
                + " ".join(f"{v:>13.4f}" for v in line[1:])
                + f"   [{dt:.0f}s/arm]"
            )

    a = np.array([r[1] for r in rows])
    print("-" * len(hdr))
    print(f"{'mean':<16} " + " ".join(f"{v:>13.4f}" for v in a.mean(axis=0)))

    s = args.stride
    ncol = 1 + len(args.methods)
    fig, axes = plt.subplots(
        len(panels),
        ncol,
        figsize=(4.4 * ncol, 4.2 * len(panels)),
        constrained_layout=True,
        squeeze=False,
    )
    for r, (name, base, amb0, amb_maps, mask) in enumerate(panels):
        for c, (tag, val, amb) in enumerate(
            [("bridge (no fill)", base, amb0)] + [(m, v, a_) for m, v, a_ in amb_maps]
        ):
            aa = np.where(mask[::s, ::s], amb[::s, ::s], np.nan)
            lim = max(1.0, float(np.nanpercentile(np.abs(aa), 99)))
            im = axes[r][c].imshow(
                aa, cmap="RdBu", vmin=-lim, vmax=lim, interpolation="nearest"
            )
            axes[r][c].set_title(f"{name}\n{tag}: {val:.3f}", fontsize=9)
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            fig.colorbar(im, ax=axes[r][c], shrink=0.8)
    fig.suptitle(
        "Connecting the subswath gaps before unwrapping "
        "(0 = agrees with production)",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nWrote {args.out.resolve()}")


def _amb(z, mask, unw):
    """Recompute the ambiguity-difference map for plotting."""
    phase = z["ig"]
    prod = z["prod_unw"]
    u = np.where(mask, np.asarray(unw, np.float32), 0.0)
    off = int(np.rint(np.nanmedian((u[mask] - prod[mask]) / TWOPI)))
    return np.rint((u - off * TWOPI - phase) / TWOPI) - np.rint((prod - phase) / TWOPI)


if __name__ == "__main__":
    main()
