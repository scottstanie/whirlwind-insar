"""Verification figure for the #35 CRLB anchor/cascade refactor on the Capella
Mexico City scene. Renders K-fields (round((unw-wrapped)/2π), masked to each
method's conncomp) for: spurt reference, coherence path, CRLB-before (BFS-median,
saved as crlb_eval/before.npz), CRLB-after (anchor/cascade, after.npz), plus the
after−before delta (what the anchor/cascade changed). Lets you eyeball that the
cut-rate / K-match table entries are reasonable.

    env -u CONDA_PREFIX uv run --with rasterio --with matplotlib \
        python scripts/plot_crlb_mexico.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

D = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/mexico_city/e2e_output/dolphin"
)
IGDIR = D / "interferograms"
UNWDIR = D / "unwrapped"
EVAL = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/crlb_eval")
STEM = "20240626_20240629"
STRIDE = 3  # plot downsample
TAU = float(2 * np.pi)


def rd(p, dt):
    import rasterio

    with rasterio.open(p) as s:
        return s.read(1).astype(dt)


def kfield(unw, wrapped, cc):
    k = np.round((unw - wrapped) / TAU)
    k = np.where((cc > 0) & np.isfinite(unw), k, np.nan)
    k = k - np.nanmedian(k)
    return k


def main():
    import matplotlib.pyplot as plt
    import whirlwind as ww

    ig = np.ascontiguousarray(rd(IGDIR / f"{STEM}.int.tif", np.complex64))
    coh = np.ascontiguousarray(rd(IGDIR / f"{STEM}.int.cor.tif", np.float32))
    mask = np.isfinite(coh) & (coh > 0) & (np.abs(ig) > 0)
    wrapped = np.angle(np.where(mask, ig, 0)).astype(np.float32)

    spurt = rd(UNWDIR / f"{STEM}.unw.tif", np.float32)
    spurt_cc = rd(UNWDIR / f"{STEM}.unw.conncomp.tif", np.float32)

    bef = np.load(EVAL / "before.npz")
    aft = np.load(EVAL / "after.npz")
    coh_unw, coh_cc = ww.unwrap(ig, coh, 1.0, mask=mask, goldstein_alpha=0)

    s = (slice(None, None, STRIDE), slice(None, None, STRIDE))
    panels = [
        (
            "wrapped phase",
            np.where(mask, wrapped, np.nan)[s],
            "twilight",
            (-np.pi, np.pi),
        ),
        ("spurt K (ref)", kfield(spurt, wrapped, spurt_cc)[s], "viridis", None),
        ("coherence-cost K", kfield(coh_unw, wrapped, coh_cc)[s], "viridis", None),
        (
            "CRLB before K (BFS-median)",
            kfield(bef["unw"], wrapped, bef["cc"])[s],
            "viridis",
            None,
        ),
        (
            "CRLB after K (anchor/cascade)",
            kfield(aft["unw"], wrapped, aft["cc"])[s],
            "viridis",
            None,
        ),
        (
            "CRLB after − before (cycles)",
            (np.round((aft["unw"] - bef["unw"]) / TAU))[s],
            "RdBu",
            (-3, 3),
        ),
    ]
    # Shared K color range (robust percentiles over the K panels).
    kvals = np.concatenate([p[1][np.isfinite(p[1])].ravel() for p in panels[1:5]])
    klo, khi = np.nanpercentile(kvals, [2, 98])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for ax, (name, arr, cmap, lim) in zip(axes.ravel(), panels):
        kw = dict(cmap=cmap, interpolation="nearest")
        if lim is not None:
            kw.update(vmin=lim[0], vmax=lim[1])
        elif cmap == "viridis":
            kw.update(vmin=klo, vmax=khi)
        im = ax.imshow(arr, **kw)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(
        f"Capella Mexico City {STEM} - CRLB #35 verification (stride {STRIDE})"
    )
    EVAL.mkdir(parents=True, exist_ok=True)
    out = EVAL / "crlb_mexico_verification.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"plot: {out}")


if __name__ == "__main__":
    main()
