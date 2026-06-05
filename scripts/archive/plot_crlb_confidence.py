"""Plots for the #58 CRLB work:
  Row 1 - .cor-as-confidence vs pseudo-coherence on IG 20240626_20240629
          (from saved crlb_eval/after.npz [pseudo-coh] and after_cor.npz [.cor]).
  Row 2 - CRLB cost vs coherence cost on IG 20240626_20240705, the case where the
          coherence cost is 50x worse (cut-rate 3.3e-2 vs 6.4e-4) - re-run live.

    env -u CONDA_PREFIX uv run --with rasterio --with matplotlib \
        python scripts/plot_crlb_confidence.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

D = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/mexico_city/e2e_output/dolphin"
)
IGDIR = D / "interferograms"
EVAL = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/crlb_eval")
STRIDE = 3
TAU = float(2 * np.pi)


def rd(p, dt):
    import rasterio

    with rasterio.open(p) as s:
        return s.read(1).astype(dt)


def kfield(unw, wrapped, cc_or_mask):
    k = np.round((unw - wrapped) / TAU)
    k = np.where(cc_or_mask & np.isfinite(unw), k, np.nan)
    return k - np.nanmedian(k)


def main():
    import matplotlib.pyplot as plt
    import whirlwind as ww

    s = (slice(None, None, STRIDE), slice(None, None, STRIDE))

    # Row 1: IG 0629, pseudo-coh vs .cor confidence (saved npz).
    stem1 = "20240626_20240629"
    ig1 = rd(IGDIR / f"{stem1}.int.tif", np.complex64)
    m1 = np.abs(ig1) > 0
    w1 = np.angle(np.where(m1, ig1, 0)).astype(np.float32)
    a = np.load(EVAL / "after.npz")
    ac = np.load(EVAL / "after_cor.npz")
    kp = kfield(a["unw"], w1, a["cc"] > 0)
    kc = kfield(ac["unw"], w1, ac["cc"] > 0)
    dk = np.round((ac["unw"] - a["unw"]) / TAU)
    dk = np.where(
        (a["cc"] > 0) & (ac["cc"] > 0),
        dk - np.nanmedian(dk[(a["cc"] > 0) & (ac["cc"] > 0)]),
        np.nan,
    )

    # Row 2: IG 0705, CRLB vs coherence cost (live).
    stem2 = "20240626_20240705"
    a2, b2 = stem2.split("_")
    ig2 = np.ascontiguousarray(rd(IGDIR / f"{stem2}.int.tif", np.complex64))
    var2 = np.ascontiguousarray(
        np.nan_to_num(
            rd(IGDIR / f"crlb_{a2}.tif", np.float32)
            + rd(IGDIR / f"crlb_{b2}.tif", np.float32)
        ),
        np.float32,
    )
    cor2 = np.ascontiguousarray(
        np.clip(np.nan_to_num(rd(IGDIR / f"{stem2}.int.cor.tif", np.float32)), 0, 1),
        np.float32,
    )
    m2 = np.isfinite(cor2) & (cor2 > 0) & (np.abs(ig2) > 0) & (var2 > 0)
    w2 = np.angle(np.where(m2, ig2, 0)).astype(np.float32)
    crlb_unw, crlb_cc = ww.unwrap_crlb(ig2, var2, coherence=cor2)
    coh_unw, coh_cc = ww.unwrap(ig2, cor2, 1.0, mask=m2, goldstein_alpha=0)
    kr = kfield(crlb_unw, w2, crlb_cc > 0)
    kh = kfield(coh_unw, w2, coh_cc > 0)

    panels = [
        ("IG0629 CRLB, pseudo-coh conf", kp, "viridis", None),
        ("IG0629 CRLB, .cor conf (4x faster)", kc, "viridis", None),
        ("IG0629  (.cor − pseudo) cycles", dk, "RdBu", (-3, 3)),
        ("IG0705 CRLB cost (cut 6.4e-4)", kr, "viridis", None),
        ("IG0705 coherence cost (cut 3.3e-2)", kh, "viridis", None),
        (None, None, None, None),
    ]
    kall = np.concatenate(
        [
            p[1][np.isfinite(p[1])].ravel()
            for p in panels
            if p[1] is not None and p[2] == "viridis"
        ]
    )
    klo, khi = np.nanpercentile(kall, [2, 98])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for ax, (name, arr, cmap, lim) in zip(axes.ravel(), panels):
        if name is None:
            ax.axis("off")
            continue
        kw = dict(cmap=cmap, interpolation="nearest")
        if lim:
            kw.update(vmin=lim[0], vmax=lim[1])
        elif cmap == "viridis":
            kw.update(vmin=klo, vmax=khi)
        im = ax.imshow(arr[s], **kw)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(
        "CRLB #58: .cor-as-confidence (row 1) + CRLB-cost vs coherence-cost (row 2)"
    )
    out = EVAL / "crlb_confidence_verification.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"plot: {out}")


if __name__ == "__main__":
    main()
