"""Atlanta report figure: the failure (fine tiled, vertical stripes, 26%) and
the fix (multilook-8 + tiled+anchor+cascade, 97.66%) vs the OPERA reference.
Rows: K fields | phase + conncomp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = float(2 * np.pi)
S = 4


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def pad_to(a, shape):
    out = np.full(shape, np.nan, np.float32)
    h = min(a.shape[0], shape[0])
    w = min(a.shape[1], shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def ww_conncomp(unw, mask, stride):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    u = unw[::stride, ::stride].astype(np.float32)
    m = mask[::stride, ::stride] & np.isfinite(u)
    h, w = u.shape
    idx = np.full((h, w), -1, np.int64)
    idx[m] = np.arange(int(m.sum()))
    nn = int(m.sum())
    if nn == 0:
        return np.zeros((h, w), np.int32)
    a = m[:, :-1] & m[:, 1:] & (np.abs(u[:, :-1] - u[:, 1:]) < np.pi)
    b = m[:-1, :] & m[1:, :] & (np.abs(u[:-1, :] - u[1:, :]) < np.pi)
    r = np.concatenate([idx[:, :-1][a], idx[:-1, :][b]])
    c = np.concatenate([idx[:, 1:][a], idx[1:, :][b]])
    g = coo_matrix((np.ones(r.size, np.uint8), (r, c)), shape=(nn, nn))
    ncomp, lab = connected_components(g, directed=False)
    counts = np.bincount(lab)
    order = np.argsort(counts)[::-1]
    remap = np.zeros(ncomp, np.int32)
    for new, old in enumerate(order, 1):
        remap[old] = new
    out = np.zeros((h, w), np.int32)
    out[m] = remap[lab]
    return out, ncomp


def comp_show(cc, ncolors=12):
    d = np.full(cc.shape, np.nan, np.float32)
    sel = cc > 0
    d[sel] = ((cc[sel] - 1) % ncolors) + 1
    return d


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kref = np.load(OUT / "atlanta_kref.npy")
    cc = np.load(OUT / "atlanta_cc.npy").astype(np.int32)
    mask = np.load(OUT / "atlanta_mask.npy")
    wrapped = np.load(OUT / "atlanta_wrapped.npy")
    fine = np.load(OUT / "atlanta_anchor_unw.npy")  # fine tiled (26%)
    ml8 = pad_to(
        np.load(OUT / "atlanta_ml8api_unw.npy"), mask.shape
    )  # multilook=8 API, 97.7%
    refu = np.nan_to_num(kref) * TAU + wrapped
    labels, counts = np.unique(cc[cc > 0], return_counts=True)
    mainland = mask & (cc == int(labels[np.argmax(counts)]))

    def kf(unw):
        k = np.round((unw - wrapped) / TAU)
        k[~mask] = np.nan
        d = (k - kref)[mainland]
        return k - modal(d[np.isfinite(d)])

    kfine, kml = kf(fine), kf(ml8)

    def m_(k):
        d = (k - kref)[mainland]
        d = d[np.isfinite(d)]
        return float((np.abs(d) < 0.5).sum()) / d.size * 100

    sd = kref.astype(np.float32).copy()
    sd[~mask] = np.nan
    lo, hi = np.nanpercentile(sd[mask], [1, 99])
    ds = lambda a: a[::S, ::S]

    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    for ax, (t, k) in zip(
        axes[0],
        [
            ("OPERA/SNAPHU reference K", sd),
            (f"whirlwind fine tiled  ({m_(kfine):.1f}% - FAILS on noise)", kfine),
            (f"whirlwind multilook-8 + tiled  ({m_(kml):.1f}%)", kml),
        ],
    ):
        im = ax.imshow(
            ds(k), vmin=lo, vmax=hi, cmap="twilight", interpolation="nearest"
        )
        ax.set_title(t, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    def ph(u):
        x = u.astype(np.float32).copy()
        x[~mask] = np.nan
        return x - np.nanmedian(u[mainland])

    pr = ph(refu)
    pw = ph(
        ml8
    )  # median-centering aligns the fields; no extra TAU*c (would double-count)
    plo, phi = np.nanpercentile(pr[mask], [1, 99])
    im = axes[1, 0].imshow(
        ds(pr), vmin=plo, vmax=phi, cmap="twilight", interpolation="nearest"
    )
    axes[1, 0].set_title("OPERA unwrapped phase", fontsize=12)
    im2 = axes[1, 1].imshow(
        ds(pw), vmin=plo, vmax=phi, cmap="twilight", interpolation="nearest"
    )
    axes[1, 1].set_title("whirlwind multilook-8 unwrapped phase", fontsize=12)
    for ax, im_ in [(axes[1, 0], im), (axes[1, 1], im2)]:
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im_, ax=ax, fraction=0.046, pad=0.02)
    wwcc, ncomp = ww_conncomp(ml8, mask, S)
    nref = int(np.unique(cc[cc > 0]).size)
    axes[1, 2].imshow(
        comp_show(wwcc), cmap="tab20", interpolation="nearest", vmin=0, vmax=12
    )
    axes[1, 2].set_title(
        f"whirlwind conncomp (largest-first)\nOPERA has {nref} comps", fontsize=12
    )
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    fig.suptitle(
        "Atlanta S-1 OPERA: whirlwind fine-tiled FAILS on noise (26%); multilook-8 + tiled recovers it (97.7%, matches SNAPHU 97.9%)",
        fontsize=14,
    )
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "report_atlanta_ml8.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
