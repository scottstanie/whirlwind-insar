"""Report figures for the no-Goldstein tiled+anchor+cascade unwrapper.

For each scene (NISAR, Atlanta) produces a 3-row figure:
  Row 1  K = round((unw-wrapped)/2pi):  reference | whirlwind | |dK| error
  Row 2  unwrapped phase:               reference | whirlwind | (ww - ref) diff
  Row 3  connected components:          reference (native) | whirlwind (computed) | coverage

Whirlwind conncomp is computed from the unwrapped result: two valid neighbours
are in the same component iff |d(unw)| < pi (no 2pi tear). Computed on a
stride-downsampled grid via scipy.sparse connected components (the conncomp
panel is downsampled anyway), then components relabelled largest-first.

Reads the .npy arrays saved by run_nisar_anchor.py / run_nisar_cascade.py /
run_atlanta_anchor.py. Usage:  make_report_figures.py [nisar] [atlanta]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
TAU = float(2 * np.pi)


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def ww_conncomp(unw, mask, stride):
    """Connected components of the unwrapped surface (tear = |d unw| >= pi),
    computed on a stride-downsampled grid. Labels: 0 = invalid, 1 = largest."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    u = unw[::stride, ::stride].astype(np.float32)
    m = mask[::stride, ::stride] & np.isfinite(u)
    h, w = u.shape
    idx = np.full((h, w), -1, np.int64)
    idx[m] = np.arange(int(m.sum()))
    nnodes = int(m.sum())
    if nnodes == 0:
        return np.zeros((h, w), np.int32)

    rows, cols = [], []
    # right edges
    a = m[:, :-1] & m[:, 1:] & (np.abs(u[:, :-1] - u[:, 1:]) < np.pi)
    ia, ib = idx[:, :-1][a], idx[:, 1:][a]
    rows.append(ia); cols.append(ib)
    # down edges
    b = m[:-1, :] & m[1:, :] & (np.abs(u[:-1, :] - u[1:, :]) < np.pi)
    ja, jb = idx[:-1, :][b], idx[1:, :][b]
    rows.append(ja); cols.append(jb)
    r = np.concatenate(rows); c = np.concatenate(cols)
    g = coo_matrix((np.ones(r.size, np.uint8), (r, c)), shape=(nnodes, nnodes))
    ncomp, lab = connected_components(g, directed=False)

    # relabel largest-first
    counts = np.bincount(lab)
    order = np.argsort(counts)[::-1]
    remap = np.zeros(ncomp, np.int32)
    for new, old in enumerate(order, start=1):
        remap[old] = new
    out = np.zeros((h, w), np.int32)
    out[m] = remap[lab]
    return out


def comp_show(cc, ncolors=12):
    """Map a conncomp label image to a small palette (largest-first), 0->nan."""
    disp = np.full(cc.shape, np.nan, np.float32)
    sel = cc > 0
    disp[sel] = ((cc[sel] - 1) % ncolors) + 1
    return disp


def plot_scene(name, ref_unw, ref_cc, ww_unw, wrapped, mask, kref, title, stride, dpi=130):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    region = mask & (ref_cc == _main_label(ref_cc))
    kww = np.round((ww_unw - wrapped) / TAU)
    kww[~mask] = np.nan
    c = modal((kww - kref)[region])
    kww_c = kww - c
    match = float((np.abs((kww_c - kref)[region]) < 0.5).sum()) / int(region.sum()) * 100
    # Honest full-image match (same mainland-aligned global offset). The full
    # number is much lower than mainland because cc<1 / low-coh pixels are
    # per-pixel-noisy and the reference itself is uncertain there.
    fm = (kww_c - kref)[mask]
    full_match = float((np.abs(fm[np.isfinite(fm)]) < 0.5).sum()) / int(np.isfinite(fm).sum()) * 100

    sk_disp = kref.astype(np.float32).copy(); sk_disp[~mask] = np.nan
    lo, hi = np.nanpercentile(sk_disp[mask], [1, 99])
    ds = lambda a: a[::stride, ::stride]

    def ref_phase():
        u = ref_unw.astype(np.float32).copy(); u[~mask] = np.nan
        return u - np.nanmedian(ref_unw[region])

    def ww_phase():
        # Median-center only: the two fields are equal up to a global cycle
        # constant, which the median subtraction removes — do NOT also subtract
        # TAU*c (that double-counts the constant and blue-shifts the display).
        u = ww_unw.astype(np.float32).copy(); u[~mask] = np.nan
        return u - np.nanmedian(ww_unw[region])

    pr, pw = ref_phase(), ww_phase()
    plo, phi = np.nanpercentile(pr[mask], [1, 99])

    fig, axes = plt.subplots(3, 3, figsize=(19, 17))
    # Row 1: K
    for ax, (t, k, cmap, vlo, vhi) in zip(axes[0], [
            ("reference K (SNAPHU/OPERA)", sk_disp, "twilight", lo, hi),
            (f"whirlwind K  ({match:.2f}% mainland / {full_match:.1f}% full image)", kww_c, "twilight", lo, hi),
            ("|dK| error vs reference (mainland)", None, "inferno", 0, 3)]):
        if k is None:
            e = np.abs(kww_c - kref).astype(np.float32); e[~region] = np.nan
            im = ax.imshow(ds(e), vmin=vlo, vmax=vhi, cmap=cmap, interpolation="nearest")
        else:
            im = ax.imshow(ds(k), vmin=vlo, vmax=vhi, cmap=cmap, interpolation="nearest")
        ax.set_title(t, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    # Row 2: phase
    for ax, (t, u) in zip(axes[1], [("reference unwrapped phase", pr),
                                    ("whirlwind unwrapped phase", pw)]):
        im = ax.imshow(ds(u), vmin=plo, vmax=phi, cmap="twilight", interpolation="nearest")
        ax.set_title(t, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    d = pw - pr
    d -= TAU * round(float(np.nanmedian(d[region])) / TAU)  # strip any residual global cycle
    im = axes[1, 2].imshow(ds(d), vmin=-TAU, vmax=TAU, cmap="RdBu_r", interpolation="nearest")
    rms = float(np.sqrt(np.nanmean((d[region])**2)))
    axes[1, 2].set_title(f"whirlwind - reference  (mainland RMS={rms:.2f} rad)", fontsize=12)
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.02)
    # Row 3: conncomp
    print(f"  [{name}] computing whirlwind conncomp (stride {stride})...", flush=True)
    wwcc = ww_conncomp(ww_unw, mask, stride)
    n_ww = int(wwcc.max())
    refcc_d = ref_cc[::stride, ::stride].astype(np.float32)
    n_ref = int(np.unique(ref_cc[ref_cc > 0]).size)
    for ax, (t, c_disp) in zip(axes[2], [
            (f"reference conncomp ({n_ref} comps)", comp_show(refcc_d.astype(np.int32))),
            (f"whirlwind conncomp ({n_ww} comps, largest-first)", comp_show(wwcc))]):
        im = ax.imshow(c_disp, cmap="tab20", interpolation="nearest", vmin=0, vmax=12)
        ax.set_title(t, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    cov = mask[::stride, ::stride].astype(np.float32)
    axes[2, 2].imshow(cov, cmap="Greys_r", interpolation="nearest")
    axes[2, 2].set_title(f"valid input mask ({100*mask.mean():.1f}%)", fontsize=12)
    axes[2, 2].set_xticks([]); axes[2, 2].set_yticks([])

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / f"report_{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)
    return match


_main_label_cache = {}
def _main_label(cc):
    key = id(cc)
    if key not in _main_label_cache:
        labels, counts = np.unique(cc[cc > 0], return_counts=True)
        _main_label_cache[key] = int(labels[np.argmax(counts)]) if labels.size else 0
    return _main_label_cache[key]


def do_nisar():
    sk = np.load(OUT / "nisar_anchor_sk.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy").astype(np.int32)
    mask = np.load(OUT / "nisar_anchor_mask.npy")
    wrapped = np.load(OUT / "nisar_anchor_wrapped.npy")
    # Current default path: tiled512+anchor+cascade + bounded sliver cleanup.
    wwu = np.load(OUT / "nisar_tileconvex_linear_unw.npy")
    sunw = sk * TAU + wrapped
    plot_scene("nisar", sunw, scc, wwu, wrapped, mask, sk,
               "NISAR no-Goldstein: whirlwind tiled+anchor+cascade+cleanup (6s) vs SNAPHU 9x9 (17 min)",
               stride=4)


def do_atlanta():
    # Current honest noisy-scene path: multilook=8 (coherent averaging suppresses
    # the noise the fine solve can't route), reference arrays from run_atlanta_report.py.
    kref = np.load(OUT / "atlanta_rep_kref.npy")
    cc = np.load(OUT / "atlanta_rep_cc.npy").astype(np.int32)
    mask = np.load(OUT / "atlanta_rep_mask.npy")
    wrapped = np.load(OUT / "atlanta_rep_wrapped.npy")
    wwu = np.load(OUT / "atlanta_rep_unw.npy")
    refu = np.nan_to_num(kref) * TAU + wrapped
    plot_scene("atlanta", refu, cc, wwu, wrapped, mask, np.nan_to_num(kref),
               "Atlanta S-1 OPERA: whirlwind multilook=8 + tiled+anchor+cascade vs OPERA/SNAPHU reference",
               stride=4)


if __name__ == "__main__":
    which = sys.argv[1:] or ["nisar", "atlanta"]
    if "nisar" in which:
        do_nisar()
    if "atlanta" in which:
        do_atlanta()
