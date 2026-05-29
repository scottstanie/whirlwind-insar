"""Prototype the thin-line heal in Python on the saved NISAR result, before
porting to Rust. A pixel is a 1px ghost strip if BOTH opposite neighbors
(left+right, or up+down) agree it is off by the SAME nonzero integer #cycles;
snap it. Can't trigger on real fringes (there left/right offsets differ).
Coherence-gated (skip <min_coh). Measures mainland K-match + ghost count +
visualizes a smooth-region line before/after.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

N = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
OUT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/outputs")
PLOTS = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/plots")
TAU = float(2 * np.pi)


def modal(d):
    d = d[np.isfinite(d)].astype(np.int64)
    return int(np.bincount(d - d.min()).argmax() + d.min())


def kround(a):
    return np.round(a / TAU)


def heal(unw, valid, iters=6):
    u = unw.copy()
    total = 0
    for _ in range(iters):
        fix = np.zeros(u.shape, np.float32)
        done = np.zeros(u.shape, bool)
        # horizontal strip: left and right neighbors agree on same nonzero k
        kl = kround(u[:, :-2] - u[:, 1:-1])   # (left - center) for j in 1..n-2
        kr = kround(u[:, 2:] - u[:, 1:-1])    # (right - center)
        vc = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
        gh = vc & (kl == kr) & (kl != 0)
        fix[:, 1:-1] = np.where(gh, kl * TAU, fix[:, 1:-1])
        done[:, 1:-1] |= gh
        # vertical strip: up and down agree on same nonzero k (only where not already fixed)
        ku = kround(u[:-2, :] - u[1:-1, :])
        kd = kround(u[2:, :] - u[1:-1, :])
        vc2 = valid[1:-1, :] & valid[:-2, :] & valid[2:, :]
        gv = vc2 & (ku == kd) & (ku != 0)
        gv &= ~done[1:-1, :]
        fix[1:-1, :] = np.where(gv, ku * TAU, fix[1:-1, :])
        done[1:-1, :] |= gv
        n = int(done.sum())
        if n == 0:
            break
        u += fix
        total += n
    return u, total


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coh = rasterio.open(N / "20251224_20260117.int.coh.looked.cleaned.tif").read(1).astype(np.float32)
    mask = np.load(OUT / "nisar_anchor_mask.npy")
    wrapped = np.load(OUT / "nisar_anchor_wrapped.npy")
    sk = np.load(OUT / "nisar_anchor_sk.npy")
    scc = np.load(OUT / "nisar_anchor_scc.npy")
    unw = np.load(OUT / "nisar_cascade_unw.npy")
    mainland = (scc == 1) & mask

    def mp(u):
        k = kround(u - wrapped); k[~mask] = np.nan
        d = (k - sk)[mainland]; d = d[np.isfinite(d)]; d = d - modal(d)
        return float((d == 0).sum())/d.size*100, float((np.abs(d) >= 2).sum())/d.size*100

    valid = mask & np.isfinite(unw) & (coh > 0.2)
    healed, nfix = heal(unw, valid)
    print(f"healed {nfix:,} ghost-strip pixels (coh>0.2)", flush=True)
    print(f"mainland match: before={mp(unw)[0]:.3f}%  after={mp(healed)[0]:.3f}%  "
          f"(|dK|>=2 {mp(unw)[1]:.3f} -> {mp(healed)[1]:.3f})", flush=True)
    np.save(OUT / "nisar_healed_unw.npy", healed.astype(np.float32))

    # Visualize a smooth-region line: find a column with many ghost pixels in a
    # smooth (low local-gradient) area. Use the per-column ghost count.
    kl = kround(unw[:, :-2] - unw[:, 1:-1]); kr = kround(unw[:, 2:] - unw[:, 1:-1])
    vc = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
    gh = vc & (kl == kr) & (kl != 0)
    gcol = gh.sum(axis=0)
    j = int(np.argmax(gcol)) + 1
    print(f"worst ghost column j={j} ({int(gcol[j-1])} ghost px)", flush=True)
    rows = np.where(gh[:, j-1])[0]
    r0 = max(0, int(np.median(rows)) - 200); r1 = min(unw.shape[0], r0 + 400)
    c0 = max(0, j-60); c1 = min(unw.shape[1], j+60)
    cr = lambda a: a[r0:r1, c0:c1]
    mk = cr(mainland)
    b = np.where(mk, cr(unw) - np.nanmedian(unw[mainland]), np.nan)
    a = np.where(mk, cr(healed) - np.nanmedian(healed[mainland]), np.nan)
    vlo, vhi = np.nanpercentile(b[np.isfinite(b)], [2, 98])
    fig, axes = plt.subplots(1, 3, figsize=(15, 9))
    axes[0].imshow(cr(np.where(mainland, coh, np.nan)), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"coherence @ col {j}", fontsize=11)
    axes[1].imshow(b, cmap="twilight", vmin=vlo, vmax=vhi, interpolation="nearest")
    axes[1].set_title("before heal", fontsize=11)
    axes[2].imshow(a, cmap="twilight", vmin=vlo, vmax=vhi, interpolation="nearest")
    axes[2].set_title("after heal", fontsize=11)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"NISAR thin-line heal @ col {j}", fontsize=13)
    fig.tight_layout()
    out = PLOTS / "nisar_heal_proto.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
