"""Distinguish decorrelated-noise fragments from coherent islands.

For each tear-component in the whirlwind unwrapped result, compute its size AND
its mean coherence. Decorrelated speckle = low coherence (it tears because phase
is random, not because it's a self-consistent disconnected region). A real island
= high coherence cluster that happens to be spatially separated. The min-size floor
should be set so it removes the former without the latter; but the cleaner lever is
coherence, not just size. We characterise the size/coherence joint distribution to
pick an absolute floor that is below real islands but above the noise-speckle bulk.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def tear_labels(unw, mask):
    u = unw.astype(np.float32)
    m = mask & np.isfinite(u)
    h, w = u.shape
    idx = np.full((h, w), -1, np.int64)
    nnodes = int(m.sum())
    idx[m] = np.arange(nnodes)
    rows, cols = [], []
    a = m[:, :-1] & m[:, 1:] & (np.abs(u[:, :-1] - u[:, 1:]) < np.pi)
    rows.append(idx[:, :-1][a]); cols.append(idx[:, 1:][a])
    b = m[:-1, :] & m[1:, :] & (np.abs(u[:-1, :] - u[1:, :]) < np.pi)
    rows.append(idx[:-1, :][b]); cols.append(idx[1:, :][b])
    r = np.concatenate(rows); c = np.concatenate(cols)
    g = coo_matrix((np.ones(r.size, np.uint8), (r, c)), shape=(nnodes, nnodes))
    ncomp, lab = connected_components(g, directed=False)
    full = np.full((h, w), -1, np.int64)
    full[m] = lab
    return ncomp, full, m


def analyze(name, unw, coh, mask):
    print(f"\n===== {name} =====")
    ncomp, lab, m = tear_labels(unw, mask)
    labvals = lab[m]
    cohvals = coh[m].astype(np.float64)
    sizes = np.bincount(labvals, minlength=ncomp)
    csum = np.bincount(labvals, weights=cohvals, minlength=ncomp)
    cmean = np.divide(csum, sizes, out=np.zeros_like(csum), where=sizes > 0)
    # bucket by size, show coherence stats per bucket
    edges = [1, 5, 10, 25, 50, 100, 200, 500, 1000, 5000, 10**9]
    print(f"  global coh: p10={np.percentile(cohvals,10):.3f} p50={np.percentile(cohvals,50):.3f} "
          f"p90={np.percentile(cohvals,90):.3f}")
    print("  size-bucket   ncomp   meancoh(of-comp-means)  median  frac-comps-coh>0.5")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sizes >= lo) & (sizes < hi)
        sel[0] = False  # skip background if present
        n = int(sel.sum())
        if n == 0:
            print(f"   [{lo:>5},{hi if hi<10**8 else 'inf':>5}): n=0")
            continue
        cm = cmean[sel]
        frac_hi = float((cm > 0.5).mean())
        print(f"   [{lo:>5},{str(hi) if hi<10**8 else 'inf':>6}): n={n:>6}  "
              f"mean={cm.mean():.3f}  median={np.median(cm):.3f}  coh>0.5frac={frac_hi:.2f}")
    # of small fragments (size<100), how many are HIGH coherence (real islands)?
    for thr in (0.4, 0.5, 0.6):
        small_isl = (sizes >= 10) & (sizes < 100) & (cmean > thr)
        small_isl[0] = False
        mid_isl = (sizes >= 100) & (sizes < 5000) & (cmean > thr)
        mid_isl[0] = False
        print(f"  coh>{thr}: islands 10-100px n={int(small_isl.sum())}, "
              f"100-5000px n={int(mid_isl.sum())} (these are the ones 1% drops!)")


def main():
    d = np.load('/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/a016_diag/a016_default_fixed.npz')
    analyze('A_016 whirlwind', d['unw'], d['coh'], d['mask'])

    g = ('NISAR_L2_PR_GUNW_003_005_A_013_004_4000_SH_20251017T124836_'
         '20251017T124857_20251029T124836_20251029T124858_X05010_N_P_J_001')
    base = '/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_gunw_reuse/' + g
    dc = np.load(base + '/full_arrays.npz')
    analyze('CLEAN A_013 whirlwind', dc['ww_unw'], dc['coh'], dc['mask'])


if __name__ == '__main__':
    main()
