"""2-level vs N-level pyramid: when does the full cascade beat a single jump?

Answers the open question from ``paper/pyramid_aliasing.md``: is the N-level
cascade (base -> base/2 -> ... -> 1) worth its machinery over a 2-level scheme
(coarse solve at `base`, then ONE residual pass straight to full resolution)?

Both share the coarse solve and the residual-against-prediction trick; they
differ only in whether the refinement steps by octaves or jumps. Each level's
residual is solved with the reuse base solver (corner-safe).

Finding (see the doc): they tie on clean data and in mild noise, but in the
**extreme-noise** regime the N-level cascade wins decisively, and by more the
larger the base. Mechanism: 2-level's single full-resolution residual pass sees
the FULL per-pixel noise (its effective looks are not multiplied), so under heavy
noise it drowns exactly like plain full-res. N-level's intermediate residual
solves run on down-looked grids with effective looks scaled by f^2, so each
residual stays unwrappable and the prediction handed to the next octave is clean.
A single base->full jump skips that progressive denoising.

    uv run python scripts/pyramid_2_vs_n.py

Deterministic; ~1 min. Synthetic Goodman-noise cones (see the doc's caveats).
"""

from __future__ import annotations

import numpy as np

import whirlwind as ww

TWOPI = 2 * np.pi
FN = {"reuse": ww.unwrap_reuse, "convex": ww.unwrap_convex, "linear": ww.unwrap}


def kpct(u, t):
    d = u - t
    d -= TWOPI * round(float(np.median(d)) / TWOPI)
    return float(np.mean(np.round(d / TWOPI) == 0)) * 100


def cone(shape, g):
    m, n = shape
    ci, cj = (m - 1) / 2, (n - 1) / 2
    i, j = np.ogrid[:m, :n]
    return (g * np.sqrt((i - ci) ** 2 + (j - cj) ** 2)).astype(np.float32)


def ml_cplx(ig, f):
    m, n = ig.shape
    cm, cn = m // f, n // f
    return ig[: cm * f, : cn * f].reshape(cm, f, cn, f).mean(axis=(1, 3))


def ml_coh(coh, f):
    m, n = coh.shape
    cm, cn = m // f, n // f
    return coh[: cm * f, : cn * f].reshape(cm, f, cn, f).mean(axis=(1, 3))


def upsample(c, m, n):
    cm, cn = c.shape
    yi = np.clip((np.arange(m) + 0.5) * cm / m - 0.5, 0, cm - 1)
    xi = np.clip((np.arange(n) + 0.5) * cn / n - 0.5, 0, cn - 1)
    y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, cm - 1); wy = (yi - y0)[:, None]
    x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, cn - 1); wx = (xi - x0)[None, :]
    top = c[y0][:, x0] * (1 - wx) + c[y0][:, x1] * wx
    bot = c[y1][:, x0] * (1 - wx) + c[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def _solve(z, coh, nlooks, solver):
    zc = z / np.where(np.abs(z) > 0, np.abs(z), 1)
    return FN[solver](zc.astype(np.complex64), coh.astype(np.float32), nlooks=float(nlooks))


def two_level(ig, coh, base, nlooks, solver="reuse"):
    m, n = ig.shape
    cu = _solve(ml_cplx(ig, base), ml_coh(coh, base), nlooks * base * base, solver)
    pred = upsample(cu, m, n)
    resid = _solve(ig * np.exp(-1j * pred), coh, nlooks, solver)  # full-res, full noise
    return pred + resid


def n_level(ig, coh, base, nlooks, solver="reuse"):
    m, n = ig.shape
    factors = []
    f = base
    while f > 1:
        factors.append(f); f //= 2
    factors.append(1)
    prev = None
    for f in factors:
        z = ml_cplx(ig, f) if f > 1 else ig
        c = ml_coh(coh, f) if f > 1 else coh
        cm, cn = z.shape
        if prev is None:
            prev = _solve(z, c, nlooks * f * f, solver)
        else:
            pred = upsample(prev, cm, cn)
            resid = _solve(z * np.exp(-1j * pred), c, nlooks * f * f, solver)
            prev = pred + resid
    return prev


def main():
    shape = (256, 256)
    t = cone(shape, 0.08 * np.pi)  # gentle: unaliased even at base=16 (16*0.08pi<pi)
    print("solver=reuse, gentle cone g=0.08π, mean over 3 seeds.")
    print(f"{'base':>4} {'gamma':>6} {'looks':>6} {'full':>6} {'2lvl':>6} {'Nlvl':>6} {'N-2':>6}")
    for base in [8, 16]:
        for gamma, nlooks in [(0.6, 4), (0.18, 4), (0.15, 4), (0.12, 4), (0.12, 2)]:
            a2, aN, af = [], [], []
            for seed in range(3):
                ig, coh = ww.simulate_ifg(t, np.full(shape, gamma, np.float32), nlooks, seed)
                ig, coh = ig.astype(np.complex64), coh.astype(np.float32)
                a2.append(kpct(two_level(ig, coh, base, nlooks), t))
                aN.append(kpct(n_level(ig, coh, base, nlooks), t))
                af.append(kpct(ww.unwrap(ig, coh, nlooks=float(nlooks)), t))
            m2, mN, mf = np.mean(a2), np.mean(aN), np.mean(af)
            flag = "  <-- N>2 wins" if mN > m2 + 3 else ""
            print(f"{base:>4} {gamma:>6.2f} {nlooks:>6} {mf:>6.1f} {m2:>6.1f} {mN:>6.1f} {mN - m2:>+6.1f}{flag}")


if __name__ == "__main__":
    main()
