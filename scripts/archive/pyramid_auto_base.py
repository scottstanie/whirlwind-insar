"""Is the pyramid's one knob (`base`) auto-selectable, or another SNAPHU-style
trial-and-error parameter?

The argument that it IS auto-selectable: unlike SNAPHU's many interacting knobs,
the pyramid has essentially ONE knob with a *physical, measurable* ceiling
(`base·g < π`, the coarsest-level Nyquist limit), and it degrades
asymmetrically - too-small `base` only loses some denoising, too-large `base`
aliases. So "pick the largest non-aliasing base" is a single well-posed
estimation problem.

This script tests that empirically over a (steepness g, coherence γ) grid by
comparing, per cell, the K-correct of:

  oracle   : best base in {1,2,4,8,16} for that cell  [unknowable in practice]
  probe    : largest base the Itoh-violation-rate probe calls unaliased (the
             trend rule from whirlwind_core::pyramid::auto_base_factor)
  fixed=B  : a single hand-chosen default

and reports each strategy's *regret* = oracle_K − strategy_K. Small probe regret
⇒ auto-selection works; large fixed-base regret ⇒ no single default suffices.

    uv run python scripts/pyramid_auto_base.py

Deterministic, ~3 min. Synthetic Goodman-noise cones; g spans 0.05π–0.45π (the
probe's near-Nyquist constant-ramp blind spot at g≳0.8π is out of range here and
is discussed in paper/pyramid_aliasing.md).
"""

from __future__ import annotations

import itertools

import numpy as np

import whirlwind as ww

TWOPI = 2 * np.pi
BASES = [1, 2, 4, 8, 16]
SHAPE = (256, 256)
NLOOKS = 4
GFRACS = [0.05, 0.1, 0.2, 0.3, 0.45]
GAMMAS = [0.7, 0.3, 0.15]
SEEDS = 3


def kpct(u, t):
    d = u - t
    d -= TWOPI * round(float(np.median(d)) / TWOPI)
    return float(np.mean(np.round(d / TWOPI) == 0)) * 100


def cone(s, g):
    m, n = s
    ci, cj = (m - 1) / 2, (n - 1) / 2
    i, j = np.ogrid[:m, :n]
    return (g * np.sqrt((i - ci) ** 2 + (j - cj) ** 2)).astype(np.float32)


def ml_c(ig, f):
    m, n = ig.shape
    cm, cn = m // f, n // f
    return ig[: cm * f, : cn * f].reshape(cm, f, cn, f).mean(axis=(1, 3))


def ml_h(coh, f):
    m, n = coh.shape
    cm, cn = m // f, n // f
    return coh[: cm * f, : cn * f].reshape(cm, f, cn, f).mean(axis=(1, 3))


def up(c, m, n):
    cm, cn = c.shape
    yi = np.clip((np.arange(m) + 0.5) * cm / m - 0.5, 0, cm - 1)
    xi = np.clip((np.arange(n) + 0.5) * cn / n - 0.5, 0, cn - 1)
    y0 = np.floor(yi).astype(int)
    y1 = np.minimum(y0 + 1, cm - 1)
    wy = (yi - y0)[:, None]
    x0 = np.floor(xi).astype(int)
    x1 = np.minimum(x0 + 1, cn - 1)
    wx = (xi - x0)[None, :]
    top = c[y0][:, x0] * (1 - wx) + c[y0][:, x1] * wx
    bot = c[y1][:, x0] * (1 - wx) + c[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def _solve(z, coh, nlooks):
    zc = z / np.where(np.abs(z) > 0, np.abs(z), 1)
    return ww.unwrap_reuse(
        zc.astype(np.complex64), coh.astype(np.float32), nlooks=float(nlooks)
    )


def n_level(ig, coh, base, nlooks):
    if base == 1:
        return _solve(ig, coh, nlooks)
    fs = []
    f = base
    while f > 1:
        fs.append(f)
        f //= 2
    fs.append(1)
    prev = None
    for f in fs:
        z = ml_c(ig, f) if f > 1 else ig
        c = ml_h(coh, f) if f > 1 else coh
        cm, cn = z.shape
        if prev is None:
            prev = _solve(z, c, nlooks * f * f)
        else:
            pred = up(prev, cm, cn)
            prev = pred + _solve(z * np.exp(-1j * pred), c, nlooks * f * f)
    return prev


def itoh_rate(ig, f, thr=0.6 * np.pi):
    ph = np.angle(ml_c(ig, f))
    dr = np.abs((ph[1:, :] - ph[:-1, :] + np.pi) % TWOPI - np.pi)
    dc = np.abs((ph[:, 1:] - ph[:, :-1] + np.pi) % TWOPI - np.pi)
    return (np.sum(dr > thr) + np.sum(dc > thr)) / (dr.size + dc.size)


def probe_base(ig, maxf=16):
    """Trend rule mirroring whirlwind_core::pyramid::auto_base_factor: keep
    doubling while the violation rate is below a benign FLOOR or still falling."""
    floor, decr = 0.05, 0.02
    m, n = ig.shape
    best, prev, f = 1, itoh_rate(ig, 1), 2
    while f <= maxf and m // f >= 4 and n // f >= 4:
        r = itoh_rate(ig, f)
        if r < floor or r <= prev - decr:
            best, prev, f = f, r, f * 2
        else:
            break
    return best


def main():
    print(f"reuse, {SHAPE}, nlooks={NLOOKS}, mean over {SEEDS} seeds.")
    header = f"{'g/π':>5} {'γ':>5} | {'oracle':>13} | {'probe':>12} | " + " ".join(
        f"fix{b:<4}" for b in BASES
    )
    print(header)
    regret = {("probe",): []} | {("fix", b): [] for b in BASES}
    worst = dict(regret)
    worst = {k: 0.0 for k in regret}
    for gfrac, gamma in itertools.product(GFRACS, GAMMAS):
        t = cone(SHAPE, gfrac * np.pi)
        kby = {b: [] for b in BASES}
        pbs = []
        for seed in range(SEEDS):
            ig, coh = ww.simulate_ifg(
                t, np.full(SHAPE, gamma, np.float32), NLOOKS, seed
            )
            ig, coh = ig.astype(np.complex64), coh.astype(np.float32)
            for b in BASES:
                kby[b].append(kpct(n_level(ig, coh, b, NLOOKS), t))
            pbs.append(probe_base(ig))
        mk = {b: float(np.mean(kby[b])) for b in BASES}
        ok = max(mk.values())
        ob = max(BASES, key=lambda b: mk[b])
        pb = int(round(np.median(pbs)))
        regret[("probe",)].append(ok - mk[pb])
        worst[("probe",)] = max(worst[("probe",)], ok - mk[pb])
        for b in BASES:
            regret[("fix", b)].append(ok - mk[b])
            worst[("fix", b)] = max(worst[("fix", b)], ok - mk[b])
        print(
            f"{gfrac:>5.2f} {gamma:>5.2f} | base={ob:<2d} K={ok:5.1f} | base={pb:<2d} K={mk[pb]:5.1f} | "
            + " ".join(f"{mk[b]:5.1f}" for b in BASES)
        )

    print("\nregret vs oracle (K-points; lower is better):")
    print(
        f"  {'probe':8s} mean={np.mean(regret[('probe',)]):5.1f}  worst={worst[('probe',)]:5.1f}"
    )
    for b in BASES:
        print(
            f"  fix={b:<5d} mean={np.mean(regret[('fix', b)]):5.1f}  worst={worst[('fix', b)]:5.1f}"
        )


if __name__ == "__main__":
    main()
