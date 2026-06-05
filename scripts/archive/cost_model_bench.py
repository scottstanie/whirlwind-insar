"""Cost-model speed + accuracy benchmark: linear vs reuse vs convex.

Substantiates the "Why the better cost is better, and when" / "Cost of the
better cost" tables in ``paper/pyramid_aliasing.md``. Runs single-threaded so
the comparison is the *cost model*, not parallelism.

  uv run python scripts/cost_model_bench.py

Deterministic (seed 0). Synthetic Goodman-noise cones - see the doc's caveats
about i.i.d. noise vs real spatially-correlated scenes.
"""

from __future__ import annotations

import time

import numpy as np

import whirlwind as ww


def cone(shape, g):
    m, n = shape
    ci, cj = (m - 1) / 2, (n - 1) / 2
    i, j = np.ogrid[:m, :n]
    return (g * np.sqrt((i - ci) ** 2 + (j - cj) ** 2)).astype(np.float32)


def k_correct(u, t):
    d = u - t
    d -= 2 * np.pi * round(float(np.median(d)) / (2 * np.pi))
    return float(np.mean(np.round(d / (2 * np.pi)) == 0))


def res_count(ig):
    return int((ww.compute_residues(np.angle(ig).astype(np.float32)) != 0).sum())


def _timed(fn, *a):
    s = time.perf_counter()
    fn(*a)
    return time.perf_counter() - s


SOLVERS = [
    ("linear", lambda ig, co: ww.unwrap(ig, co, nlooks=8.0)[0]),
    ("reuse", lambda ig, co: ww.unwrap_reuse(ig, co, nlooks=8.0)),
    ("convex", lambda ig, co: ww.unwrap_convex(ig, co, nlooks=8.0)),
]


def row(ig, co, truth, label, extra=""):
    res = {}
    for name, fn in SOLVERS:
        fn(ig, co)  # warm up
        dt = min(_timed(fn, ig, co) for _ in range(3))
        res[name] = (dt, k_correct(fn(ig, co), truth))
    base = res["linear"][0]
    cells = "  ".join(
        f"{n}={res[n][0] * 1000:7.1f}ms({res[n][0] / base:.2f}x,K={res[n][1] * 100:.0f})"
        for n, _ in SOLVERS
    )
    print(f"{label:28s}{extra}  {cells}")


def main():
    ww.set_num_threads(1)
    print(f"threads={ww.num_threads()}\n--- size sweep (mild cone, gamma=0.7) ---")
    for sz in [256, 512, 1024]:
        t = cone((sz, sz), 0.15 * np.pi)
        ig, co = ww.simulate_ifg(t, np.full((sz, sz), 0.7, np.float32), 8, 0)
        row(ig.astype(np.complex64), co.astype(np.float32), t, f"{sz}x{sz}")

    print("--- noise sweep (512^2, mild cone) ---")
    for gamma in [0.9, 0.6, 0.4, 0.3]:
        t = cone((512, 512), 0.15 * np.pi)
        ig, co = ww.simulate_ifg(t, np.full((512, 512), gamma, np.float32), 8, 0)
        ig = ig.astype(np.complex64)
        row(
            ig,
            co.astype(np.float32),
            t,
            f"gamma={gamma:.1f}",
            extra=f" res={res_count(ig):6d}",
        )


if __name__ == "__main__":
    main()
