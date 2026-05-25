"""Benchmark whirlwind-rs (Rust) vs whirlwind (C++) — the WW author asked.

Single-IG 2D unwrap only. Stage 2 (closure correction) has no C++ counterpart,
so it's out of scope here.

Comparison axes:
  - Speed (wall clock per unwrap)
  - Output agreement (max abs diff, fraction of pixels within 1e-3 rad)
  - Peak RSS (best effort via resource.getrusage)

Datasets:
  (1) Synthetic ramps at several sizes — pixel-perfect agreement expected
  (2) A small Palos Verdes tile of phase-linked IGs — realistic noise

Run:
    uv run python scripts/bench_vs_whirlwind_cpp.py
"""

from __future__ import annotations

import functools
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

import whirlwind          # the C++ implementation
import whirlwind_rs       # the Rust implementation

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    rasterio = None


DOLPHIN = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/"
    "palos-verdes/Palos_Verdes_C13_RO23_SP/e2e_output_20260519/dolphin"
)
SYNTHETIC_SIZES = [256, 512, 1024, 2048]
REAL_TILE = (1000, 1500, 2024, 2524)   # (i0, j0, i1, j1) — 1024×1024
REAL_MAX_IGS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    """Best-effort peak RSS in MiB. On macOS rusage returns bytes."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def time_unwrap(fn, *args, warmup_runs: int = 1, timing_runs: int = 3) -> tuple[float, float, np.ndarray]:
    """Run `fn(*args)` once for warmup, then `timing_runs` times. Return
    (median wall clock seconds, max RSS during measurement, last output)."""
    rss_before = rss_mb()
    last_out = None
    for _ in range(warmup_runs):
        last_out = fn(*args)
    timings = []
    for _ in range(timing_runs):
        t0 = time.perf_counter()
        last_out = fn(*args)
        timings.append(time.perf_counter() - t0)
    rss_after = rss_mb()
    return float(np.median(timings)), max(rss_before, rss_after), last_out


def compare(unw_cpp: np.ndarray, unw_rust: np.ndarray) -> dict:
    """Per-pixel agreement between two unwrapped outputs (mod 2π)."""
    valid = np.isfinite(unw_cpp) & np.isfinite(unw_rust)
    if not valid.any():
        return {"valid_pixels": 0}
    # Whirlwind defines unwrapped phase only up to a global integer multiple
    # of 2π. The two implementations may pick different global offsets, so
    # we compare modulo 2π.
    d = unw_cpp[valid] - unw_rust[valid]
    d_mod = np.angle(np.exp(1j * d))
    abs_mod = np.abs(d_mod)
    return {
        "valid_pixels":   int(valid.sum()),
        "max_diff_mod":   float(abs_mod.max()),
        "mean_diff_mod":  float(abs_mod.mean()),
        "rms_mod":        float(np.sqrt(np.mean(d_mod ** 2))),
        "pct_within_1e3": float(100 * np.mean(abs_mod < 1e-3)),
        "pct_within_1e1": float(100 * np.mean(abs_mod < 1e-1)),
    }


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------

def gen_noisy_bump(size: int, gamma: float = 0.7, nlooks: int = 4, seed: int = 0):
    """A noisy Gaussian-bump IG. Matches the existing tests/bench style."""
    rng = np.random.default_rng(seed)
    i = np.arange(size)[:, None] - size / 2
    j = np.arange(size)[None, :] - size / 2
    r2 = i ** 2 + j ** 2
    truth = 8.0 * np.exp(-r2 / ((size / 6) ** 2)).astype(np.float32)
    g = np.full((size, size), gamma, dtype=np.float32)
    # Lee 1994 noise: σ²_φ ≈ (1 - γ²) / (2 N γ²) for boxcar averaging
    sigma = np.sqrt((1 - g ** 2) / (2 * nlooks * g ** 2))
    noise = rng.standard_normal(truth.shape).astype(np.float32) * sigma
    igram = np.exp(1j * (truth + noise)).astype(np.complex64)
    return igram, g, truth


def synthetic_bench() -> list[dict]:
    results = []
    for size in SYNTHETIC_SIZES:
        print(f"\n=== synthetic {size}x{size} ===")
        igram, cor, _ = gen_noisy_bump(size)
        nlooks = 4.0
        t_cpp, _, unw_cpp = time_unwrap(whirlwind.unwrap, igram, cor, nlooks)
        t_rust, _, unw_rust = time_unwrap(whirlwind_rs.unwrap, igram, cor, nlooks)
        agree = compare(unw_cpp, unw_rust)
        print(f"  C++:   {t_cpp*1000:7.1f} ms")
        print(f"  Rust:  {t_rust*1000:7.1f} ms  ({t_cpp/t_rust:.2f}x speedup)")
        print(f"  agreement: max|Δ mod 2π| = {agree['max_diff_mod']:.4g}, "
              f"pct < 1e-3 = {agree['pct_within_1e3']:.2f}%")
        results.append({
            "case": f"synthetic_{size}",
            "size": size,
            "cpp_ms": t_cpp * 1000,
            "rust_ms": t_rust * 1000,
            "speedup": t_cpp / t_rust,
            "agree": agree,
        })
    return results


# ---------------------------------------------------------------------------
# Real-data benchmark (Palos Verdes)
# ---------------------------------------------------------------------------

def real_bench() -> list[dict]:
    if rasterio is None:
        print("\n[skipped] rasterio not available")
        return []
    if not DOLPHIN.exists():
        print(f"\n[skipped] dolphin dir not found: {DOLPHIN}")
        return []
    ig_dir = DOLPHIN / "interferograms"
    igs = sorted(ig_dir.glob("*.int.tif"))[:REAL_MAX_IGS]
    if not igs:
        print(f"\n[skipped] no IGs found in {ig_dir}")
        return []

    i0, j0, i1, j1 = REAL_TILE
    win = Window(j0, i0, j1 - j0, i1 - i0)  # type: ignore[call-arg]
    print(f"\n=== Palos Verdes real IGs ({i1-i0}x{j1-j0}, {len(igs)} IGs) ===")

    results = []
    for path in igs:
        stem = path.name.removesuffix(".int.tif")
        cor_path = ig_dir / f"{stem}.int.cor.tif"
        mask_path = ig_dir / f"{stem}.int.mask.tif"
        if not cor_path.exists():
            continue

        with rasterio.open(path) as src:
            igram = src.read(1, window=win)
            igram = np.nan_to_num(igram, nan=0.0).astype(np.complex64)
        with rasterio.open(cor_path) as src:
            cor = src.read(1, window=win).astype(np.float32)
            cor = np.nan_to_num(cor, nan=0.0).clip(0.0, 0.999)
        mask = None
        if mask_path.exists():
            with rasterio.open(mask_path) as src:
                mask = src.read(1, window=win).astype(bool)

        nlooks = 4.0
        # warmup=1, timing=1 to keep total runtime manageable
        t_cpp, _, unw_cpp = time_unwrap(
            whirlwind.unwrap, igram, cor, nlooks, warmup_runs=0, timing_runs=1
        )
        if mask is not None:
            t_rust, _, unw_rust = time_unwrap(
                whirlwind_rs.unwrap, igram, cor, nlooks, mask,
                warmup_runs=0, timing_runs=1,
            )
        else:
            t_rust, _, unw_rust = time_unwrap(
                whirlwind_rs.unwrap, igram, cor, nlooks,
                warmup_runs=0, timing_runs=1,
            )
        agree = compare(unw_cpp, unw_rust)
        print(f"  {stem[:25]}... cpp={t_cpp*1000:7.1f} ms, rust={t_rust*1000:7.1f} ms, "
              f"speedup={t_cpp/t_rust:.2f}x, agree pct<1e-3={agree['pct_within_1e3']:6.2f}%")
        results.append({
            "case": "real",
            "ig": stem,
            "shape": list(igram.shape),
            "cpp_ms": t_cpp * 1000,
            "rust_ms": t_rust * 1000,
            "speedup": t_cpp / t_rust,
            "agree": agree,
        })
    return results


def main() -> None:
    print(f"C++ Whirlwind: {whirlwind.__version__}")
    print("Rust Whirlwind: whirlwind_rs v0.1.0 (this repo)")
    print()
    syn = synthetic_bench()
    real = real_bench()

    out_path = Path("/tmp/whirlwind-bench.json")
    out_path.write_text(json.dumps({"synthetic": syn, "real": real}, indent=2))
    print(f"\nResults written to {out_path}")

    # Summary
    print("\n--- summary ---")
    if syn:
        speedups = [r["speedup"] for r in syn]
        print(f"synthetic ({len(syn)} sizes): median speedup {np.median(speedups):.2f}x, "
              f"max {max(speedups):.2f}x")
    if real:
        speedups = [r["speedup"] for r in real]
        agree = [r["agree"]["pct_within_1e3"] for r in real]
        print(f"real ({len(real)} IGs):       median speedup {np.median(speedups):.2f}x, "
              f"median pct<1e-3 agreement {np.median(agree):.2f}%")


if __name__ == "__main__":
    main()
