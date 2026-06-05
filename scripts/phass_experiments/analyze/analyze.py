"""Tabulate and plot phass-experiment results.

Reads:
  <OUT>/<scene>_{baseline,hard_cut,phass_cost,phass_full}.npz
  <OUT>/pv_snaphu.npz
  (NISAR SNAPHU reference is read from .snaphu_9x9.{unw,cc}.tif in the
  input directory.)

Writes:
  <PLOTS>/<scene>_k_panel.png        side-by-side K-field comparison
  <OUT>/results.md                   markdown summary table

`OUT` and `PLOTS` resolve to
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments/{outputs,plots}/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

ROOT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/phass_experiments")
OUT = ROOT / "outputs"
PLOTS = ROOT / "plots"
NISAR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
PV = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/palos-verdes"
    "/Palos_Verdes_C13_RO23_SP/network_output/20251129_20251205"
)

MODES = ["baseline", "hard_cut", "hard_cut_lo", "phass_cost", "phass_full"]
LABELS = {
    "baseline": "default Carballo",
    "hard_cut": "Carballo + cut @2.0 rad",
    "hard_cut_lo": "Carballo + cut @1.0 rad",
    "phass_cost": "PHASS γ²",
    "phass_full": "PHASS γ² + cut @2.0",
}
TAU = np.float32(2 * np.pi)


def load_scene(scene: str):
    if scene == "nisar":
        with rasterio.open(NISAR / "20251224_20260117.int.looked.tif") as src:
            ig = src.read(1).astype(np.complex64)
        with rasterio.open(
            NISAR / "20251224_20260117.int.coh.looked.cleaned.tif"
        ) as src:
            coh = src.read(1).astype(np.float32)
        with rasterio.open(NISAR / "20251224_20260117.snaphu_9x9.unw.tif") as src:
            snaphu_unw = src.read(1).astype(np.float32)
        with rasterio.open(NISAR / "20251224_20260117.snaphu_9x9.cc.tif") as src:
            snaphu_cc = src.read(1).astype(np.uint32)
        wrapped = np.angle(ig).astype(np.float32)
        mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0) & (np.abs(ig) > 0)
        snaphu_k = np.round((snaphu_unw - wrapped) / TAU).astype(np.int32)
    elif scene == "pv":
        with rasterio.open(
            PV / "CAPELLA_C13_C13_SP_PHS_HH_20251129T183328_20251205T162657.tif"
        ) as src:
            wrapped = src.read(1).astype(np.float32)
        with rasterio.open(
            PV / "CAPELLA_C13_C13_SP_COH_HH_20251129T183328_20251205T162657.tif"
        ) as src:
            coh = src.read(1).astype(np.float32)
        ref = np.load(OUT / "pv_snaphu.npz")
        snaphu_unw = ref["unw"].astype(np.float32)
        snaphu_cc = ref["cc"].astype(np.uint32)
        snaphu_k = ref["k"].astype(np.int32)
        mask = np.isfinite(coh) & (coh > 0) & (coh < 1.0)
    else:
        raise ValueError(scene)
    return dict(
        wrapped=wrapped,
        coh=coh,
        mask=mask,
        snaphu_unw=snaphu_unw,
        snaphu_cc=snaphu_cc,
        snaphu_k=snaphu_k,
    )


def tabulate(scene: str, lines: list[str]) -> None:
    inp = load_scene(scene)
    snaphu_main = (inp["snaphu_cc"] == 1) & inp["mask"]
    n_main = int(snaphu_main.sum())
    lines.append(f"## {scene}")
    lines.append(
        f"shape={inp['wrapped'].shape}  "
        f"mask={int(inp['mask'].sum()):,}  "
        f"SNAPHU cc=1 mainland={n_main:,} px "
        f"({n_main/inp['wrapped'].size*100:.1f}% of frame)"
    )
    lines.append("")
    header = (
        "mode",
        "wall",
        "n_cc",
        "cov%",
        "shared",
        "K=match%",
        "|dK|=1%",
        "|dK|≥2%",
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    # Compare K on SNAPHU's cc=1 mainland (∩ whirlwind input mask), not on
    # whirlwind's conncomp. We want to measure unwrap *quality* on the land
    # area SNAPHU trusts, independent of how strict whirlwind's conncomp
    # threshold happened to be set.
    common_full = snaphu_main  # whirlwind's mask is the same input mask
    n_common = int(common_full.sum())
    for mode in MODES:
        path = OUT / f"{scene}_{mode}.npz"
        if not path.exists():
            lines.append(
                "| "
                + " | ".join([LABELS[mode], "—", "—", "—", "—", "—", "—", "—"])
                + " |"
            )
            continue
        d = np.load(path)
        cc = d["cc"]
        k_ww = d["k"].astype(np.int32)
        elapsed = float(d["elapsed"])
        dk = k_ww[common_full] - inp["snaphu_k"][common_full]
        center = int(np.bincount(dk - dk.min()).argmax() + dk.min())
        dk_c = dk - center
        m0 = float((dk_c == 0).sum()) / n_common * 100
        m1 = float((np.abs(dk_c) == 1).sum()) / n_common * 100
        m2 = float((np.abs(dk_c) >= 2).sum()) / n_common * 100
        lines.append(
            f"| {LABELS[mode]} | {elapsed:.1f}s | {int(cc.max())} | "
            f"{(cc>0).mean()*100:.2f} | {n_common:,} | "
            f"{m0:.2f} | {m1:.2f} | {m2:.2f} |"
        )
    lines.append("")


def plot_k_panel(scene: str) -> None:
    import matplotlib.pyplot as plt

    inp = load_scene(scene)
    panels = []
    panels.append(
        (
            "SNAPHU 9x9" if scene == "nisar" else "SNAPHU smooth",
            inp["snaphu_k"],
            inp["snaphu_cc"] > 0,
        )
    )
    for mode in MODES:
        path = OUT / f"{scene}_{mode}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        panels.append((LABELS[mode], d["k"].astype(np.int32), d["cc"] > 0))

    if not panels:
        return
    k_all = np.concatenate([p[1][p[2]] for p in panels if p[2].any()])
    if k_all.size == 0:
        return
    lo, hi = float(np.quantile(k_all, 0.005)), float(np.quantile(k_all, 0.995))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (label, k, valid) in zip(axes, panels):
        kp = k.astype(np.float32).copy()
        kp[~valid] = np.nan
        im = ax.imshow(kp, vmin=lo, vmax=hi, cmap="twilight", interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"{scene}: integer cycles K = round((unw − wrapped)/2π)", fontsize=11)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / f"{scene}_k_panel.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def main() -> None:
    lines: list[str] = ["# PHASS-cost experiment results", ""]
    for scene in ["pv", "nisar"]:
        try:
            tabulate(scene, lines)
        except FileNotFoundError as e:
            lines.append(f"## {scene}\n_missing input_: {e}\n")
            continue
        try:
            plot_k_panel(scene)
        except Exception as e:
            lines.append(f"_plot failed_: {e}\n")
    summary = OUT / "results.md"
    OUT.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines))
    print(f"wrote {summary}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
