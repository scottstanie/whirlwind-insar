"""Compare whirlwind's bridge post-pass against isce3's bridge_unwrapped_phase
(the NISAR GUNW workflow's bridging) on a single frame.

The per-component agreement metric used elsewhere aligns each production
component independently, so it CANNOT see a wrong inter-region 2pi offset - the
exact thing bridging fixes. Here we score the *absolute* agreement: after
removing one global cycle offset (taken on the largest integration region), what
fraction of valid pixels land on the same integer cycle as the production unwrap.
That number drops precisely when disconnected regions are bridged to the wrong
relative level.

Runs ONE whirlwind solve (bridge=False) and then applies, in Python:
  - raw            : no bridging
  - ww-bridge      : whirlwind's _bridge_components (coarse 8x anchor)
  - isce3-bridge   : isce3.unwrap.bridge_unwrapped_phase (MST, local endpoints),
                     with the NISAR GUNW default knobs.

Usage: python scripts/diag_bridge_isce3_compare.py [FRAME=A_016]
Run in the mapping-312 env (has isce3 + whirlwind).
"""
import glob
import sys

import h5py
import numpy as np

sys.path.insert(0, "scripts")
from tophu_compare import gunw_layers, water_only_mask, wrap_phase, percomp_match

import whirlwind as ww
from whirlwind._bridge import _bridge_components
from isce3.unwrap.bridge_phase import bridge_unwrapped_phase

TWOPI = 2.0 * np.pi
H5DIR = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw"
# NISAR GUNW defaults (nisar/workflows/defaults/insar.yaml -> unwrap.bridge).
ISCE3_BRIDGE = dict(
    radius=500, min_num_pixel=14, erosion_size=2,
    ramp_type=None, deramp_max_num_sample=int(1e6),
)


def absolute_agreement(u, prod_unw, region, ref_lab, valid):
    """Fraction of valid pixels on the same integer cycle as production after
    removing ONE global offset (median ambiguity of the reference region)."""
    amb = np.rint((u - prod_unw) / TWOPI)
    ref = valid & (region == ref_lab)
    g = np.rint(np.median(amb[ref])) if ref.any() else 0.0
    return float(np.mean((amb[valid] - g) == 0))


def region_offsets(u, prod_unw, region, ref_lab, valid, min_px=500):
    """Per-integration-region cycle offset vs production, relative to the
    reference region. 0 = correctly bridged. Returns list of (lab, npx, off)."""
    amb = np.rint((u - prod_unw) / TWOPI)
    ref = valid & (region == ref_lab)
    g = np.rint(np.median(amb[ref])) if ref.any() else 0.0
    out = []
    for lab in range(1, int(region.max()) + 1):
        m = valid & (region == lab)
        npx = int(m.sum())
        if npx < min_px:
            continue
        off = int(np.rint(np.median(amb[m])) - g)
        out.append((lab, npx, off))
    return out


ALL_FRAMES = [
    "A_013", "A_016", "A_018", "A_020", "A_022", "A_025", "A_028",
    "A_030", "A_035", "D_074", "D_075", "D_077", "D_078",
]


def run_frame(frame):
    h5 = glob.glob(f"{H5DIR}/*_{frame}_*.h5")[0]
    with h5py.File(h5, "r") as h:
        pol, prod_unw, coh, prod_cc, mask_arr = gunw_layers(h)
    mask = (
        water_only_mask(mask_arr, prod_unw.shape)
        & np.isfinite(prod_unw)
        & np.isfinite(coh)
    )
    wrapped = np.where(mask, wrap_phase(prod_unw), 0.0).astype(np.float32)
    ig = np.exp(1j * wrapped).astype(np.complex64)
    coh_in = np.where(mask, np.clip(np.nan_to_num(coh), 0, 1), 0.0).astype(np.float32)

    # One heavy solve, no bridging.
    raw, _ = ww.unwrap(ig, coh_in, 16.0, mask, bridge=False)
    raw = np.asarray(raw, np.float32)

    # whirlwind's coarse-anchor bridge.
    ww_b = np.asarray(_bridge_components(raw.copy(), ig, coh_in, 16.0, mask), np.float32)

    # isce3's MST bridge on the same raw output (zero outside mask = its cluster mask).
    raw_z = np.where(mask, raw, 0.0).astype(np.float32)
    isce3_b = np.asarray(
        bridge_unwrapped_phase(raw_z, **ISCE3_BRIDGE), np.float32
    )

    region, n_region = ww.label_components(np.ascontiguousarray(mask))
    sizes = np.bincount(region.ravel())
    ref_lab = int(np.argmax(sizes[1:]) + 1)
    valid = mask & np.isfinite(prod_unw)

    print(f"{frame}: pol={pol} shape={mask.shape} integration-regions={n_region} "
          f"(>=500px: {int((sizes[1:] >= 500).sum())})", flush=True)
    print(f"{'method':12s} {'per-comp%':>9s} {'absolute%':>9s}  wrong-region-offsets",
          flush=True)
    for name, u in [("raw", raw), ("ww-bridge", ww_b), ("isce3-bridge", isce3_b)]:
        pc = percomp_match(u, prod_unw, wrapped, prod_cc, valid) * 100
        ab = absolute_agreement(u, prod_unw, region, ref_lab, valid) * 100
        offs = region_offsets(u, prod_unw, region, ref_lab, valid)
        wrong = [(lab, npx, o) for (lab, npx, o) in offs if o != 0]
        wrong_px = sum(npx for _, npx, _ in wrong)
        wrong_str = (
            f"{len(wrong)}/{len(offs)} regions, {wrong_px} px"
            + ("" if not wrong else "  " + ", ".join(
                f"L{lab}:{o:+d}({npx})" for lab, npx, o in sorted(
                    wrong, key=lambda t: -t[1])[:6]))
        )
        print(f"{name:12s} {pc:9.2f} {ab:9.2f}  {wrong_str}", flush=True)

    np.savez_compressed(
        f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/ww_4way_final/{frame}_bridge_compare.npz",
        wrapped=wrapped, coh=coh_in, mask=mask, prod_unw=prod_unw.astype(np.float32),
        prod_cc=prod_cc.astype(np.int64), raw=raw, ww_bridge=ww_b, isce3_bridge=isce3_b,
        region=region.astype(np.int32),
    )
    print(f"{frame}: saved arrays for plotting", flush=True)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "A_016"
    frames = ALL_FRAMES if arg.lower() == "all" else sys.argv[1:]
    for fr in frames:
        run_frame(fr)


if __name__ == "__main__":
    main()
