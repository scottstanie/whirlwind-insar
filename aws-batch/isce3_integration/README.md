# Running whirlwind inside the isce3 GUNW workflow

This adds whirlwind as a selectable unwrapping algorithm in the isce3 RUNW step,
alongside `icu` / `phass` / `snaphu`. Select it with:

```yaml
runconfig:
  groups:
    processing:
      phase_unwrap:
        algorithm: whirlwind
        whirlwind:
          nlooks:                  # blank -> isce3's effective-looks estimate (see below)
          mask:                    # optional valid-pixel mask file (nonzero = valid)
          bridge: true             # whirlwind's own region re-leveling post-pass
          downsample: 1            # >1 for noisy scenes (coarse-solve cycles)
          conncomp_min_coherence: auto
          goldstein_alpha: 0.0
```

Everything else (crossmul, looks, preprocess/water/subswath masking, RUNW output
datasets, statistics) is unchanged — whirlwind reads the same wrapped
interferogram + coherence and writes the same `unwrappedPhase` /
`connectedComponents` datasets that snaphu does.

`nlooks` is the effective number of independent looks in the coherence estimate.
When blank, the branch calls isce3's `get_effective_looks`, which computes it from
the interferogram range/azimuth sample spacing and the SAR range/azimuth
bandwidths (`n_e = k_r k_a d_r d_a / (rho_r rho_a)`) — the same value the snaphu
branch falls back to. It is usually a non-integer (e.g. ~20-24 for a NISAR GUNW),
lower than the raw multilook-window product because of oversampling and spatial
correlation. whirlwind uses it for the coherence cost model and the conncomp
coherence floor, so set it explicitly only if you know the effective looks of
your coherence.

`whirlwind` must be importable in the isce3 environment
(`pip install whirlwind-insar`); it is imported lazily, only when this algorithm
is selected, so isce3 has no hard dependency on it.

## What changed in isce3

Three files (already edited in `~/repos/isce3`):

| File | Change |
|---|---|
| `share/nisar/schemas/insar.yaml` | `whirlwind` added to the `algorithm` enum; new `whirlwind: include('whirlwind_options')` and a `whirlwind_options` schema block. |
| `share/nisar/defaults/insar.yaml` | new `phase_unwrap.whirlwind` defaults block. |
| `python/packages/nisar/workflows/unwrap.py` | new `elif algorithm == "whirlwind":` branch; the generic isce3 `bridge` post-pass is skipped for whirlwind (it bridges internally). |

`unwrap_runconfig.py` needed no change: its `yaml_check` already allocates an
empty config dict for whatever `algorithm` is chosen, and the branch reads its
options with `.get(...)` defaults.

## How the branch works

The branch (in `unwrap.py`) does, per frequency/polarization:

```python
elif algorithm == "whirlwind":
    info_channel.log("Unwrapping with whirlwind")
    import whirlwind as ww

    ww_cfg = unwrap_args["whirlwind"]
    igram_array = open_raster(igram_path)   # complex wrapped interferogram
    coh_array = open_raster(corr_path)      # coherence in [0, 1]

    # Valid-pixel mask (True = valid): optional mask file (nonzero = valid)
    # combined with the preprocess invalid-pixel mask, if either is present.
    valid = None
    if ww_cfg.get("mask") is not None:
        valid = open_raster(ww_cfg["mask"]) != 0
    if (unwrap_args["preprocess_wrapped_phase"]["enabled"] and mask is not None):
        valid = ~mask if valid is None else (valid & ~mask)

    # nlooks: from config, else isce3's get_effective_looks (same as snaphu).
    if ww_cfg.get("nlooks") is not None:
        nlooks = ww_cfg["nlooks"]
    else:
        ...  # get_effective_looks(ref_slc, ref_orbit, rg_spacing, az_spacing, ...)

    unw_array, conncomp_array = ww.unwrap(
        igram_array.astype(np.complex64, copy=False),
        coh_array.astype(np.float32, copy=False),
        float(nlooks),
        valid,
        bridge=ww_cfg.get("bridge", True),
        downsample=ww_cfg.get("downsample", 1),
        conncomp_min_coherence=ww_cfg.get("conncomp_min_coherence", "auto"),
        goldstein_alpha=ww_cfg.get("goldstein_alpha", 0.0),
    )
    dst_h5[unw_path][:, :] = unw_array
    dst_h5[conn_comp_path][:, :] = conncomp_array
    unw_raster = isce3.io.Raster(unw_raster_path)
    compute_stats_real_data(unw_raster, unw_dataset)
```

Mask convention note: isce3 / SNAPHU mask files use **nonzero = valid**; whirlwind
wants a boolean **True = valid**, hence the `!= 0`. The preprocess `mask` variable
is the opposite (True = *invalid*), so it is inverted (`~mask`).

## Porting to another isce3 fork (e.g. an "isce3-scale" SDS tree)

`whirlwind_unwrap.py` in this directory is a **self-contained reference** of the
same logic as a single function, `run_whirlwind(igram, coherence, nlooks, ...)`,
returning `(unwrapped_phase, connected_components)`. A team maintaining their own
isce3 fork can either:

1. apply the same three edits above (recommended — minimal, matches upstream), or
2. drop `whirlwind_unwrap.py` into their `nisar/unwrap/` package and have the
   `unwrap.py` branch call `run_whirlwind(...)`, keeping the workflow edit to a
   couple of lines.

Both paths produce identical results; option 1 is what was applied here.
