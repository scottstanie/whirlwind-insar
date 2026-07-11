# x86_64 vs ARM64: does the cheaper architecture change the results?

Short answer: no. On real NISAR data, whirlwind produces bit-identical
unwrapping results on `linux/amd64` (x86_64) and `linux/arm64` (Graviton) over
every valid pixel — same 2π ambiguities, same connected components, same phase.
ARM64/Graviton is ~20% cheaper on Fargate at zero quality cost.

This is expected: whirlwind's minimum-cost-flow solve runs on integer Carballo
costs, so the routing is deterministic regardless of CPU. The output phase is
`wrapped_input + 2π·(integer cycle field)`, and both terms come out identical.

## The test

A real 2048×2048 center crop of the canonical 005_D_077 NISAR GUNW frame
(`...003_005_D_077_004_4000...`), run through both container images built from
the same `aws-batch/Dockerfile`:

```bash
docker buildx build --platform linux/arm64 -f aws-batch/Dockerfile -t whirlwind-gunw:arm64 --load .
docker buildx build --platform linux/amd64 -f aws-batch/Dockerfile -t whirlwind-gunw:amd64 --load .

for arch in arm64 amd64; do
  docker run --rm -v "$PWD/crop:/work" whirlwind-gunw:$arch \
    /work/NISAR_..._D077_crop2048.h5 --out-dir /work/out_$arch --nlooks 16 --sizes full
done
```

## Result

| quantity | arm64 | amd64 | equal? |
|---|---|---|---|
| `ambiguity_match_frac` (vs production) | 0.99520 | 0.99520 | identical |
| `ambiguity_match_frac_percomp` | 0.99960 | 0.99960 | identical |
| `residual_wrapped_rmse_rad` | 2.328e-06 | 2.328e-06 | identical |
| connected components | 2 | 2 | identical |
| 2π ambiguity integer, per valid pixel | — | — | identical (0 differ) |
| unwrapped phase, per valid pixel | — | — | bit-identical |

The two outputs differ at 43,518 pixels (~1%), but all of them are in the
masked-out nodata region (0 differences inside the valid mask). Masked pixels
are unconstrained — the solver assigns them arbitrary values that aren't part of
the product — so they carry no meaning.

Wall-clock here was 6.2 s (arm64, native on the M-series build host) vs 8.4 s
(amd64, emulated via QEMU on the same host). The emulation penalty is a build-
host artifact, not a Fargate property: on native x86_64 Fargate the runtime is
comparable to Graviton. The cost takeaway is purely the Fargate price difference
(~20% in Graviton's favor), since the *result* is the same either way.

## Recommendation

- Cost-sensitive runs (e.g. sweeping hundreds of products): ARM64/Graviton.
  Identical results, ~20% cheaper, and it builds natively on Apple-Silicon Macs.
  Set `runtimePlatform.cpuArchitecture: ARM64` in `job-definition.json` and build
  with `--platform linux/arm64`.
- NISAR SDS / x86 environments: x86_64. The committed `job-definition.json`
  defaults to this. Build with `--platform linux/amd64`.

Both images are validated to build and run end-to-end. Point this out to anyone
weighing the cost: the cheap option is not a quality trade-off here.
