# Comparing whirlwind against NISAR L2 GUNW products

This directory packages a self-service benchmark so anyone on the NISAR team can
run the whirlwind phase unwrapper on real NISAR L2 GUNW products and see, for
themselves, how closely it matches the production unwrapped phase and connected
components — on one product or on hundreds, via AWS Batch.

You do not need to pre-download anything: point the tool at a granule name, an
ASF download URL, or an `s3://` URI and it fetches the product first.

- `compare_gunw.py` — the comparison entrypoint (portable, one file).
- `Dockerfile` — container image (builds the whirlwind wheel from source).
- `job-definition.json` — AWS Batch job-definition template (Fargate).
- `submit_batch.py` — submit one Batch job per product.
- `sample_granules.txt` — three sample products to start with.
- **`LOCAL_BENCH.md` — running a large campaign on your own CPU server, no AWS.**
  Companion scripts: `discover_granules.py` (build a spatially spread granule
  list from the ASF catalog), `run_local.py` (parallel runner with peak-memory
  tracking and resume), `aggregate_results.py` (campaign roll-up + plots),
  `make_synthetic_gunw.py` (offline smoke test).
- `isce3_integration/` — wiring whirlwind into the isce3 GUNW workflow.
- `ARCH_COMPARISON.md` — x86_64 vs ARM64: results are identical, ARM64 is cheaper.

---

## What it compares, and how

For each GUNW product, `compare_gunw.py`:

1. Reads the production 80 m `unwrappedPhase` and re-wraps it to `[-pi, pi)`.
   That re-wrapped phase is the *only* phase input handed to whirlwind. This is
   an apples-to-apples test of the unwrapping algorithm: both unwrappers see the
   identical wrapped field on the identical grid.

   > Why re-wrap instead of using the product's 20 m `wrappedInterferogram`? The
   > beta GUNW wrapped layer has been flagged as possibly mis-georeferenced, and
   > it is on a different (finer) grid than the unwrapped product. Re-wrapping the
   > 80 m unwrapped phase removes both confounders. (`--use-product-wrapped`
   > overrides this if you want to test the product wrapped layer directly.)

2. Reads the production `coherenceMagnitude` and the GUNW `mask` (water / subswath
   validity), and builds a valid-pixel mask.

   `--mask-policy subswath` (the default) matches what the NISAR workflow itself
   masks before unwrapping: samples invalid in either RSLC, and **not** water.
   Its runconfig sets `mask_type: subswath_mask`, which isce3 reduces to
   `invalid = ~reference_valid | ~secondary_valid`; the browse imagery confirms
   it, with the subswath edges blanked and the water unwrapped. Masking water
   instead severs the valid domain along every river, splitting a frame into
   hundreds of integration regions that then have to be re-leveled against each
   other. `water_only` and `nisar_land` are available for comparison.

3. Runs `whirlwind.unwrap(igram, corr, nlooks, mask)` — the exact public API an
   external user would call. This returns an unwrapped phase and SNAPHU-style
   connected-component labels.

4. Compares whirlwind's output to the production `unwrappedPhase` and
   `connectedComponents`:

   | metric                                          | meaning                                                                                                                                                                                              |
   | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `ambiguity_match_frac`                          | fraction of valid pixels where whirlwind and the production unwrap agree on the integer 2π count, after removing a single global cycle offset. 1.0 = identical ambiguities.                      |
   | `ambiguity_match_frac_percomp`                  | same, but re-leveled within each production connected component first. This is the fair score across water/decorrelation gaps, where the absolute inter-region 2π offset is physically unobservable. |
   | `residual_wrapped_rmse_rad` / `_p95_abs_rad`    | RMSE / 95th-pct of `wrap(whirlwind − production)`; near 0 means the shapes agree.                                                                                                                    |
   | `prod_num_cc` / `ww_num_cc`                     | number of connected components in each.                                                                                                                                                              |
   | `prod_unwrapped_recall` / `ww_unwrapped_recall` | fraction of data pixels each unwrapper actually labels (conncomp > 0) — the coverage/recall lens.                                                                                                    |
   | `runtime_s`, `rss_delta_mb`                     | wall-clock and peak-RSS delta of the whirlwind call.                                                                                                                                                 |

   Outputs per product: `<crop>.json` (all metrics), an eight-panel `<crop>.png`
   (row 1: wrapped input · coherence · production unwrapped · whirlwind unwrapped;
   row 2: NISAR GUNW/SNAPHU conncomps · whirlwind conncomps · conncomp coverage
   (ww−prod) · ambiguity diff), and `<crop>_arrays.npz` (the rasters).
   Across all products: `summary.csv` and `summary.md`.

This is a regression / agreement benchmark on the production grid, not a test
of the full NISAR wrapped-product geocoding.

### Connected-component coverage

The "conncomp coverage (ww−prod)" panel shows which unwrapper labeled each pixel:
red = whirlwind only, blue = production only, gray = both.

whirlwind labels every pixel it unwrapped self-consistently, a larger set than
production SNAPHU labels (production also drops pixels by tile cost and region-size
rules). On the cryo A_140 frame (median coherence 0.15) whirlwind labels ~99% of
valid pixels vs production's ~70%, so those regions show red. The 2π solution still
agrees with production at 99.8%.

`--conncomp-min-coherence` sets the coherence below which conncomp labels a pixel
0. The default `auto` is `0.32/sqrt(nlooks)` (0.045 at 50 looks); raise it to drop
more low-coherence pixels, at the cost of more components. On A_140:

| `--conncomp-min-coherence` | labeled fraction |
| -------------------------- | ---------------- |
| 0.08                       | 0.99             |
| 0.10                       | 0.82             |
| 0.12                       | 0.67             |
| 0.15                       | 0.35             |

Raising this floor is the most promising open knob for closing the remaining
label gap against production. It is **not** changed from `auto` by default —
see [`CONNCOMP_FLOOR_EXPERIMENT.md`](CONNCOMP_FLOOR_EXPERIMENT.md) for why, and
for the experiment to run at campaign scale before touching the default.

---

## Quick start without AWS (one machine, `uv`)

The script has an inline dependency header, so [`uv`](https://docs.astral.sh/uv/)
runs it with no manual environment setup. It pulls `whirlwind-insar` from PyPI;
to test unreleased code, build from source instead (see the Docker section).

```bash
# Earthdata Login credentials (any one of these):
export EARTHDATA_TOKEN=...                       # an EDL bearer token, or
# export EARTHDATA_USERNAME=...  EARTHDATA_PASSWORD=...   ; or a ~/.netrc entry

uv run aws-batch/compare_gunw.py \
  https://nisar.asf.earthdatacloud.nasa.gov/.../<ID>.h5 \
  --out-dir out --nlooks 50
```

Run all three samples at once:

```bash
uv run aws-batch/compare_gunw.py --inputs-file aws-batch/sample_granules.txt --out-dir out
```

> Memory note: a full NISAR frame uses several GB and is solved single-threaded
> at peak. The tool runs products one at a time on purpose; don't fan out
> multiple full-frame solves on one machine.

---

## Build and run the container

The image bundles the Python `whirlwind` package (the CLI alone can't read HDF5)
and builds the wheel from this repo's source, so reviewers test the latest code.

```bash
# From the repo root (build context = whole repo):
docker build -f aws-batch/Dockerfile -t whirlwind-gunw .

docker run --rm -e EARTHDATA_TOKEN="$EARTHDATA_TOKEN" \
  -v "$PWD/out:/work/out" whirlwind-gunw \
  https://nisar.asf.earthdatacloud.nasa.gov/.../<ID>.h5 --out-dir /work/out
```

Apple Silicon / architecture: the committed `job-definition.json` targets
`X86_64`. Either build for that target (`docker build --platform linux/amd64 ...`,
emulated and slower on an M-series Mac), or — recommended for your own runs —
switch to ARM64/Graviton, which is ~20% cheaper on Fargate and builds natively on
Apple Silicon: set `runtimePlatform.cpuArchitecture` to `ARM64` in
`job-definition.json` and build with `--platform linux/arm64`.

---

## Run on AWS Batch (lowest cost)

The single biggest cost lever is the region. The ASF DAAC NISAR cloud
holdings live in `us-west-2`; running Batch there keeps every download
in-region, so there is no data-egress charge — you pay only for compute.
Then: Fargate Spot (≈70% off on-demand, no instances to manage), right-sized
memory, and optionally ARM64/Graviton (≈20% cheaper still).

Below, replace `ACCOUNT_ID`, the bucket, and the subnet/SG with your values.
Commands use the `my-profile` profile for our sample run.

### 1. Push the image to ECR

```bash
AWS="aws --profile my-profile --region us-west-2"
ACCOUNT_ID=$($AWS sts get-caller-identity --query Account --output text)
ECR="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com"

$AWS ecr create-repository --repository-name whirlwind-gunw || true
$AWS ecr get-login-password | docker login --username AWS --password-stdin "$ECR"

docker build -f aws-batch/Dockerfile -t whirlwind-gunw .         # add --platform if needed
docker tag whirlwind-gunw:latest "$ECR/whirlwind-gunw:latest"
docker push "$ECR/whirlwind-gunw:latest"
```

### 2. Store Earthdata Login credentials in Secrets Manager

```bash
$AWS secretsmanager create-secret --name earthdata-login \
  --secret-string '{"username":"YOUR_EDL_USERNAME","password":"YOUR_EDL_PASSWORD"}'
```

### 3. Create the two IAM roles

```bash
cat > /tmp/ecs-trust.json <<'JSON'
{ "Version": "2012-10-17", "Statement": [
  { "Effect": "Allow", "Principal": { "Service": "ecs-tasks.amazonaws.com" },
    "Action": "sts:AssumeRole" } ] }
JSON

# Execution role: pull the image + write logs + read the EDL secret.
$AWS iam create-role --role-name whirlwind-gunw-exec-role \
  --assume-role-policy-document file:///tmp/ecs-trust.json
$AWS iam attach-role-policy --role-name whirlwind-gunw-exec-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
$AWS iam put-role-policy --role-name whirlwind-gunw-exec-role \
  --policy-name read-edl-secret --policy-document "{\"Version\":\"2012-10-17\",
   \"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"secretsmanager:GetSecretValue\",
   \"Resource\":\"arn:aws:secretsmanager:us-west-2:$ACCOUNT_ID:secret:earthdata-login-*\"}]}"

# Job role: write results to the output bucket (and read s3:// inputs if used).
$AWS iam create-role --role-name whirlwind-gunw-job-role \
  --assume-role-policy-document file:///tmp/ecs-trust.json
$AWS iam put-role-policy --role-name whirlwind-gunw-job-role \
  --policy-name s3-io --policy-document '{"Version":"2012-10-17","Statement":[
   {"Effect":"Allow","Action":["s3:PutObject","s3:GetObject","s3:ListBucket"],
    "Resource":["arn:aws:s3:::YOUR_BUCKET","arn:aws:s3:::YOUR_BUCKET/*"]}]}'
```

### 4. Register the job definition

Edit `aws-batch/job-definition.json`: set `image` to `$ECR/whirlwind-gunw:latest`,
the two role ARNs, and the secret ARNs (use the real ARN from step 2, keeping the
`:username::` / `:password::` JSON-key suffixes). Tune `VCPU` / `MEMORY` /
`ephemeralStorage` for your largest frames (4 vCPU / 30 GB / 50 GiB is a safe
default).

```bash
$AWS batch register-job-definition --cli-input-json file://aws-batch/job-definition.json
```

### 5. Create the Fargate Spot compute environment + queue

Use a public subnet in your default VPC (so `assignPublicIp: ENABLED` reaches
the internet for EDL) and a security group that allows outbound (the default SG
does).

```bash
SUBNET=subnet-xxxxxxxx      # a public subnet in us-west-2
SG=sg-xxxxxxxx              # default SG is fine (outbound all)

$AWS batch create-compute-environment \
  --compute-environment-name whirlwind-gunw-ce --type MANAGED --state ENABLED \
  --compute-resources "type=FARGATE_SPOT,maxvCpus=64,subnets=$SUBNET,securityGroupIds=$SG"

$AWS batch create-job-queue --job-queue-name whirlwind-gunw-queue \
  --priority 1 --state ENABLED \
  --compute-environment-order order=1,computeEnvironment=whirlwind-gunw-ce
```

### 6. Submit jobs (one per product)

```bash
uv run aws-batch/submit_batch.py \
  --inputs-file aws-batch/sample_granules.txt \
  --job-queue whirlwind-gunw-queue \
  --job-definition whirlwind-gunw \
  --s3-out s3://YOUR_BUCKET/ww-gunw-bench \
  --region us-west-2 --profile my-profile
```

Add `--dry-run` first to print what would be submitted. Each job uploads to
`s3://YOUR_BUCKET/ww-gunw-bench/<id>/`.

### 7. Collect results

```bash
aws --profile my-profile s3 sync s3://YOUR_BUCKET/ww-gunw-bench ./gunw_results
open gunw_results/*/full.png      # the eight-panel comparison figures
column -s, -t < gunw_results/*/summary.csv | less -S
```

### A single sample run (`my-profile`)

To validate the whole pipeline on just the cryosphere frame before fanning out:

```bash
uv run aws-batch/submit_batch.py \
  "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_BETA_V1/NISAR_L2_PR_GUNW_009_163_A_140_010_7700_SH_20260108T130215_20260108T130251_20260120T130216_20260120T130252_X05010_N_P_J_001/NISAR_L2_PR_GUNW_009_163_A_140_010_7700_SH_20260108T130215_20260108T130251_20260120T130216_20260120T130252_X05010_N_P_J_001.h5" \
  --job-queue whirlwind-gunw-queue --job-definition whirlwind-gunw \
  --s3-out s3://YOUR_BUCKET/ww-gunw-bench --region us-west-2 --profile my-profile
```

### Rough cost

In `us-west-2`, a full GUNW frame solves in single-digit minutes on 4 vCPU. At
Fargate Spot rates that is on the order of a cent or two per product, plus
a few MB of S3 for the figures/arrays and zero egress (in-region download).
A few hundred products is dollars, not hundreds of dollars.

---

## Reproducing with the standalone Rust CLI (optional)

`compare_gunw.py --dump-flat` also writes the solver inputs as flat binary
(`<crop>.phase`, `<crop>.cor`, `<crop>.mask`) plus a `<crop>.cli.txt` with the
equivalent command for the pure-Rust `whirlwind` binary (the top-level
`./Dockerfile`). Useful for MATLAB users or anyone who wants to drive the CLI
directly on the exact same data without HDF5.

---

## isce3 GUNW workflow integration

To run whirlwind *inside* the isce3 RUNW step (instead of SNAPHU/ICU/PHASS), see
[`isce3_integration/README.md`](isce3_integration/README.md). It adds an
`algorithm: whirlwind` branch to `nisar/workflows/unwrap.py` plus the runconfig
and defaults plumbing.
