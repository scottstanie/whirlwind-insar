# Running the NISAR GUNW comparison on your own server

This is the no-cloud path: instead of one AWS Batch job per granule, run the
same comparison across a pool of worker slots on a CPU box you already have.
Nothing here needs S3 access or an AWS account -- products are pulled from ASF
over authenticated HTTPS.

| script | what it does |
| --- | --- |
| `discover_granules.py` | Pull the ASF catalog, join it to the track/frame land table, pick a spatially spread subset -> `manifest.txt` |
| `run_local.py` | Run one `compare_gunw.py` per granule across N workers; records wall time + true peak RSS; resumable |
| `compare_gunw.py` | The per-granule comparison (unchanged; also what Batch runs) |
| `aggregate_results.py` | Roll the campaign into `campaign.csv` / `campaign.md` / `campaign_summary.png` |
| `make_synthetic_gunw.py` | Tiny fake products for smoke-testing the chain without downloads |

## 0. Prerequisites

**An environment with your whirlwind build installed**, plus `h5py`, `numpy`,
`matplotlib`, `pandas`, `requests`, `psutil`, `asf-search`. `run_local.py`
launches `compare_gunw.py` with the *same interpreter that runs it*, and checks
up front that the imports work -- so activate the env you want to benchmark and
use plain `python`, not `uv run`.

> `compare_gunw.py`'s inline dependency header pins `whirlwind-insar` from PyPI.
> That is what the Docker image uses. If you launch with `--uv`, you are
> benchmarking the *published wheel*, not your local build. For a dev build,
> install it into an env and let the default interpreter path be used.

**Earthdata credentials**, either way:

```bash
export EARTHDATA_TOKEN=<your EDL bearer token>
# or a ~/.netrc entry:
#   machine urs.earthdata.nasa.gov login <user> password <pass>
```

## 1. Smoke-test the chain first (30 seconds, no downloads)

Do this before committing to hundreds of GB. It exercises discovery-free
running, the peak-memory sampler, resume, and the aggregator.

```bash
cd aws-batch
python make_synthetic_gunw.py --out-dir /tmp/synth --count 4
python run_local.py --manifest /tmp/synth/manifest.txt --root /tmp/smoke --workers 2
python aggregate_results.py --root /tmp/smoke
```

You should get 4/4 succeeded and a `campaign_summary.png`. The agreement
numbers are meaningless here (the "production" field is synthetic truth) --
this only proves the plumbing.

## 2. Build the granule list

Pull the whole NISAR GUNW catalog once (cached to CSV), keep frames that the
land table says are land, and take a couple per track spread along the orbit:

```bash
python discover_granules.py \
  --land-frames ~/repos/virtual-sar/src/virtual_sar/data/nisar_land_frames.csv \
  --per-track 2 \
  --min-land 0.2 \
  --inventory-csv nisar_gunw_inventory.csv \
  --out manifest.txt
```

This writes `manifest.txt` (URLs, one per line) and `manifest.meta.csv`
(track/frame/bounding box, used later for the coverage map), and prints the
total download volume.

Knobs worth knowing:

- `--per-track 1..4` -- the main size dial. At `2` you get roughly 2 x the
  number of track/direction pairs (~400 products, ~900 GB of downloads).
- `--max-results 2000` -- cap the catalog query for a quick trial.
- `--limit 25` -- truncate the manifest; the fastest way to a real end-to-end run.
- `--prefer short-baseline|recent` -- which repeat pass to keep per frame.
- `--min-land 0.2` -- raise to skip mostly-ocean frames.

**Start small.** Do a 25-granule run first and look at the output before
launching the full campaign:

```bash
python discover_granules.py --land-frames <...> --per-track 1 --limit 25 --out manifest_pilot.txt
```

## 3. Run the campaign

```bash
python run_local.py \
  --manifest manifest.txt \
  --root /data/ww-bench \
  --workers 8 \
  --delete-after \
  --timeout 10800
```

`--delete-after` removes each product once its job succeeds, which is what
keeps a ~900 GB manifest inside ~20 GB of working disk. Drop it only if you
want the products kept for re-runs.

Everything lands under `--root`:

```
/data/ww-bench/
  runs.jsonl                     # one record per job: rc, wall_s, peak_rss_mb
  logs/<granule>.log             # full stdout/stderr per job
  results/<granule>/<granule>/   # full.json, plots, from compare_gunw.py
  downloads/                     # transient with --delete-after
```

### Choosing workers and threads

Each worker gets `cores // workers` threads by default (override with
`--threads-per-worker`), passed down as `WHIRLWIND_NUM_THREADS` so N workers do
not each grab every core. Two limits to respect:

- **Memory.** Budget from the measured rate: `peak_GB ≈ 0.2 x megapixels` per
  frame, so `workers x per-frame GB` must fit in RAM with room to spare. On
  earlier NISAR benchmarking a full frame ran a few GB. After your pilot run,
  read the real number out of `campaign.csv` (`megapixels`, `peak_rss_mb`) and
  size the full campaign from that.
- **Network.** 8 concurrent 2 GB downloads is usually the actual bottleneck,
  not the CPU.

8-10 workers is a sensible default on a large box; drop to 4 if the pilot shows
frames near your memory ceiling.

### A caveat on runtime numbers

With several unwraps in flight, per-job wall time includes contention and is
**not** a clean single-frame benchmark. Use the parallel mode to get through the
campaign and to compare accuracy, memory, and components. For headline runtime,
re-run the frames you care about serially:

```bash
python run_local.py --manifest interesting.txt --root /data/ww-bench-timing --workers 1 --force
```

### Interrupting and resuming

Ctrl-C is safe: each job appends to `runs.jsonl` as it finishes. Re-running the
same command skips everything already recorded as successful. Jobs that failed
are retried automatically on the next run; `--force` re-runs even the
successful ones.

## 4. Aggregate

```bash
python aggregate_results.py --root /data/ww-bench --meta manifest.meta.csv
```

Writes into the campaign root:

- `campaign.csv` -- one row per granule: agreement, runtime, peak RSS,
  component counts, coherence, track/frame, geometry.
- `campaign.md` -- headline numbers and the worst 15 frames to look at first.
- `campaign_summary.png` -- six panels: agreement histogram and ECDF, runtime
  vs size, peak memory vs size, whirlwind vs production component counts, and a
  lon/lat coverage map coloured by agreement.

The headline metric is `ambiguity_match_frac_percomp`: agreement with the
production unwrap on the 2*pi integer, re-levelled within each production
connected component (the only fair comparison across water and decorrelation
gaps, since a region's absolute cycle is unobservable).

## 5. Triage

Start with the worst-frames table in `campaign.md`, then open that granule's
plot in `results/<granule>/<granule>/` and its log in `logs/`. Failed jobs:

```bash
python -c "
import json
for l in open('/data/ww-bench/runs.jsonl'):
    r = json.loads(l)
    if r['returncode'] != 0:
        print(r['returncode'], r['job_id'], r['log'])
"
```

Common causes: expired Earthdata token (401 in the log), disk full, or a job
killed by `--timeout`.
