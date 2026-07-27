# Running the NISAR GUNW comparison on your own server

Run a series of comparisons to NISAR GUNW products on a single machine.

## Quickstart:

```bash
# Discover granules
# First download https://gist.github.com/scottstanie/1fe9a5d8b8f8880163f9626f624474f8/raw/7f7d10a8f54bef5183067026237908811fd32365/nisar_land_frames.csv
python discover_granules.py \
  --land-frames nisar_land_frames.csv \
  --per-track 100 \
  --min-land 0.25 \
  --inventory-csv nisar_gunw_inventory.csv \
  --out manifest.txt

# Run the campaign
python run_local.py \
  --manifest manifest.txt \
  --root ./ww-bench \
  --workers 8 \
  --delete-after \
  --timeout 10800

# Aggregate results
python aggregate_results.py --root ./ww-bench --meta manifest.meta.csv
```

Products are pulled from ASF over HTTPS.

| script                   | what it does                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `discover_granules.py`   | Pull the ASF catalog, join it to the track/frame land table, pick a spatially spread subset -> `manifest.txt` |
| `run_local.py`           | Run one `compare_gunw.py` per granule across N workers; records wall time + true peak RSS; resumable          |
| `compare_gunw.py`        | The per-granule comparison (unchanged; also what Batch runs)                                                  |
| `aggregate_results.py`   | Roll the campaign into `campaign.csv` / `campaign.md` / `campaign_summary.png`                                |
| `make_synthetic_gunw.py` | Tiny fake products for smoke-testing the chain without downloads                                              |

## 0. Prerequisites

**An environment with your whirlwind build installed**, plus `h5py`, `numpy`, `matplotlib`, `pandas`, `requests`, `psutil`, `asf-search`. `run_local.py` launches `compare_gunw.py`, and checks up front that the imports work. Environment with whirlwin must be active.

Earthdata credentials must be available if listing granules to be downloaded, rather than local HDF5 files that are already downloaded.

## 1. Build the granule list

Pull the whole NISAR GUNW catalog once (cached to CSV), keep frames that the land table says are land, and take a couple per track spread along the orbit.
This uses a summarized version of NISAR track/frame database here: https://gist.github.com/scottstanie/1fe9a5d8b8f8880163f9626f624474f8


```bash
wget https://gist.github.com/scottstanie/1fe9a5d8b8f8880163f9626f624474f8/raw/7f7d10a8f54bef5183067026237908811fd32365/nisar_land_frames.csv
python discover_granules.py \
  --land-frames nisar_land_frames.csv \
  --per-track 100 \
  --min-land 0.25 \
  --inventory-csv nisar_gunw_inventory.csv \
  --out manifest.txt
```

This writes `manifest.txt` (URLs, one per line) and `manifest.meta.csv`
(track/frame/bounding box, retained in the campaign table), and prints the
total download volume.

Knobs worth knowing:

- `--per-track 1..4` -- the main size dial. At `2` you get roughly 2 x the
  number of track/direction pairs (~400 products, ~900 GB of downloads).
- `--max-results 2000` -- cap the catalog query for a quick trial.
- `--limit 25` -- truncate the manifest; the fastest way to a real end-to-end run.
- `--prefer short-baseline|recent` -- which repeat pass to keep per frame.
- `--min-land 0.2` -- raise to skip mostly-ocean frames.

To doo a 25-granule run first and look at the output before launching the full campaign:

```bash
python discover_granules.py --land-frames <...> --per-track 1 --limit 25 --out manifest_pilot.txt
```

## 2. Run the campaign

```bash
python run_local.py \
  --manifest manifest.txt \
  --root ./ww-bench \
  --workers 8 \
  --delete-after \
  --timeout 10800
```

`--delete-after` removes each product once its job succeeds, which is what
keeps a ~900 GB manifest inside ~20 GB of working disk. Drop it only if you
want the products kept for re-runs.

The comparison wrapper also exposes the main preprocessing A/B controls. Use a
fresh `--root` (or pass through `--force`) so resumability does not reuse the
baseline JSON:

```bash
# Valid low-coherence pixels only; masked water remains masked.
python run_local.py ... \
  --compare-arg=--interpolate --compare-arg=--interp-cutoff --compare-arg=0.2

# Coarse solve or Goldstein-informed solve.
python run_local.py ... --compare-arg=--downsample --compare-arg=4
python run_local.py ... --compare-arg=--goldstein-alpha --compare-arg=0.7
```

Everything lands under `--root`:

```
./ww-bench/
  runs.jsonl                     # one record per job: rc, wall_s, peak_rss_mb
  logs/<granule>.log             # full stdout/stderr per job
  results/<granule>/<granule>/   # full.json, plots, from compare_gunw.py
  downloads/                     # transient with --delete-after
```

## 3. Aggregate results to a CSV

```bash
python aggregate_results.py --root ./ww-bench --meta manifest.meta.csv
```

Writes into the campaign root:

- `campaign.csv` -- one row per granule: agreement, runtime, peak RSS,
  component counts, coherence, track/frame, geometry.
- `campaign.md` -- headline numbers and the worst 15 frames to look at first.
- `campaign_summary.png` -- six panels: agreement success curve, agreement vs
  coherence, whirlwind vs production labeled-pixel coverage, runtime and
  peak-memory scaling, and component counts.

The headline metric is `ambiguity_match_frac_percomp`: agreement with the
production unwrap on the 2*pi integer, re-levelled within each production
connected component (the only fair comparison across water and decorrelation
gaps, since a region's absolute cycle is unobservable).
