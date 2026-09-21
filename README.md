# FireComp: A Global Wildfire Dataset and Benchmark at 375 m Resolution

FireComp is a global-scale dataset and benchmark for **next-day wildfire spread
prediction** from 375 m VIIRS active-fire detections. Given today's fire state,
terrain embeddings, observed weather and a weather forecast, a model predicts
tomorrow's fire extent as a dense binary segmentation map over 256×256 patches.

Existing sub-kilometre spread datasets cover single regions (USA, Canada,
Mediterranean); global products such as SeasFire are far too coarse (0.25°) for
local fire behaviour. FireComp covers **9 regions on 6 continents, 2017–2025**,
with region-balanced sampling so that no single fire-prone area dominates.

This repository contains the full pipeline: data sourcing, dataset construction,
the benchmark models, baselines, the evaluation protocol and a browser-based
viewer for inspecting predictions on the globe.

**Dataset:** [huggingface.co/datasets/justuskarlsson/FireComp](https://huggingface.co/datasets/justuskarlsson/FireComp) (CC BY 4.0)

---

## The task

| | |
|---|---|
| **Input** | current fire mask, accumulated burn-time maps, Alpha Earth terrain embeddings (PCA-5), ERA5 weather for day *T*, GFS forecast for day *T+1*; optional canopy height |
| **Target** | `next_mask` — all fire pixels on day *T+1* (comparable to prior work) <br> `new_fires` — only pixels that are newly burning on *T+1* (harder, no persistence shortcut) |
| **Patch** | 256 × 256 px at 375 m (~96 km) |
| **Loss mask** | pixels must be cloud-free and observed on both days |
| **Split** | stratified temporal split: within each region samples are sorted by date and cut 60 / 15 / 25 into train / val / test. The test set is identical across every experiment |
| **Metrics** | F1, IoU and AUC-PR (discrimination) plus Brier score (calibration) |

Fires are classified by land cover into *vegetation*, *crop* and *static*
(industrial/persistent heat sources); the benchmark trains and evaluates on
vegetation fires by default (`fire_type` in the config).

### Temporal alignment (hours from day *T* 00:00 UTC)

- `cur_mask` / `accum_*`: fire detections during day *T* (0–24 h)
- ERA5: observed weather during day *T* (0–24 h)
- GFS: forecast for day *T+1* (24–48 h)
- `next_mask` / `new_fires`: fire detections during day *T+1* (24–48 h)

---

## Installation

Python ≥ 3.12 with GDAL is required; the easiest route is conda for the
geospatial stack, then pip for the rest:

```bash
conda create -n firecomp python=3.12 gdal=3.9 -c conda-forge
conda activate firecomp
pip install -e .            # add [viewer] for the FastAPI backend, [dev] for pytest
```

PyTorch should be installed following the instructions for your CUDA version.
Data lives under `data/` in the repo root by default; set `FIRECOMP_DATA` to
point elsewhere.

Google Earth Engine (Alpha Earth, ERA5, GFS, canopy height) requires an
authenticated `earthengine` account, and NASA Earthdata (VIIRS) requires
`earthaccess` credentials in `~/.netrc`.

---

## Getting the dataset

The built dataset (130 649 samples, 8 zstd-compressed HDF5 shards + metadata),
the region raster and the LA 2025 case-study samples are on Hugging Face and
mirror the `data/` layout this code expects:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download justuskarlsson/FireComp --repo-type dataset --local-dir data
```

This gives `data/next_day_v3/` (main dataset), `data/next_day_v3_case_study/`,
`data/regions/` and `data/fire_areas.npz`. With that in place you can skip
straight to [Training and evaluation](#training-and-evaluation). Building from
scratch (below) is only needed to reproduce or extend the dataset.

---

## Building the dataset from scratch

```bash
# 1. Choose VIIRS images per region and download VNP03 geolocation granules (resume-safe, shardable with -ji/-jn)
python -m firecomp.next_day.download --manifest --top 1000
python -m firecomp.next_day.download --top 1000
python -m firecomp.next_day.download --supplemental --top 1000   # T+1 granules for uncovered pairs
python -m firecomp.next_day.download --pick --top 1000           # writes data/next_day_v3/samples.json

# 2. Build the per-sample H5 shards in three phases (phase 1 first, then 2 and 3 in any order)
python -m firecomp.next_day.preprocess build-accum    --dataset-dir data/next_day_v3
python -m firecomp.next_day.preprocess build-lossmask --dataset-dir data/next_day_v3
python -m firecomp.next_day.preprocess build-fields   --dataset-dir data/next_day_v3

# Optional: small subset for development
python -m firecomp.next_day.preprocess distill --source data/next_day_v3 --dest data/next_day_small --n-fires 100
```

The region polygons (`data/WildfireRegions.kml`) are rasterised once with
`python -m firecomp.core.regions`. Land-cover cluster priors used for fire-type
classification are generated with `python -m firecomp.core.clusters`.

---

## Training and evaluation

All entry points are dataclass-driven: every `NextDayConfig` field
(`firecomp/next_day/config.py`) can be set from a JSON file or `--flag` override.

```bash
# Single model
python -m firecomp.next_day.implementations.dl_2d train --model-type unet++ --encoder-name resnet34 --loss-type bce --target-type next_mask --tag my_run
python -m firecomp.next_day.implementations.dl_2d eval data/runs/my_run/best.pt

# Benchmark grid: {UNet, UNet++, ViT} × loss configs × {next_mask, new_fires}
python -m firecomp.next_day.implementations.grid_search --num-gpus 4

# Baselines
python -m firecomp.next_day.implementations.persistence   --target next_mask --split test
python -m firecomp.next_day.implementations.morphological --target next_mask --split test --radii 1 2 3

# Cross-region transfer matrix and leave-one-region-out
python -m firecomp.next_day.implementations.transfer matrix
python -m firecomp.next_day.implementations.transfer loo

# Input-group ablation and fire-type composition study
python -m firecomp.next_day.implementations.ablation
python -m firecomp.next_day.implementations.fire_type_study

# Paper figures and LaTeX tables
python -m firecomp.next_day.run_figures
python -m firecomp.next_day.tables
```

### Models

| Model | Type | Notes |
|-------|------|-------|
| UNet | CNN, from scratch | Anchor model — what most fire-ML papers use |
| UNet++ (ResNet encoder) | Dense-skip CNN | via `segmentation_models_pytorch` |
| ViT (SETR-style) | Transformer | Global receptive field from layer 1 |

Losses: BCE (with `pos_weight`), focal, Dice. The loss → calibration trade-off
is a central finding: Dice maximises F1 but yields overconfident probabilities;
BCE gives honest uncertainty at lower overlap.

---

## Research viewer

A Cesium globe for browsing test-set predictions per fire event, comparing
models side by side. See [viewer/README.md](viewer/README.md).

---

## Repository layout

```
firecomp/
├── core/        losses, metrics, regions, fire statistics & typing, cluster priors, CLI, checkpoints, GPU queue
├── dsrc/        data sources: VIIRS (VNP14/VNP03), Alpha Earth, ERA5, GFS, WeatherNext 2, canopy height
├── next_day/    config, dataset, download, preprocess, implementations/ (models, baselines, experiments), figures, tables
├── models/      UNet, UNet++/ViT wrappers
├── cpp/         C++ BFS for clustering fire detections into fire events
└── viewer/      export of test predictions for the viewer
viewer/          FastAPI backend + Cesium/Vite frontend
tests/           pytest suite
docs/data.md     data sources in detail
```

---

## License

MIT
