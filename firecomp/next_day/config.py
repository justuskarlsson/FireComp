"""
next_day/config.py — NextDayConfig dataclass.

What the dataset + training script need to know. Plain dataclass with a
from_file() classmethod for loading a JSON override.

Dataset-only fields (terrain, include_* flags, padding)
sit alongside training fields (lr, num_epochs) so one JSON file configures
the whole pipeline.

Dataset versioning: `dataset_version` selects an immutable on-disk dataset
variant ("latest" = base dir, "v2" = cleaned subset, etc.).  Default is
"latest" — all files live directly in `data/next_day_v3/`.

Fire type filtering: `fire_type` controls which fires to include at load
time.  "vegetation" (default) = wildfire only, "all" = everything,
"crop" / "static" = specific categories.  Uses the per-sample fire_type
field from the LC-based classifier (see core/fire_filter.py).

Subsampling: `max_samples` (0 = all) randomly selects N samples at load time
so you can train on a 10K dataset with only 1K samples — no distill step.
"""

import json
from dataclasses import dataclass, fields as dc_fields
from typing import Literal


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Paper 1 losses only — Dice/Hybrid still in firecomp/core/losses.py
SchedulerType = Literal["none", "onecycle"]
LossType = Literal["bce", "focal"]
# Paper 1 models only — full registry lives in firecomp/models/segmentation_models.py
ModelType = Literal["unet", "unet++", "vit"]
# Paper 1 encoders — unet/unet++ default to resnet18; vit always uses mit_b2
EncoderType = Literal["resnet18", "resnet34", "resnet50", "efficientnet-b3"]


# ---------------------------------------------------------------------------
# NextDayConfig
# ---------------------------------------------------------------------------

@dataclass
class NextDayConfig:
    """
    Config for the next-day fire spread segmentation task: dataset fields
    and training knobs in one place.
    """

    # --- dataset selection ---
    dataset_dir: str = "data/next_day_v3"
    dataset_version: str = "latest"
    fire_type: str = "vegetation"       # "all" | "vegetation" | "crop" | "static"

    # --- input fields ---
    terrain_type: str = "ae_pca_5"          # ae_pca_5 | ae_12345 | vnp02_terrain
    target_type: str = "next_mask"          # next_mask | new_fires
    include_terrain: bool = True
    include_cur_mask: bool = True
    include_accum_min: bool = True
    include_accum_max: bool = True
    include_accum_count: bool = True
    include_weather: bool = True
    include_weather_pct: bool = False
    include_gfs: bool = True
    include_canopy: bool = False
    include_pos_encoding: bool = False

    # --- task geometry ---
    img_size: int = 256
    padding: int = 16

    # --- data loading ---
    batch_size: int = 128
    num_workers: int = 8
    device: str = "cuda"

    # --- subsampling (replaces distill) ---
    max_samples: int = 0                    # 0 = use all; >0 = random subsample
    subsample_seed: int = 42                # for reproducible subsampling

    # --- training ---
    lr: float = 5e-4
    num_epochs: int = 15
    patience: int = 5           # early-stop after N epochs without improvement (0=off)
    scheduler: SchedulerType = "none"   # "none" = flat LR, "onecycle" = OneCycleLR
    grad_clip: float = 50.0

    # --- loss ---
    loss_type: LossType = "bce"
    pos_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # --- model ---
    model_type: ModelType = "unet++"
    encoder_name: EncoderType = "resnet18"

    # --- run bookkeeping ---
    tag: str = "test"
    run_dir: str = ""               # if set, save checkpoint + result here
    max_batches: int = 0            # 0 = full epoch; >0 = cap batches per epoch (smoke test)
    perf: bool = False              # print per-epoch data/fwd/bwd timing (adds cuda.sync overhead)

    # --- transfer experiments ---
    train_regions: list | None = None    # if set, filter training to these region names
    exclude_regions: list | None = None  # if set, exclude these regions from ALL splits (LOO)

    # ---------------- helpers ----------------

    @classmethod
    def from_file(cls, path: str) -> "NextDayConfig":
        """Load JSON, drop unknown keys, instantiate."""
        with open(path) as f:
            raw = json.load(f)
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in valid})
