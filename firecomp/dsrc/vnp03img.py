"""VNP03IMG Geolocation data source.

VNP03IMG contains geolocation data (lat/lon) for VIIRS imagery.
Cloud/quality masks come from VNP14's fire_mask variable.

This module handles:
- Downloading VNP03IMG files for geolocation
- Extracting loss masks from VNP14 fire_mask (cloud, no-data, water, etc.)
"""

from collections import defaultdict
from dataclasses import dataclass
import logging
import os
import time
from typing import Optional, Set

import h5py
import netCDF4
import numpy as np
from pqdm.processes import pqdm
from tqdm import tqdm

from firecomp.config import config
from firecomp.dsrc.earthaccess import safe_download

# Suppress verbose logging from urllib3 and earthaccess
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("earthaccess").setLevel(logging.WARNING)
from firecomp.dsrc.vnp14 import DEG_CELL_SIZE, Fires
from firecomp.dsrc.viirs import (
    extract_vnp14_metadata,
    extract_vnp14_gring,
    bbox_intersects_gring,
    group_vnp14_by_date,
    viirs_granule_prefix,
    results_to_prefix_map,
    result_filename,
    group_prefixes_by_date,
)
from datetime import timedelta


# VNP14 fire_mask values (np.uint8)
class FireMaskValue:
    NOT_PROCESSED_NO_DATA = 0  # Not processed (no data or poor quality)
    NOT_PROCESSED_BOWTIE = 1  # Not processed (bowtie deletion)
    UNUSED = 2  # Unused
    WATER = 3  # Water
    CLOUD = 4  # Cloud
    LAND = 5  # Land (clear, no fire)
    UNCLASSIFIED = 6  # Unclassified
    FIRE_LOW = 7  # Low confidence fire pixel
    FIRE_NOMINAL = 8  # Nominal confidence fire pixel
    FIRE_HIGH = 9  # High confidence fire pixel


# Values to mask out in loss computation (don't penalize model for these)
# Note: WATER is NOT masked - model should learn to predict water areas
LOSS_MASK_VALUES = {
    FireMaskValue.NOT_PROCESSED_NO_DATA,
    FireMaskValue.NOT_PROCESSED_BOWTIE,
    FireMaskValue.UNUSED,
    FireMaskValue.CLOUD,
    FireMaskValue.UNCLASSIFIED,
}


@dataclass
class GeoData:
    """Geolocation data from VNP03IMG."""

    latitude: np.ndarray  # 2D array (6464, 6400)
    longitude: np.ndarray  # 2D array (6464, 6400)


def read_fire_mask(vnp14_path: str) -> np.ndarray:
    """Read fire_mask from VNP14 file.

    Args:
        vnp14_path: Path to VNP14IMG NetCDF file

    Returns:
        fire_mask array (6464, 6400) uint8
    """
    ds = netCDF4.Dataset(vnp14_path, "r")
    try:
        return ds.variables["fire mask"][:]
    finally:
        ds.close()


def _check_date_batch(
    dates_batch: list[tuple[str, list[str]]],  # [(date_str, vnp14_filenames), ...]
    fire_id_by_date: dict[str, list[int]],
    geod_bbox_by_id: dict[int, list[float]],
    vnp14_dir: str,
):
    """Check which VNP14 files overlap with fires for a batch of dates. (Module-level for pickling)"""
    vnp14_used = set()
    prefixes_by_date: dict[str, list[str]] = defaultdict(list)
    fire_ids_by_prefix: dict[str, list[int]] = defaultdict(list)

    for date_str, vnp14s in tqdm(dates_batch, desc="Checking VNP14 files"):
        fire_ids_for_date = fire_id_by_date.get(date_str, [])
        if not fire_ids_for_date:
            continue
        for vnp14_fn in vnp14s:
            prefix = viirs_granule_prefix(vnp14_fn)
            vnp14_path = os.path.join(vnp14_dir, vnp14_fn)

            # Use GRing polygon for more accurate coverage check
            gring = extract_vnp14_gring(vnp14_path)
            if not gring:
                # Fallback to bbox if GRing not available
                bbox = extract_vnp14_metadata(vnp14_path)
                if not bbox:
                    continue
                gring_lons = [bbox[0], bbox[0], bbox[2], bbox[2]]
                gring_lats = [bbox[1], bbox[3], bbox[3], bbox[1]]
            else:
                gring_lons, gring_lats = gring

            matching_fires = []
            for fire_id in fire_ids_for_date:
                geod_bbox = geod_bbox_by_id[fire_id]
                # Check if fire bbox intersects the swath polygon
                if bbox_intersects_gring(geod_bbox, gring_lons, gring_lats):
                    matching_fires.append(fire_id)
            if matching_fires:
                vnp14_used.add(vnp14_path)
                prefixes_by_date[date_str].append(prefix)
                fire_ids_by_prefix[prefix].extend(matching_fires)

    return (vnp14_used, dict(prefixes_by_date), dict(fire_ids_by_prefix))


def get_vnp14_on_fires(fire_ids, start_dates, end_dates, min_x, min_y, max_x, max_y):
    """Get VNP14 file paths for each fire."""
    print("Creating date ranges")

    fire_id_by_date: dict[str, list[int]] = defaultdict(list)
    for i in range(len(start_dates)):
        cur_dt = start_dates[i].date()
        end_dt = end_dates[i].date()
        while cur_dt <= end_dt:
            dt = cur_dt.strftime("%Y-%m-%d")
            fire_id_by_date[dt].append(fire_ids[i])
            cur_dt += timedelta(days=1)
    fire_id_by_date = dict(fire_id_by_date)  # convert for pickling
    print("Creating geod bboxes")

    geod_bbox_by_id: dict[int, list[float]] = {
        fire_ids[i]: [
            min_x[i],
            min_y[i],
            max_x[i],
            max_y[i],
        ]
        for i in range(len(fire_ids))
    }

    print("Listing vnp14 paths...")
    vnp14_paths = os.listdir(config.vnp14_dir)
    vnp14_by_date = group_vnp14_by_date(vnp14_paths)

    # Split dates into n_jobs chunks to minimize serialization overhead
    dates_list = list(vnp14_by_date.items())

    used_vnp14, prefixes_by_date, fire_ids_by_prefix = _check_date_batch(
        dates_list, fire_id_by_date, geod_bbox_by_id, config.vnp14_dir
    )

    return list(used_vnp14), prefixes_by_date, fire_ids_by_prefix


class VNP03IMG:
    """Handler for VNP03IMG geolocation data."""

    def __init__(self):
        import earthaccess

        earthaccess.login()
        self._earthaccess = earthaccess

    def download(
        self,
        vnp14_paths: list[str],
        date_strs: list[str],
        output_dir: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        """Download VNP03IMG files for the given VNP14 file paths.

        Args:
            vnp14_paths: List of VNP14 file paths on the filesystem
            output_dir: Directory to save files (defaults to config.vnp03img_dir)
            max_retries: Number of retries on failure

        Returns:
            List of (vnp14_path, vnp03_path) tuples
        """
        output_dir = output_dir or config.vnp03img_dir
        os.makedirs(output_dir, exist_ok=True)

        # Extract granule prefixes from VNP14 paths and track mapping
        prefix_to_vnp14: dict[str, str] = {}
        for vnp14_path in vnp14_paths:
            prefix = viirs_granule_prefix(vnp14_path)
            if prefix:
                prefix_to_vnp14[prefix] = vnp14_path

        print(f"Searching {len(vnp14_paths)} granules across {len(date_strs)} days")

        # Track vnp03 paths by their prefix for pairing later
        all_results: list = []
        vnp03_found = set()

        for date_str in tqdm(date_strs, desc="Getting VNP03IMG download urls by date"):
            results = self._earthaccess.search_data(
                short_name="VNP03IMG",
                temporal=(date_str, date_str),
                count=-1,
            )
            for r in results:
                prefix = viirs_granule_prefix(result_filename(r))
                if prefix in prefix_to_vnp14 and prefix not in vnp03_found:
                    all_results.append(r)
                    vnp03_found.add(prefix)

        print(f"Downloading VNP03IMG files")
        prefix_to_vnp03 = safe_download(
            all_results, output_dir, delete_invalid=True, max_pending_wait=-1
        )

        # Build result pairs: (vnp14_path, vnp03_path)
        result_pairs: list[tuple[str, str]] = []
        for prefix, vnp14_path in prefix_to_vnp14.items():
            vnp03_path = prefix_to_vnp03.get(prefix)
            if vnp03_path:
                result_pairs.append((vnp14_path, vnp03_path))

        return result_pairs


if __name__ == "__main__":
    h5 = h5py.File(config.vnp14_path, "r")
    fires: Fires.Stats = Fires.load(h5, "stats_100")
    fires = Fires.mask(fires, fires.start_date > np.datetime64("2025-01-26"))
    print(f"Num fires: {len(fires.id)}")
    vnp14_on_fires = get_vnp14_on_fires(fires)
    print(f"Num VNP14 files on fires: {len(vnp14_on_fires)}")
