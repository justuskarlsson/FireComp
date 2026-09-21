from PIL import Image
import h5py
from tqdm import tqdm
from firecomp.dsrc.dsrc import DSRC, pipeline
from firecomp.config import config
import os
from datetime import date, datetime, timedelta
import numpy as np
from firecomp.dsrc.vnp14 import Fires
from firecomp.dsrc.weather import Grid


class FiresDaily(DSRC):
    name = "fires_daily"
    channel_names = ["fire"]
    no_data = 0.0

    def __init__(self):
        super().__init__()
        self.data_dir = os.path.join(config.data_dir, "fires_daily")
        os.makedirs(self.data_dir, exist_ok=True)
        self.grid = Grid.from_nx(8)

    def get_for_day(self, day: date, x, y, size: int, ret_component: bool = False):
        h5 = h5py.File(os.path.join(config.vnp14_fires_dir, "vnp14_fires_2012.h5"), "r")
        meta: Fires.Meta = Fires.load(h5, "meta")
        t = (day - meta.start_date.item().date()).days
        proj: Fires.Projection | None = Fires.load(h5, "projection_by_t", str(t))
        if proj is None:
            return None
        res = []
        for xx, yy in zip(x, y):
            x0 = xx - size // 2
            y0 = yy - size // 2
            x1 = xx + size // 2
            y1 = yy + size // 2
            mask = (proj.x >= x0) & (proj.x < x1) & (proj.y >= y0) & (proj.y < y1)
            if ret_component:
                fire = np.full((x1 - x0, y1 - y0), -1, dtype=np.int32)
            else:
                fire = np.zeros((x1 - x0, y1 - y0), dtype=np.int32)
            xx = proj.x[mask]
            yy = proj.y[mask]
            if ret_component:
                fire[xx - x0, yy - y0] = proj.component[mask]
            else:
                fire[xx - x0, yy - y0] = 1
            res.append(fire)
        return res

    def get_png(self, arr: np.ndarray, **kwargs) -> Image.Image:
        mapping = {
            0: (0, 0, 0, 0),
            1: (255, 0, 0, 255),
        }
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        for key, val in mapping.items():
            rgba[arr == key] = val
        return Image.fromarray(rgba)


class FiresDailyNextDay(FiresDaily):
    """
    TODO:
    - Polygon stuff
    - SDF stuff (depends on Poylgon)

    """

    name = "fires_next_day"
    channel_names = ["fire"]
    no_data = 0.0

    def get_for_day(self, day: date, x, y, size: int, ret_component: bool = False):
        day = day + timedelta(days=1)
        return super().get_for_day(day, x, y, size, ret_component)


class FiresAccum(DSRC):
    name = "fires_accum"
    channel_names = ["accum", "cur_spread", "next_accum", "next_ignition"]
    no_data = 0.0

    def __init__(self):
        super().__init__()
        self._h5 = None  # Lazy initialization
    
    @property
    def h5(self):
        """Lazy load HDF5 file only when accessed."""
        if self._h5 is None:
            self._h5 = h5py.File(config.vnp14_path, "r")
        return self._h5

    def get_for_day(
        self,
        accum_by_fire: dict[int, np.ndarray],
        accum_t_by_fire: dict[int, np.ndarray],
        bbox_by_fire: dict[int, tuple[int, int, int, int]],
        t: int,
        fire_mask_by_id: dict[int, np.ndarray],
    ):
        """
        bbox, non-inclusive ends.

        """
        proj: Fires.Projection | None = Fires.load(self.h5, "projection_by_t", str(t))
        if proj is None:
            return None, None
        cur_by_fire = dict()
        ignition_by_fire = dict()
        missing_mask_ids = []
        for fire_id in accum_by_fire:
            bbox = bbox_by_fire[fire_id]

            height = bbox[3] - bbox[1]
            if fire_id in fire_mask_by_id:
                fire_bool = fire_mask_by_id[fire_id][0] >= 7
            else:
                missing_mask_ids.append(fire_id)
                # No fire mask data - allow all fire pixels
                fire_bool = np.ones((height, bbox[2] - bbox[0]), dtype=np.bool_)
            mask = (
                (proj.x >= bbox[0])
                & (proj.x < bbox[2])
                & (proj.y >= bbox[1])
                & (proj.y < bbox[3])
            )
            x = proj.x[mask] - bbox[0]
            y = proj.y[mask] - bbox[1]
            # fire_bool is flipped (y=0 at top), but proj.y is not (y=0 at south)
            # Flip y to match fire_bool coordinate system
            c = proj.component[mask]
            accum_c = accum_by_fire[fire_id]
            accum_t = accum_t_by_fire[fire_id]
            # &-operation with more accurate fire mask
            y_flipped = (height - 1) - y
            is_fire = fire_bool[y_flipped, x]
            y = y[is_fire]
            x = x[is_fire]
            c = c[is_fire]
            cur_spread = np.full(
                (bbox[3] - bbox[1], bbox[2] - bbox[0]), -1, dtype=np.int32
            )
            ignition = np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0]), dtype=np.int32)
            if len(y):
                accum_c[y, x] = c
                # Debug: show first assignment
                if len(cur_by_fire) < 2:
                    unique_c = np.unique(c)
                    print(
                        f"  [DEBUG assign] fire_id={fire_id}, assigning {len(y)} pixels, component values: {unique_c[:5]}"
                    )
                # Shouldn't we do the highest t for the fire?
                # More prior on where it will spread: No, instead include cur_spread in data
                new_ts = accum_t[y, x]
                new_ts[new_ts == -1] = t
                accum_t[y, x] = new_ts
                cur_spread[y, x] = c
                ign = proj.ignition[mask][is_fire]
                ignition[y, x] = ign
            cur_by_fire[fire_id] = cur_spread
            ignition_by_fire[fire_id] = ignition

        if missing_mask_ids:
            print(
                f"  [get_for_day t={t}] {len(missing_mask_ids)}/{len(accum_by_fire)} fires missing fire_mask (using fallback)"
            )
        return cur_by_fire, ignition_by_fire
