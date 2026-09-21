"""
tests/test_figures.py — Smoke tests for paper figure functions.

Generates figures with mock data and saves them to data/tests/figures/.
Channel arrays are mocked (random); region raster is the real
wildfire_regions.tif — only the sample counts dict is synthetic.

Run:
    pytest tests/test_figures.py -v
"""

import numpy as np
import matplotlib.pyplot as plt

from firecomp.next_day.figures import (
    fig_cluster_burnability,
    fig_cluster_lc,
    fig_input_grid,
    fig_input_group,
    fig_fire_types,
    fig_lc_ratio,
    fig_prediction_basemap,
    fig_prediction_samples,
    fig_region_extents,
    fig_region_samples,
    fig_region_stats,
    fig_sample_density,
    fig_size_and_region_f1,
    assign_size_bucket,
    REGION_SHORT,
    SIZE_BUCKETS,
)

OUT_DIR = "data/tests/figures"
RNG = np.random.default_rng(42)
H, W = 256, 256


# ---------------------------------------------------------------------------
# fig:inputs — one PDF per channel group
# ---------------------------------------------------------------------------

class TestFigInputGroup:
    """fig_input_group — simple channel grid for one group."""

    def test_fire_state(self):
        accum_t = np.full((H, W), -1.0, dtype=np.float32)
        yy, xx = np.ogrid[:H, :W]
        for day in range(1, 9):
            radius = 15 + day * 8
            accum_t[((yy - H // 2) ** 2 + (xx - W // 2) ** 2) < radius ** 2] = day

        cur_mask = np.zeros((H, W), dtype=np.float32)
        cur_mask[((yy - H // 2) ** 2 + (xx - W // 2) ** 2) < 40 ** 2] = 1.0

        fig = fig_input_group(
            {"accum_t": accum_t, "cur_mask": cur_mask},
            "Fire State",
            cmaps={"accum_t": "YlOrRd", "cur_mask": "Reds"},
            nodata={"accum_t": -1.0},
            ncols=2,
            out_path=f"{OUT_DIR}/inputs_fire_state",
        )
        assert fig is not None
        plt.close(fig)

    def test_terrain(self):
        channels = {
            "PCA 1": RNG.normal(0, 1, (H, W)).astype(np.float32),
            "PCA 2": RNG.normal(0, 1, (H, W)).astype(np.float32),
            "PCA 3": RNG.normal(0, 1, (H, W)).astype(np.float32),
            "PCA 4": RNG.normal(0, 1, (H, W)).astype(np.float32),
            "PCA 5": RNG.normal(0, 1, (H, W)).astype(np.float32),
            "Pos sin(lat)": RNG.uniform(-1, 1, (H, W)).astype(np.float32),
            "Pos cos(lat)": RNG.uniform(-1, 1, (H, W)).astype(np.float32),
            "Pos sin(lon)": RNG.uniform(-1, 1, (H, W)).astype(np.float32),
            "Pos cos(lon)": RNG.uniform(-1, 1, (H, W)).astype(np.float32),
        }
        cmaps = {k: "bwr" for k in channels if k.startswith("Pos")}
        fig = fig_input_group(
            channels, "Terrain",
            cmaps=cmaps, ncols=3,
            out_path=f"{OUT_DIR}/inputs_terrain",
        )
        assert fig is not None
        plt.close(fig)

    def test_weather(self):
        channels = {
            # ERA5 (current day)
            "Current · VPD": RNG.uniform(0, 40, (H, W)).astype(np.float32),
            "Current · Soil Moist.": RNG.uniform(0, 1, (H, W)).astype(np.float32),
            "Current · Soil Ratio": RNG.uniform(0, 3, (H, W)).astype(np.float32),
            "Current · Wind Mag.": RNG.uniform(0, 20, (H, W)).astype(np.float32),
            "Current · Wind sin": RNG.uniform(-1, 1, (H, W)).astype(np.float32),
            "Current · Wind cos": RNG.uniform(-1, 1, (H, W)).astype(np.float32),
            # GFS (next-day forecast)
            "Forecast · Temp Max": RNG.uniform(290, 320, (H, W)).astype(np.float32),
            "Forecast · RH Min": RNG.uniform(10, 90, (H, W)).astype(np.float32),
            "Forecast · U Wind Max": RNG.uniform(-15, 15, (H, W)).astype(np.float32),
            "Forecast · V Wind Max": RNG.uniform(-15, 15, (H, W)).astype(np.float32),
            "Forecast · Precip Sum": RNG.uniform(0, 20, (H, W)).astype(np.float32),
        }
        cmaps = {
            "Current · VPD": "YlOrRd", "Current · Soil Moist.": "YlGnBu",
            "Current · Wind sin": "bwr", "Current · Wind cos": "bwr",
            "Forecast · U Wind Max": "bwr", "Forecast · V Wind Max": "bwr",
            "Forecast · Precip Sum": "YlGnBu",
        }
        fig = fig_input_group(
            channels, "Weather",
            cmaps=cmaps, ncols=3,
            out_path=f"{OUT_DIR}/inputs_weather",
        )
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig:regions — world map coloured by sample count (real raster)
# ---------------------------------------------------------------------------

class TestFigRegionSamples:
    """fig_region_samples — Cartopy rasterised heatmap, real region raster."""

    def test_train(self):
        region_samples = {
            "Western Europe": 1200,
            "Eastern Europe": 800,
            "MENA": 350,
            "Africa": 950,
            "North Asia": 1500,
            "South Asia": 600,
            "Oceania": 1100,
            "North NA": 1800,
            "Central NA": 2200,
            "South America": 700,
        }
        fig = fig_region_samples(
            region_samples,
            title="Train — samples per region",
            out_path=f"{OUT_DIR}/regions_train",
        )
        assert fig is not None
        plt.close(fig)

    def test_test_split(self):
        """Test split — smaller counts, checks unbalance is visible."""
        region_samples = {
            "Western Europe": 200,
            "Eastern Europe": 150,
            "MENA": 50,
            "Africa": 180,
            "North Asia": 300,
            "South Asia": 100,
            "Oceania": 220,
            "North NA": 350,
            "Central NA": 400,
            "South America": 120,
        }
        fig = fig_region_samples(
            region_samples,
            title="Test — samples per region",
            out_path=f"{OUT_DIR}/regions_test",
        )
        assert fig is not None
        plt.close(fig)

    def test_few_regions(self):
        """Only a subset of regions present."""
        fig = fig_region_samples(
            {"North NA": 500, "Oceania": 300},
            title="Subset",
        )
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig:region_stats — class balance bar chart
# ---------------------------------------------------------------------------

class TestFigRegionStats:
    """fig_region_stats — horizontal bar chart with pos ratio + nodata frac."""

    def test_basic(self):
        stats = {
            "Western Europe": {"pos_ratio": 0.0032, "nodata_ratio": 0.35, "n_samples": 1200},
            "North NA":       {"pos_ratio": 0.0018, "nodata_ratio": 0.22, "n_samples": 1800},
            "Central NA":     {"pos_ratio": 0.0041, "nodata_ratio": 0.18, "n_samples": 2200},
            "Africa":         {"pos_ratio": 0.0085, "nodata_ratio": 0.45, "n_samples": 950},
            "Oceania":        {"pos_ratio": 0.0055, "nodata_ratio": 0.30, "n_samples": 1100},
            "MENA":         {"pos_ratio": 0.0012, "nodata_ratio": 0.52, "n_samples": 350},
        }
        fig = fig_region_stats(
            stats,
            title="Train — class balance by region",
            out_path=f"{OUT_DIR}/region_stats_train",
        )
        assert fig is not None
        plt.close(fig)

    def test_single_region(self):
        stats = {
            "North NA": {"pos_ratio": 0.002, "nodata_ratio": 0.2, "n_samples": 500},
        }
        fig = fig_region_stats(stats, title="Single region")
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig:fire_types — 3×3 grid of fire type examples
# ---------------------------------------------------------------------------

def _mock_fire_raster(h: int, w: int, n_days: int = 8) -> np.ndarray:
    """Synthetic fire detection raster (expanding disc)."""
    raster = np.full((h, w), np.nan, dtype=np.float32)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    for day in range(n_days):
        radius = 3 + day * max(1, min(h, w) // (2 * n_days))
        new = ((yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2) & np.isnan(raster)
        raster[new] = day
    return raster


class TestFigFireTypes:
    """fig_fire_types — 3×3 grid of fire type detection rasters."""

    def test_basic(self):
        # Static: small, compact
        static = [_mock_fire_raster(20, 20, 3) for _ in range(3)]
        # Crop: medium, scattered
        crop = [_mock_fire_raster(40, 60, 4) for _ in range(3)]
        # Wild: large, spreading
        wild = [_mock_fire_raster(80, 120, 10) for _ in range(3)]

        panels = [static, crop, wild]
        type_labels = ["Static", "Crop Burn", "Wildfire"]
        fire_labels = [
            ["Fire 1001 (50 det.)", "Fire 1002 (45 det.)", "Fire 1003 (40 det.)"],
            ["Fire 2001 (200 det.)", "Fire 2002 (180 det.)", "Fire 2003 (150 det.)"],
            ["Fire 3001 (5000 det.)", "Fire 3002 (3000 det.)", "Fire 3003 (2000 det.)"],
        ]

        fig = fig_fire_types(
            panels, type_labels, fire_labels,
            title="Fire types — detection spread",
            out_path=f"{OUT_DIR}/fire_types",
        )
        assert fig is not None
        assert len(fig.axes) >= 9  # 3×3 panels + colorbar
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig:clusters — world map coloured by tame-to-wild ratio (mock data)
# ---------------------------------------------------------------------------

def _mock_ratio_region(
    ratio_map: np.ndarray,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    *,
    rng: np.random.Generator,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> None:
    """Fill a lat/lon box in the ratio_map with lognormal ratios (in-place)."""
    r0 = int((90 - lat_range[1]) / 0.1)
    r1 = int((90 - lat_range[0]) / 0.1)
    c0 = int((lon_range[0] + 180) / 0.1)
    c1 = int((lon_range[1] + 180) / 0.1)
    ratio_map[r0:r1, c0:c1] = rng.lognormal(mu, sigma, (r1 - r0, c1 - c0)
                                              ).clip(0.01, 100).astype(np.float32)


class TestFigClusterBurnability:
    """fig_cluster_burnability — world map coloured by tame-to-wild ratio."""

    def test_basic(self):
        """Synthetic ratio map with several continental regions filled."""
        ratio_map = np.full((1800, 3600), np.nan, dtype=np.float32)
        rng = np.random.default_rng(42)

        # Western Europe: mixed tame/wild (centered around 1)
        _mock_ratio_region(ratio_map, (35, 60), (-10, 30), rng=rng, mu=0.0, sigma=1.0)
        # North America: more wildfire-dominated (lower ratios)
        _mock_ratio_region(ratio_map, (25, 60), (-130, -60), rng=rng, mu=-0.5, sigma=1.2)
        # Australia: moderately tame
        _mock_ratio_region(ratio_map, (-40, -10), (110, 155), rng=rng, mu=0.3, sigma=0.8)
        # Sub-Saharan Africa: high tame (crop burns)
        _mock_ratio_region(ratio_map, (-35, 15), (-15, 50), rng=rng, mu=1.0, sigma=0.6)
        # North Asia / Siberia: wildfire-dominated
        _mock_ratio_region(ratio_map, (50, 70), (60, 150), rng=rng, mu=-1.0, sigma=0.8)

        fig = fig_cluster_burnability(
            ratio_map,
            title="Cluster burnability — tame-to-wild ratio (mock)",
            out_path=f"{OUT_DIR}/cluster_burnability",
        )
        assert fig is not None
        plt.close(fig)

    def test_uniform_ratio(self):
        """All valid cells at ratio=1 — colorbar should still render."""
        ratio_map = np.full((1800, 3600), np.nan, dtype=np.float32)
        ratio_map[300:550, 1700:2100] = 1.0

        fig = fig_cluster_burnability(ratio_map, title="Uniform ratio")
        assert fig is not None
        plt.close(fig)

    def test_empty(self):
        """All NaN — no valid cells, should not crash."""
        ratio_map = np.full((1800, 3600), np.nan, dtype=np.float32)
        fig = fig_cluster_burnability(ratio_map, title="No data")
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig:cluster_lc — world map coloured by dominant land-cover class
# ---------------------------------------------------------------------------

class TestFigClusterLC:
    """fig_cluster_lc — world map with categorical LC classes and legend."""

    LC_NAMES = [
        "EG_Needle", "EG_Broad", "Shrub", "Grass",
        "Cropland", "Bare", "Water",
    ]

    def _fill_box(
        self, lc_map: np.ndarray,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        cls_idx: int,
    ) -> None:
        """Fill a lat/lon box in lc_map with a given class index."""
        r0 = int((90 - lat_range[1]) / 0.1)
        r1 = int((90 - lat_range[0]) / 0.1)
        c0 = int((lon_range[0] + 180) / 0.1)
        c1 = int((lon_range[1] + 180) / 0.1)
        lc_map[r0:r1, c0:c1] = cls_idx

    def test_basic(self):
        """Several continental regions filled with different LC classes."""
        lc_map = np.full((1800, 3600), -1, dtype=np.int8)

        # Boreal needle-leaf across Canada / Siberia
        self._fill_box(lc_map, (50, 65), (-130, -60), 0)   # EG_Needle
        self._fill_box(lc_map, (55, 70), (60, 150), 0)     # EG_Needle
        # Tropical broadleaf — Amazon + Congo
        self._fill_box(lc_map, (-15, 5), (-75, -45), 1)    # EG_Broad
        self._fill_box(lc_map, (-5, 10), (15, 30), 1)      # EG_Broad
        # Shrub — Australia interior
        self._fill_box(lc_map, (-30, -15), (120, 145), 2)  # Shrub
        # Grass — East African savanna
        self._fill_box(lc_map, (-15, 10), (25, 42), 3)     # Grass
        # Cropland — Western Europe + India
        self._fill_box(lc_map, (40, 55), (-5, 25), 4)      # Cropland
        self._fill_box(lc_map, (10, 30), (70, 90), 4)      # Cropland
        # Bare — Sahara
        self._fill_box(lc_map, (15, 35), (-15, 40), 5)     # Bare
        # Water — just a small patch
        self._fill_box(lc_map, (55, 65), (25, 50), 6)      # Water

        fig = fig_cluster_lc(
            lc_map, self.LC_NAMES,
            title="Dominant LC class per cell (mock)",
            out_path=f"{OUT_DIR}/cluster_lc",
        )
        assert fig is not None
        plt.close(fig)

    def test_single_class(self):
        """Only one class present — legend should have one entry."""
        lc_map = np.full((1800, 3600), -1, dtype=np.int8)
        self._fill_box(lc_map, (30, 60), (-10, 40), 3)  # Grass
        fig = fig_cluster_lc(lc_map, self.LC_NAMES, title="Single class")
        assert fig is not None
        plt.close(fig)

    def test_empty(self):
        """All -1 — no valid cells, should not crash."""
        lc_map = np.full((1800, 3600), -1, dtype=np.int8)
        fig = fig_cluster_lc(lc_map, self.LC_NAMES, title="No data")
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig:lc_ratio — tame-to-wild ratio bar chart per LC class
# ---------------------------------------------------------------------------

LC_NAMES_MOCK = [
    "Bare", "Cropland", "DC_Broad", "DC_Needle", "EG_Broad",
    "EG_Needle", "Grass", "Mixed_Forest", "Shrub",
]


class TestFigLcRatio:
    """fig_lc_ratio — horizontal bar chart of T:W ratio per LC class."""

    def test_weighted(self):
        ratios = {
            "EG_Needle": 0.15, "EG_Broad": 0.08, "DC_Broad": 0.25,
            "DC_Needle": 0.12, "Mixed_Forest": 0.20, "Shrub": 0.80,
            "Grass": 1.50, "Cropland": 12.0, "Bare": 0.40,
        }
        n_wild = {
            "EG_Needle": 8500, "EG_Broad": 3200, "DC_Broad": 1200,
            "DC_Needle": 600, "Mixed_Forest": 2100, "Shrub": 4500,
            "Grass": 9800, "Cropland": 1100, "Bare": 350,
        }
        n_tame = {
            "EG_Needle": 200, "EG_Broad": 50, "DC_Broad": 80,
            "DC_Needle": 30, "Mixed_Forest": 150, "Shrub": 900,
            "Grass": 6200, "Cropland": 18000, "Bare": 60,
        }
        fig = fig_lc_ratio(
            LC_NAMES_MOCK, ratios,
            n_wild=n_wild, n_tame=n_tame,
            title="T:W ratio per LC (sqrt-weighted, mock)",
            out_path=f"{OUT_DIR}/lc_ratio_weighted",
        )
        assert fig is not None
        plt.close(fig)

    def test_unweighted(self):
        ratios = {
            "EG_Needle": 0.02, "EG_Broad": 0.01, "DC_Broad": 0.07,
            "Grass": 0.63, "Cropland": 16.0, "Shrub": 0.20,
        }
        fig = fig_lc_ratio(
            LC_NAMES_MOCK, ratios,
            title="T:W ratio per LC (unweighted, mock)",
            out_path=f"{OUT_DIR}/lc_ratio_unweighted",
        )
        assert fig is not None
        plt.close(fig)

    def test_empty(self):
        """No ratios — should not crash."""
        fig = fig_lc_ratio(LC_NAMES_MOCK, {}, title="Empty")
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig:lc_region — LC × region heatmap
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# fig:input_grid — 3-col grid with colorbars
# ---------------------------------------------------------------------------

class TestFigInputGrid:
    def test_basic(self):
        channels = {
            "accum_t": RNG.uniform(-1, 5, (H, W)).astype(np.float32),
            "cur_mask": RNG.uniform(0, 1, (H, W)).astype(np.float32),
            "PCA 1": RNG.normal(0, 1, (H, W)).astype(np.float32),
            "next_mask": RNG.uniform(0, 1, (H, W)).astype(np.float32),
        }
        fig = fig_input_grid(
            channels,
            nodata={"accum_t": -1.0},
            out_path=f"{OUT_DIR}/input_grid",
        )
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig:region_extents — single map with region colours + legend
# ---------------------------------------------------------------------------

class TestFigRegionExtents:
    def test_basic(self):
        region_names = {
            1: "Western Europe", 2: "MENA", 3: "Africa",
            4: "North Asia", 5: "South Asia", 6: "Oceania",
            7: "North NA", 8: "Central NA", 9: "South America",
            10: "Eastern Europe",
        }
        fig = fig_region_extents(
            region_names,
            title="Study regions",
            out_path=f"{OUT_DIR}/region_extents",
        )
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig:size_and_region_f1 — scatter of F1 by region × fire-size bucket
# ---------------------------------------------------------------------------

class TestFigSizeAndRegionF1:
    """fig_size_and_region_f1 — one dot per (region, bucket) group."""

    def _make_data(self, rng, *, sparse=False):
        """One entry per (region, bucket) group with pooled pixel-level F1.

        F1 values are kept in the 0–0.6 range to test y-axis auto-scaling.
        When *sparse*, some groups have n_samples below min_count.
        """
        data = []
        regions = list(REGION_SHORT.keys())
        buckets = [b[0] for b in SIZE_BUCKETS]
        for region in regions:
            for bucket in buckets:
                n = rng.integers(3, 12) if sparse else rng.integers(15, 80)
                # F1 varies by bucket: large fires → higher pooled F1
                if "5k" in bucket:
                    f1 = float(rng.uniform(0.30, 0.60))
                elif "500" in bucket:
                    f1 = float(rng.uniform(0.15, 0.45))
                else:
                    f1 = float(rng.uniform(0.02, 0.25))
                data.append({
                    "region": region,
                    "f1": f1,
                    "size_bucket": bucket,
                    "n_samples": int(n),
                })
        return data

    def test_basic(self):
        rng = np.random.default_rng(42)
        data = self._make_data(rng)
        fig = fig_size_and_region_f1(
            data,
            min_count=10,
            title="Per-pixel F1 by region and fire size (mock)",
            out_path=f"{OUT_DIR}/size_and_region_f1",
        )
        assert fig is not None
        plt.close(fig)

    def test_sparse_data(self):
        """Some groups have fewer than min_count — should be omitted."""
        rng = np.random.default_rng(99)
        data = self._make_data(rng, sparse=True)
        fig = fig_size_and_region_f1(
            data, min_count=10,
            title="Sparse data (mock)",
        )
        assert fig is not None
        plt.close(fig)

    def test_single_region(self):
        data = [{"region": "Africa", "f1": 0.35,
                 "size_bucket": "500–5k", "n_samples": 30}]
        fig = fig_size_and_region_f1(data, min_count=5)
        assert fig is not None
        plt.close(fig)

    def test_assign_bucket(self):
        assert assign_size_bucket(0) == "< 500"
        assert assign_size_bucket(499) == "< 500"
        assert assign_size_bucket(500) == "500–5k"
        assert assign_size_bucket(4999) == "500–5k"
        assert assign_size_bucket(5000) == "5k+"
        assert assign_size_bucket(100000) == "5k+"
