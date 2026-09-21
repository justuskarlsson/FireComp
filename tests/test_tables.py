"""
tests/test_tables.py — Smoke tests for LaTeX table generation.

Tests use a mock dataset object with fake samples so they don't need
real H5 files on disk.  Transfer and grid tests use minimal mock JSON.

Run:
    pytest tests/test_tables.py -v
"""

from dataclasses import dataclass
from firecomp.next_day.tables import (
    table_split_summary,
    table_region_split,
    table_region_year,
    table_fire_type_region,
    _table_transfer_matrix,
    table_loo,
    table_transfer_summary,
    table_grid_models,
    table_grid_losses,
    table_grid_targets,
    table_grid_regions,
    table_grid_all,
    table_solidity,
)
from firecomp.next_day.latex import tex_escape_text


# ---------------------------------------------------------------------------
# Mock dataset — just enough to satisfy the table functions
# ---------------------------------------------------------------------------

@dataclass
class MockSample:
    fire_id: int
    dt: str
    region_id: int
    xi: int = 0
    yi: int = 0
    lon: float = 0.0
    lat: float = 0.0
    img_size: int = 256
    fire_type: int = 0


class MockDataset:
    """Minimal stand-in for NextDayDataset with pre-built sample lists."""

    def __init__(self, train, val, test):
        self._train = train
        self._val = val
        self._test = test

    @property
    def train_samples(self):
        return self._train

    @property
    def val_samples(self):
        return self._val

    @property
    def test_samples(self):
        return self._test


def _make_ds():
    """Build a small mock dataset spanning 2 regions × 3 years."""
    train = [
        MockSample(1, "2019-06-01", 7, fire_type=0),
        MockSample(1, "2019-06-02", 7, fire_type=0),
        MockSample(2, "2020-03-15", 3, fire_type=2),   # crop
        MockSample(3, "2020-07-10", 7, fire_type=0),
        MockSample(4, "2021-01-05", 6, fire_type=0),
        MockSample(5, "2021-08-20", 3, fire_type=1),   # static
    ]
    val = [
        MockSample(6, "2019-09-12", 7, fire_type=0),
        MockSample(7, "2020-04-01", 3, fire_type=2),   # crop
    ]
    test = [
        MockSample(8, "2022-01-10", 7, fire_type=0),
        MockSample(9, "2022-02-20", 3, fire_type=0),
        MockSample(10, "2022-06-15", 6, fire_type=1),  # static
    ]
    return MockDataset(train, val, test)


# ---------------------------------------------------------------------------
# Dataset table tests
# ---------------------------------------------------------------------------

class TestSplitSummary:
    def test_basic(self):
        ds = _make_ds()
        tex = table_split_summary(None, ds, None)
        assert r"\begin{table}" in tex
        assert r"\caption{#1}" in tex
        assert r"\label{tab:split-summary}" in tex
        assert "Train" in tex
        assert "Val" in tex
        assert "Test" in tex
        assert "2019-06-01" in tex   # train start date
        assert "Total" in tex
        assert r"\end{table}" in tex

    def test_counts(self):
        ds = _make_ds()
        tex = table_split_summary(None, ds, None)
        # 6 train + 2 val + 3 test = 11 total
        assert "11" in tex


class TestRegionSplit:
    def test_basic(self):
        ds = _make_ds()
        tex = table_region_split(None, ds, None)
        assert r"\begin{table}" in tex
        assert "Train" in tex
        assert "Val" in tex
        assert "Test" in tex
        assert "Total" in tex
        assert r"\end{table}" in tex

    def test_has_region_names(self):
        ds = _make_ds()
        tex = table_region_split(None, ds, None)
        # Region 7 = North NA, 3 = Africa, 6 = Oceania
        assert "N.~NA" in tex
        assert "Afr" in tex
        assert "Oce" in tex


class TestRegionYear:
    def test_basic(self):
        ds = _make_ds()
        tex = table_region_year(None, ds, None)
        assert r"\begin{table}[H]" in tex
        assert "2019" in tex
        assert "2020" in tex
        assert "2021" in tex
        assert "2022" in tex
        assert "Total" in tex
        assert r"\end{table}" in tex
        assert r"table*" not in tex

    def test_has_dashes_for_zero(self):
        ds = _make_ds()
        tex = table_region_year(None, ds, None)
        # Oceania (region 6) only has samples in 2021 and 2022
        # So 2019 and 2020 should show "--"
        assert "--" in tex


class TestFireTypeRegion:
    def test_basic(self):
        ds = _make_ds()
        tex = table_fire_type_region(None, ds, None)
        assert r"\begin{table}" in tex
        assert "Veg" in tex
        assert "Static" in tex
        assert "Crop" in tex
        assert "Total" in tex
        assert r"\end{table}" in tex

    def test_has_region_names(self):
        ds = _make_ds()
        tex = table_fire_type_region(None, ds, None)
        assert "N.~NA" in tex
        assert "Afr" in tex
        assert "Oce" in tex


# ---------------------------------------------------------------------------
# Transfer table tests (mock results)
# ---------------------------------------------------------------------------

# {target_type: {train_label: {eval_region: F1}}} — the shape build_transfer_matrix
# produces and save_matrix writes.
MOCK_MATRIX = {
    "next_mask": {
        "global": {"Western Europe": 0.35, "Africa": 0.42, "North NA": 0.38, "global": 0.40},
        "Western Europe": {"Western Europe": 0.40, "Africa": 0.30, "North NA": 0.25, "global": 0.32},
        "loo_Western Europe": {"Western Europe": 0.33, "Africa": 0.41, "North NA": 0.37, "global": 0.39},
    },
    "new_fires": {
        "global": {"Western Europe": 0.22, "Africa": 0.28, "North NA": 0.25, "global": 0.26},
        "loo_Western Europe": {"Western Europe": 0.20, "Africa": 0.27, "North NA": 0.24, "global": 0.25},
    },
}


class TestTransferMatrix:
    def test_next_mask(self):
        tex = _table_transfer_matrix(MOCK_MATRIX["next_mask"], "next_mask")
        assert r"\begin{table}[H]" in tex
        assert r"\caption{#1}" in tex
        assert r"\label{tab:transfer-next-mask}" in tex
        assert "W.~Eur" in tex
        assert r"\end{table}" in tex

    def test_new_fires(self):
        tex = _table_transfer_matrix(MOCK_MATRIX["new_fires"], "new_fires")
        assert r"\label{tab:transfer-new-fires}" in tex


class TestTransferSummary:
    def test_basic(self):
        tex = table_transfer_summary(None, None, None, matrix=MOCK_MATRIX)
        assert r"\begin{table}" in tex
        assert "W.~Eur" in tex
        assert "Global" in tex


# ---------------------------------------------------------------------------
# Grid search table tests (mock results)
# ---------------------------------------------------------------------------

MOCK_GRID = [
    {
        "tag": "grid_unet_bce_pw10_newf",
        "model": "unet",
        "loss": "bce",
        "target": "new_fires",
        "val_f1": 0.2348,
        "threshold": 0.6,
        "test_f1": 0.2558,
        "test_precision": 0.2426,
        "test_recall": 0.2705,
        "test_iou": 0.1467,
        "test_brier": 0.0084,
        "by_region": {
            "Oceania": 0.1765,
            "South America": 0.2999,
            "Africa": 0.0393,
            "Western Europe": 0.1382,
            "North Asia": 0.3318,
            "North NA": 0.3435,
            "unknown_0": 0.0,
        },
    },
    {
        "tag": "grid_unet_bce_pw10_next",
        "model": "unet",
        "loss": "bce",
        "target": "next_mask",
        "val_f1": 0.2836,
        "threshold": 0.65,
        "test_f1": 0.3041,
        "test_precision": 0.2909,
        "test_recall": 0.3186,
        "test_iou": 0.1793,
        "test_brier": 0.0101,
        "nf_equiv_f1": 0.2574,
        "nf_equiv_precision": 0.2365,
        "nf_equiv_recall": 0.2823,
        "nf_equiv_iou": 0.1477,
        "nf_equiv_brier": 0.009,
        "nf_equiv_threshold": 0.6,
        "by_region": {
            "Oceania": 0.1315,
            "South America": 0.3445,
            "Africa": 0.0317,
            "Western Europe": 0.1423,
            "North Asia": 0.356,
            "North NA": 0.4011,
            "unknown_0": 0.5263,
        },
        "nf_equiv_by_region": {
            "Oceania": 0.1609,
            "South America": 0.2974,
            "Africa": 0.0452,
            "Western Europe": 0.134,
            "North Asia": 0.327,
            "North NA": 0.3412,
        },
    },
    {
        "tag": "grid_unetpp_focal_a25_newf",
        "model": "unet++",
        "loss": "focal",
        "target": "new_fires",
        "val_f1": 0.2100,
        "threshold": 0.55,
        "test_f1": 0.2200,
        "test_precision": 0.2100,
        "test_recall": 0.2310,
        "test_iou": 0.1240,
        "test_brier": 0.0090,
        "by_region": {
            "Oceania": 0.15,
            "South America": 0.28,
            "Africa": 0.03,
            "Western Europe": 0.12,
            "North Asia": 0.30,
            "North NA": 0.31,
        },
    },
    {
        "tag": "grid_unetpp_focal_a25_next",
        "model": "unet++",
        "loss": "focal",
        "target": "next_mask",
        "val_f1": 0.2500,
        "threshold": 0.60,
        "test_f1": 0.2700,
        "test_precision": 0.2600,
        "test_recall": 0.2810,
        "test_iou": 0.1560,
        "test_brier": 0.0110,
        "by_region": {
            "Oceania": 0.11,
            "South America": 0.32,
            "Africa": 0.02,
            "Western Europe": 0.13,
            "North Asia": 0.33,
            "North NA": 0.38,
        },
    },
]


class TestGridModels:
    def test_basic(self):
        tex = table_grid_models(None, None, MOCK_GRID)
        assert r"\begin{table}" in tex
        assert r"\caption{#1}" in tex
        assert r"\label{tab:grid-models}" in tex
        assert "UNet" in tex
        assert "UNet++" in tex
        assert r"\textbf" in tex  # best should be bolded
        assert r"\end{table}" in tex

    def test_has_metrics(self):
        tex = table_grid_models(None, None, MOCK_GRID)
        assert "F1" in tex
        assert "Prec" in tex
        assert "IoU" in tex
        assert ".270" in tex  # 3 dp, table style (.xxx)


class TestGridLosses:
    def test_basic(self):
        tex = table_grid_losses(None, None, MOCK_GRID)
        assert r"\begin{table}" in tex
        assert "BCE" in tex
        assert "Focal" in tex
        assert r"\end{table}" in tex


class TestGridTargets:
    def test_basic(self):
        tex = table_grid_targets(None, None, MOCK_GRID)
        assert r"\begin{table}" in tex
        assert tex_escape_text("next_mask") in tex
        assert tex_escape_text("new_fires") in tex
        assert r"\end{table}" in tex

    def test_has_nf_equiv(self):
        tex = table_grid_targets(None, None, MOCK_GRID)
        assert tex_escape_text("nf_equiv") in tex


class TestGridRegions:
    def test_basic(self):
        tex = table_grid_regions(None, None, MOCK_GRID)
        assert r"\begin{table}" in tex
        assert "N.~NA" in tex
        assert "Afr" in tex
        assert "Global" in tex
        assert r"\end{table}" in tex


class TestGridAll:
    def test_basic(self):
        tex = table_grid_all(None, None, MOCK_GRID)
        assert r"\begin{table}[H]" in tex
        assert "UNet" in tex
        assert "UNet++" in tex
        # Should have all 4 mock runs
        assert "BCE pw10" in tex
        assert r"\end{table}" in tex
        assert r"table*" not in tex

    def test_has_threshold(self):
        tex = table_grid_all(None, None, MOCK_GRID)
        assert "0.60" in tex or "0.65" in tex


# ---------------------------------------------------------------------------
# Solidity table tests (mock results)
# ---------------------------------------------------------------------------

MOCK_SOLIDITY = [
    {"fire_id": 1, "region_id": 7, "region_name": "North NA",
     "solidity": 0.72, "n_components": 2, "num_pixels": 5000,
     "mean_component_size": 2500.0},
    {"fire_id": 2, "region_id": 7, "region_name": "North NA",
     "solidity": 0.65, "n_components": 3, "num_pixels": 3000,
     "mean_component_size": 1000.0},
    {"fire_id": 3, "region_id": 3, "region_name": "Africa",
     "solidity": 0.18, "n_components": 12, "num_pixels": 2000,
     "mean_component_size": 166.7},
    {"fire_id": 4, "region_id": 3, "region_name": "Africa",
     "solidity": 0.22, "n_components": 8, "num_pixels": 1500,
     "mean_component_size": 187.5},
    {"fire_id": 5, "region_id": 1, "region_name": "Western Europe",
     "solidity": 0.55, "n_components": 4, "num_pixels": 4000,
     "mean_component_size": 1000.0},
]


class TestSolidity:
    def test_basic(self):
        tex = table_solidity(None, None, None, solidity=MOCK_SOLIDITY)
        assert r"\begin{table}" in tex
        assert "N.~NA" in tex
        assert "Afr" in tex
        assert "W.~Eur" in tex
        assert "Comp." in tex
        assert r"\end{table}" in tex

    def test_sorted_by_solidity(self):
        tex = table_solidity(None, None, None, solidity=MOCK_SOLIDITY)
        # Africa should appear before North NA (lower solidity)
        afr_pos = tex.index("Afr")
        nna_pos = tex.index("N.~NA")
        assert afr_pos < nna_pos


# ---------------------------------------------------------------------------
# tex_escape_text + write_tex tests
# ---------------------------------------------------------------------------

class TestTexEscapeText:
    def test_underscores(self):
        assert tex_escape_text("next_mask") == r"next\_mask"

    def test_ampersand(self):
        assert tex_escape_text("A & B") == r"A \& B"

    def test_percent(self):
        assert tex_escape_text("100%") == r"100\%"

    def test_hash(self):
        assert tex_escape_text("#1") == r"\#1"

    def test_backslash(self):
        # Backslash is replaced first, then braces in the replacement
        # also get escaped — this is correct for pure data text.
        assert tex_escape_text("a\\b") == r"a\textbackslash\{\}b"

    def test_tilde_caret(self):
        assert tex_escape_text("~") == r"\textasciitilde{}"
        assert tex_escape_text("^") == r"\textasciicircum{}"

    def test_braces(self):
        assert tex_escape_text("{x}") == r"\{x\}"

    def test_dollar(self):
        assert tex_escape_text("$5") == r"\$5"

    def test_plain_text(self):
        assert tex_escape_text("hello world") == "hello world"


class TestWriteTex:
    def test_wraps_in_define(self, tmp_path):
        from firecomp.next_day.latex import write_tex
        tex_path = tmp_path / "test.tex"
        sections = [("tab:foo", r"\begin{table}[t]\end{table}")]
        write_tex(tex_path, sections)
        content = tex_path.read_text()
        assert r"\definegeneratedartifact{tab:foo}" in content
        assert r"\begin{table}" in content

    def test_multiple_sections(self, tmp_path):
        from firecomp.next_day.latex import write_tex
        tex_path = tmp_path / "test.tex"
        sections = [
            ("fig:a", "body-a"),
            ("fig:b", "body-b"),
        ]
        write_tex(tex_path, sections)
        content = tex_path.read_text()
        assert r"\definegeneratedartifact{fig:a}" in content
        assert r"\definegeneratedartifact{fig:b}" in content
        assert "body-a" in content
        assert "body-b" in content
