"""
Baseline tests: run persistence + morphological on the 100-sample desktop dataset.

Plumbing only — verifies no crashes and correct output structure, not result
quality. Both baselines run on CPU (no model weights).

Run:  pytest tests/test_baselines.py -v -s
"""

from pathlib import Path

import pytest

DATASET_DIR = Path("data/tasks/next_day/100/")
H5_FILE = DATASET_DIR / "v2" / "dataset_0.h5"

pytestmark = pytest.mark.skipif(
    not H5_FILE.exists(),
    reason="Desktop dataset not available",
)


# ---------------------------------------------------------------------------
# Shared dataset fixture — load once, reuse across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds():
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.dataset import NextDayDataset

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        dataset_version="v2",
        fire_type="all",  # v2 dataset has no fire_type metadata
        device="cpu",
        batch_size=16,
        num_workers=0,
    )
    return NextDayDataset(cfg)


@pytest.fixture(scope="module")
def ds_new_fires():
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.dataset import NextDayDataset

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        dataset_version="v2",
        target_type="new_fires",
        fire_type="all",  # v2 dataset has no fire_type metadata
        device="cpu",
        batch_size=16,
        num_workers=0,
    )
    return NextDayDataset(cfg)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_persistence_next_mask(ds):
    """Persistence baseline runs on next_mask and returns valid metrics."""
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.implementations.persistence import eval

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        dataset_version="v2",
        target_type="next_mask",
        device="cpu",
        batch_size=16,
        num_workers=0,
    )
    result = eval(cfg, ds=ds)

    assert "test_f1" in result
    assert "test_precision" in result
    assert "test_recall" in result
    assert result["test_f1"] >= 0.0
    assert result["baseline"] == "persistence"


def test_persistence_new_fires(ds_new_fires):
    """Persistence baseline on new_fires target — should run, F1 likely low."""
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.implementations.persistence import eval

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        dataset_version="v2",
        target_type="new_fires",
        device="cpu",
        batch_size=16,
        num_workers=0,
    )
    result = eval(cfg, ds=ds_new_fires)

    assert "test_f1" in result
    assert result["baseline"] == "persistence"
    assert result["target"] == "new_fires"


# ---------------------------------------------------------------------------
# Morphological
# ---------------------------------------------------------------------------

def test_morphological_next_mask(ds):
    """Morphological baseline runs with multiple radii and picks the best."""
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.implementations.morphological import eval

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        dataset_version="v2",
        target_type="next_mask",
        device="cpu",
        batch_size=16,
        num_workers=0,
    )
    result = eval(cfg, ds=ds, radii=[1, 2])

    assert "test_f1" in result
    assert "best_radius" in result
    assert result["best_radius"] in [1, 2]
    assert result["baseline"] == "morphological"


def test_morphological_new_fires(ds_new_fires):
    """Morphological baseline on new_fires — expansion ring only."""
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.implementations.morphological import eval

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        dataset_version="v2",
        target_type="new_fires",
        device="cpu",
        batch_size=16,
        num_workers=0,
    )
    result = eval(cfg, ds=ds_new_fires, radii=[1, 2])

    assert "test_f1" in result
    assert result["baseline"] == "morphological"
    assert result["target"] == "new_fires"


def test_morphological_differs_from_persistence(ds):
    """Morphological predictions (radius>0) must differ from persistence (radius=0 equivalent)."""
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.implementations.persistence import eval as persistence_eval
    from firecomp.next_day.implementations.morphological import eval as morphological_eval

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        dataset_version="v2",
        target_type="next_mask",
        device="cpu",
        batch_size=16,
        num_workers=0,
    )
    p_result = persistence_eval(cfg, ds=ds)
    m_result = morphological_eval(cfg, ds=ds, radii=[2])

    # Morphological with dilation should differ from persistence
    # (unless the dataset has zero fire pixels, which is unlikely)
    assert (
        p_result["test_f1"] != m_result["test_f1"]
        or p_result["test_precision"] != m_result["test_precision"]
    ), "Morphological predictions should differ from persistence"
