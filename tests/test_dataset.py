"""Tests for MosaicDataset generation, serialization, and properties."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.simulate_dataset import dataset_from_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

CONFIG_NAMES = [
    "example",
    "single_mosaic",
    "high_overlap",
    "elongated_clusters",
]


def _load_cfg(name: str) -> dict:
    with open(CONFIGS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(params=CONFIG_NAMES)
def config_and_dataset(request):
    cfg = _load_cfg(request.param)
    ds = dataset_from_config(cfg)
    return cfg, ds


# ---------------------------------------------------------------------------
# shape and dtype checks
# ---------------------------------------------------------------------------


def test_groups_shape(config_and_dataset):
    cfg, ds = config_and_dataset
    n_cells = len(ds)
    assert ds.groups.shape == (n_cells,)
    assert np.issubdtype(ds.groups.dtype, np.integer)


def test_centers_shape(config_and_dataset):
    _, ds = config_and_dataset
    assert ds.centers.shape == (len(ds), 2)
    assert np.issubdtype(ds.centers.dtype, np.floating)


def test_clipped_shape(config_and_dataset):
    _, ds = config_and_dataset
    assert ds.clipped.shape == (len(ds),)
    assert ds.clipped.dtype == bool


def test_polygons_length_and_shape(config_and_dataset):
    _, ds = config_and_dataset
    assert len(ds.polygons) == len(ds)
    for poly in ds.polygons:
        assert poly.ndim == 2
        assert poly.shape[1] == 2


def test_feature_matrix_shape(config_and_dataset):
    cfg, ds = config_and_dataset
    n_dims = cfg.get("features", {}).get("n_dims", 2)
    assert ds.X.shape == (len(ds), n_dims)
    assert ds.y.shape == (len(ds),)


# ---------------------------------------------------------------------------
# semantic checks
# ---------------------------------------------------------------------------


def test_n_mosaics_property(config_and_dataset):
    cfg, ds = config_and_dataset
    expected = cfg["mosaic"]["n_mosaics"]
    assert ds.n_mosaics == expected


def test_group_labels_match_n_mosaics(config_and_dataset):
    cfg, ds = config_and_dataset
    n = cfg["mosaic"]["n_mosaics"]
    assert set(np.unique(ds.groups)) == set(range(n))


def test_feature_labels_match_n_mosaics(config_and_dataset):
    cfg, ds = config_and_dataset
    n = cfg["mosaic"]["n_mosaics"]
    assert set(np.unique(ds.y)) == set(range(n))


def test_n_feature_dims_property(config_and_dataset):
    cfg, ds = config_and_dataset
    expected = cfg.get("features", {}).get("n_dims", 2)
    assert ds.n_feature_dims == expected


def test_nonempty(config_and_dataset):
    _, ds = config_and_dataset
    assert len(ds) > 0


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(config_and_dataset):
    _, ds = config_and_dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dataset.npz"
        ds.save(path)
        loaded = MosaicDataset.load(path)

    assert len(loaded) == len(ds)
    np.testing.assert_array_equal(loaded.groups, ds.groups)
    np.testing.assert_array_equal(loaded.centers, ds.centers)
    np.testing.assert_array_equal(loaded.clipped, ds.clipped)
    np.testing.assert_array_equal(loaded.X, ds.X)
    np.testing.assert_array_equal(loaded.y, ds.y)
    assert len(loaded.polygons) == len(ds.polygons)
    for a, b in zip(loaded.polygons, ds.polygons):
        np.testing.assert_array_equal(a, b)


def test_save_creates_parent_dirs():
    cfg = _load_cfg("example")
    ds = dataset_from_config(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "dir" / "out.npz"
        ds.save(path)
        assert path.exists()
