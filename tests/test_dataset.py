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
    "single",
    "circles",
    "moons",
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
    cfg = _load_cfg("single")
    ds = dataset_from_config(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "dir" / "out.npz"
        ds.save(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# hull features
# ---------------------------------------------------------------------------


def test_hull_features_absent_when_disabled(config_and_dataset):
    cfg, ds = config_and_dataset
    if cfg.get("hull_features", {}).get("enabled", False):
        pytest.skip("hull_features enabled for this config")
    assert ds.hull_features is None
    assert ds.hull_feature_names == []


def test_hull_features_present_when_enabled(config_and_dataset):
    cfg, ds = config_and_dataset
    if not cfg.get("hull_features", {}).get("enabled", False):
        pytest.skip("hull_features not enabled for this config")
    assert ds.hull_features is not None
    assert ds.hull_features.shape == (len(ds), 3)
    assert ds.hull_feature_names == ["area", "perimeter", "circularity"]


def test_hull_features_values(config_and_dataset):
    cfg, ds = config_and_dataset
    if not cfg.get("hull_features", {}).get("enabled", False):
        pytest.skip("hull_features not enabled for this config")
    # Area and perimeter must be positive for non-clipped interior cells
    interior = ~ds.clipped
    assert np.all(ds.hull_features[interior, 0] > 0), "area should be positive"
    assert np.all(ds.hull_features[interior, 1] > 0), "perimeter should be positive"
    # Circularity bounded (0, 1] for convex shapes
    circ = ds.hull_features[interior, 2]
    assert np.all(circ > 0) and np.all(circ <= 1.0 + 1e-6), "circularity out of range"


def test_hull_features_roundtrip(config_and_dataset):
    cfg, ds = config_and_dataset
    if not cfg.get("hull_features", {}).get("enabled", False):
        pytest.skip("hull_features not enabled for this config")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dataset.npz"
        ds.save(path)
        loaded = MosaicDataset.load(path)
    np.testing.assert_array_equal(loaded.hull_features, ds.hull_features)
    assert loaded.hull_feature_names == ds.hull_feature_names
