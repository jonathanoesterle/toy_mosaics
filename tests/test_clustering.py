"""Tests for clustering strategies: KMeans, GMM, and LeidenMosaicStrategy."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from toy_mosaics.clustering import ClusteringResult, GMMStrategy, KMeansStrategy, LeidenMosaicStrategy
from toy_mosaics.simulate_dataset import dataset_from_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

CONFIG_NAMES = ["example", "circles", "moons"]

STRATEGIES = [KMeansStrategy, GMMStrategy]


def _load_cfg(name: str) -> dict:
    with open(CONFIGS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(params=CONFIG_NAMES)
def dataset(request):
    cfg = _load_cfg(request.param)
    return dataset_from_config(cfg)


@pytest.mark.parametrize("StrategyClass", STRATEGIES)
def test_labels_shape(dataset, StrategyClass):
    result = StrategyClass(n_clusters=dataset.n_mosaics).fit(dataset)
    assert result.labels.shape == (len(dataset),)


@pytest.mark.parametrize("StrategyClass", STRATEGIES)
def test_labels_dtype(dataset, StrategyClass):
    result = StrategyClass(n_clusters=dataset.n_mosaics).fit(dataset)
    assert np.issubdtype(result.labels.dtype, np.integer)


@pytest.mark.parametrize("StrategyClass", STRATEGIES)
def test_n_unique_labels(dataset, StrategyClass):
    result = StrategyClass(n_clusters=dataset.n_mosaics).fit(dataset)
    assert len(np.unique(result.labels)) == dataset.n_mosaics


@pytest.mark.parametrize("StrategyClass", STRATEGIES)
def test_result_type(dataset, StrategyClass):
    result = StrategyClass(n_clusters=dataset.n_mosaics).fit(dataset)
    assert isinstance(result, ClusteringResult)
    assert result.model is not None


def test_kmeans_reproducible():
    cfg = _load_cfg("example")
    ds = dataset_from_config(cfg)
    r1 = KMeansStrategy(n_clusters=ds.n_mosaics, random_state=42).fit(ds)
    r2 = KMeansStrategy(n_clusters=ds.n_mosaics, random_state=42).fit(ds)
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_gmm_reproducible():
    cfg = _load_cfg("example")
    ds = dataset_from_config(cfg)
    r1 = GMMStrategy(n_clusters=ds.n_mosaics, random_state=42).fit(ds)
    r2 = GMMStrategy(n_clusters=ds.n_mosaics, random_state=42).fit(ds)
    np.testing.assert_array_equal(r1.labels, r2.labels)


# ---------------------------------------------------------------------------
# LeidenMosaicStrategy
# ---------------------------------------------------------------------------

# spatial_radius large enough to capture inter-cell overlap in the example config
# (mean_diameters ≈ 10–15 → use 30 as a safe upper bound)
_LEIDEN_KWARGS = dict(k=10, spatial_radius=30.0, tau_low_global=0.0, n_iter=2, resolution=1.0)


@pytest.fixture
def leiden_dataset():
    cfg = _load_cfg("example")
    return dataset_from_config(cfg)


def test_leiden_labels_shape(leiden_dataset):
    result = LeidenMosaicStrategy(n_clusters=leiden_dataset.n_mosaics, **_LEIDEN_KWARGS).fit(leiden_dataset)
    assert result.labels.shape == (len(leiden_dataset),)


def test_leiden_labels_dtype(leiden_dataset):
    result = LeidenMosaicStrategy(n_clusters=leiden_dataset.n_mosaics, **_LEIDEN_KWARGS).fit(leiden_dataset)
    assert np.issubdtype(result.labels.dtype, np.integer)


def test_leiden_result_type(leiden_dataset):
    result = LeidenMosaicStrategy(n_clusters=leiden_dataset.n_mosaics, **_LEIDEN_KWARGS).fit(leiden_dataset)
    assert isinstance(result, ClusteringResult)
    assert isinstance(result.model, dict)
    assert "converged" in result.model
    assert "n_iters_run" in result.model
    assert "n_merges" in result.model


def test_leiden_reproducible(leiden_dataset):
    ds = leiden_dataset
    kwargs = dict(n_clusters=ds.n_mosaics, random_state=42, **_LEIDEN_KWARGS)
    r1 = LeidenMosaicStrategy(**kwargs).fit(ds)
    r2 = LeidenMosaicStrategy(**kwargs).fit(ds)
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_leiden_exclude_clipped_flag(leiden_dataset):
    ds = leiden_dataset
    base = dict(n_clusters=ds.n_mosaics, random_state=0, **_LEIDEN_KWARGS)
    # smoke-test that both flag values run without error
    LeidenMosaicStrategy(**base, exclude_clipped=True).fit(ds)
    LeidenMosaicStrategy(**base, exclude_clipped=False).fit(ds)


def test_leiden_n_iters_single(leiden_dataset):
    result = LeidenMosaicStrategy(
        n_clusters=leiden_dataset.n_mosaics, n_iter=1, **{k: v for k, v in _LEIDEN_KWARGS.items() if k != "n_iter"}
    ).fit(leiden_dataset)
    assert result.model["n_iters_run"] == 1


def test_leiden_merge_disabled(leiden_dataset):
    result = LeidenMosaicStrategy(
        n_clusters=leiden_dataset.n_mosaics, merge=False, **_LEIDEN_KWARGS
    ).fit(leiden_dataset)
    assert result.model["n_merges"] == 0
    assert result.labels.shape == (len(leiden_dataset),)


def test_leiden_merge_reduces_clusters(leiden_dataset):
    """Merge step should produce <= as many clusters as the no-merge version."""
    base = dict(n_clusters=leiden_dataset.n_mosaics, random_state=0, resolution=2.0,
                **{k: v for k, v in _LEIDEN_KWARGS.items() if k != "resolution"})
    r_no_merge = LeidenMosaicStrategy(**base, merge=False).fit(leiden_dataset)
    r_merge    = LeidenMosaicStrategy(**base, merge=True, theta_paga=0.1, delta_r=0.01).fit(leiden_dataset)
    assert len(np.unique(r_merge.labels)) <= len(np.unique(r_no_merge.labels))
