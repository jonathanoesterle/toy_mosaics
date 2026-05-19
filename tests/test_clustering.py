"""Tests for baseline clustering strategies (KMeans, GMM)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from toy_mosaics.clustering import ClusteringResult, GMMStrategy, KMeansStrategy
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
