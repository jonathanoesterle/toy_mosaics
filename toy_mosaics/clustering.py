"""Clustering strategies: baselines (KMeans, GMM) and Leiden with mosaic repulsion."""
from __future__ import annotations

from typing import Any

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from toy_mosaics._result import ClusteringResult  # noqa: F401 — re-exported
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.leiden_mosaic import LeidenMosaicStrategy, LeidenProximityStrategy  # noqa: F401 — re-exported
from toy_mosaics.ward_mosaic import WardMosaicStrategy  # noqa: F401 — re-exported
from toy_mosaics.mrf_mosaic import MRFMosaicStrategy  # noqa: F401 — re-exported


class KMeansStrategy:
    """K-Means clustering on the feature matrix X."""

    def __init__(self, n_clusters: int, random_state: int = 0, **kwargs: Any) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kwargs = kwargs

    def fit(self, dataset: MosaicDataset) -> ClusteringResult:
        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            **self._kwargs,
        )
        labels = model.fit_predict(dataset.X)
        return ClusteringResult(labels=labels, model=model)


class GMMStrategy:
    """Gaussian Mixture Model clustering on the feature matrix X."""

    def __init__(self, n_clusters: int, random_state: int = 0, **kwargs: Any) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kwargs = kwargs

    def fit(self, dataset: MosaicDataset) -> ClusteringResult:
        model = GaussianMixture(
            n_components=self.n_clusters,
            random_state=self.random_state,
            **self._kwargs,
        )
        model.fit(dataset.X)
        labels = model.predict(dataset.X)
        return ClusteringResult(labels=labels, model=model)
