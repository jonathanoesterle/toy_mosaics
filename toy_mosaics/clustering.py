"""Baseline clustering strategies: KMeans and GMM operating on feature vectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from toy_mosaics.dataset import MosaicDataset


@dataclass
class ClusteringResult:
    """Output of a clustering strategy.

    Attributes
    ----------
    labels:
        Integer cluster assignment per cell, shape (n_cells,).
    model:
        Fitted sklearn estimator.
    """

    labels: NDArray[np.int_]
    model: Any


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
