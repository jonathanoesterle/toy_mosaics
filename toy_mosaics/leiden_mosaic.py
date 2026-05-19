"""Leiden clustering with iterative mosaic-overlap repulsion.

Algorithm outline
-----------------
1. Build a k-NN feature graph with RBF-kernel edge weights.
2. Compute a sparse coverage-fraction map for all cell pairs within a
   spatial radius (coverage fraction = intersection / min-area of the two
   convex hulls; see overlap.coverage_fraction).
3. Run the iterative loop:
   a. Apply repulsion: multiply each edge weight by (1 - ramp(C, τ_low, τ_high)).
      The ramp has a dead zone [0, τ_low] that leaves normal same-type touching
      (e.g. ~10 % for BCs) unpenalised.
   b. Run Leiden on the repulsion-weighted graph.
   c. Estimate per-cluster τ_low as the 75th percentile of within-cluster
      spatial-pair coverage fractions.
   d. Repeat until ARI between consecutive label sets exceeds 1 - convergence_tol
      or n_iter is reached.
"""
from __future__ import annotations

from typing import Optional

import igraph
import leidenalg
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.spatial import KDTree
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from toy_mosaics._result import ClusteringResult
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.overlap import coverage_fraction


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spatial_pairs(centers: NDArray, radius: float) -> tuple[NDArray, NDArray]:
    """Index pairs (i < j) whose centres are within *radius* of each other."""
    pairs = KDTree(centers).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    return pairs[:, 0], pairs[:, 1]


def _build_coverage_map(
    polygons: list[NDArray],
    centers: NDArray,
    radius: float,
    clipped: NDArray,
    exclude_clipped: bool,
) -> dict[tuple[int, int], float]:
    """Coverage fraction for every spatially close, non-clipped pair."""
    ii, jj = _spatial_pairs(centers, radius)
    cf_map: dict[tuple[int, int], float] = {}
    for i, j in zip(ii.tolist(), jj.tolist()):
        if exclude_clipped and (clipped[i] or clipped[j]):
            continue
        cf = coverage_fraction(polygons[i], polygons[j])
        if cf > 0.0:
            cf_map[(i, j)] = cf
    return cf_map


def _build_feature_graph(X: NDArray, k: int) -> sp.csr_matrix:
    """Symmetric k-NN graph with RBF-kernel weights (bandwidth = median kNN dist)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    distances, indices = nn.kneighbors(X)
    distances, indices = distances[:, 1:], indices[:, 1:]  # drop self

    sigma = float(np.median(distances)) or 1.0
    weights = np.exp(-(distances**2) / (2 * sigma**2))

    n = len(X)
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((weights.ravel(), (rows, indices.ravel())), shape=(n, n))
    return W.maximum(W.T)  # symmetrise (take max of w_ij, w_ji)


def _apply_repulsion(
    W: sp.csr_matrix,
    cf_map: dict[tuple[int, int], float],
    tau_low_per_cell: NDArray,
    tau_high: float,
) -> sp.csr_matrix:
    """Return W with edge weights scaled by (1 - ramp(C, τ_low_ij, τ_high))."""
    if not cf_map:
        return W
    W_lil = W.tolil()
    for (i, j), cf in cf_map.items():
        tau_low = (tau_low_per_cell[i] + tau_low_per_cell[j]) / 2.0
        span = tau_high - tau_low
        repulsion = float(np.clip((cf - tau_low) / span, 0.0, 1.0)) if span > 0 else float(cf > tau_low)
        factor = 1.0 - repulsion
        W_lil[i, j] = W_lil[i, j] * factor
        W_lil[j, i] = W_lil[j, i] * factor
    return W_lil.tocsr()


def _to_igraph(W: sp.csr_matrix) -> igraph.Graph:
    """Convert a sparse symmetric weight matrix to a weighted undirected igraph."""
    coo = sp.triu(W, k=1).tocoo()
    g = igraph.Graph(n=W.shape[0], directed=False)
    g.add_edges(list(zip(coo.row.tolist(), coo.col.tolist())))
    g.es["weight"] = coo.data.tolist()
    return g


def _run_leiden(g: igraph.Graph, resolution: float, seed: int) -> NDArray:
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = np.empty(g.vcount(), dtype=np.int64)
    for cluster_id, members in enumerate(partition):
        labels[members] = cluster_id
    return labels


def _estimate_tau_low(
    labels: NDArray,
    cf_map: dict[tuple[int, int], float],
    min_pairs: int,
    global_tau_low: float,
) -> NDArray:
    """Per-cell τ_low = 75th-percentile of within-cluster spatial-pair CFs.

    Clusters with fewer than *min_pairs* within-cluster pairs keep *global_tau_low*.
    The 75th percentile is used so the dead zone covers the bulk of normal
    same-type contact rather than just the median.
    """
    n_clusters = int(labels.max()) + 1
    bucket: list[list[float]] = [[] for _ in range(n_clusters)]
    for (i, j), cf in cf_map.items():
        if labels[i] == labels[j]:
            bucket[labels[i]].append(cf)

    tau_low_per_cluster = np.full(n_clusters, global_tau_low)
    for c, vals in enumerate(bucket):
        if len(vals) >= min_pairs:
            tau_low_per_cluster[c] = float(np.percentile(vals, 75))

    return tau_low_per_cluster[labels]


# ---------------------------------------------------------------------------
# Public strategy class
# ---------------------------------------------------------------------------

class LeidenMosaicStrategy:
    """Leiden clustering with iterative mosaic-overlap repulsion.

    Parameters
    ----------
    n_clusters:
        Expected number of cell types.  Stored for reference; tune
        *resolution* to obtain the desired number of Leiden communities.
    k:
        Number of feature-space nearest neighbours in the base graph.
    spatial_radius:
        Distance threshold for computing pairwise coverage fractions.
        Rule of thumb: ~1.5 × expected mosaic spacing (≈ mean cell diameter).
    tau_low_global:
        Initial dead-zone onset applied uniformly before per-cluster
        estimation.  Keep small (e.g. 0.05) to start conservative.
    tau_high:
        Coverage fraction at which full repulsion is applied.
    resolution:
        Leiden resolution parameter — increase for more communities.
    n_iter:
        Maximum adaptive iterations.  Stops early when ARI between
        consecutive runs exceeds 1 − convergence_tol.
    convergence_tol:
        ARI gap below 1.0 that is accepted as converged (default 0.01).
    min_pairs_for_estimate:
        Minimum within-cluster spatial pairs needed to update a cluster's
        τ_low.  Clusters below this threshold keep global_tau_low.
    exclude_clipped:
        Skip repulsion for pairs where either cell is clipped (truncated
        hull → unreliable coverage fraction).
    random_state:
        Seed for Leiden's internal RNG.
    """

    def __init__(
        self,
        n_clusters: int,
        k: int = 15,
        spatial_radius: float = 20.0,
        tau_low_global: float = 0.05,
        tau_high: float = 0.40,
        resolution: float = 1.0,
        n_iter: int = 3,
        convergence_tol: float = 0.01,
        min_pairs_for_estimate: int = 5,
        exclude_clipped: bool = True,
        random_state: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.k = k
        self.spatial_radius = spatial_radius
        self.tau_low_global = tau_low_global
        self.tau_high = tau_high
        self.resolution = resolution
        self.n_iter = n_iter
        self.convergence_tol = convergence_tol
        self.min_pairs_for_estimate = min_pairs_for_estimate
        self.exclude_clipped = exclude_clipped
        self.random_state = random_state

    def fit(self, dataset: MosaicDataset) -> ClusteringResult:
        n_cells = len(dataset)

        W_feat = _build_feature_graph(dataset.X, self.k)
        cf_map = _build_coverage_map(
            dataset.polygons,
            dataset.centers,
            self.spatial_radius,
            dataset.clipped,
            self.exclude_clipped,
        )

        tau_low_per_cell = np.full(n_cells, self.tau_low_global)
        prev_labels: Optional[NDArray] = None
        labels: Optional[NDArray] = None
        converged = False
        n_iters_run = 0

        for iteration in range(self.n_iter):
            n_iters_run = iteration + 1
            W_rep = _apply_repulsion(W_feat, cf_map, tau_low_per_cell, self.tau_high)
            labels = _run_leiden(_to_igraph(W_rep), self.resolution, seed=self.random_state + iteration)

            if prev_labels is not None:
                ari = adjusted_rand_score(prev_labels, labels)
                if ari >= 1.0 - self.convergence_tol:
                    converged = True
                    break

            prev_labels = labels.copy()

            # update τ_low for the next iteration (skip on final pass)
            if iteration < self.n_iter - 1:
                tau_low_per_cell = _estimate_tau_low(
                    labels, cf_map, self.min_pairs_for_estimate, self.tau_low_global
                )

        return ClusteringResult(
            labels=labels,
            model={
                "tau_low_per_cell": tau_low_per_cell,
                "tau_high": self.tau_high,
                "n_iters_run": n_iters_run,
                "converged": converged,
            },
        )
