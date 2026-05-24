"""Ward hierarchical clustering with spatial mosaic-violation penalty.

Modified Ward: each merge step minimises
    Δ_mosaic(A, B) = ΔW(A, B) + lam_scaled · V_norm(A, B)

where ΔW is the standard Ward variance increase and V_norm is the mean
ramp(score(i, j)) over all (i ∈ A, j ∈ B) cross-cluster spatial pairs.
lam_scaled = lam × median_leaf_ΔW so that lam=1 means the spatial penalty
equals a typical leaf-level Ward cost per unit of V_norm.

Unlike Leiden, the penalty is evaluated directly on the candidate merge pair —
no graph, no indirect paths.
"""
from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray

from toy_mosaics._result import ClusteringResult
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.leiden_mosaic import _build_coverage_map, _build_proximity_map


def _ramp(score: float, tau_low: float, tau_high: float) -> float:
    span = tau_high - tau_low
    if span <= 0.0:
        return float(score > tau_low)
    return float(np.clip((score - tau_low) / span, 0.0, 1.0))


def _ward_delta(mean_a: NDArray, mean_b: NDArray, n_a: int, n_b: int) -> float:
    diff = mean_a - mean_b
    return float((n_a * n_b) / (n_a + n_b) * np.dot(diff, diff))


class WardMosaicStrategy:
    """Ward hierarchical clustering with spatial mosaic-violation penalty.

    Parameters
    ----------
    n_clusters:
        Target number of clusters.
    spatial_radius:
        Neighbourhood radius for spatial pair detection.
    lam:
        Dimensionless penalty weight. 0 = standard Ward.
        lam=1 sets the penalty equal to the median leaf-level Ward cost
        per unit of normalised violation score.
    tau_low:
        Dead-zone onset: ramp = 0 for score ≤ tau_low.
    tau_high:
        Ramp reaches 1 at tau_high (full penalty).
    use_coverage:
        If True, use polygon coverage fraction as the spatial signal;
        if False (default), use proximity score 1 − d/radius.
    exclude_clipped:
        Skip spatial pairs where either cell is boundary-clipped.
    """

    def __init__(
        self,
        n_clusters: int,
        spatial_radius: float = 20.0,
        lam: float = 1.0,
        tau_low: float = 0.05,
        tau_high: float = 0.40,
        use_coverage: bool = False,
        exclude_clipped: bool = True,
    ) -> None:
        self.n_clusters = n_clusters
        self.spatial_radius = spatial_radius
        self.lam = lam
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.use_coverage = use_coverage
        self.exclude_clipped = exclude_clipped

    def fit(self, dataset: MosaicDataset) -> ClusteringResult:
        X = dataset.X
        n_cells = len(dataset)

        # Build spatial score map
        if self.use_coverage:
            raw_map = _build_coverage_map(
                dataset.polygons, dataset.centers, self.spatial_radius,
                dataset.clipped, self.exclude_clipped,
            )
        else:
            raw_map = _build_proximity_map(
                dataset.centers, self.spatial_radius,
                dataset.clipped, self.exclude_clipped,
            )

        # Precompute ramp values; build per-cell adjacency for sparse V_norm
        # ramp_map[(i, j)] with i < j is the canonical key
        ramp_map: dict[tuple[int, int], float] = {}
        spatial_nbrs: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_cells)}
        for (i, j), score in raw_map.items():
            rv = _ramp(score, self.tau_low, self.tau_high)
            if rv > 0.0:
                ramp_map[(i, j)] = rv
                spatial_nbrs[i].append((j, rv))
                spatial_nbrs[j].append((i, rv))

        # Calibrate λ: scale to median leaf-level Ward cost
        # For singletons n_A = n_B = 1 → ΔW = 0.5 · ‖X_i − X_j‖²
        sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        upper = np.triu_indices(n_cells, k=1)
        median_leaf_cost = float(np.median(0.5 * sq_dists[upper]))
        lam_scaled = self.lam * median_leaf_cost

        # Initialize singleton clusters
        members: dict[int, list[int]] = {i: [i] for i in range(n_cells)}
        sets: dict[int, set[int]] = {i: {i} for i in range(n_cells)}
        means: dict[int, NDArray] = {i: X[i].copy() for i in range(n_cells)}
        sizes: dict[int, int] = {i: 1 for i in range(n_cells)}
        active: set[int] = set(range(n_cells))
        next_id = n_cells

        def _cost(a: int, b: int) -> float:
            dw = _ward_delta(means[a], means[b], sizes[a], sizes[b])
            if lam_scaled == 0.0:
                return dw
            sb = sets[b]
            total = sum(rv for i in members[a] for j, rv in spatial_nbrs[i] if j in sb)
            return dw + lam_scaled * total / (sizes[a] * sizes[b])

        # Build initial heap for all N(N-1)/2 leaf pairs
        # For singletons: V_norm(i, j) = ramp_map.get((i,j), 0.0) (single pair)
        heap: list[tuple[float, int, int]] = []
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                dw = float(0.5 * sq_dists[i, j])
                cost = dw + lam_scaled * ramp_map.get((i, j), 0.0)
                heapq.heappush(heap, (cost, i, j))

        # Greedy merge loop with lazy deletion
        while len(active) > self.n_clusters:
            # Pop until we find a valid (both-active) merge
            while True:
                if not heap:
                    break
                cost, a, b = heapq.heappop(heap)
                if a in active and b in active:
                    break

            cid = next_id
            next_id += 1
            na, nb = sizes[a], sizes[b]
            ns = na + nb
            new_mean = (na * means[a] + nb * means[b]) / ns

            members[cid] = members[a] + members[b]
            sets[cid] = sets[a] | sets[b]
            means[cid] = new_mean
            sizes[cid] = ns

            active.discard(a)
            active.discard(b)
            active.add(cid)
            del members[a], members[b], sets[a], sets[b]
            del means[a], means[b], sizes[a], sizes[b]

            # Push new merge costs for the new cluster vs all other active clusters
            for other in active:
                if other == cid:
                    continue
                heapq.heappush(heap, (_cost(cid, other), cid, other))

        # Assign compact labels 0..k-1
        labels = np.empty(n_cells, dtype=np.int64)
        for lbl, cid in enumerate(sorted(active)):
            for cell in members[cid]:
                labels[cell] = lbl

        return ClusteringResult(
            labels=labels,
            model={
                "lam": self.lam,
                "lam_scaled": lam_scaled,
                "median_leaf_cost": median_leaf_cost,
                "n_clusters_found": len(active),
            },
        )
