"""Post-processing merge step for over-split Leiden clusters.

Two conditions must both be met for a merge to be accepted:
  1. PAGA weight >= theta_paga  (clusters are well-connected in feature space)
  2. Regularity gain  >= delta_r (merging improves the joint NND-CV)

The algorithm is iterative: after each merge the PAGA and NND-CV scores are
recomputed for the new super-cluster, which naturally handles 3+ fragments of
the same type in successive iterations.

The NND-CV function is the primary plug-in point: replace it with any
regularity metric that takes (centers, boolean mask) -> float where lower
means more regular.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Plug-in regularity metric
# ---------------------------------------------------------------------------

def nnd_cv(centers: NDArray, mask: NDArray) -> float:
    """Coefficient of variation of within-cluster nearest-neighbour distances.

    Lower = more regular.  Returns np.inf when fewer than 2 cells are present.
    Swap this function to use any other regularity metric (Ripley K, Voronoi
    area CV, …) without touching the rest of the merge algorithm.
    """
    pts = centers[mask]
    if len(pts) < 2:
        return np.inf
    dists, _ = KDTree(pts).query(pts, k=2)
    nn_dists = dists[:, 1]
    mean = float(nn_dists.mean())
    return float(nn_dists.std() / mean) if mean > 0 else 0.0


def regularity_gain(centers: NDArray, mask_a: NDArray, mask_b: NDArray) -> float:
    """ΔR = NND-CV(A) + NND-CV(B) − 2 × NND-CV(A ∪ B).

    Positive means the merge improves spatial regularity.
    Returns -inf if either cluster is too small for a reliable estimate.
    """
    cv_a = nnd_cv(centers, mask_a)
    cv_b = nnd_cv(centers, mask_b)
    if np.isinf(cv_a) or np.isinf(cv_b):
        return -np.inf
    cv_ab = nnd_cv(centers, mask_a | mask_b)
    return cv_a + cv_b - 2.0 * cv_ab


# ---------------------------------------------------------------------------
# PAGA-style inter-cluster connectivity
# ---------------------------------------------------------------------------

def paga_weight(W: sp.csr_matrix, idx_a: NDArray, idx_b: NDArray) -> float:
    """Normalised inter-cluster edge weight (PAGA-style cosine similarity).

    Score = Σ W[a→b] / sqrt(Σ W[a→*] · Σ W[b→*]).

    Analogous to a cosine similarity between the cluster row-sum vectors.
    Bounded in [0, 1]; 0 = no more connected than a disconnected cluster;
    1 = all inter-cluster weight goes between a and b.
    """
    W_a = W[idx_a]
    W_b = W[idx_b]
    cross = float(W_a[:, idx_b].sum())
    denom = float(np.sqrt(float(W_a.sum()) * float(W_b.sum())))
    return cross / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Main merge loop
# ---------------------------------------------------------------------------

def iterative_merge(
    centers: NDArray,
    labels: NDArray,
    W: sp.csr_matrix,
    theta_paga: float = 0.3,
    delta_r: float = 0.05,
    min_size: int = 5,
) -> tuple[NDArray, int]:
    """Greedily merge over-split clusters using PAGA connectivity + NND-CV gain.

    At each round, find the pair (A, B) with the highest regularity gain ΔR
    among all pairs that also pass the PAGA connectivity threshold.  Merge
    that pair, then repeat.  Handles 3+ fragment chains naturally: once A and
    B merge into AB, the next round evaluates (AB, C) with updated scores.

    Parameters
    ----------
    centers:
        Cell spatial coordinates, shape (n_cells, 2).
    labels:
        Over-split cluster assignments, shape (n_cells,).
    W:
        Feature k-NN graph (symmetric, weighted), shape (n_cells, n_cells).
        Use the raw feature graph (W_feat), not the repulsion-weighted one,
        so that PAGA reflects pure feature similarity.
    theta_paga:
        Minimum PAGA weight for a pair to enter the regularity test.
    delta_r:
        Minimum regularity gain ΔR to accept a merge.
    min_size:
        Clusters smaller than this are skipped (NND-CV is unreliable).

    Returns
    -------
    merged_labels:
        Updated assignments, relabelled to consecutive integers starting at 0.
    n_merges:
        Number of merges performed.
    """
    labels = labels.copy().astype(np.int64)
    n_merges = 0

    while True:
        cluster_ids = np.unique(labels)
        if len(cluster_ids) <= 1:
            break

        best_score = delta_r  # strict inequality required to merge
        best_pair: tuple[int, int] | None = None

        for ci, a in enumerate(cluster_ids):
            idx_a = np.where(labels == a)[0]
            if len(idx_a) < min_size:
                continue
            mask_a = labels == a

            for b in cluster_ids[ci + 1:]:
                idx_b = np.where(labels == b)[0]
                if len(idx_b) < min_size:
                    continue

                if paga_weight(W, idx_a, idx_b) < theta_paga:
                    continue

                gain = regularity_gain(centers, mask_a, labels == b)
                if gain > best_score:
                    best_score = gain
                    best_pair = (a, b)

        if best_pair is None:
            break

        a, b = best_pair
        labels[labels == b] = a
        n_merges += 1

    # Relabel to consecutive integers
    old_ids = np.unique(labels)
    remap = np.zeros(int(old_ids.max()) + 1, dtype=np.int64)
    for new_id, old_id in enumerate(old_ids):
        remap[old_id] = new_id

    return remap[labels], n_merges
