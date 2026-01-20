import numpy as np


def cluster_separation_loss(means, covariances, weights, X):
    """
    Encourage clusters to be well-separated.
    Returns negative of minimum pairwise distance between cluster means.
    """
    n_components = means.shape[0]
    min_dist = np.inf

    for i in range(n_components):
        for j in range(i + 1, n_components):
            dist = np.linalg.norm(means[i] - means[j])
            min_dist = min(min_dist, dist)

    return -min_dist  # Negative because we want to maximize separation


def cluster_size_balance_loss(means, covariances, weights, X):
    """
    Encourage balanced cluster sizes.
    Penalizes deviation from uniform weights.
    """
    uniform_weight = 1.0 / len(weights)
    return np.sum((weights - uniform_weight) ** 2)


def cluster_compactness_loss(means, covariances, weights, X):
    """
    Encourage compact clusters (small covariances).
    """
    if isinstance(covariances, np.ndarray):
        if covariances.ndim == 3:  # full covariance
            # Sum of trace of all covariances
            return np.sum([np.trace(cov) for cov in covariances])
        elif covariances.ndim == 2:  # diagonal covariance
            return np.sum(covariances)
        else:  # spherical
            return np.sum(covariances)
    return 0.0
