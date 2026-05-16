import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors


def spatial_separation_loss(means, covariances, weights, X, positions, min_distance=1.0,
                            return_penalty=False, return_responsibilities=False,
                            reference_positions=None, reference_responsibilities=None):
    """
    Penalize cells that are close in spatial position (xy) but assigned to same cluster.
    """
    n_samples = X.shape[0]
    n_components = means.shape[0]

    if reference_responsibilities is None:
        # Compute responsibilities (soft cluster assignments) for current X
        # We'll use a local simplified version of responsibility calculation to avoid dependency
        log_prob = np.zeros((n_samples, n_components))
        for k in range(n_components):
            diff = X - means[k]
            if covariances.ndim == 3:  # full
                cov = covariances[k]
                L = np.linalg.cholesky(cov + 1e-6 * np.eye(X.shape[1]))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                y = solve_triangular(L, diff.T, lower=True)
                mahalanobis = np.sum(y ** 2, axis=0)
                log_prob[:, k] = -0.5 * (mahalanobis + log_det + X.shape[1] * np.log(2 * np.pi))
            elif covariances.ndim == 2:  # diag
                cov = covariances[k] + 1e-6
                log_det = np.sum(np.log(cov))
                mahalanobis = np.sum((diff ** 2) / cov, axis=1)
                log_prob[:, k] = -0.5 * (mahalanobis + log_det + X.shape[1] * np.log(2 * np.pi))

            log_prob[:, k] += np.log(weights[k] + 1e-15)

        log_prob_norm = logsumexp(log_prob, axis=1, keepdims=True)
        responsibilities = np.exp(log_prob - log_prob_norm)
    else:
        # If reference_responsibilities matches X, use it.
        # Otherwise (e.g. predicting on new data), we need current responsibilities for total_loss.
        if reference_responsibilities.shape[0] == n_samples:
            responsibilities = reference_responsibilities
        else:
            # Recompute responsibilities for current X (which corresponds to positions)
            log_prob = np.zeros((n_samples, n_components))
            for k in range(n_components):
                diff = X - means[k]
                if covariances.ndim == 3:  # full
                    cov = covariances[k]
                    L = np.linalg.cholesky(cov + 1e-6 * np.eye(X.shape[1]))
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    y = solve_triangular(L, diff.T, lower=True)
                    mahalanobis = np.sum(y ** 2, axis=0)
                    log_prob[:, k] = -0.5 * (mahalanobis + log_det + X.shape[1] * np.log(2 * np.pi))
                elif covariances.ndim == 2:  # diag
                    cov = covariances[k] + 1e-6
                    log_det = np.sum(np.log(cov))
                    mahalanobis = np.sum((diff ** 2) / cov, axis=1)
                    log_prob[:, k] = -0.5 * (mahalanobis + log_det + X.shape[1] * np.log(2 * np.pi))

                log_prob[:, k] += np.log(weights[k] + 1e-15)

            log_prob_norm = logsumexp(log_prob, axis=1, keepdims=True)
            responsibilities = np.exp(log_prob - log_prob_norm)

    if reference_positions is None:
        reference_positions = positions
        reference_resps = responsibilities
    else:
        reference_resps = reference_responsibilities
        if reference_resps is None:
            # This case shouldn't happen with the way we call it from GMM,
            # but for completeness we'd need to compute it for reference_positions too.
            raise ValueError("reference_responsibilities must be provided if reference_positions is provided")

    # Use KDTree for efficient neighbor search
    nn = NearestNeighbors(radius=min_distance)
    nn.fit(reference_positions)
    distances, indices = nn.radius_neighbors(positions)

    penalty_matrix = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        neighbor_indices = indices[i]
        neighbor_distances = distances[i]

        # Remove self if we are comparing with the same set
        if reference_positions is positions:
            mask = neighbor_indices != i
            neighbor_indices = neighbor_indices[mask]
            neighbor_distances = neighbor_distances[mask]

        if len(neighbor_indices) > 0:
            # dist_weight = (min_distance - distance)
            dist_weight = min_distance - neighbor_distances

            # For each cluster k: penalty += sum_j(resp_j,k * dist_weight_j)
            neighbor_resps = reference_resps[neighbor_indices]
            penalty_matrix[i] = np.sum(neighbor_resps * dist_weight[:, np.newaxis], axis=0)

    # total_loss = sum_i sum_k (resp_i,k * penalty_matrix_i,k)
    total_loss = np.sum(responsibilities * penalty_matrix)

    # Normalize
    total_loss /= (n_samples * (n_samples - 1)) if n_samples > 1 else 1.0

    if return_penalty and return_responsibilities:
        return total_loss, penalty_matrix, responsibilities
    elif return_penalty:
        return total_loss, penalty_matrix
    elif return_responsibilities:
        return total_loss, responsibilities

    return total_loss
