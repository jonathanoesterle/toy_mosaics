import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from gmm_losses import cluster_size_balance_loss


class GMMWithCustomLoss:
    """
    Gaussian Mixture Model with custom cluster-level loss terms.

    This extends sklearn's GaussianMixture by allowing users to define
    custom loss functions that operate on cluster-level statistics.
    """

    def __init__(self, n_components=3, covariance_type='full',
                 max_iter=100, reg_covar=1e-6, tol=1e-3,
                 custom_loss_fn=None, custom_loss_weight=1.0,
                 optimize_with_custom_loss=True):
        """
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        covariance_type : str
            Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
        max_iter : int
            Maximum number of EM iterations
        reg_covar : float
            Regularization added to covariance diagonal
        tol : float
            Convergence threshold
        custom_loss_fn : callable or None
            Custom loss function taking (means, covariances, weights, X)
            Should return a scalar loss value
        custom_loss_weight : float
            Weight for the custom loss term
        optimize_with_custom_loss : bool
            If True, use gradient-based optimization with custom loss
            If False, use standard EM then evaluate custom loss
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.tol = tol
        self.custom_loss_fn = custom_loss_fn
        self.custom_loss_weight = custom_loss_weight
        self.optimize_with_custom_loss = optimize_with_custom_loss

        # Initialize base GMM
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            reg_covar=reg_covar,
            tol=tol,
            random_state=42
        )

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.converged_ = False
        self.lower_bound_ = None

    def fit(self, X, y=None, init_means=None, init_covs=None, init_weights=None):
        """
        X : array-like, shape (n_samples, n_features)
        init_means : array-like (n_components, n_features), optional
        """
        # If no initial guess is provided, use standard GMM as before
        if init_means is None:
            self.gmm.fit(X)
            start_means = self.gmm.means_
            start_covs = self.gmm.covariances_
            start_weights = self.gmm.weights_
        else:
            # Use provided guess (ensure they are numpy arrays)
            start_means = np.asarray(init_means)
            start_covs = np.asarray(init_covs) if init_covs is not None else self._init_covs_from_data(X)
            start_weights = np.asarray(init_weights) if init_weights is not None else np.ones(
                self.n_components) / self.n_components

        if not self.optimize_with_custom_loss or self.custom_loss_fn is None:
            self.means_, self.covariances_, self.weights_ = start_means, start_covs, start_weights
        else:
            # Seeding the custom optimizer with our guess
            self._optimize_with_custom_loss(X, start_means, start_covs, start_weights)

        return self

    def params_from_labels(self, X, labels):
        """Calculate empirical means/covs from hard labels."""
        n_samples, n_features = X.shape
        means = np.array([X[labels == k].mean(axis=0) for k in range(self.n_components)])
        weights = np.array([np.sum(labels == k) / n_samples for k in range(self.n_components)])

        covs = []
        for k in range(self.n_components):
            diff = X[labels == k] - means[k]
            c = (diff.T @ diff) / np.sum(labels == k)
            covs.append(c + np.eye(n_features) * self.reg_covar)

        return means, np.array(covs), weights

    def _optimize_with_custom_loss(self, X, start_means, start_covs, start_weights):
        # Pack the provided initial guess
        init_params = self._pack_parameters(start_means, start_covs, start_weights)

        def objective(params):
            means, covs, weights = self._unpack_parameters(params, X.shape[1])
            nll = -self._compute_log_likelihood(X, means, covs, weights)

            custom_loss = 0.0
            if self.custom_loss_fn is not None:
                custom_loss = self.custom_loss_fn(means, covs, weights, X)

            return nll + self.custom_loss_weight * custom_loss

        result = minimize(objective, init_params, method='L-BFGS-B', options={'maxiter': self.max_iter})

        # Unpack optimized parameters
        self.means_, self.covariances_, self.weights_ = self._unpack_parameters(
            result.x, X.shape[1]
        )
        self.converged_ = result.success
        self.lower_bound_ = -result.fun

    def _pack_parameters(self, means, covariances, weights):
        """Pack GMM parameters into a single vector."""
        params = [means.flatten()]

        if self.covariance_type == 'full':
            # Store Cholesky decomposition L such that Sigma = L @ L.T
            # To ensure L[i,i] > 0, we store log(L[i,i])
            for cov in covariances:
                L = np.linalg.cholesky(cov + np.eye(cov.shape[0]) * self.reg_covar)
                diag_idx = np.diag_indices(L.shape[0])
                L[diag_idx] = np.log(L[diag_idx])
                params.append(L[np.tril_indices(L.shape[0])])
        elif self.covariance_type == 'diag':
            # Store log-variance to ensure positivity
            params.append(np.log(covariances.flatten()))
        elif self.covariance_type == 'spherical':
            # Store log-variance
            params.append(np.log(covariances))

        # Use softmax-like unconstrained representation for weights
        # We'll use log(weights) and then normalize with logsumexp
        params.append(np.log(weights + 1e-15))

        return np.concatenate(params)

    def _unpack_parameters(self, params, n_features):
        """Unpack parameter vector into means, covariances, weights."""
        idx = 0

        # Means
        means_size = self.n_components * n_features
        means = params[idx:idx + means_size].reshape(self.n_components, n_features)
        idx += means_size

        # Covariances
        if self.covariance_type == 'full':
            n_cov_params = n_features * (n_features + 1) // 2
            covariances = np.zeros((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                l_params = params[idx:idx + n_cov_params]
                idx += n_cov_params
                
                L = np.zeros((n_features, n_features))
                L[np.tril_indices(n_features)] = l_params
                diag_idx = np.diag_indices(n_features)
                L[diag_idx] = np.exp(L[diag_idx])
                
                covariances[k] = L @ L.T + np.eye(n_features) * self.reg_covar
        elif self.covariance_type == 'diag':
            cov_size = self.n_components * n_features
            log_cov = params[idx:idx + cov_size].reshape(self.n_components, n_features)
            covariances = np.exp(log_cov) + self.reg_covar
            idx += cov_size
        elif self.covariance_type == 'spherical':
            log_cov = params[idx:idx + self.n_components]
            covariances = np.exp(log_cov) + self.reg_covar
            idx += self.n_components

        # Weights
        log_weights = params[idx:idx + self.n_components]
        # Softmax to get weights
        weights = np.exp(log_weights - logsumexp(log_weights))

        return means, covariances, weights

    def _compute_log_likelihood(self, X, means, covariances, weights):
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            diff = X - means[k]

            if self.covariance_type == 'full':
                cov = covariances[k]
                try:
                    L = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    return -np.inf

                log_det = 2 * np.sum(np.log(np.diag(L)))
                y = solve_triangular(L, diff.T, lower=True)
                mahalanobis = np.sum(y ** 2, axis=0)
                log_prob[:, k] = -0.5 * (mahalanobis + log_det + n_features * np.log(2 * np.pi))

            elif self.covariance_type == 'diag':
                cov = covariances[k]
                log_det = np.sum(np.log(cov))
                mahalanobis = np.sum((diff ** 2) / cov, axis=1)
                log_prob[:, k] = -0.5 * (mahalanobis + log_det + n_features * np.log(2 * np.pi))
            
            elif self.covariance_type == 'spherical':
                cov = covariances[k]
                log_det = n_features * np.log(cov)
                mahalanobis = np.sum(diff ** 2, axis=1) / cov
                log_prob[:, k] = -0.5 * (mahalanobis + log_det + n_features * np.log(2 * np.pi))

            log_prob[:, k] += np.log(weights[k] + 1e-15)

        return np.sum(logsumexp(log_prob, axis=1))

    def predict(self, X):
        """Predict cluster labels."""
        log_prob = self._compute_responsibilities(X)
        return np.argmax(log_prob, axis=1)

    def predict_proba(self, X):
        """Predict cluster probabilities."""
        log_prob = self._compute_responsibilities(X)
        # Log-sum-exp trick for normalization
        log_prob_norm = logsumexp(log_prob, axis=1, keepdims=True)
        return np.exp(log_prob - log_prob_norm)

    def _compute_responsibilities(self, X):
        """Compute log responsibilities for each sample."""
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            diff = X - self.means_[k]
            if self.covariance_type == 'full':
                cov = self.covariances_[k]
                L = np.linalg.cholesky(cov)
                log_det = 2 * np.sum(np.log(np.diag(L)))
                y = solve_triangular(L, diff.T, lower=True)
                mahalanobis = np.sum(y ** 2, axis=0)
                log_prob[:, k] = -0.5 * (mahalanobis + log_det + n_features * np.log(2 * np.pi))
            elif self.covariance_type == 'diag':
                cov = self.covariances_[k]
                log_det = np.sum(np.log(cov))
                mahalanobis = np.sum((diff ** 2) / cov, axis=1)
                log_prob[:, k] = -0.5 * (mahalanobis + log_det + n_features * np.log(2 * np.pi))
            elif self.covariance_type == 'spherical':
                cov = self.covariances_[k]
                log_det = n_features * np.log(cov)
                mahalanobis = np.sum(diff ** 2, axis=1) / cov
                log_prob[:, k] = -0.5 * (mahalanobis + log_det + n_features * np.log(2 * np.pi))

            log_prob[:, k] += np.log(self.weights_[k] + 1e-15)

        return log_prob


def spatial_separation_loss(means, covariances, weights, X, positions, min_distance=1.0, 
                            return_penalty=False, reference_positions=None, reference_responsibilities=None):
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

    if return_penalty:
        return total_loss, penalty_matrix
    
    return total_loss


class SpatialAwareGMM(GMMWithCustomLoss):
    """
    GMM that considers spatial positions when clustering.
    Prevents spatially proximate cells from being assigned to the same cluster.
    """

    def __init__(self, n_components=3, covariance_type='full',
                 max_iter=100, reg_covar=1e-6, tol=1e-3,
                 spatial_loss_weight=1.0, min_spatial_distance=1.0,
                 spatial_inference_weight=1.0,
                 spatial_loss_fn=spatial_separation_loss,
                 additional_loss_fn=None, additional_loss_weight=0.0):
        """
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        covariance_type : str
            Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
        max_iter : int
            Maximum number of EM iterations
        reg_covar : float
            Regularization added to covariance diagonal
        tol : float
            Convergence threshold
        spatial_loss_weight : float
            Weight for spatial separation loss during training
        min_spatial_distance : float
            Minimum spatial distance threshold for penalty
        spatial_inference_weight : float
            Weight for spatial term during inference/prediction.
            Higher values make predictions more influenced by spatial proximity.
        spatial_loss_fn : callable
            Function to compute spatial separation loss.
            Should take (means, covariances, weights, X, positions, min_distance)
            and return a scalar loss or (loss, penalty_matrix) if return_penalty=True.
        additional_loss_fn : callable or None
            Optional additional custom loss function
        additional_loss_weight : float
            Weight for additional custom loss
        """
        self.spatial_loss_weight = spatial_loss_weight
        self.min_spatial_distance = min_spatial_distance
        self.spatial_inference_weight = spatial_inference_weight
        self.spatial_loss_fn = spatial_loss_fn
        self.additional_loss_fn = additional_loss_fn
        self.additional_loss_weight = additional_loss_weight
        self.positions_ = None

        # Create combined loss function
        def combined_loss(means, covariances, weights, X):
            loss = 0.0

            # Spatial separation loss
            if self.positions_ is not None and self.spatial_loss_weight > 0:
                s_loss = self.spatial_loss_fn(
                    means, covariances, weights, X,
                    self.positions_, self.min_spatial_distance
                )
                # Handle potential tuple return (loss, penalty)
                if isinstance(s_loss, tuple):
                    s_loss = s_loss[0]
                loss += self.spatial_loss_weight * s_loss

            # Additional custom loss
            if self.additional_loss_fn is not None and self.additional_loss_weight > 0:
                loss += self.additional_loss_weight * self.additional_loss_fn(
                    means, covariances, weights, X
                )

            return loss

        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            reg_covar=reg_covar,
            tol=tol,
            custom_loss_fn=combined_loss,
            custom_loss_weight=1.0,  # Already weighted in combined_loss
            optimize_with_custom_loss=True
        )

    def fit(self, X, positions, y=None, init_means=None, init_covs=None, init_weights=None):
        """
        X : array-like, shape (n_samples, n_features)
            Feature data
        positions : array-like, shape (n_samples, 2)
            Spatial xy positions for each cell
        """
        self.positions_ = np.asarray(positions)

        if self.positions_.shape[0] != X.shape[0]:
            raise ValueError("X and positions must have same number of samples")

        if self.positions_.shape[1] != 2:
            raise ValueError("positions must be 2D (xy coordinates)")

        # Store training features for later use in prediction
        self.X_train_ = np.asarray(X)

        result = super().fit(X, y, init_means=init_means, init_covs=init_covs, init_weights=init_weights)

        # Compute and store training cluster assignments
        self.train_labels_ = super().predict(X)
        self.train_responsibilities_ = super().predict_proba(X)

        return result

    def fit_predict(self, X, positions, y=None, fit_kws=None, predict_kws=None):
        """
        Fit the model and predict cluster labels with spatial awareness.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature data
        positions : array-like, shape (n_samples, 2)
            Spatial xy positions for each cell

        Returns:
        --------
        labels : array (n_samples,)
            Cluster labels
        """
        if fit_kws is None:
            fit_kws = dict()
        if predict_kws is None:
            predict_kws = dict()

        self.fit(X, positions, y, **fit_kws)
        return self.predict(X, positions, **predict_kws)

    def predict(self, X, positions=None):
        """
        Predict cluster labels with optional spatial constraint.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature data
        positions : array-like, shape (n_samples, 2) or None
            Spatial xy positions. If None, uses feature-only prediction.

        Returns:
        --------
        labels : array (n_samples,)
            Cluster labels
        """
        if positions is None or self.spatial_inference_weight == 0:
            # Standard prediction without spatial constraint
            return super().predict(X)

        positions = np.asarray(positions)
        if positions.shape[0] != X.shape[0]:
            raise ValueError("X and positions must have same number of samples")

        # Compute feature-based log probabilities
        log_prob_features = self._compute_responsibilities(X)

        # Compute spatial penalty term for each sample-cluster pair
        spatial_penalty = self._compute_spatial_penalty(positions)

        # Combine feature probability with spatial penalty
        # Higher penalty = lower probability of assignment
        log_prob_combined = log_prob_features - self.spatial_inference_weight * spatial_penalty

        return np.argmax(log_prob_combined, axis=1)

    def predict_proba(self, X, positions=None):
        """
        Predict cluster probabilities with optional spatial constraint.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature data
        positions : array-like, shape (n_samples, 2) or None
            Spatial xy positions. If None, uses feature-only prediction.

        Returns:
        --------
        proba : array (n_samples, n_components)
            Cluster probabilities
        """
        if positions is None or self.spatial_inference_weight == 0:
            return super().predict_proba(X)

        positions = np.asarray(positions)
        if positions.shape[0] != X.shape[0]:
            raise ValueError("X and positions must have same number of samples")

        # Compute feature-based log probabilities
        log_prob_features = self._compute_responsibilities(X)

        # Compute spatial penalty
        spatial_penalty = self._compute_spatial_penalty(positions)

        # Combine and normalize
        log_prob_combined = log_prob_features - self.spatial_inference_weight * spatial_penalty

        # Convert to probabilities with logsumexp trick
        log_prob_norm = logsumexp(log_prob_combined, axis=1, keepdims=True)
        return np.exp(log_prob_combined - log_prob_norm)

    def _compute_spatial_penalty(self, positions):
        """
        Compute spatial penalty for each sample-cluster assignment.

        For each sample, computes how much penalty it would incur if assigned
        to each cluster, based on proximity to other samples likely in that cluster.

        Parameters:
        -----------
        positions : array (n_samples_new, 2)
            Spatial positions for samples to predict

        Returns:
        --------
        penalty : array (n_samples_new, n_components)
            Spatial penalty for each sample-cluster pair
        """
        if self.means_ is None:
            return np.zeros((positions.shape[0], self.n_components))

        # Determine reference positions and responsibilities
        if hasattr(self, 'positions_') and self.positions_ is not None and hasattr(self, 'train_responsibilities_'):
            reference_positions = self.positions_
            reference_resps = self.train_responsibilities_
        else:
            reference_positions = positions
            reference_resps = self.predict_proba(positions=None)

        # Delegate to spatial_loss_fn
        # We need an X for the signature, even if it's not used when reference_responsibilities is provided
        # We'll use reference_positions as X if it has the right shape, or positions.
        # But actually, if we provide reference_responsibilities, X is only used for n_samples.
        X_placeholder = np.zeros((positions.shape[0], self.means_.shape[1]))

        _, penalty = self.spatial_loss_fn(
            self.means_, self.covariances_, self.weights_,
            X_placeholder, positions, self.min_spatial_distance,
            return_penalty=True,
            reference_positions=reference_positions,
            reference_responsibilities=reference_resps
        )
        return penalty


# Demo usage
if __name__ == "__main__":
    # Generate sample data with features and spatial positions
    np.random.seed(42)
    n_samples = 300

    # Create feature data (3 clusters in feature space)
    X, y_true = make_blobs(n_samples=n_samples, centers=3, n_features=2,
                           cluster_std=1.0, random_state=42)

    # Create spatial positions (independent of features to demonstrate the issue)
    # Some cells from different feature clusters will be spatially close
    positions = np.random.randn(n_samples, 2) * 3

    # Standard GMM (ignores spatial positions)
    print("Fitting standard GMM...")
    gmm_standard = GMMWithCustomLoss(n_components=3, covariance_type='full',
                                     optimize_with_custom_loss=False)
    gmm_standard.fit(X)

    # Spatial-aware GMM
    print("Fitting spatial-aware GMM...")
    gmm_spatial = SpatialAwareGMM(
        n_components=3,
        covariance_type='full',
        spatial_loss_weight=10.0,
        min_spatial_distance=1.5,
        spatial_inference_weight=10.0
    )
    gmm_spatial.fit(X, positions)

    # Spatial-aware GMM with additional balance loss
    print("Fitting spatial-aware GMM with balance loss...")
    gmm_spatial_balanced = SpatialAwareGMM(
        n_components=3,
        covariance_type='full',
        spatial_loss_weight=10.0,
        min_spatial_distance=1.5,
        spatial_inference_weight=10.0,
        additional_loss_fn=cluster_size_balance_loss,
        additional_loss_weight=10.0
    )
    gmm_spatial_balanced.fit(X, positions)

    # Demo: predict on new data with spatial constraint
    print("\nDemo: Predicting with spatial constraint...")
    # Create a small test set
    X_test = np.array([[0, 0], [1, 1], [5, 5]])
    pos_test = np.array([[0, 0], [0.5, 0.5], [10, 10]])

    print("Test predictions (with spatial constraint):")
    labels_with_spatial = gmm_spatial.predict(X_test, pos_test)
    print(f"  Labels: {labels_with_spatial}")

    print("Test predictions (without spatial constraint):")
    labels_without_spatial = gmm_spatial.predict(X_test, positions=None)
    print(f"  Labels: {labels_without_spatial}")
    print("  Note: May differ if spatial proximity changes assignments")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    models = [
        (gmm_standard, "Standard GMM"),
        (gmm_spatial, "Spatial-Aware GMM"),
        (gmm_spatial_balanced, "Spatial + Balance")
    ]

    for col, (model, title) in enumerate(models):
        # Use spatial-aware prediction if available
        if isinstance(model, SpatialAwareGMM):
            labels = model.predict(X, positions)
        else:
            labels = model.predict(X)

        # Plot in feature space
        ax = axes[0, col]
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
        ax.scatter(model.means_[:, 0], model.means_[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2)
        ax.set_title(f"{title}\n(Feature Space)")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        # Plot in spatial space
        ax = axes[1, col]
        scatter = ax.scatter(positions[:, 0], positions[:, 1],
                             c=labels, cmap='viridis', alpha=0.6)
        ax.set_title(f"{title}\n(Spatial XY)")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # Add colorbar for spatial plot
        plt.colorbar(scatter, ax=ax, label='Cluster')

    plt.tight_layout()
    plt.show()

    print("\nCluster weights comparison:")
    print(f"Standard: {gmm_standard.weights_.round(3)}")
    print(f"Spatial: {gmm_spatial.weights_.round(3)}")
    print(f"Spatial+Balance: {gmm_spatial_balanced.weights_.round(3)}")

    # Analyze spatial proximity within clusters
    print("\n--- Spatial Proximity Analysis ---")
    for i, (model, name) in enumerate([(gmm_standard, "Standard"),
                                       (gmm_spatial, "Spatial-Aware")]):
        if isinstance(model, SpatialAwareGMM):
            labels = model.predict(X, positions)
        else:
            labels = model.predict(X)
        
        close_pairs_same_cluster = 0
        total_close_pairs = 0

        # Efficient proximity analysis using KDTree
        nn = NearestNeighbors(radius=1.5)
        nn.fit(positions)
        distances, indices = nn.radius_neighbors(positions)
        
        for idx1 in range(n_samples):
            neighbor_indices = indices[idx1]
            for idx2 in neighbor_indices:
                if idx1 < idx2: # Only count each pair once
                    total_close_pairs += 1
                    if labels[idx1] == labels[idx2]:
                        close_pairs_same_cluster += 1

        if total_close_pairs > 0:
            pct = 100 * close_pairs_same_cluster / total_close_pairs
            print(f"{name}: {close_pairs_same_cluster}/{total_close_pairs} "
                  f"({pct:.1f}%) spatially close pairs in same cluster")
