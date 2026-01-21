import numpy as np
import warnings

from sklearn.base import _fit_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky
from sklearn.utils._array_api import (
    get_namespace,
)
from sklearn.utils.validation import validate_data, check_random_state


class PopulationConstrainedGMM(GaussianMixture):

    def __init__(
            self,
            n_components=1,
            *,
            covariance_type="full",
            tol=1e-3,
            reg_covar=1e-6,
            max_iter=100,
            n_init=1,
            init_params="kmeans",
            weights_init=None,
            means_init=None,
            precisions_init=None,
            random_state=None,
            warm_start=False,
            verbose=0,
            verbose_interval=10,
            spatial_penalty_weight=1.0,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
            covariance_type=covariance_type,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
        )

        self.spatial_penalty_weight = spatial_penalty_weight
        self.pair_dists = None

    def _compute_within_cluster_nn_distances(self, resp, cluster_idx):
        """
        Compute nearest neighbor distances for samples within a specific cluster.

        Uses soft assignment: distances are weighted by responsibilities.

        Parameters
        ----------
        resp : array of shape (n_samples, n_components)
            Responsibility matrix (probabilities)
        cluster_idx : int
            Which cluster to compute NN distances for

        Returns
        -------
        nn_distances : array of shape (n_samples,)
            For each sample, the nearest neighbor distance IF it were in this cluster
        """
        n_samples = resp.shape[0]
        nn_distances = np.zeros(n_samples)

        # Get responsibilities for this cluster
        cluster_weights = resp[:, cluster_idx]

        for i in range(n_samples):
            # For sample i, find nearest neighbor among samples in this cluster
            # Weight distances by how much each sample belongs to the cluster

            min_dist = np.inf

            for j in range(n_samples):
                if i == j:
                    continue

                # Distance to sample j, weighted by j's membership in this cluster
                # We want to find the nearest neighbor among cluster members
                if cluster_weights[j] > 1e-6:  # Only consider samples with non-negligible membership
                    dist = self.pair_dists[i, j]
                    if dist < min_dist:
                        min_dist = dist

            nn_distances[i] = min_dist if min_dist < np.inf else 0.0

        return nn_distances

    def _compute_spatial_penalty(self, log_resp, xp=None):
        """
        Compute spatial penalty based on within-cluster nearest-neighbor distance consistency.

        For each cluster k:
        1. Compute NN distances AS IF each sample belonged to cluster k
        2. Compute mean/std of these NN distances (weighted by current responsibilities)
        3. Penalize samples whose NN distance deviates from the cluster's pattern

        Parameters
        ----------
        log_resp : array-like of shape (n_samples, n_components)
            Log responsibilities from E-step

        Returns
        -------
        penalty : array of shape (n_samples, n_components)
            Penalty values for each sample-cluster assignment
        """
        if xp is None:
            xp = np

        resp = xp.exp(log_resp)
        n_samples = resp.shape[0]
        n_components = resp.shape[1]

        # Convert to numpy for easier computation
        resp_np = np.asarray(resp)

        penalty = np.zeros((n_samples, n_components))

        # For each cluster, compute the penalty
        for k in range(n_components):
            # Compute NN distances within this cluster
            nn_distances_k = self._compute_within_cluster_nn_distances(resp_np, k)

            # Compute cluster statistics weighted by responsibilities
            weights_k = resp_np[:, k]

            if np.sum(weights_k) < 1e-10:
                # Cluster is empty, no penalty
                continue

            # Weighted mean and std of NN distances for this cluster
            mean_nn_dist = np.average(nn_distances_k, weights=weights_k)
            variance = np.average((nn_distances_k - mean_nn_dist) ** 2, weights=weights_k)
            std_nn_dist = np.sqrt(variance) + 1e-10  # Add epsilon for stability

            # Penalize deviation from cluster's mean NN distance
            deviation = (nn_distances_k - mean_nn_dist) / std_nn_dist
            penalty[:, k] = deviation ** 2

        return xp.asarray(penalty)

    def _e_step(self, X, xp=None):
        """E step with spatial penalty.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        xp, _ = get_namespace(X, xp=xp)
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, xp=xp)

        # Add spatial penalty if pair_dists is available
        if self.pair_dists is not None and self.spatial_penalty_weight > 0:
            spatial_penalty = self._compute_spatial_penalty(log_resp, xp=xp)

            # Subtract penalty from log responsibilities (lower is better)
            log_resp = log_resp - self.spatial_penalty_weight * spatial_penalty

            # Re-normalize responsibilities
            from scipy.special import logsumexp
            log_resp_norm = logsumexp(log_resp, axis=1, keepdims=True)
            log_resp = log_resp - log_resp_norm

        return xp.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp, xp=None):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        xp, _ = get_namespace(X, log_resp, xp=xp)
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, xp.exp(log_resp), self.reg_covar, self.covariance_type, xp=xp
        )
        self.weights_ /= xp.sum(self.weights_)
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type, xp=xp
        )

    def fit_spatial(self, X, positions):
        self.fit_predict_spatial(X=X, positions=positions)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict_spatial(self, X, positions):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        positions : array-like of shape (n_samples, 2)
            Spatial positions (e.g., x, y coordinates) corresponding to each data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        # Precompute pairwise distances (this doesn't change)
        self.pair_dists = pairwise_distances(positions, metric='euclidean')

        xp, _ = get_namespace(X)
        X = validate_data(self, X, dtype=[xp.float64, xp.float32], ensure_min_samples=2)

        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X, xp=xp)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -xp.inf
        best_lower_bounds = []
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state, xp=xp)

            lower_bound = -xp.inf if do_init else self.lower_bound_
            current_lower_bounds = []

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                converged = False
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X, xp=xp)
                    self._m_step(X, log_resp, xp=xp)
                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)
                    current_lower_bounds.append(lower_bound)

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        converged = True
                        break

                self._print_verbose_msg_init_end(lower_bound, converged)

                if lower_bound > max_lower_bound or max_lower_bound == -xp.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter
                    best_lower_bounds = current_lower_bounds
                    self.converged_ = converged

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                (
                    "Best performing initialization did not converge. "
                    "Try different init parameters, or increase max_iter, "
                    "tol, or check for degenerate data."
                ),
                ConvergenceWarning,
            )

        self._set_parameters(best_params, xp=xp)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound
        self.lower_bounds_ = best_lower_bounds

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X, xp=xp)

        return xp.argmax(log_resp, axis=1)