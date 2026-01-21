import warnings

import numpy as np
from sklearn.base import _fit_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
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
            membership_threshold=0.2,
            debug_spatial=False,
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
        self.membership_threshold = membership_threshold
        self.debug_spatial = debug_spatial
        self.spatial_quality_history_ = []  # Track spatial quality over iterations

    def _build_knn_graph(self, positions, k=15):
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(positions)
        dists, indices = nbrs.kneighbors(positions)

        # Remove self-neighbor (distance 0)
        self.knn_dists = dists[:, 1:]
        self.knn_indices = indices[:, 1:]

    def _compute_within_cluster_nn_distances(
            self,
            resp,
            cluster_idx,
    ):
        cluster_mask = (resp[:, cluster_idx] > self.membership_threshold) | (resp[:, cluster_idx] == np.max(resp, axis=1))
        n_samples = resp.shape[0]

        nn_distances = np.full(n_samples, np.inf)

        if not np.any(cluster_mask):
            return np.zeros(n_samples)

        # For each sample, check its k nearest neighbors
        for i in range(n_samples):
            neighbors = self.knn_indices[i]
            neighbor_dists = self.knn_dists[i]

            # Filter neighbors belonging to this cluster
            valid = cluster_mask[neighbors]

            if np.any(valid):
                nn_distances[i] = neighbor_dists[valid].min()

        nn_distances[~np.isfinite(nn_distances)] = 0.0
        return nn_distances

    def evaluate_spatial_quality(self, resp=None, log_resp=None, return_details=False):
        """
        Evaluate the spatial quality of the current clustering.

        Measures how consistent the within-cluster NN distances are.
        Lower coefficient of variation = better spatial consistency.

        Parameters
        ----------
        resp : array of shape (n_samples, n_components), optional
            Responsibility matrix. If None, uses current log_resp
        log_resp : array of shape (n_samples, n_components), optional
            Log responsibility matrix
        return_details : bool
            If True, return detailed statistics per cluster

        Returns
        -------
        quality_score : float
            Overall spatial quality score (lower = more consistent)
            This is the weighted average of CV across clusters
        details : dict (optional)
            Detailed statistics if return_details=True
        """
        if resp is None:
            if log_resp is None:
                raise ValueError("Must provide either resp or log_resp")
            resp = np.exp(log_resp)

        resp = np.asarray(resp)
        n_components = resp.shape[1]

        cluster_stats = []
        total_weighted_cv = 0.0
        total_weight = 0.0

        for k in range(n_components):
            # Compute NN distances within this cluster
            nn_distances_k = self._compute_within_cluster_nn_distances(resp, k)

            # Get weights for this cluster
            weights_k = resp[:, k]
            cluster_size = np.sum(weights_k)

            if cluster_size < 1e-6:
                continue

            # Compute weighted statistics
            mean_nn = np.average(nn_distances_k, weights=weights_k)
            variance = np.average((nn_distances_k - mean_nn) ** 2, weights=weights_k)
            std_nn = np.sqrt(variance)

            # Coefficient of variation (CV): std/mean
            # Lower CV = more consistent distances
            cv = std_nn / (mean_nn + 1e-10)

            cluster_stats.append({
                'cluster': k,
                'size': cluster_size,
                'mean_nn_dist': mean_nn,
                'std_nn_dist': std_nn,
                'cv': cv,
                'min_nn_dist': np.min(nn_distances_k[weights_k > 1e-6]) if np.any(weights_k > 1e-6) else 0,
                'max_nn_dist': np.max(nn_distances_k[weights_k > 1e-6]) if np.any(weights_k > 1e-6) else 0,
            })

            # Weighted average of CV (weighted by cluster size)
            total_weighted_cv += cv * cluster_size
            total_weight += cluster_size

        # Overall quality score: weighted average CV
        quality_score = total_weighted_cv / (total_weight + 1e-10)

        if return_details:
            return quality_score, cluster_stats
        return quality_score

    def print_spatial_quality(self, resp=None, log_resp=None, iteration=None):
        """Print detailed spatial quality information."""
        quality_score, cluster_stats = self.evaluate_spatial_quality(
            resp=resp, log_resp=log_resp, return_details=True
        )

        prefix = f"[Iter {iteration}] " if iteration is not None else ""
        print(f"\n{prefix}=== Spatial Quality Report ===")
        print(f"Overall Quality Score (avg CV): {quality_score:.4f}")

        # Also compute and show what happens WITHOUT spatial penalty
        if self.knn_indices is not None and iteration == 1:
            print("\n[DEBUG] Computing NN distances for cluster 0...")
            if log_resp is not None:
                resp = np.exp(log_resp)
            self._compute_within_cluster_nn_distances(resp, 0)

        print("\nPer-Cluster Statistics:")
        print(f"{'Cluster':<8} {'Size':<8} {'Mean NN':<10} {'Std NN':<10} {'CV':<8} {'Min NN':<10} {'Max NN':<10}")
        print("-" * 74)

        for stats in cluster_stats:
            print(f"{stats['cluster']:<8} "
                  f"{stats['size']:<8.2f} "
                  f"{stats['mean_nn_dist']:<10.4f} "
                  f"{stats['std_nn_dist']:<10.4f} "
                  f"{stats['cv']:<8.4f} "
                  f"{stats['min_nn_dist']:<10.4f} "
                  f"{stats['max_nn_dist']:<10.4f}")
        print("=" * 74)

    def _compute_spatial_penalty(self, log_resp, xp=None, annealing_factor=1.0):
        if xp is None: xp = np
        resp = xp.exp(log_resp)
        n_samples, n_components = resp.shape
        resp_np = np.asarray(resp)
        penalty = np.zeros((n_samples, n_components))

        for k in range(n_components):
            nn_distances_k = self._compute_within_cluster_nn_distances(resp_np, k)
            weights_k = resp_np[:, k]

            sum_w = np.sum(weights_k)
            if sum_w < 1e-10:
                penalty[:, k] = 10.0  # Standardize high penalty
                continue

            # Use Robust Statistics: Weighted Median instead of Mean
            # Sorting is expensive, so we can use a 75th percentile clipping
            # to prevent outliers from blowing up the variance.
            threshold = np.percentile(nn_distances_k, 90)
            nn_clipped = np.clip(nn_distances_k, 0, threshold)

            robust_mean = np.average(nn_clipped, weights=weights_k)
            robust_std = np.sqrt(np.average((nn_clipped - robust_mean) ** 2, weights=weights_k)) + 1e-10

            # Deviation using the robust mean
            deviation = np.abs(nn_distances_k - robust_mean) / (robust_mean + 1e-10)

            # We square the deviation to heavily penalize "spatial outliers"
            # while the annealing_factor scales the overall influence
            penalty[:, k] = (deviation ** 2) + (robust_std / (robust_mean + 1e-10))

        return xp.asarray(penalty) * annealing_factor

    def _e_step(self, X, xp=None, annealing_factor=1.0):
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

        if self.knn_indices is not None and self.spatial_penalty_weight > 0:
            spatial_penalty = self._compute_spatial_penalty(log_resp, xp=xp, annealing_factor=annealing_factor)

            # Subtract penalty from log responsibilities (lower is better)
            log_resp = log_resp - self.spatial_penalty_weight * spatial_penalty

            # Re-normalize responsibilities
            from scipy.special import logsumexp
            log_resp_norm = logsumexp(log_resp, axis=1, keepdims=True)
            log_resp = log_resp - log_resp_norm

        return xp.mean(log_prob_norm), log_resp

    def fit_spatial(self, X, positions):
        self.fit_predict_spatial(X=X, positions=positions)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict_spatial(self, X, positions, knn=15):
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
        self._build_knn_graph(positions, k=knn)

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
        self.spatial_quality_history_ = []  # Reset history

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
                    annealing_factor = n_iter / self.max_iter

                    log_prob_norm, log_resp = self._e_step(X, xp=xp, annealing_factor=annealing_factor)

                    # Debug: print spatial quality
                    if self.debug_spatial:
                        spatial_quality = self.evaluate_spatial_quality(log_resp=log_resp)
                        self.spatial_quality_history_.append(spatial_quality)

                        if n_iter == 1 or n_iter % 5 == 0 or n_iter == self.max_iter:
                            self.print_spatial_quality(log_resp=log_resp, iteration=n_iter)

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

        # Final spatial quality report
        if self.debug_spatial:
            print("\n" + "=" * 74)
            print("FINAL CLUSTERING RESULTS")
            self.print_spatial_quality(log_resp=log_resp, iteration="FINAL")

        return xp.argmax(log_resp, axis=1)
