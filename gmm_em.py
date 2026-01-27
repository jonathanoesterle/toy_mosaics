import warnings

import numpy as np
from scipy.special import logsumexp
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
            use_duplicate_detection=False,
            duplicate_threshold_q=10.,
            duplicate_push_value=10.0,
            duplicate_keep_value=1.0,

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
        self.use_duplicate_detection = use_duplicate_detection
        self.duplicate_threshold_q = duplicate_threshold_q
        self.duplicate_keep_value = duplicate_keep_value
        self.duplicate_push_value = duplicate_push_value
        self.spatial_quality_history_ = []


    def _compute_expected_nn_distances(self, resp):
        """
        Compute expected nearest neighbor distance for each sample and component.

        For sample i and component j:
        - Iterate through k-nearest neighbors (sorted by distance)
        - Compute probability that neighbor n is the first in cluster j
        - Expected distance = sum of (probability × distance)

        Parameters
        ----------
        resp : array-like of shape (n_samples, n_components)
            Responsibility matrix (posterior probabilities)

        Returns
        -------
        expected_nn : array of shape (n_samples, n_components)
            Expected nearest neighbor distance for each sample-component pair
        """
        n_samples, n_components = resp.shape
        k_neighbors = self.knn_indices.shape[1]

        expected_nn = np.zeros((n_samples, n_components))

        for i in range(n_samples):
            for j in range(n_components):
                # Get responsibilities of k-nearest neighbors for cluster j
                neighbor_indices = self.knn_indices[i]
                neighbor_distances = self.knn_dists[i]
                neighbor_probs = resp[neighbor_indices, j]

                # Compute probability that each neighbor is the first in cluster j
                expected_dist = 0.0
                cumulative_prob_none = 1.0  # P(none of the previous neighbors are in j)

                for n in range(k_neighbors):
                    q_n = neighbor_probs[n]
                    d_n = neighbor_distances[n]

                    if n < k_neighbors - 1:
                        # For neighbors 0 to k-2: P(this is first) = P(none before) × q_n
                        prob_this_is_first = cumulative_prob_none * q_n
                        expected_dist += prob_this_is_first * d_n
                        cumulative_prob_none *= (1.0 - q_n)
                    else:
                        # For k-th neighbor: assign all remaining probability
                        expected_dist += cumulative_prob_none * d_n

                expected_nn[i, j] = expected_dist

        return expected_nn

    def _compute_spatial_penalty(self, log_resp=None, resp=None, xp=None):
        """
        Compute spatial penalty based on expected nearest neighbor distances.

        For each cluster:
        1. Compute expected NN distance for each sample
        2. Compute weighted mean expected NN distance for the cluster
        3. Penalize deviations from cluster mean

        Parameters
        ----------
        log_resp : array-like of shape (n_samples, n_components), optional
            Log responsibilities
        resp : array-like of shape (n_samples, n_components), optional
            Responsibilities (if log_resp not provided)
        xp : module, optional
            Array namespace (numpy or cupy)

        Returns
        -------
        penalty : array of shape (n_samples, n_components)
            Spatial penalty for each sample-component pair
        """
        if xp is None:
            xp = np
        if resp is None:
            resp = xp.exp(log_resp)
        else:
            resp = xp.asarray(resp)

        n_samples, n_components = resp.shape
        resp_np = np.asarray(resp)
        penalty = np.zeros((n_samples, n_components))

        # Compute expected NN distances for all samples and components
        expected_nn = self._compute_expected_nn_distances(resp_np)

        # Compute cluster-level mean expected NN distance (weighted by responsibilities)
        for j in range(n_components):
            weights = resp_np[:, j]
            total_weight = np.sum(weights)

            if total_weight < 1e-6:
                continue

            # Weighted mean expected NN distance for cluster j
            mean_nn = np.sum(expected_nn[:, j] * weights) / total_weight

            # Coherence penalty: normalized squared deviation from cluster mean
            dev = np.abs(expected_nn[:, j] - mean_nn) / (mean_nn + 1e-10)
            penalty[:, j] = dev ** 2

        # Optional: Duplicate detection (if enabled)
        if self.use_duplicate_detection:
            duplicate_threshold = np.percentile(self.knn_dists, self.duplicate_threshold_q)

            for j in range(n_components):
                # Use hard membership for duplicate detection
                cluster_mask = (resp_np[:, j] > self.membership_threshold) | \
                               (resp_np[:, j] == np.max(resp_np, axis=1))

                for i in range(n_samples):
                    if not cluster_mask[i]:
                        continue

                    # Find neighbors in same cluster
                    neighbors_in_cluster = cluster_mask[self.knn_indices[i]]

                    # Check if any are too close (duplicates)
                    duplicate_mask = (self.knn_dists[i] < duplicate_threshold) & neighbors_in_cluster

                    if np.any(duplicate_mask):
                        # Found duplicates
                        duplicate_indices = self.knn_indices[i][duplicate_mask]

                        # Check if I'm the least confident among the duplicates
                        my_confidence = resp_np[i, j]
                        duplicate_confidences = resp_np[duplicate_indices, j]

                        if my_confidence <= np.min(duplicate_confidences):
                            penalty[i, j] += self.duplicate_push_value  # Strong penalty to push me out
                        if my_confidence >= np.max(duplicate_confidences):
                            penalty[i, j] -= self.duplicate_keep_value  # Keep me in


        return xp.asarray(penalty)

    def evaluate_spatial_quality(self, resp=None, log_resp=None, return_details=False):
        """
        Evaluate spatial quality based on expected NN distances.

        Parameters
        ----------
        resp : array-like, optional
            Responsibility matrix
        log_resp : array-like, optional
            Log responsibility matrix
        return_details : bool, default=False
            If True, return detailed per-cluster statistics

        Returns
        -------
        quality_score : float
            Overall spatial quality score (lower is better)
        cluster_stats : list of dict, optional
            Per-cluster statistics (if return_details=True)
        """
        if resp is None:
            resp = np.exp(log_resp)

        resp = np.asarray(resp)
        n_samples, n_components = resp.shape

        # Compute expected NN distances
        expected_nn = self._compute_expected_nn_distances(resp)

        cluster_stats = []
        total_penalty = 0.0
        total_weight = 0.0

        for j in range(n_components):
            weights = resp[:, j]
            cluster_size = np.sum(weights)

            if cluster_size < 1e-6:
                continue

            # Weighted mean expected NN distance
            mean_nn = np.sum(expected_nn[:, j] * weights) / cluster_size

            # Weighted standard deviation
            variance = np.sum(((expected_nn[:, j] - mean_nn) ** 2) * weights) / cluster_size
            std_nn = np.sqrt(variance)

            # Coefficient of variation
            cv = std_nn / (mean_nn + 1e-10)

            # Coherence penalty (matches what we optimize)
            dev = np.abs(expected_nn[:, j] - mean_nn) / (mean_nn + 1e-10)
            coherence_penalty = np.sum((dev ** 2) * weights) / cluster_size

            cluster_stats.append({
                'cluster': j,
                'size': cluster_size,
                'mean_nn': mean_nn,
                'std_nn': std_nn,
                'cv': cv,
                'coherence_penalty': coherence_penalty,
            })

            total_penalty += coherence_penalty * cluster_size
            total_weight += cluster_size

        # Overall quality = weighted average penalty
        quality_score = total_penalty / (total_weight + 1e-10)

        if return_details:
            return quality_score, cluster_stats
        return quality_score

    def print_spatial_quality(self, resp=None, log_resp=None, iteration=None):
        """Print simplified spatial quality information."""
        quality_score, cluster_stats = self.evaluate_spatial_quality(
            resp=resp, log_resp=log_resp, return_details=True
        )

        prefix = f"[Iter {iteration}] " if iteration is not None else ""
        print(f"\n{prefix}=== Spatial Quality Report ===")
        print(f"Overall Quality Score: {quality_score:.4f}")
        print("\nPer-Cluster Statistics:")
        print(f"{'Cluster':<8} {'Size':<10} {'Mean NN':<12} {'Std NN':<12} {'CV':<10} {'Penalty':<10}")
        print("-" * 72)

        for stats in cluster_stats:
            print(f"{stats['cluster']:<8} "
                  f"{stats['size']:<10.2f} "
                  f"{stats['mean_nn']:<12.4f} "
                  f"{stats['std_nn']:<12.4f} "
                  f"{stats['cv']:<10.4f} "
                  f"{stats['coherence_penalty']:<10.4f}")

        print("=" * 72)

    def _build_knn_graph(self, positions, k=15):
        """Build k-nearest neighbor graph from spatial positions."""
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(positions)
        dists, indices = nbrs.kneighbors(positions)

        # Remove self-neighbor (distance 0)
        self.knn_dists = dists[:, 1:]
        self.knn_indices = indices[:, 1:]

    def _e_step(self, X, xp=None, annealing_factor=1.0):
        """
        E step with spatial penalty based on expected NN distances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        xp : module, optional
            Array namespace
        annealing_factor : float, default=1.0
            Annealing factor for spatial penalty (0 to 1)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X
        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities
        """
        xp, _ = get_namespace(X, xp=xp)
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, xp=xp)

        if self.knn_indices is not None and self.spatial_penalty_weight > 0:
            spatial_penalty = self._compute_spatial_penalty(log_resp, xp=xp)

            # Subtract penalty from log responsibilities (lower is better)
            log_resp = log_resp - self.spatial_penalty_weight * spatial_penalty * annealing_factor

            log_resp_norm = logsumexp(log_resp, axis=1, keepdims=True)
            log_resp = log_resp - log_resp_norm

        return xp.mean(log_prob_norm), log_resp

    def fit_spatial(self, X, positions):
        """Fit the model with spatial constraints."""
        self.fit_predict_spatial(X=X, positions=positions)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict_spatial(self, X, positions, knn=15):
        """
        Estimate model parameters using X with spatial constraints and predict labels.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        positions : array-like of shape (n_samples, 2)
            Spatial positions (e.g., x, y coordinates) corresponding to each data point.
        knn : int, default=15
            Number of nearest neighbors to consider for spatial penalty.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        # Precompute k-nearest neighbor graph
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

        # If we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -xp.inf
        best_lower_bounds = []
        self.converged_ = False
        self.spatial_quality_history_ = []

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
                    annealing_factor = min(1.0, (n_iter - 1) / (self.max_iter // 2))

                    log_prob_norm, log_resp = self._e_step(X, xp=xp, annealing_factor=annealing_factor)

                    # Debug: track and print spatial quality
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

        # Should only warn about convergence if max_iter > 0
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

        # Always do a final e-step to guarantee consistent labels
        _, log_resp = self._e_step(X, xp=xp)

        # Final spatial quality report
        if self.debug_spatial:
            print("\n" + "=" * 72)
            print("FINAL CLUSTERING RESULTS")
            self.print_spatial_quality(log_resp=log_resp, iteration="FINAL")

        return xp.argmax(log_resp, axis=1)