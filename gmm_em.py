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
        self._cached_cluster_stats = None
        self._iteration_counter = 0

    def _compute_cluster_statistics(self, resp_np, k, cluster_mask, weights):
        """
        Compute and cache statistics for a cluster.
        Only called during statistics update phase.
        """
        if np.sum(weights) < 1e-6:
            return None

        nn_dists = self._compute_within_cluster_nn_distances(resp_np, k)

        # Filter out distances that are "duplicate-range"
        nn_dists_low = np.percentile(nn_dists, 10)
        nn_dists_high = np.percentile(nn_dists, 90)

        valid_dist_mask = (nn_dists >= nn_dists_low) & (nn_dists <= nn_dists_high)
        if np.sum(valid_dist_mask & cluster_mask) > 5:
            filtered_dists = nn_dists[valid_dist_mask & cluster_mask]
            filtered_weights = weights[valid_dist_mask & cluster_mask]
            mean_nn = np.average(filtered_dists, weights=filtered_weights)
        else:
            mean_nn = np.median(nn_dists[cluster_mask])

        return {
            'mean_nn': mean_nn,
            'nn_dists': nn_dists  # Cache this to avoid recomputation
        }

    def evaluate_spatial_quality(self, resp=None, log_resp=None, return_details=False):
        """Evaluate spatial quality - should match what we're optimizing!"""

        if resp is None:
            resp = np.exp(log_resp)

        resp = np.asarray(resp)
        n_components = resp.shape[1]

        duplicate_threshold = np.percentile(self.knn_dists, 10)

        cluster_stats = []
        total_penalty = 0.0
        total_weight = 0.0

        for k in range(n_components):
            cluster_mask = (resp[:, k] > self.membership_threshold) | \
                           (resp[:, k] == np.max(resp, axis=1))
            weights = resp[:, k]
            cluster_size = np.sum(weights)

            if cluster_size < 1e-6:
                continue

            # Count duplicates
            n_duplicates = 0
            for i in range(len(resp)):
                if cluster_mask[i]:
                    neighbors_in_cluster = cluster_mask[self.knn_indices[i]]
                    duplicate_mask = (self.knn_dists[i] < duplicate_threshold) & neighbors_in_cluster
                    if np.any(duplicate_mask):
                        n_duplicates += 1

            # Compute coherence (matching the penalty calculation)
            nn_dists = self._compute_within_cluster_nn_distances(resp, k)

            # Filter like we do in penalty
            valid_dist_mask = nn_dists >= duplicate_threshold
            if np.sum(valid_dist_mask & cluster_mask) > 5:
                filtered_dists = nn_dists[valid_dist_mask & cluster_mask]
                filtered_weights = weights[valid_dist_mask & cluster_mask]
                mean_nn = np.average(filtered_dists, weights=filtered_weights)
            else:
                mean_nn = np.median(nn_dists[cluster_mask])

            # Coherence penalty (matching optimization)
            dev = np.abs(nn_dists - mean_nn) / (mean_nn + 1e-10)
            coherence_penalty = np.average((dev ** 2), weights=weights)

            # Total penalty for this cluster
            duplicate_penalty = 10.0 * (n_duplicates / (cluster_size + 1e-10))
            total_cluster_penalty = duplicate_penalty + coherence_penalty

            cluster_stats.append({
                'cluster': k,
                'size': cluster_size,
                'n_duplicates': n_duplicates,
                'mean_nn_dist': mean_nn,
                'coherence_penalty': coherence_penalty,
                'duplicate_penalty': duplicate_penalty,
                'total_penalty': total_cluster_penalty,
            })

            total_penalty += total_cluster_penalty * cluster_size
            total_weight += cluster_size

        # Overall quality = weighted average penalty (matches what we optimize!)
        quality_score = total_penalty / (total_weight + 1e-10)

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
        print(cluster_stats)

        print("=" * 74)

    def _build_knn_graph(self, positions, k=15):
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(positions)
        dists, indices = nbrs.kneighbors(positions)

        # Remove self-neighbor (distance 0)
        self.knn_dists = dists[:, 1:]
        self.knn_indices = indices[:, 1:]

    def _compute_within_cluster_nn_distances(self, resp, cluster_idx):
        # Determine which points belong to this cluster
        cluster_mask = (resp[:, cluster_idx] > self.membership_threshold) | \
                       (resp[:, cluster_idx] == np.max(resp, axis=1))

        # Vectorized lookup: Get the cluster membership of all neighbors at once
        # neighbors_in_cluster shape: (n_samples, k)
        neighbors_in_cluster = cluster_mask[self.knn_indices]

        # Mask distances: if a neighbor is NOT in the cluster, set distance to infinity
        masked_dists = np.where(neighbors_in_cluster, self.knn_dists, np.nan)

        # Find minimum distance to a neighbor in the same cluster
        nn_distances = np.nanmin(masked_dists, axis=1)

        # CRITICAL FIX: If no neighbors are in the cluster, use a large distance
        # (e.g., the global maximum distance in the KNN graph) instead of 0.0.
        # This penalizes spatial isolation.
        global_max = np.max(self.knn_dists)
        nn_distances[~np.isfinite(nn_distances)] = global_max

        return nn_distances

    def _compute_spatial_penalty(self, log_resp=None, resp=None, xp=None, update_stats=None):
        """
        Compute spatial penalty with two-phase approach.

        Args:
            update_stats: If None, alternates automatically based on iteration counter.
                         If True, forces statistics update.
                         If False, uses cached statistics.
        """
        if xp is None: xp = np
        if resp is None:
            resp = xp.exp(log_resp)
        else:
            resp = xp.asarray(resp)
        n_samples, n_components = resp.shape
        resp_np = np.asarray(resp)
        penalty = np.zeros((n_samples, n_components))

        # Determine whether to update statistics this iteration
        if update_stats is None:
            # Alternate: update stats on even iterations, freeze on odd
            update_stats = (self._iteration_counter % 2 == 0)

        self._iteration_counter += 1

        # Define what "too close" means for duplicate detection
        duplicate_threshold = np.percentile(self.knn_dists, 10)

        # Initialize cache if needed
        if self._cached_cluster_stats is None or update_stats:
            self._cached_cluster_stats = {}

        for k in range(n_components):
            cluster_mask = (resp_np[:, k] > self.membership_threshold) | \
                           (resp_np[:, k] == np.max(resp_np, axis=1))

            weights = resp_np[:, k]

            if np.sum(weights) < 1e-6:
                continue

            # PHASE 1: Update or retrieve statistics
            if update_stats:
                # Update phase: recompute statistics based on current assignments
                stats = self._compute_cluster_statistics(resp_np, k, cluster_mask, weights)
                if stats is not None:
                    self._cached_cluster_stats[k] = stats

            # Use cached statistics (either just computed or from previous iteration)
            if k not in self._cached_cluster_stats:
                continue

            stats = self._cached_cluster_stats[k]
            mean_nn = stats['mean_nn']

            # PHASE 2: Apply penalties using (possibly frozen) statistics
            # Recompute nn_dists if we didn't just compute them
            if update_stats:
                nn_dists = stats['nn_dists']
            else:
                nn_dists = self._compute_within_cluster_nn_distances(resp_np, k)

            # Apply normal coherence penalty
            dev_raw = np.abs(nn_dists - mean_nn)
            dev = dev_raw / (mean_nn + 1e-10)
            penalty[:, k] += (dev ** 2)

            # DUPLICATE DETECTION (always active)
            # For each sample in cluster, check for duplicates
            for i in range(n_samples):
                if not cluster_mask[i]:
                    continue

                # Find neighbors in same cluster
                neighbors_in_cluster = cluster_mask[self.knn_indices[i]]

                # Check if any are TOO close (duplicates)
                duplicate_mask = (self.knn_dists[i] < duplicate_threshold) & neighbors_in_cluster

                if np.any(duplicate_mask):
                    # Found duplicates!
                    duplicate_indices = self.knn_indices[i][duplicate_mask]

                    # Check if I'm the least confident among the duplicates
                    my_confidence = resp_np[i, k]
                    duplicate_confidences = resp_np[duplicate_indices, k]

                    if my_confidence <= np.min(duplicate_confidences):
                        penalty[i, k] += 10.  # Strong penalty to push me out
                    if my_confidence >= np.max(duplicate_confidences):
                        penalty[i, k] -= 10.  # keep me in

        return xp.asarray(penalty)

    def reset_spatial_stats(self):
        """
        Reset cached statistics and iteration counter.
        Call this when restarting the EM algorithm or reinitializing.
        """
        self._cached_cluster_stats = None
        self._iteration_counter = 0

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
            spatial_penalty = self._compute_spatial_penalty(log_resp, xp=xp)

            # Subtract penalty from log responsibilities (lower is better)
            log_resp = log_resp - self.spatial_penalty_weight * spatial_penalty * annealing_factor

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
        self.reset_spatial_stats()

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
