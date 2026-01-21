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

        self.pair_dists = None

    def _e_step(self, X, xp=None):
        """E step.

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
