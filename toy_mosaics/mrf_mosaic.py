"""MRF spatial label refinement: GMM + ICM post-processing.

Algorithm
---------
1. Fit a GMM to get initial labels and per-component log-likelihoods.
   Using log-likelihood (not log-posterior) as the unary cost avoids the
   zero-posterior problem: cells far from their correct cluster mean still
   get a finite cost.

2. Build a pairwise weight map from polygon coverage fractions, ramped
   through the same tau_low/tau_high dead zone used in the Leiden approach.
   After the ramp, same-type adjacent cells (C ~ 0.07) contribute zero;
   cross-type overlapping cells (C ~ 0.5+) contribute weight in [0, 1].

3. Run ICM (Iterated Conditional Modes):
      E_local(i, k) = unary[i, k] + lam * sum_j w_ij * [labels[j] == k]
   Sweep over unfrozen cells (those below a confidence threshold) until no
   cell changes label.

4. Return refined labels.

Key parameters
--------------
lam:
    Penalty strength.  Natural scale: lam ~ (log-likelihood gap between the
    correct and assigned cluster for a boundary cell).  For anisotropic toy
    data this is ~30 nats, but the pairwise conflict per cell is ~1-2,
    requiring lam ~ 20-25 to flip genuinely ambiguous cells.  Higher lam
    enforces the mosaic constraint more aggressively at the cost of
    potentially over-correcting.
tau_low / tau_high:
    Dead zone for the coverage fraction ramp.  tau_low=0.10 filters out the
    normal same-type contact level (~7-10 % for typical mosaics).
conf_threshold:
    Cells with GMM max-posterior above this are frozen — never reassigned.
    This prevents the algorithm from disturbing confident assignments.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from toy_mosaics._result import ClusteringResult
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.leiden_mosaic import _build_coverage_map


def _ramp(score: float, tau_low: float, tau_high: float) -> float:
    span = tau_high - tau_low
    if span <= 0.0:
        return float(score > tau_low)
    return float(np.clip((score - tau_low) / span, 0.0, 1.0))


class MRFMosaicStrategy:
    """GMM + ICM spatial label refinement via coverage-fraction pairwise term.

    Parameters
    ----------
    n_clusters:
        Number of cell types.
    spatial_radius:
        Neighbourhood radius for coverage fraction computation.
    lam:
        ICM penalty weight.  See module docstring for calibration guidance.
    tau_low:
        Dead-zone onset for the pairwise ramp (filters same-type contact).
    tau_high:
        Full-penalty threshold for the ramp.
    conf_threshold:
        GMM posterior confidence above which cells are frozen (not updated).
    max_iters:
        Maximum ICM sweep iterations.
    exclude_clipped:
        Skip pairs where either cell is boundary-clipped.
    random_state:
        GMM random seed.
    """

    def __init__(
        self,
        n_clusters: int,
        spatial_radius: float = 20.0,
        lam: float = 20.0,
        tau_low: float = 0.10,
        tau_high: float = 0.40,
        conf_threshold: float = 0.90,
        max_iters: int = 30,
        exclude_clipped: bool = True,
        random_state: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.spatial_radius = spatial_radius
        self.lam = lam
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.conf_threshold = conf_threshold
        self.max_iters = max_iters
        self.exclude_clipped = exclude_clipped
        self.random_state = random_state

    def fit(self, dataset: MosaicDataset) -> ClusteringResult:
        X = dataset.X
        K = self.n_clusters
        n_cells = len(dataset)

        # --- GMM: initial labels + posteriors ---
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            n_init=3,
            random_state=self.random_state,
        )
        gmm.fit(X)
        labels = gmm.predict(X).astype(np.int64)
        posteriors = gmm.predict_proba(X)  # (n_cells, K) — used for freezing

        # Unary cost = negative unnormalized log-posterior
        #   = -[log P(X_i | cluster k) + log pi_k]
        # This is equivalent to log P(cluster k | X_i) up to the per-cell
        # normalizing constant (which cancels in argmin).  Including log pi_k
        # ensures ICM at lam=0 reproduces the GMM's initial assignment exactly,
        # preventing false flips for cells where mixing-weight differences matter.
        log_liks = np.zeros((n_cells, K))
        for k in range(K):
            mvn = multivariate_normal(mean=gmm.means_[k], cov=gmm.covariances_[k])
            log_liks[:, k] = mvn.logpdf(X)
        unary = -(log_liks + np.log(gmm.weights_))  # (n_cells, K)

        # --- Spatial pairwise weights (coverage fraction + ramp) ---
        raw_map = _build_coverage_map(
            dataset.polygons, dataset.centers, self.spatial_radius,
            dataset.clipped, self.exclude_clipped,
        )

        # Per-cell adjacency: nbrs[i] = [(j, effective_weight), ...]
        nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]
        for (i, j), cf in raw_map.items():
            w = _ramp(cf, self.tau_low, self.tau_high)
            if w > 0.0:
                nbrs[i].append((j, w))
                nbrs[j].append((i, w))

        # --- Freeze high-confidence cells ---
        frozen: NDArray = posteriors.max(axis=1) >= self.conf_threshold

        # --- ICM loop ---
        n_iters = 0
        n_changed_total = 0
        violations_before = sum(
            1 for (i, j) in raw_map if _ramp(raw_map[(i, j)], self.tau_low, self.tau_high) > 0
            and labels[i] == labels[j]
        )

        for iteration in range(self.max_iters):
            n_iters = iteration + 1
            n_changed_this_iter = 0
            for i in range(n_cells):
                if frozen[i]:
                    continue
                # E_local(i, k) = unary[i,k] + lam * total_weight_of_k-labeled_nbrs
                e_local = unary[i].copy()
                for j, w in nbrs[i]:
                    e_local[labels[j]] += self.lam * w
                best_k = int(np.argmin(e_local))
                if best_k != labels[i]:
                    labels[i] = best_k
                    n_changed_this_iter += 1
            n_changed_total += n_changed_this_iter
            if n_changed_this_iter == 0:
                break

        violations_after = sum(
            1 for (i, j) in raw_map if _ramp(raw_map[(i, j)], self.tau_low, self.tau_high) > 0
            and labels[i] == labels[j]
        )

        return ClusteringResult(
            labels=labels,
            model={
                "gmm": gmm,
                "lam": self.lam,
                "n_iters": n_iters,
                "n_frozen": int(frozen.sum()),
                "n_changed": n_changed_total,
                "violations_before": violations_before,
                "violations_after": violations_after,
            },
        )
