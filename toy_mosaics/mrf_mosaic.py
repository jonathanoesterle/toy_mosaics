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

3. Calibrate tau_low and tau_high from the data if not supplied explicitly
   (see _calibrate_tau).  Uses frozen-cell pairs as a proxy for the true
   same-type / diff-type CF distributions.

4. Run ICM (Iterated Conditional Modes):
      E_local(i, k) = unary[i, k] + lam * sum_j w_ij * [labels[j] == k]
   Sweep over unfrozen cells (those below a confidence threshold) until no
   cell changes label.

5. Return refined labels.

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
    Thresholds for the coverage-fraction ramp.  Pass None (default) to
    auto-calibrate from the frozen-cell CF distribution (Option B in the
    MRF brainstorm doc).  Pass explicit floats to fix the values.

    Calibration hierarchy (each threshold independently):
      1. Frozen same/diff-label pairs (n >= 10): most reliable
      2. All same/diff-label pairs: fallback when too few frozen pairs
      3. Hard-coded prior: last resort

    With signed_ramp=False: prior tau_low=0.10 (dead zone above normal same-tile
    contact level ~7-10 %).  With signed_ramp=True: prior tau_low=0.30 (covers
    intra-tile CF range 0.20-0.26 in the attractive zone).
signed_ramp:
    When False (default): pairwise weights are in [0, 1] — pure repulsion.
    When True: weights are in [-1, 1] — attractive below tau_low, repulsive
    above.  The attractive term encodes that low-CF same-type adjacency
    indicates shared territory rather than violation.  Best used with
    tau_low=0.25-0.35 so genuine intra-tile contacts become attractors.
conf_threshold:
    Cells with GMM max-posterior above this are frozen — never reassigned.
    This prevents the algorithm from disturbing confident assignments.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from toy_mosaics._result import ClusteringResult
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.leiden_mosaic import _build_coverage_map

# Hard-coded priors used when calibration data are insufficient
_PRIOR_TAU_LOW_PLAIN = 0.10
_PRIOR_TAU_LOW_SIGNED = 0.30
_PRIOR_TAU_HIGH = 0.40
_MIN_CALIB_PAIRS = 10   # minimum pairs required for a reliable quantile estimate


def _ramp(score: float, tau_low: float, tau_high: float) -> float:
    span = tau_high - tau_low
    if span <= 0.0:
        return float(score > tau_low)
    return float(np.clip((score - tau_low) / span, 0.0, 1.0))


def _signed_ramp(score: float, tau_low: float, tau_high: float) -> float:
    """Signed pairwise weight: attractive (negative) for CF < tau_low, repulsive above.

    Below tau_low, same-type adjacency is interpreted as same-tile territorial
    contact and attracts.  Above tau_low it is a cross-tile violation and repels.
    Set tau_low high enough to cover the expected intra-tile CF range (~0.25-0.35).
    """
    if tau_low > 0.0 and score < tau_low:
        return (score - tau_low) / tau_low  # in (-1, 0): attraction
    span = tau_high - tau_low
    if span <= 0.0:
        return 1.0 if score > tau_low else 0.0
    return float(np.clip((score - tau_low) / span, 0.0, 1.0))


def _calibrate_tau(
    raw_map: dict,
    frozen: NDArray,
    labels: NDArray,
    *,
    signed_ramp: bool,
    min_pairs: int = _MIN_CALIB_PAIRS,
) -> tuple[float, float, dict]:
    """Infer tau_low and tau_high from the coverage-fraction distribution.

    Fallback hierarchy for each threshold (applied independently):
      1. Frozen same/diff-label pairs  — high-confidence proxy for true type labels
      2. All same/diff-label pairs     — more data, noisier labels
      3. Hard-coded prior              — last resort

    tau_low  = p99 of same-label CF  (covers 99% of intra-tile contacts)
    tau_high = p25 of diff-label CF  (75% of cross-type pairs at full repulsion)

    Returns (tau_low, tau_high, info_dict).
    """
    prior_low = _PRIOR_TAU_LOW_SIGNED if signed_ramp else _PRIOR_TAU_LOW_PLAIN
    prior_high = _PRIOR_TAU_HIGH
    min_gap = 0.05

    # Partition pairs by frozen status and label relationship
    same_frozen: list[float] = []
    diff_frozen: list[float] = []
    same_all:    list[float] = []
    diff_all:    list[float] = []
    for (i, j), cf in raw_map.items():
        if labels[i] == labels[j]:
            same_all.append(cf)
            if frozen[i] and frozen[j]:
                same_frozen.append(cf)
        else:
            diff_all.append(cf)
            if frozen[i] and frozen[j]:
                diff_frozen.append(cf)

    # --- tau_low ---
    if len(same_frozen) >= min_pairs:
        tau_low = float(np.quantile(same_frozen, 0.99))
        low_source = f"frozen_same (n={len(same_frozen)})"
    elif len(same_all) >= min_pairs:
        tau_low = float(np.quantile(same_all, 0.99))
        low_source = (
            f"all_same (n={len(same_all)}; "
            f"frozen_same n={len(same_frozen)} < {min_pairs})"
        )
    else:
        tau_low = prior_low
        low_source = f"prior (same-label pairs n={len(same_all)} < {min_pairs})"

    # --- tau_high ---
    if len(diff_frozen) >= min_pairs:
        tau_high = float(np.quantile(diff_frozen, 0.25))
        high_source = f"frozen_diff (n={len(diff_frozen)})"
    elif len(diff_all) >= min_pairs:
        tau_high = float(np.quantile(diff_all, 0.25))
        high_source = (
            f"all_diff (n={len(diff_all)}; "
            f"frozen_diff n={len(diff_frozen)} < {min_pairs})"
        )
    else:
        tau_high = prior_high
        high_source = f"prior (diff-label pairs n={len(diff_all)} < {min_pairs})"

    # --- Safety: tau_low must be meaningfully below tau_high ---
    if tau_low >= tau_high - min_gap:
        if tau_low >= tau_high:
            bad = f"inverted: tau_low={tau_low:.3f} > tau_high={tau_high:.3f}"
        else:
            bad = f"gap too small: tau_low={tau_low:.3f}, tau_high={tau_high:.3f} (gap={tau_high - tau_low:.3f} < {min_gap})"
        tau_low, tau_high = prior_low, prior_high
        low_source = high_source = f"prior ({bad})"

    # --- Clamp to physically sensible range ---
    tau_low  = float(np.clip(tau_low,  0.01, 0.49))
    tau_high = float(np.clip(tau_high, tau_low + min_gap, 0.99))

    return tau_low, tau_high, {
        "tau_low_source":  low_source,
        "tau_high_source": high_source,
        "n_same_frozen": len(same_frozen),
        "n_diff_frozen": len(diff_frozen),
        "n_same_all":    len(same_all),
        "n_diff_all":    len(diff_all),
    }


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
        Dead-zone onset (plain ramp) or attractive-zone boundary (signed ramp).
        Pass None (default) to auto-calibrate from the frozen-cell CF
        distribution; pass a float to fix it.
    tau_high:
        Full-penalty threshold for the ramp.  Pass None (default) to
        auto-calibrate; pass a float to fix it.
    conf_threshold:
        GMM posterior confidence above which cells are frozen (not updated).
    max_iters:
        Maximum ICM sweep iterations.
    exclude_clipped:
        Skip pairs where either cell is boundary-clipped.
    random_state:
        GMM random seed.
    signed_ramp:
        If True, use a signed pairwise weight: same-type neighbours with
        CF < tau_low attract (negative energy), those with CF > tau_low repel.
        Encodes that low-CF same-type adjacency means shared territory, not
        violation.  When enabled, set tau_low to cover the expected intra-tile
        CF range (typically 0.25-0.35 for convex-hull-based coverage fractions).
    n_workers:
        Number of parallel workers.  1 (default) = single-threaded, identical
        to prior behaviour.  -1 = all CPUs.  When != 1:
        - Coverage map is built in parallel with joblib.
        - ICM uses synchronous (Jacobi) sparse-matrix updates via scipy.sparse,
          which calls BLAS and picks up available threads automatically.
        Jacobi ICM has the same monotone convergence guarantee as sequential ICM
        but updates all cells simultaneously per sweep; the fixed point may differ
        slightly from the sequential result.
    """

    def __init__(
        self,
        n_clusters: int,
        spatial_radius: float = 20.0,
        lam: float = 20.0,
        tau_low: float | None = None,
        tau_high: float | None = None,
        conf_threshold: float = 0.90,
        max_iters: int = 30,
        exclude_clipped: bool = True,
        random_state: int = 0,
        signed_ramp: bool = False,
        n_workers: int = 1,
    ) -> None:
        self.n_clusters = n_clusters
        self.spatial_radius = spatial_radius
        self.lam = lam
        self.tau_low = tau_low
        self.signed_ramp = signed_ramp
        self.tau_high = tau_high
        self.conf_threshold = conf_threshold
        self.max_iters = max_iters
        self.exclude_clipped = exclude_clipped
        self.random_state = random_state
        self.n_workers = n_workers

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
            n_workers=self.n_workers,
        )

        # --- Freeze high-confidence cells ---
        # Computed here (before ramp) so _calibrate_tau can use frozen status.
        frozen: NDArray = posteriors.max(axis=1) >= self.conf_threshold

        # --- Calibrate tau thresholds ---
        # Use the frozen-cell CF distribution as a proxy for true same/diff-type.
        # Individual thresholds can be fixed by passing explicit floats; None
        # means calibrate from data with a graceful fallback hierarchy.
        tau_calib: dict = {}
        tau_low  = self.tau_low
        tau_high = self.tau_high
        if tau_low is None or tau_high is None:
            cal_low, cal_high, tau_calib = _calibrate_tau(
                raw_map, frozen, labels, signed_ramp=self.signed_ramp,
            )
            if tau_low is None:
                tau_low = cal_low
            if tau_high is None:
                tau_high = cal_high

        # Safety: if the final combination is invalid (e.g. user fixed tau_low and
        # calibrated tau_high ended up below it), push tau_high up to tau_low + min_gap.
        _min_gap = 0.05
        if tau_high < tau_low + _min_gap:
            tau_high = float(np.clip(tau_low + _min_gap, tau_low + _min_gap, 0.99))
            tau_calib["tau_high_adjusted"] = f"pushed to tau_low+{_min_gap} ({tau_high:.3f})"

        # --- Build per-cell adjacency from calibrated thresholds ---
        # Signed ramp: negative weights attract same-type low-CF neighbours.
        ramp_fn = _signed_ramp if self.signed_ramp else _ramp
        nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]
        for (i, j), cf in raw_map.items():
            w = ramp_fn(cf, tau_low, tau_high)
            if w != 0.0:
                nbrs[i].append((j, w))
                nbrs[j].append((i, w))

        # Save GMM labels before ICM for diagnostics / frozen-cell test
        labels_initial = labels.copy()

        # --- ICM loop ---
        n_iters = 0
        n_changed_total = 0
        violations_before = sum(
            1 for (i, j) in raw_map
            if _ramp(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
        )

        if self.n_workers != 1:
            # Vectorised Jacobi ICM: build a sparse weight matrix W, then each
            # sweep is one sparse matmul.  scipy.sparse calls BLAS internally,
            # picking up available CPU threads without explicit parallelism code.
            rows, cols, data = [], [], []
            for i, nbr_list in enumerate(nbrs):
                for j, w in nbr_list:
                    rows.append(i); cols.append(j); data.append(w)
            W = csr_matrix(
                (np.array(data), (np.array(rows), np.array(cols))),
                shape=(n_cells, n_cells),
            ) if rows else csr_matrix((n_cells, n_cells))

            for iteration in range(self.max_iters):
                n_iters = iteration + 1
                # one_hot[i, k] = 1 iff labels[i] == k
                one_hot = (labels[:, None] == np.arange(K)).astype(np.float64)
                # pairwise[i, k] = lam * sum_j W[i,j] * (labels[j] == k)
                pairwise = self.lam * (W @ one_hot)
                new_labels = np.argmin(unary + pairwise, axis=1).astype(np.int64)
                new_labels[frozen] = labels[frozen]
                n_changed_this_iter = int(np.sum(new_labels != labels))
                n_changed_total += n_changed_this_iter
                labels = new_labels
                if n_changed_this_iter == 0:
                    break
        else:
            # Sequential Gauss-Seidel ICM: each cell immediately sees label
            # changes from earlier in the same sweep (current behaviour).
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
            1 for (i, j) in raw_map
            if _ramp(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
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
                # diagnostic fields
                "labels_initial": labels_initial,
                "frozen": frozen,
                "tau_low": tau_low,
                "tau_high": tau_high,
                "tau_calibration": tau_calib,  # empty dict if tau was fixed by caller
                "raw_map": raw_map,
                "spatial_radius": self.spatial_radius,
                "exclude_clipped": self.exclude_clipped,
                "conf_threshold": self.conf_threshold,
                "signed_ramp": self.signed_ramp,
            },
        )
