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
_LOG_RAMP_CF_CLIP = 0.99  # clip CF before log-ramp to avoid boundary effects at CF=1


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


def _log_ramp(score: float, tau_low: float, tau_high: float, alpha: float = 10.0) -> float:
    """Plain log-ramp: zero below tau_low, super-linear repulsion above.

    Penalty = log(1 + α·(CF−τ_low)) / log(1 + α·(1−τ_low)), in [0, 1].
    CF is clipped at _LOG_RAMP_CF_CLIP before evaluation.
    tau_high is unused — the ramp runs from tau_low to CF=1.
    """
    score = min(score, _LOG_RAMP_CF_CLIP)
    if score <= tau_low:
        return 0.0
    denom = np.log1p(alpha * (1.0 - tau_low))
    if denom <= 0.0:
        return 1.0
    return float(np.clip(np.log1p(alpha * (score - tau_low)) / denom, 0.0, 1.0))


def _signed_log_ramp(score: float, tau_low: float, tau_high: float, alpha: float = 10.0) -> float:
    """Signed log-ramp: attractive below tau_low, super-linear repulsion above.

    Attractive zone (CF < tau_low): same formula as _signed_ramp — in (-1, 0).
    Repulsive zone  (CF > tau_low): log(1+α·(CF−τ_low)) / log(1+α·(1−τ_low)), in (0, 1].
    CF is clipped at _LOG_RAMP_CF_CLIP before evaluation.
    tau_high is unused — repulsive ramp is defined over [tau_low, 1].
    """
    score = min(score, _LOG_RAMP_CF_CLIP)
    if tau_low > 0.0 and score < tau_low:
        return (score - tau_low) / tau_low  # in (-1, 0): attraction
    if score <= tau_low:
        return 0.0
    denom = np.log1p(alpha * (1.0 - tau_low))
    if denom <= 0.0:
        return 1.0
    return float(np.clip(np.log1p(alpha * (score - tau_low)) / denom, 0.0, 1.0))


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


def _connected_components(cells: np.ndarray, adj: dict[int, list[int]]) -> list[list[int]]:
    """Return connected components of `cells` via BFS over `adj`."""
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in cells:
        s = int(start)
        if s in visited:
            continue
        component: list[int] = []
        stack = [s]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(n for n in adj.get(node, []) if n not in visited)
        components.append(component)
    return components


def _split_fragmented_clusters(
    labels: np.ndarray,
    centers: np.ndarray,
    K: int,
    spatial_radius: float,
) -> tuple[np.ndarray, int, dict[int, int]]:
    """Split clusters whose cells are disconnected in the spatial proximity graph.

    Uses KDTree-based proximity (pairs within spatial_radius) for connectivity,
    so cells without polygon overlap are still considered connected when they
    are spatially close.  Only genuinely spatially separated groups of cells
    trigger a split.

    Returns (new_labels, K_split, parent_map) where parent_map maps each
    new label k >= K to its original parent in 0..K-1.
    """
    from scipy.spatial import KDTree
    tree = KDTree(centers)
    pairs = tree.query_pairs(spatial_radius)
    n = len(labels)
    full_adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in pairs:
        full_adj[i].append(j)
        full_adj[j].append(i)

    new_labels = labels.copy()
    next_label = K
    parent_map: dict[int, int] = {}
    for k in range(K):
        cells = np.where(labels == k)[0]
        if len(cells) < 2:
            continue
        cell_set = set(int(c) for c in cells)
        cluster_adj = {c: [nb for nb in full_adj[c] if nb in cell_set] for c in cell_set}
        comps = _connected_components(cells, cluster_adj)
        if len(comps) <= 1:
            continue
        for comp in comps[1:]:
            new_labels[np.array(comp)] = next_label
            parent_map[next_label] = k
            next_label += 1
    return new_labels, next_label, parent_map


def _merge_clusters_to_k(
    labels: np.ndarray,
    X: np.ndarray,
    raw_map: dict,
    tau_low: float,
    K_target: int,
) -> np.ndarray:
    """Greedily merge sub-clusters to K_target.

    Each step picks the pair of clusters with the smallest L2 feature-mean
    distance that passes the territorial-overlap veto (mean CF < tau_low).
    Labels in the result are remapped to 0..K_final-1.
    """
    labels = labels.copy()
    active = sorted(np.unique(labels).tolist())

    while len(active) > K_target:
        means = {k: X[labels == k].mean(axis=0) for k in active}

        best: tuple[int, int] | None = None
        best_dist = np.inf
        for idx, a in enumerate(active):
            for b in active[idx + 1:]:
                cells_a = np.where(labels == a)[0]
                cells_b = np.where(labels == b)[0]
                cfs = [
                    raw_map[(min(int(i), int(j)), max(int(i), int(j)))]
                    for i in cells_a
                    for j in cells_b
                    if (min(int(i), int(j)), max(int(i), int(j))) in raw_map
                ]
                if cfs and float(np.mean(cfs)) >= tau_low:
                    continue
                dist = float(np.linalg.norm(means[a] - means[b]))
                if dist < best_dist:
                    best_dist = dist
                    best = (a, b)

        if best is None:
            break

        a, b = best
        labels[labels == b] = a
        active.remove(b)

    mapping = {old: new for new, old in enumerate(active)}
    return np.array([mapping[int(lbl)] for lbl in labels], dtype=np.int64)


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
    log_ramp:
        If True, replace the linear repulsive zone with a log-ramp:
          penalty = log(1 + α·(CF−τ_low)) / log(1 + α·(1−τ_low))
        This makes the penalty super-linear in CF, applying much stronger force
        for near-complete territorial overlaps (CF → 1) than the linear ramp.
        CF is clipped at 0.99 before evaluation.  tau_high is ignored in the
        repulsive zone — the ramp runs from tau_low to CF=1.
        Combine with signed_ramp=True to keep the attractive zone below tau_low.
    log_ramp_alpha:
        Curvature of the log-ramp (default 10.0).  At α=1 the ramp is nearly
        linear; at α=10 there is a noticeable upward bend; at α=50 the ramp
        behaves like a soft step.  Only used when log_ramp=True.
    theta_hard:
        CF threshold above which same-label adjacency is treated as a
        near-certain labelling error (H1 pre-pass).  Before ICM, for every
        pair with CF > theta_hard and the same current label, the cell with the
        lower GMM max-posterior is force-unfrozen so ICM can correct it.
        None (default) disables the pre-pass.  Recommended value: 0.85.
        The number of cells unfrozen is stored as ``n_force_unfrozen`` in the
        model dict.
    split_merge:
        If True, wrap ICM with a split–merge pass (§14 pipeline).
        Before ICM: clusters whose cells are disconnected in the spatial graph
        are split into their connected components (Option B), and the unary
        cost matrix is expanded by duplicating the parent component's column.
        After ICM: sub-clusters are greedily merged back to ``n_clusters``
        using L2 feature-mean distance as the merge key and a territorial-
        overlap veto (mean CF >= tau_low blocks a merge) as the constraint.
        Useful when the GMM under-clusters (merges two spatially separate
        types) or over-clusters (splits one type across multiple components).
        ``n_splits`` and ``n_merges`` in the model dict report the number of
        each operation performed.
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
        log_ramp: bool = False,
        log_ramp_alpha: float = 10.0,
        theta_hard: float | None = None,
        split_merge: bool = False,
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
        self.log_ramp = log_ramp
        self.log_ramp_alpha = log_ramp_alpha
        self.theta_hard = theta_hard
        self.split_merge = split_merge

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

        # --- Split fragmented clusters before ICM (Option B) ---
        K_eff = K
        n_splits = 0
        n_merges = 0
        if self.split_merge:
            labels, K_eff, _parent_map = _split_fragmented_clusters(
                labels, dataset.centers, K, self.spatial_radius
            )
            n_splits = K_eff - K
            if K_eff > K:
                extra = np.column_stack(
                    [unary[:, _parent_map[k]] for k in range(K, K_eff)]
                )
                unary = np.hstack([unary, extra])

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
        # Select ramp variant; log-ramp uses a closure to bind alpha.
        if self.log_ramp:
            _alpha = self.log_ramp_alpha
            if self.signed_ramp:
                ramp_fn = lambda cf, tl, th: _signed_log_ramp(cf, tl, th, _alpha)
            else:
                ramp_fn = lambda cf, tl, th: _log_ramp(cf, tl, th, _alpha)
        else:
            ramp_fn = _signed_ramp if self.signed_ramp else _ramp
        nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]
        for (i, j), cf in raw_map.items():
            w = ramp_fn(cf, tau_low, tau_high)
            if w != 0.0:
                nbrs[i].append((j, w))
                nbrs[j].append((i, w))

        # --- H1 pre-pass: force-unfreeze cells in near-certain labelling errors ---
        # CF > theta_hard between two same-label cells is geometrically impossible
        # for genuinely same-type pairs; at least one cell is wrong.  Unfreezing
        # the less-confident cell lets ICM correct it even when it would otherwise
        # be locked by the confidence threshold.
        n_force_unfrozen = 0
        if self.theta_hard is not None:
            # Sort descending by CF so the worst violations are processed first.
            high_cf_pairs = sorted(
                ((cf, i, j) for (i, j), cf in raw_map.items()
                 if cf > self.theta_hard and labels[i] == labels[j]),
                reverse=True,
            )
            for cf, i, j in high_cf_pairs:
                if labels[i] != labels[j]:
                    continue  # already resolved by an earlier pair in this pass
                less_conf = i if posteriors[i].max() < posteriors[j].max() else j
                if frozen[less_conf]:
                    frozen[less_conf] = False
                    n_force_unfrozen += 1

        # Save GMM labels before ICM for diagnostics / frozen-cell test
        labels_initial = labels.copy()

        # --- ICM loop ---
        n_iters = 0
        n_changed_total = 0
        violations_before = sum(
            1 for (i, j) in raw_map
            if ramp_fn(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
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
                one_hot = (labels[:, None] == np.arange(K_eff)).astype(np.float64)
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
            if ramp_fn(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
        )

        # --- Merge sub-clusters back to n_clusters (Option A) ---
        if self.split_merge and K_eff > K:
            labels = _merge_clusters_to_k(labels, X, raw_map, tau_low, K)
            n_merges = K_eff - len(np.unique(labels))
            violations_after = sum(
                1 for (i, j) in raw_map
                if ramp_fn(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
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
                "log_ramp": self.log_ramp,
                "log_ramp_alpha": self.log_ramp_alpha if self.log_ramp else None,
                "theta_hard": self.theta_hard,
                "n_force_unfrozen": n_force_unfrozen,
                "split_merge": self.split_merge,
                "n_splits": n_splits,
                "n_merges": n_merges,
            },
        )
