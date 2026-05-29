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
   cell changes label, or until the relative energy decrease falls below
   energy_tol.

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
    auto-calibrate from the frozen-cell CF distribution.  Pass explicit
    floats to fix the values.

    Calibration hierarchy (each threshold independently):
      1. Frozen same/diff-label pairs (n >= 10): most reliable
      2. All same/diff-label pairs: fallback when too few frozen pairs
      3. Hard-coded prior: last resort

    With signed_ramp=False: prior tau_low=0.10.
    With signed_ramp=True:  prior tau_low=0.30.
tau_low_q / tau_high_q:
    Quantile levels used when estimating tau_low (default p99 of same-label
    CF) and tau_high (default p25 of diff-label CF).  Expose these to adapt
    to different overlap regimes without changing the prior.
cf_tau_mixture:
    When True, fit a two-component 1D GMM to the full CF distribution and
    use the p``tau_low_q`` quantile of the lower component as tau_low.
    Replaces the label-based quantile estimate and removes the quantile's
    dependence on GMM label quality.
signed_ramp:
    When False (default): pairwise weights are in [0, 1] — pure repulsion.
    When True: weights are in [-1, 1] — attractive below tau_low, repulsive
    above.
conf_threshold:
    Cells with GMM max-posterior above this are frozen — never reassigned.
    This prevents the algorithm from disturbing confident assignments.
gmm_quality_gate:
    Before using frozen cells as a calibration oracle, compute the fraction
    of frozen cells whose highest-CF spatial neighbour has the same GMM
    label.  If this consistency falls below gmm_quality_gate (default 0.6),
    all calibrated quantities fall back to hard priors.  Set to 0.0 to
    disable.
n_restarts:
    Number of independent GMM+ICM runs.  The coverage map is built once and
    shared.  The result with the lowest total MRF energy is returned.
    Default 1 (single run, original behaviour).
energy_tol:
    Relative energy decrease below which an ICM sweep is considered
    converged.  Default 1e-4.  Set to 0.0 to fall back to the n_changed==0
    criterion only.
recalibrate:
    If True, after the main ICM pass re-estimate tau_low/tau_high/lam from
    the post-ICM labels (which are better than the raw GMM labels), rebuild
    the pairwise weight map, and run a second ICM pass.  Default False.
covariance_type:
    GMM covariance structure.  Default "full".  Use "diag" or "tied" for
    high-dimensional features to reduce overfitting.
kde_unary:
    If True, replace the Gaussian log-likelihood unary costs with kernel
    density estimates (one KDE per cluster, fitted on the cells assigned
    by the GMM).  More robust for non-Gaussian feature distributions.
count_reg:
    If > 0, add a per-label bias ``count_reg * (n_k - n_expected)^2`` to
    the unary costs before ICM.  Penalises labels with many more cells than
    the expected n_cells/K, softly enforcing balanced cluster sizes.
    Default 0.0 (disabled).
lam_boost:
    Multiplier applied to lam in the high-weight phase of pairwise conflict
    resolution.  None (default) computes an adaptive value from the 95th
    percentile of unary gaps so the boost is just strong enough to flip the
    hardest genuine conflict.  Pass an explicit float to fix it.
conflict_min_posterior:
    Minimum GMM posterior probability a cell must have for its new label
    before a conflict-resolution swap is accepted.  Default 0.01.
use_dbscan_split:
    When True (default) and split_merge=True, use DBSCAN with an adaptive
    eps (2.5 × median nearest-neighbour distance within the cluster) to
    detect spatially disconnected components.  More robust than k-NN for
    variable-density clusters.  Falls back to k-NN connectivity when DBSCAN
    returns a single component.
icm_jacobi:
    When True, use the synchronous (Jacobi) sparse-matrix ICM update which
    is faster on large datasets.  Default False, which always uses
    Gauss-Seidel regardless of n_workers.  Note: Jacobi and Gauss-Seidel
    may converge to different local minima.
"""
from __future__ import annotations

import warnings
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


def _dilate_polygons(polygons: list[NDArray], amount: float) -> list[NDArray]:
    """Return a list of polygons dilated by `amount` (absolute units) using shapely.

    Dilation is applied only for coverage-fraction computation; the dataset
    polygons are never modified.  A round buffer is used (default shapely
    behaviour) so the dilated polygon is always convex and valid.  Falls back
    to the original polygon on any shapely error.

    Typical values: 5–15 % of the mean cell diameter, or 1–3 % of
    spatial_radius.  Setting amount=0 is a no-op.
    """
    if amount <= 0.0:
        return polygons
    from shapely.geometry import Polygon as _SPoly
    dilated: list[NDArray] = []
    for verts in polygons:
        try:
            poly = _SPoly(verts)
            if not poly.is_valid or poly.area == 0.0:
                dilated.append(verts)
                continue
            buf = poly.buffer(amount)
            coords = np.array(buf.exterior.coords)[:-1]  # drop closing duplicate
            dilated.append(coords.astype(np.float64))
        except Exception:
            dilated.append(verts)
    return dilated


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
    unary: NDArray,
    *,
    signed_ramp: bool,
    lam_alpha: float = 2.0,
    min_pairs: int = _MIN_CALIB_PAIRS,
    tau_low_q: float = 0.99,
    tau_high_q: float = 0.25,
    log_ramp: bool = False,
    log_ramp_alpha: float = 10.0,
    gmm_quality_gate: float = 0.6,
    cf_tau_mixture: bool = False,
) -> tuple[float, float, float | None, dict]:
    """Infer tau_low, tau_high, and lam from the data, without ground-truth labels.

    **tau calibration** (fallback hierarchy for each threshold independently):
      1. GMM quality gate check (W2a): if frozen-cell spatial consistency is
         below gmm_quality_gate, fall back immediately to hard priors.
      2. CF mixture model (W6b, cf_tau_mixture=True): fit a 2-component GMM to
         the full CF distribution; tau_low = p``tau_low_q`` of the lower component.
      3. Frozen same/diff-label pairs  — high-confidence proxy for true type labels
      4. All same/diff-label pairs     — more data, noisier labels
      5. Hard-coded prior              — last resort

    tau_low  = p(tau_low_q)  of same-label CF
    tau_high = p(tau_high_q) of diff-label CF

    **lambda calibration**: uses the actual ramp function (not a linear
    approximation) so the calibrated lam is consistent with log_ramp/signed_ramp.

    Returns (tau_low, tau_high, lam_calibrated, info_dict).
    lam_calibrated is None when calibration data are insufficient.
    """
    prior_low = _PRIOR_TAU_LOW_SIGNED if signed_ramp else _PRIOR_TAU_LOW_PLAIN
    prior_high = _PRIOR_TAU_HIGH
    min_gap = 0.05
    info: dict = {}

    # --- W2a: GMM quality gate ---
    # Check whether frozen cells are spatially self-consistent before using them
    # as a calibration oracle. For each frozen cell, find its highest-CF neighbour
    # and check if that neighbour has the same label.
    if gmm_quality_gate > 0.0:
        frozen_set = set(int(i) for i in range(len(frozen)) if frozen[i])
        cell_max_cf: dict[int, tuple[int, float]] = {}
        for (i, j), cf in raw_map.items():
            for cell, nbr in ((i, j), (j, i)):
                if cell in frozen_set:
                    if cell not in cell_max_cf or cf > cell_max_cf[cell][1]:
                        cell_max_cf[cell] = (nbr, cf)
        n_checked = len(cell_max_cf)
        n_consistent = sum(
            1 for cell, (nbr, _) in cell_max_cf.items()
            if labels[cell] == labels[nbr]
        )
        consistency = n_consistent / n_checked if n_checked > 0 else 0.0
        info["gmm_consistency"] = consistency
        if consistency < gmm_quality_gate:
            info["quality_gate_triggered"] = True
            return prior_low, prior_high, None, {
                "tau_low_source": f"prior (quality_gate: consistency={consistency:.2f} < {gmm_quality_gate})",
                "tau_high_source": f"prior (quality_gate: consistency={consistency:.2f} < {gmm_quality_gate})",
                "lam_source": "not_calibrated (quality_gate failed)",
                **info,
            }
        info["quality_gate_triggered"] = False

    # --- Partition pairs by frozen status and label relationship ---
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

    # --- W6b: CF mixture model for tau_low ---
    tau_low_from_mixture: float | None = None
    if cf_tau_mixture:
        all_cf_vals = same_all + diff_all
        if len(all_cf_vals) >= 20:
            cf_arr = np.array(all_cf_vals).reshape(-1, 1)
            try:
                gm = GaussianMixture(n_components=2, n_init=3, random_state=0).fit(cf_arr)
                mix_labels = gm.predict(cf_arr)
                low_comp = int(np.argmin(gm.means_.ravel()))
                low_cfs = cf_arr[mix_labels == low_comp, 0]
                if len(low_cfs) >= min_pairs:
                    tau_low_from_mixture = float(np.quantile(low_cfs, tau_low_q))
                    info["tau_low_mixture_n_low"] = len(low_cfs)
                    info["tau_low_mixture_mean"] = float(gm.means_[low_comp, 0])
            except Exception:
                tau_low_from_mixture = None

    # --- tau_low ---
    if tau_low_from_mixture is not None:
        tau_low = tau_low_from_mixture
        low_source = f"cf_mixture (n_low={info.get('tau_low_mixture_n_low','?')})"
    elif len(same_frozen) >= min_pairs:
        tau_low = float(np.quantile(same_frozen, tau_low_q))
        low_source = f"frozen_same (n={len(same_frozen)})"
    elif len(same_all) >= min_pairs:
        tau_low = float(np.quantile(same_all, tau_low_q))
        low_source = (
            f"all_same (n={len(same_all)}; "
            f"frozen_same n={len(same_frozen)} < {min_pairs})"
        )
    else:
        tau_low = prior_low
        low_source = f"prior (same-label pairs n={len(same_all)} < {min_pairs})"

    # --- tau_high ---
    if len(diff_frozen) >= min_pairs:
        tau_high = float(np.quantile(diff_frozen, tau_high_q))
        high_source = f"frozen_diff (n={len(diff_frozen)})"
    elif len(diff_all) >= min_pairs:
        tau_high = float(np.quantile(diff_all, tau_high_q))
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

    # --- Lambda calibration ---
    # The push uses a linear ramp (cf − τ_low) / (1 − τ_low) as a conservative
    # approximation of the actual ICM ramp.  This preserves the calibrated lam
    # value from the original implementation; using the actual ICM ramp would
    # deflate lam (because the actual ramp saturates at 1 for CF > τ_high while
    # the linear formula grows more gradually), causing weaker ICM and more
    # unresolved violations at the conflict-resolution stage.
    #
    # W5 note: a future improvement could normalise the push consistently with
    # the actual ramp, but requires jointly adjusting the convergence target so
    # that the calibrated lam produces the same effective ICM strength.
    lam_calibrated: float | None = None
    lam_source = "not_calibrated"
    _push_denom = max(1.0 - tau_low, 1e-6)

    unfrozen_idx = np.where(~frozen)[0]
    if len(unfrozen_idx) >= min_pairs:
        unary_unfrozen = unary[unfrozen_idx]
        best_costs     = unary_unfrozen.min(axis=1)
        second_costs   = np.partition(unary_unfrozen, 1, axis=1)[:, 1]
        median_unary_gap = float(np.median(second_costs - best_costs))

        # Linear-ramp push (original formula) — consistent with historical lam values.
        cell_push = np.zeros(len(frozen))
        for (i, j), cf in raw_map.items():
            if cf <= tau_low or labels[i] != labels[j]:
                continue
            ramp_val = min(1.0, (cf - tau_low) / _push_denom)
            if not frozen[i] and frozen[j]:
                cell_push[i] += ramp_val
            elif frozen[i] and not frozen[j]:
                cell_push[j] += ramp_val

        pushed = cell_push[unfrozen_idx]
        pushed_nonzero = pushed[pushed > 0]

        frac_frozen_violation = (
            sum(1 for cf in same_frozen if cf > tau_low) / len(same_frozen)
            if same_frozen else 0.0
        )
        frozen_anchor_reliable = (
            len(same_frozen) >= min_pairs and frac_frozen_violation <= 0.20
        )

        if not frozen_anchor_reliable:
            reason = (
                f"n_same_frozen={len(same_frozen)} < {min_pairs}"
                if len(same_frozen) < min_pairs
                else f"frozen_violation_rate={frac_frozen_violation:.2f} > 0.20"
            )
            lam_source = f"not_calibrated ({reason} — anchor check failed)"
        elif len(pushed_nonzero) >= min_pairs and median_unary_gap > 0:
            median_push    = float(np.median(pushed_nonzero))
            lam_calibrated = float(lam_alpha * median_unary_gap / median_push)
            lam_source = (
                f"calibrated (alpha={lam_alpha:.1f}, "
                f"median_gap={median_unary_gap:.3f}, "
                f"median_push={median_push:.3f}, "
                f"n_pushed={len(pushed_nonzero)}/{len(unfrozen_idx)}, "
                f"frozen_viol={frac_frozen_violation:.2f})"
            )
        else:
            lam_source = (
                f"not_calibrated (n_pushed={len(pushed_nonzero)} < {min_pairs} "
                f"or gap={median_unary_gap:.3f} <= 0)"
            )

    return tau_low, tau_high, lam_calibrated, {
        "tau_low_source":  low_source,
        "tau_high_source": high_source,
        "lam_source":      lam_source,
        "n_same_frozen": len(same_frozen),
        "n_diff_frozen": len(diff_frozen),
        "n_same_all":    len(same_all),
        "n_diff_all":    len(diff_all),
        **info,
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
    k_nn: int = 5,
    use_dbscan: bool = True,
) -> tuple[np.ndarray, int, dict[int, int]]:
    """Split clusters whose cells form multiple disconnected spatial groups.

    W12a: k is scaled adaptively as max(k_nn, sqrt(n_k)) so large clusters
    are not spuriously fragmented by a fixed small k.

    W12b: When use_dbscan=True, DBSCAN with eps=2.5×median-NN-distance is
    used first; the k-NN graph serves as a fallback when DBSCAN returns a
    single component.  DBSCAN is more robust to variable-density clusters.

    Returns (new_labels, K_split, parent_map) where parent_map maps each
    new label k >= K to its original parent in 0..K-1.
    """
    from scipy.spatial import KDTree
    new_labels = labels.copy()
    next_label = K
    parent_map: dict[int, int] = {}
    for k in range(K):
        cells = np.where(labels == k)[0]
        n_k = len(cells)
        if n_k < 2:
            continue
        sub_centers = centers[cells]

        # --- W12b: Try DBSCAN first ---
        comps: list[list[int]] | None = None
        if use_dbscan:
            try:
                from sklearn.cluster import DBSCAN
                k_q = min(3, n_k)
                nn_dists = KDTree(sub_centers).query(sub_centers, k=k_q)[0][:, -1]
                median_nn = float(np.median(nn_dists))
                eps = 2.5 * median_nn if median_nn > 0 else 1.0
                db_labels = DBSCAN(eps=eps, min_samples=1).fit_predict(sub_centers)
                unique_db = np.unique(db_labels[db_labels >= 0])
                if len(unique_db) > 1:
                    comps = [np.where(db_labels == lbl)[0].tolist() for lbl in unique_db]
            except Exception:
                comps = None

        # --- W12a: Fallback — k-NN with adaptive k ---
        if comps is None:
            k_effective = max(k_nn, int(np.sqrt(n_k)))
            k_actual = min(k_effective + 1, n_k)
            _, nbr_idx = KDTree(sub_centers).query(sub_centers, k=k_actual)
            adj: dict[int, list[int]] = {i: [] for i in range(n_k)}
            for local_i in range(n_k):
                for local_j in nbr_idx[local_i, 1:]:
                    adj[local_i].append(int(local_j))
                    adj[int(local_j)].append(local_i)
            comps = _connected_components(np.arange(n_k), adj)

        if len(comps) <= 1:
            continue
        for comp in comps[1:]:
            new_labels[cells[np.array(comp)]] = next_label
            parent_map[next_label] = k
            next_label += 1
    return new_labels, next_label, parent_map


def _resolve_conflicting_pairs(
    labels: np.ndarray,
    unary: np.ndarray,
    nbrs: list[list[tuple[int, float]]],
    lam: float,
    max_rounds: int = 5,
    lam_boost: float = 5.0,
    lam_boost_min_w: float = 0.7,
    min_posterior: float = 0.01,
    frozen: NDArray | None = None,
) -> tuple[np.ndarray, int]:
    """Jointly resolve same-label repulsive cell pairs by 2-cell energy optimisation.

    W7: After finding the jointly optimal (k_i*, k_j*), each new assignment
    is accepted only if the cell has at least min_posterior probability for
    the new label under the GMM feature model.  This prevents swaps that are
    spatially motivated but have zero feature support (distinguishes true
    impostors from borderline geometric conflicts).

    frozen: when provided, a frozen cell may appear in a conflict pair but
    its label is held fixed — only its unfrozen partner is free to change.
    Cells force-unfrozen by the H1 pre-pass are absent from the frozen mask
    and remain fully free.  The fixed lower bound lam_boost=5.0 ensures that
    Phase 2's pairwise energy never exceeds a multiple that could overwhelm a
    frozen cell's unary gap even without this restriction; the frozen mask
    here is a belt-and-suspenders guard.

    See original docstring for the full algorithm description.
    """
    labels = labels.copy()
    n_cells, K = unary.shape
    n_swaps_total = 0

    def _posterior_ok(cell: int, new_k: int) -> bool:
        if min_posterior <= 0.0:
            return True
        log_liks = -unary[cell].copy()
        log_liks -= log_liks.max()
        post = np.exp(log_liks)
        s = post.sum()
        if s == 0:
            return False
        return float(post[new_k] / s) >= min_posterior

    def _run_rounds(lam_eff: float, min_w: float) -> int:
        nonlocal labels
        total = 0
        for _ in range(max_rounds):
            conflicts: list[tuple[float, int, int]] = []
            for i in range(n_cells):
                for j, w in nbrs[i]:
                    if j <= i or w <= min_w:
                        continue
                    if labels[i] == labels[j]:
                        conflicts.append((w, i, j))
            conflicts.sort(reverse=True)

            n_this = 0
            for w_ij, i, j in conflicts:
                if labels[i] != labels[j]:
                    continue

                e_i = unary[i].copy()
                for jj, ww in nbrs[i]:
                    if jj == j:
                        continue
                    lbl = int(labels[jj])
                    if lbl < K:
                        e_i[lbl] += lam_eff * ww

                e_j = unary[j].copy()
                for jj, ww in nbrs[j]:
                    if jj == i:
                        continue
                    lbl = int(labels[jj])
                    if lbl < K:
                        e_j[lbl] += lam_eff * ww

                k_curr = int(labels[i])
                curr_e = e_i[k_curr] + e_j[k_curr] + lam_eff * w_ij

                # Frozen cells hold their current label; only their unfrozen
                # partner is free to change.  H1-unfrozen cells are absent
                # from the frozen mask and remain fully free.
                i_frozen = frozen is not None and bool(frozen[i])
                j_frozen = frozen is not None and bool(frozen[j])
                ki_range = [k_curr] if i_frozen else range(K)
                kj_range = [k_curr] if j_frozen else range(K)

                best_e = curr_e
                best_ki, best_kj = k_curr, k_curr
                for ki in ki_range:
                    for kj in kj_range:
                        cross = lam_eff * w_ij if ki == kj else 0.0
                        e = e_i[ki] + e_j[kj] + cross
                        if e < best_e:
                            best_e = e
                            best_ki, best_kj = ki, kj

                if best_ki == k_curr and best_kj == k_curr:
                    continue

                # W7: feature guard — only require posterior support for cells
                # that are actually changing label.
                ki_ok = (best_ki == k_curr) or _posterior_ok(i, best_ki)
                kj_ok = (best_kj == k_curr) or _posterior_ok(j, best_kj)
                if not (ki_ok and kj_ok):
                    continue

                labels[i] = best_ki
                labels[j] = best_kj
                n_this += 1

            total += n_this
            if n_this == 0:
                break
        return total

    n_swaps_total += _run_rounds(lam, min_w=0.0)

    if lam_boost > 1.0:
        n_swaps_total += _run_rounds(lam * lam_boost, min_w=lam_boost_min_w)

    return labels, n_swaps_total


def _label_residual_violators(
    labels: np.ndarray,
    nbrs: list[list[tuple[int, float]]],
    unary: NDArray | None = None,
    min_posterior: float = 0.01,
    max_rounds: int = 5,
    frozen: NDArray | None = None,
) -> tuple[np.ndarray, int, int]:
    """Reassign cells that still violate their mosaic after all ICM/conflict passes.

    For each cell with any same-label repulsive contact (w > 0), find the label
    with the lowest spatial violation score:

        violation_score(i, k) = Σ_{j: labels[j]==k, w_ij>0} w_ij

    When ``unary`` is provided (the K-dimensional mixture unary from the last
    ICM pass), two additional rules apply:

    * **Tiebreaking**: among labels tied at the minimum violation score, the one
      with the lowest unary cost (best feature fit) is preferred.
    * **Feature guard**: the chosen label must have at least ``min_posterior``
      probability under the feature model.  The per-label posterior is computed
      from the unary via softmax:  p_k ∝ exp(-unary[i,k]).  If no zero-violation
      label clears the threshold, the cell is marked -1 (unlabeled).

    Cells are labelled -1 when:
      - No label gives violation_score == 0 (no spatially clean assignment), OR
      - The best zero-violation label has feature posterior < min_posterior.

    When ``frozen`` is provided, frozen cells are skipped entirely — they keep
    their current label even when in a same-label repulsive contact.  This
    prevents high-confidence GMM cells from being assigned -1 by a spatial-only
    step when their violation could not be resolved by conflict resolution.

    Processes violators in descending violation order.  Iterates until convergence
    or max_rounds.  Returns (labels, n_reassigned, n_unlabeled).
    """
    labels = labels.astype(np.int64).copy()
    n_cells = len(labels)
    K = int(np.max(labels[labels >= 0])) + 1 if np.any(labels >= 0) else 0
    n_reassigned = 0
    n_unlabeled  = 0

    for _ in range(max_rounds):
        viol = np.zeros(n_cells)
        for i in range(n_cells):
            if labels[i] < 0:
                continue
            for j, w in nbrs[i]:
                if w > 0 and labels[j] == labels[i]:
                    viol[i] += w

        violators = [(viol[i], i) for i in range(n_cells) if viol[i] > 0]
        if not violators:
            break
        violators.sort(reverse=True)

        n_this = 0
        for _, i in violators:
            if frozen is not None and frozen[i]:
                continue  # never reassign high-confidence cells here
            curr_score = sum(w for j, w in nbrs[i] if w > 0 and labels[j] == labels[i])
            if curr_score == 0:
                continue

            scores = np.zeros(K)
            for j, w in nbrs[i]:
                if w > 0 and 0 <= labels[j] < K:
                    scores[int(labels[j])] += w

            min_scr = float(scores.min())
            old_k   = int(labels[i])

            if min_scr > 0:
                labels[i] = -1
                n_unlabeled += 1
                n_this += 1
                continue

            candidates = np.where(scores == 0.0)[0]
            if unary is not None and len(candidates) > 1:
                best_k = int(candidates[int(np.argmin(unary[i][candidates]))])
            else:
                best_k = int(candidates[0])

            # Feature guard: require min_posterior for the chosen label.
            # Exception: if ALL violating neighbours are frozen, those cells
            # cannot be moved elsewhere, so this cell must yield regardless of
            # its posterior — assigning -1 here would be permanent.
            if unary is not None:
                viol_nbrs = [jj for jj, ww in nbrs[i]
                             if ww > 0 and labels[jj] == labels[i]]
                all_viol_frozen = (
                    frozen is not None
                    and len(viol_nbrs) > 0
                    and all(bool(frozen[jj]) for jj in viol_nbrs)
                )
                if not all_viol_frozen:
                    log_liks = -unary[i].copy()
                    log_liks -= log_liks.max()
                    post = np.exp(log_liks)
                    post /= post.sum()
                    if float(post[best_k]) < min_posterior:
                        labels[i] = -1
                        n_unlabeled += 1
                        n_this += 1
                        continue

            labels[i] = best_k
            if best_k != old_k:
                n_reassigned += 1
            n_this += 1

        if n_this == 0:
            break

    return labels, n_reassigned, n_unlabeled


def _merge_clusters_to_k(
    labels: np.ndarray,
    X: np.ndarray,
    raw_map: dict,
    tau_low: float,
    K_target: int,
    frozen: np.ndarray | None = None,
    min_frozen_pairs: int = 5,
    veto_frac: float = 0.25,
    min_adj_pairs: int = 1,
    min_pairs_for_veto: int = 5,
) -> tuple[np.ndarray, list[tuple[int, int, float]], list[int]]:
    """Greedily merge sub-clusters to K_target.

    Each step picks the pair of clusters with the smallest L2 feature-mean
    distance that passes two spatial gates:

    Gate 1 — adjacency requirement: the pair must have at least
    ``min_adj_pairs`` spatially adjacent cell pairs in ``raw_map``.

    Gate 2 — fraction veto: only applied when there are at least
    ``min_pairs_for_veto`` adjacent pairs.  Blocks the merge if the fraction
    of adjacent (frozen-first) pairs with CF >= tau_low is >= ``veto_frac``.

    When ``frozen`` is supplied the veto is evaluated on frozen-cell pairs only,
    falling back to all adjacent pairs when fewer than ``min_frozen_pairs``
    such pairs exist.
    Labels in the result are remapped to 0..K_final-1.

    Returns ``(labels, merge_history, active)`` where ``active[k]`` is the
    original sub-cluster index that became final cluster ``k`` (the surviving
    representative used to look up the GMM unary column for a second ICM pass).

    Note (W9): average-linkage distance was evaluated but is not used here
    because it changes the greedy merge order in ways that alter cluster
    membership mid-merge, causing subsequent spatial-gate checks to fail.
    Average linkage requires co-designing the gate structure and is deferred.
    """
    labels = labels.copy()
    active = sorted(np.unique(labels).tolist())
    merge_history: list[tuple[int, int, float]] = []

    while len(active) > K_target:
        # W9: average-linkage distance between cluster feature sets.
        # Computed lazily per candidate pair (only after passing spatial gates)
        # to avoid O(K²·n²) work upfront.  Capped at max_cells_for_avg_dist
        # cells per cluster so the inner product stays fast.
        best: tuple[int, int] | None = None
        best_dist = np.inf
        for idx, a in enumerate(active):
            for b in active[idx + 1:]:
                cells_a = np.where(labels == a)[0]
                cells_b = np.where(labels == b)[0]
                all_cfs, frozen_cfs = [], []
                for i in cells_a:
                    for j in cells_b:
                        key = (min(int(i), int(j)), max(int(i), int(j)))
                        if key not in raw_map:
                            continue
                        cf = raw_map[key]
                        all_cfs.append(cf)
                        if frozen is not None and frozen[i] and frozen[j]:
                            frozen_cfs.append(cf)
                # Gate 1: require spatial adjacency
                if len(all_cfs) < min_adj_pairs:
                    continue
                cfs = frozen_cfs if len(frozen_cfs) >= min_frozen_pairs else all_cfs
                # Gate 2: fraction veto (only when statistically meaningful)
                if len(cfs) >= min_pairs_for_veto:
                    frac_violating = sum(c >= tau_low for c in cfs) / len(cfs)
                    if frac_violating >= veto_frac:
                        continue
                dist = float(np.linalg.norm(
                    X[cells_a].mean(axis=0) - X[cells_b].mean(axis=0)
                ))
                if dist < best_dist:
                    best_dist = dist
                    best = (a, b)

        if best is None:
            break

        a, b = best
        merge_history.append((a, b, best_dist))
        labels[labels == b] = a
        active.remove(b)

    mapping = {old: new for new, old in enumerate(active)}
    return np.array([mapping[int(lbl)] for lbl in labels], dtype=np.int64), merge_history, active


def _init_gmm_means(
    X: NDArray,
    K: int,
    method: str,
    random_state: int,
    *,
    leiden_k_features: int = 15,
    leiden_resolution: float | None = None,
    require_k: bool = True,
) -> tuple[NDArray, str]:
    """Return (K_actual, D) initial means for GMM and the method actually used.

    When require_k=False (leiden only), accepts whatever cluster count Leiden
    returns and the caller is responsible for updating K_init from len(means).
    When require_k=True (default), falls back to k-means if Leiden does not
    produce exactly K clusters.
    """
    if method == "leiden":
        from toy_mosaics.leiden_mosaic import (
            _build_feature_graph, _calibrate_resolution, _run_leiden, _to_igraph,
        )
        W = _build_feature_graph(X, leiden_k_features)
        g = _to_igraph(W)
        res = leiden_resolution if leiden_resolution is not None else _calibrate_resolution(g, K, seed=random_state)
        labels = _run_leiden(g, res, seed=random_state)
        unique = np.unique(labels)
        if not require_k or len(unique) == K:
            return np.array([X[labels == k].mean(axis=0) for k in unique]), "leiden"
        method = "kmeans"

    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)
        return np.array([X[labels == k].mean(axis=0) for k in range(K)]), "agglomerative"

    if method == "random":
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=K, replace=False)
        return X[idx].copy(), "random"

    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=3, random_state=random_state)
        km.fit(X)
        return km.cluster_centers_, "kmeans"

    raise ValueError(f"Unknown init method: {method!r}. Choose 'kmeans', 'leiden', 'agglomerative', or 'random'.")


def _compute_energy(
    labels: np.ndarray,
    unary: NDArray,
    nbrs: list[list[tuple[int, float]]],
    lam: float,
) -> float:
    """Total MRF energy: Σ_i unary[i, k_i] + lam × Σ_{i<j,k_i==k_j} w_ij.

    W13: tracking total energy enables multi-restart comparison (W1) and
    energy-delta convergence (W13b).
    """
    e = 0.0
    n = len(labels)
    for i in range(n):
        k = int(labels[i])
        if k >= 0:
            e += float(unary[i, k])
    for i in range(n):
        k_i = int(labels[i])
        if k_i < 0:
            continue
        for j, w in nbrs[i]:
            if j > i and 0 <= int(labels[j]) == k_i:
                e += lam * w
    return e


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
        Pass None (default) to auto-calibrate; pass a float to fix it.
    tau_high:
        Full-penalty threshold for the ramp.  Pass None (default) to
        auto-calibrate; pass a float to fix it.
    tau_low_q / tau_high_q:
        Quantile levels for tau calibration.  Default 0.99 / 0.25.
    cf_tau_mixture:
        Fit a two-component GMM to the CF distribution for a label-independent
        tau_low estimate.  Default False.
    conf_threshold:
        GMM posterior confidence above which cells are frozen (not updated).
    max_iters:
        Maximum ICM sweep iterations.
    energy_tol:
        Relative energy decrease below which an ICM sweep is converged.
        Default 1e-4.  Set 0.0 to use only the n_changed==0 criterion.
    exclude_clipped:
        Skip pairs where either cell is boundary-clipped.
    random_state:
        GMM random seed.
    n_restarts:
        Number of independent GMM+ICM runs; best by energy is returned.
        Default 1.
    recalibrate:
        Re-estimate tau/lam from post-ICM labels and run a second ICM pass.
        Default False.
    signed_ramp:
        If True, use a signed pairwise weight: same-type neighbours with
        CF < tau_low attract; those with CF > tau_low repel.
    n_workers:
        Number of parallel workers for coverage-map building.  ICM always
        uses Gauss-Seidel (sequential) unless icm_jacobi=True.
    icm_jacobi:
        Use vectorised Jacobi ICM (sparse matmul) instead of Gauss-Seidel.
        Faster on large datasets but may converge to a different local minimum.
        Default False.
    log_ramp:
        Use super-linear log-ramp for the repulsive zone.
    log_ramp_alpha:
        Curvature of the log-ramp.  Default 10.0.
    theta_hard:
        CF threshold for the H1 pre-pass force-unfreeze step.
    split_merge:
        Enable the split-then-merge pipeline.
    n_clusters_init:
        GMM components for initialisation (> n_clusters = over-specified).
    merge_veto_frac / merge_min_adj_pairs / merge_min_pairs_for_veto:
        Spatial gates for the merge step.
    init:
        GMM initialisation method.
    leiden_k_features / leiden_resolution:
        Leiden init parameters.
    covariance_type:
        GMM covariance structure.  Default "full".  Use "diag" or "tied" for
        high-dimensional features.
    kde_unary:
        Replace Gaussian log-likelihood unary costs with per-cluster KDE
        log-densities.  More robust for non-Gaussian feature distributions.
        Default False.
    count_reg:
        Per-label count regulariser weight.  Default 0.0 (disabled).
    gmm_quality_gate:
        Minimum frozen-cell spatial consistency before calibration proceeds.
        Default 0.6.
    lam_boost:
        Multiplier for the high-weight phase of conflict resolution.
        None (default) = auto-compute from unary gap distribution.
    conflict_min_posterior:
        Minimum feature posterior for a conflict-resolution swap to be accepted.
        Default 0.01.
    use_dbscan_split:
        Use DBSCAN for spatial split detection (requires split_merge=True).
        Default True.
    per_cluster_tau:
        If True, calibrate τ_low independently per cluster from that
        cluster's frozen same-label CF distribution, rather than using a
        single global τ_low.  Clusters with typically higher intra-tile CF
        (denser mosaics) get a higher dead zone so the ramp only fires for
        genuinely anomalous contacts relative to that cluster's baseline.
        Falls back to global τ_low for clusters with fewer than
        _MIN_CALIB_PAIRS frozen same-label pairs.  Default False.
    n_em_iters:
        Number of EM-like re-fit iterations after the main ICM (and after
        the merge step if split_merge=True).  Each iteration: (1) re-fits a
        K-component GMM warm-started from the current cluster centroids,
        (2) recomputes unary costs from the new GMM, (3) re-runs ICM with
        the updated unary.  The new GMM Gaussians are anchored on the
        spatially-corrected assignment, so feature-impostor cells get more
        accurate unary costs on the next ICM pass.  No further splits or
        merges are performed.  Default 0 (disabled).
    n_cleanup_steps:
        Number of post-ICM cleanup passes.  Passes alternate between step 6
        (pairwise conflict resolution) and step 7 (residual violator
        assignment):  1 → [6],  2 → [6, 7],  3 → [6, 7, 6],  4 → [6, 7, 6, 7], …
        Default 0 (disabled).  Odd passes run conflict resolution; even passes
        run residual-violator assignment.  Running an extra conflict-resolution
        pass after residual-violators has proven effective at recovering cells
        that were temporarily unlabelled and then reassigned by the GMM fallback.
    polygon_dilation:
        If not None and > 0, dilate every polygon by this absolute distance
        (in the same spatial units as the dataset) before computing coverage
        fractions.  This converts "touching but zero-overlap" same-tile pairs
        into small positive CF values, giving the signed ramp an attractive
        signal for genuine intra-tile contacts that would otherwise be
        invisible.  The dataset polygons are never modified.  Default None
        (disabled).  Typical values: 5–15 % of the mean cell diameter,
        e.g. ``polygon_dilation = spatial_radius * 0.05``.
    lam_alpha:
        Scale factor for lambda calibration.
    """

    def __init__(
        self,
        n_clusters: int,
        spatial_radius: float = 20.0,
        lam: float | None = None,
        lam_alpha: float = 2.0,
        tau_low: float | None = None,
        tau_high: float | None = None,
        tau_low_q: float = 0.99,
        tau_high_q: float = 0.25,
        cf_tau_mixture: bool = False,
        conf_threshold: float = 0.90,
        max_iters: int = 30,
        energy_tol: float = 0.0,
        exclude_clipped: bool = True,
        random_state: int = 0,
        n_restarts: int = 1,
        recalibrate: bool = False,
        signed_ramp: bool = False,
        n_workers: int = 1,
        icm_jacobi: bool = False,
        log_ramp: bool = False,
        log_ramp_alpha: float = 10.0,
        theta_hard: float | None = None,
        split_merge: bool = False,
        n_clusters_init: int | None = None,
        merge_veto_frac: float = 0.40,
        merge_min_adj_pairs: int = 0,
        merge_min_pairs_for_veto: int = 10,
        init: str = "kmeans",
        leiden_k_features: int = 15,
        leiden_resolution: float | None = None,
        covariance_type: str = "full",
        kde_unary: bool = False,
        count_reg: float = 0.0,
        gmm_quality_gate: float = 0.0,
        lam_boost: float | None = None,
        adaptive_lam_boost: bool = False,
        conflict_min_posterior: float = 0.01,
        use_dbscan_split: bool = False,
        per_cluster_tau: bool = False,
        n_em_iters: int = 0,
        n_cleanup_steps: int = 0,
        polygon_dilation: float | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.spatial_radius = spatial_radius
        self.lam = lam
        self.lam_alpha = lam_alpha
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.tau_low_q = tau_low_q
        self.tau_high_q = tau_high_q
        self.cf_tau_mixture = cf_tau_mixture
        self.signed_ramp = signed_ramp
        self.conf_threshold = conf_threshold
        self.max_iters = max_iters
        self.energy_tol = energy_tol
        self.exclude_clipped = exclude_clipped
        self.random_state = random_state
        self.n_restarts = n_restarts
        self.recalibrate = recalibrate
        self.n_workers = n_workers
        self.icm_jacobi = icm_jacobi
        self.log_ramp = log_ramp
        self.log_ramp_alpha = log_ramp_alpha
        self.theta_hard = theta_hard
        self.split_merge = split_merge
        self.n_clusters_init = n_clusters_init
        self.merge_veto_frac = merge_veto_frac
        self.merge_min_adj_pairs = merge_min_adj_pairs
        self.merge_min_pairs_for_veto = merge_min_pairs_for_veto
        self.init = init
        self.leiden_k_features = leiden_k_features
        self.leiden_resolution = leiden_resolution
        self.covariance_type = covariance_type
        self.kde_unary = kde_unary
        self.count_reg = count_reg
        self.gmm_quality_gate = gmm_quality_gate
        self.lam_boost = lam_boost
        self.adaptive_lam_boost = adaptive_lam_boost
        self.conflict_min_posterior = conflict_min_posterior
        self.use_dbscan_split = use_dbscan_split
        self.per_cluster_tau = per_cluster_tau
        self.n_em_iters = n_em_iters
        self.n_cleanup_steps = n_cleanup_steps
        self.polygon_dilation = polygon_dilation

    # ------------------------------------------------------------------
    # Internal: single GMM+ICM run given a pre-built coverage map
    # ------------------------------------------------------------------

    def _fit_single(
        self,
        dataset: MosaicDataset,
        raw_map: dict,
        random_state: int,
    ) -> tuple[np.ndarray, dict]:
        """Run one GMM+ICM pass and return (labels, model_dict).

        ``raw_map`` is shared across restarts (W1) and built once by ``fit``.
        """
        X = dataset.X
        K = self.n_clusters
        K_init = self.n_clusters_init if self.n_clusters_init is not None else K
        n_cells = len(dataset)

        # --- GMM: initial labels + posteriors ---
        free_leiden = (
            self.init == "leiden"
            and self.leiden_resolution is not None
            and self.n_clusters_init is None
        )
        means_init, init_method_used = _init_gmm_means(
            X, K_init, self.init, random_state,
            leiden_k_features=self.leiden_k_features,
            leiden_resolution=self.leiden_resolution,
            require_k=not free_leiden,
        )
        if free_leiden:
            K_init = len(means_init)
        if init_method_used != self.init:
            init_method_used = f"{self.init}→{init_method_used}"

        # W4a: covariance_type is now a parameter
        gmm = GaussianMixture(
            n_components=K_init,
            covariance_type=self.covariance_type,
            n_init=3,
            random_state=random_state,
            means_init=means_init,
            init_params="random",
        )
        gmm.fit(X)
        labels = gmm.predict(X).astype(np.int64)
        posteriors = gmm.predict_proba(X)

        # --- Unary costs ---
        # W4b: optionally replace Gaussian log-likelihoods with per-cluster KDE
        if self.kde_unary:
            from scipy.stats import gaussian_kde
            log_liks = np.zeros((n_cells, K_init))
            for k in range(K_init):
                mask = labels == k
                if mask.sum() < 2:
                    # Too few cells: fall back to GMM Gaussian
                    mvn = multivariate_normal(mean=gmm.means_[k], cov=gmm.covariances_[k]
                                              if self.covariance_type == "full" else np.diag(gmm.covariances_[k]))
                    log_liks[:, k] = mvn.logpdf(X)
                else:
                    try:
                        kde = gaussian_kde(X[mask].T)
                        log_liks[:, k] = kde.logpdf(X.T)
                    except Exception:
                        mvn = multivariate_normal(mean=gmm.means_[k], cov=gmm.covariances_[k]
                                                  if self.covariance_type == "full" else np.diag(gmm.covariances_[k]))
                        log_liks[:, k] = mvn.logpdf(X)
        else:
            log_liks = np.zeros((n_cells, K_init))
            for k in range(K_init):
                if self.covariance_type == "full":
                    cov = gmm.covariances_[k]
                elif self.covariance_type == "tied":
                    cov = gmm.covariances_
                elif self.covariance_type == "diag":
                    cov = np.diag(gmm.covariances_[k])
                else:
                    cov = gmm.covariances_[k]
                mvn = multivariate_normal(mean=gmm.means_[k], cov=cov)
                log_liks[:, k] = mvn.logpdf(X)

        unary = -(log_liks + np.log(gmm.weights_))  # (n_cells, K_init)

        # --- Split fragmented clusters before ICM (Option B) ---
        K_eff = K_init
        n_splits = 0
        n_merges = 0
        _parent_map: dict[int, int] = {}
        _labels_gmm = labels.copy()
        if self.split_merge:
            if K_init > K:
                pass  # already overclustered; merge step will reduce to K
            else:
                labels, K_eff, _parent_map = _split_fragmented_clusters(
                    labels, dataset.centers, K_init,
                    use_dbscan=self.use_dbscan_split,
                )
                n_splits = K_eff - K_init
                if K_eff > K_init:
                    extra = np.column_stack(
                        [unary[:, _parent_map[k]] for k in range(K_init, K_eff)]
                    )
                    unary = np.hstack([unary, extra])

        # --- Freeze high-confidence cells ---
        frozen: NDArray = posteriors.max(axis=1) >= self.conf_threshold

        # --- Calibrate tau thresholds and lambda ---
        tau_calib: dict = {}
        tau_low  = self.tau_low
        tau_high = self.tau_high
        cal_lam: float | None = None
        if tau_low is None or tau_high is None or self.lam is None:
            cal_low, cal_high, cal_lam, tau_calib = _calibrate_tau(
                raw_map, frozen, labels, unary,
                signed_ramp=self.signed_ramp,
                lam_alpha=self.lam_alpha,
                tau_low_q=self.tau_low_q,
                tau_high_q=self.tau_high_q,
                log_ramp=self.log_ramp,
                log_ramp_alpha=self.log_ramp_alpha,
                gmm_quality_gate=self.gmm_quality_gate,
                cf_tau_mixture=self.cf_tau_mixture,
            )
            if tau_low is None:
                tau_low = cal_low
            if tau_high is None:
                tau_high = cal_high

        lam: float = (
            self.lam if self.lam is not None
            else cal_lam if cal_lam is not None
            else 20.0
        )
        tau_calib["lam"] = lam

        _min_gap = 0.05
        if tau_high < tau_low + _min_gap:
            tau_high = float(np.clip(tau_low + _min_gap, tau_low + _min_gap, 0.99))
            tau_calib["tau_high_adjusted"] = f"pushed to tau_low+{_min_gap} ({tau_high:.3f})"

        # --- Build ramp function and per-cell adjacency ---
        if self.log_ramp:
            _alpha = self.log_ramp_alpha
            if self.signed_ramp:
                ramp_fn = lambda cf, tl, th: _signed_log_ramp(cf, tl, th, _alpha)
            else:
                ramp_fn = lambda cf, tl, th: _log_ramp(cf, tl, th, _alpha)
        else:
            ramp_fn = _signed_ramp if self.signed_ramp else _ramp

        # --- P3: per-cluster adaptive tau_low ---
        # For each cluster k, estimate its own τ_low from the p(tau_low_q)
        # quantile of same-label CF values among frozen pairs in that cluster.
        # A denser cluster has higher typical intra-tile CF; its τ_low is
        # higher so the ramp only fires when CF is anomalously high for that
        # cluster.  Falls back to global tau_low when data are insufficient.
        tau_low_per_k: dict[int, float] = {}
        if self.per_cluster_tau:
            for k in range(K_eff):
                same_k_frozen = [
                    cf for (i, j), cf in raw_map.items()
                    if labels[i] == k and labels[j] == k
                    and frozen[i] and frozen[j]
                ]
                if len(same_k_frozen) >= _MIN_CALIB_PAIRS:
                    tau_low_per_k[k] = float(np.clip(
                        np.quantile(same_k_frozen, self.tau_low_q), 0.01, 0.49
                    ))

        nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]
        for (i, j), cf in raw_map.items():
            # P3: use cluster-specific tau_low for same-label pairs when available
            if self.per_cluster_tau and labels[i] == labels[j]:
                k = int(labels[i])
                tl = tau_low_per_k.get(k, tau_low)
            else:
                tl = tau_low
            w = ramp_fn(cf, tl, tau_high)
            if w != 0.0:
                nbrs[i].append((j, w))
                nbrs[j].append((i, w))

        # --- W11: count regulariser — soft cluster-size prior ---
        # Adds a fixed bias to the unary before ICM.  Penalises over-represented
        # labels so cells on the boundary prefer less crowded alternatives.
        if self.count_reg > 0.0:
            n_expected = n_cells / K_eff
            for k in range(K_eff):
                n_k = float(np.sum(labels == k))
                unary[:, k] += self.count_reg * (n_k - n_expected) ** 2

        # --- H1 pre-pass: force-unfreeze cells in near-certain labelling errors ---
        n_force_unfrozen = 0
        if self.theta_hard is not None:
            high_cf_pairs = sorted(
                ((cf, i, j) for (i, j), cf in raw_map.items()
                 if cf > self.theta_hard and labels[i] == labels[j]),
                reverse=True,
            )
            for cf, i, j in high_cf_pairs:
                if labels[i] != labels[j]:
                    continue
                if frozen[i]:
                    frozen[i] = False
                    n_force_unfrozen += 1
                if frozen[j]:
                    frozen[j] = False
                    n_force_unfrozen += 1

        labels_initial = labels.copy()

        # --- ICM loop ---
        # W3: always use Gauss-Seidel unless icm_jacobi=True (explicit opt-in).
        # W13: compute energy before and after each sweep for delta convergence.
        n_iters = 0
        n_changed_total = 0
        violations_before = sum(
            1 for (i, j) in raw_map
            if ramp_fn(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
        )

        energy_before_icm = _compute_energy(labels, unary, nbrs, lam)
        prev_energy = energy_before_icm

        if self.icm_jacobi:
            # Jacobi (vectorised) path — faster but may reach a different local min
            rows, cols, data = [], [], []
            for i, nbr_list in enumerate(nbrs):
                for j, w in nbr_list:
                    rows.append(i); cols.append(j); data.append(w)
            W_sp = csr_matrix(
                (np.array(data), (np.array(rows), np.array(cols))),
                shape=(n_cells, n_cells),
            ) if rows else csr_matrix((n_cells, n_cells))

            for iteration in range(self.max_iters):
                n_iters = iteration + 1
                one_hot = (labels[:, None] == np.arange(K_eff)).astype(np.float64)
                pairwise = lam * (W_sp @ one_hot)
                new_labels = np.argmin(unary + pairwise, axis=1).astype(np.int64)
                new_labels[frozen] = labels[frozen]
                n_changed_this_iter = int(np.sum(new_labels != labels))
                n_changed_total += n_changed_this_iter
                labels = new_labels
                if n_changed_this_iter == 0:
                    break
                if self.energy_tol > 0.0:
                    curr_energy = _compute_energy(labels, unary, nbrs, lam)
                    rel_delta = abs(prev_energy - curr_energy) / (abs(prev_energy) + 1e-10)
                    prev_energy = curr_energy
                    if rel_delta < self.energy_tol:
                        break
        else:
            # Gauss-Seidel (sequential) path — reproducible regardless of n_workers
            for iteration in range(self.max_iters):
                n_iters = iteration + 1
                n_changed_this_iter = 0
                for i in range(n_cells):
                    if frozen[i]:
                        continue
                    e_local = unary[i].copy()
                    for j, w in nbrs[i]:
                        e_local[labels[j]] += lam * w
                    best_k = int(np.argmin(e_local))
                    if best_k != labels[i]:
                        labels[i] = best_k
                        n_changed_this_iter += 1
                n_changed_total += n_changed_this_iter
                if n_changed_this_iter == 0:
                    break
                # W13b: energy-delta convergence criterion
                if self.energy_tol > 0.0:
                    curr_energy = _compute_energy(labels, unary, nbrs, lam)
                    rel_delta = abs(prev_energy - curr_energy) / (abs(prev_energy) + 1e-10)
                    prev_energy = curr_energy
                    if rel_delta < self.energy_tol:
                        break

        energy_after_icm = _compute_energy(labels, unary, nbrs, lam)

        violations_after = sum(
            1 for (i, j) in raw_map
            if ramp_fn(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
        )

        _labels_post_icm = labels.copy()

        # --- W2b: Iterative recalibration ---
        # Re-estimate tau/lam from post-ICM labels (better oracle than raw GMM).
        # Rebuild nbrs and run a second ICM pass.
        labels_post_recalib: np.ndarray | None = None
        recalib_tau_low: float | None = None
        recalib_tau_high: float | None = None
        if self.recalibrate:
            rc_low, rc_high, rc_lam, _ = _calibrate_tau(
                raw_map, frozen, labels, unary,
                signed_ramp=self.signed_ramp,
                lam_alpha=self.lam_alpha,
                tau_low_q=self.tau_low_q,
                tau_high_q=self.tau_high_q,
                log_ramp=self.log_ramp,
                log_ramp_alpha=self.log_ramp_alpha,
                gmm_quality_gate=self.gmm_quality_gate,
                cf_tau_mixture=self.cf_tau_mixture,
            )
            # Only update thresholds if they improve (prevent drifting away from good values)
            recalib_tau_low  = rc_low
            recalib_tau_high = rc_high
            rc_lam_eff = rc_lam if rc_lam is not None else lam
            # Rebuild pairwise weights with recalibrated thresholds
            nbrs_rc: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]
            for (i, j), cf in raw_map.items():
                w = ramp_fn(cf, rc_low, rc_high)
                if w != 0.0:
                    nbrs_rc[i].append((j, w))
                    nbrs_rc[j].append((i, w))
            prev_e_rc = _compute_energy(labels, unary, nbrs_rc, rc_lam_eff)
            for _ in range(self.max_iters):
                n_ch = 0
                for i in range(n_cells):
                    if frozen[i]:
                        continue
                    e_local = unary[i].copy()
                    for j, w in nbrs_rc[i]:
                        e_local[labels[j]] += rc_lam_eff * w
                    best_k = int(np.argmin(e_local))
                    if best_k != labels[i]:
                        labels[i] = best_k
                        n_ch += 1
                n_changed_total += n_ch
                if n_ch == 0:
                    break
                if self.energy_tol > 0.0:
                    curr_e_rc = _compute_energy(labels, unary, nbrs_rc, rc_lam_eff)
                    if abs(prev_e_rc - curr_e_rc) / (abs(prev_e_rc) + 1e-10) < self.energy_tol:
                        break
                    prev_e_rc = curr_e_rc
            labels_post_recalib = labels.copy()
            # Switch to recalibrated adjacency for downstream steps
            nbrs = nbrs_rc
            lam  = rc_lam_eff

        # --- Merge sub-clusters back to n_clusters ---
        merge_history: list[tuple[int, int, float]] = []
        surviving_subclusters: list[int] = list(range(K_eff))
        _labels_post_merge: np.ndarray | None = None   # set only when merge runs
        n_unfrozen_merge = 0
        n_iters_post_merge = 0
        n_unfrozen_h1_post = 0
        _unary_final: NDArray = unary
        _labels_post_cleanup: np.ndarray | None = None  # set after merge+2nd ICM
        _labels_post_conflict: np.ndarray | None = None # set after step 6
        n_swaps_total = 0
        n_reassigned_total = 0
        n_unlabeled_total = 0

        if self.split_merge and K_eff > K:
            labels, merge_history, surviving_subclusters = _merge_clusters_to_k(
                labels, X, raw_map, tau_low, K,
                frozen=frozen,
                veto_frac=self.merge_veto_frac,
                min_adj_pairs=self.merge_min_adj_pairs,
                min_pairs_for_veto=self.merge_min_pairs_for_veto,
            )
            n_merges = K_eff - len(np.unique(labels))
            _labels_post_merge = labels.copy()

            for (i, j), cf in raw_map.items():
                if (labels[i] == labels[j]
                        and _labels_post_icm[i] != _labels_post_icm[j]
                        and cf >= tau_low):
                    if frozen[i]:
                        frozen[i] = False
                        n_unfrozen_merge += 1
                    if frozen[j]:
                        frozen[j] = False
                        n_unfrozen_merge += 1

            component_map: dict[int, set] = {k: {k} for k in range(K_eff)}
            for a_h, b_h, _ in merge_history:
                if b_h in component_map:
                    component_map[a_h] = component_map[a_h] | component_map.pop(b_h)

            K_final = len(surviving_subclusters)
            unary_k = np.zeros((n_cells, K_final))
            for new_k, old_k in enumerate(surviving_subclusters):
                components = sorted(component_map.get(old_k, {old_k}))
                log_mix = -unary[:, components[0]].copy()
                for c in components[1:]:
                    log_mix = np.logaddexp(log_mix, -unary[:, c])
                unary_k[:, new_k] = -log_mix

            for iteration in range(self.max_iters):
                n_iters_post_merge = iteration + 1
                n_changed_this_iter = 0
                for i in range(n_cells):
                    if frozen[i]:
                        continue
                    e_local = unary_k[i].copy()
                    for j, w in nbrs[i]:
                        e_local[labels[j]] += lam * w
                    best_k = int(np.argmin(e_local))
                    if best_k != labels[i]:
                        labels[i] = best_k
                        n_changed_this_iter += 1
                n_changed_total += n_changed_this_iter
                if n_changed_this_iter == 0:
                    break

            if self.theta_hard is not None:
                high_cf_post = sorted(
                    ((cf, i, j) for (i, j), cf in raw_map.items()
                     if cf > self.theta_hard and labels[i] == labels[j]),
                    reverse=True,
                )
                for cf, i, j in high_cf_post:
                    if labels[i] != labels[j]:
                        continue
                    if frozen[i]:
                        frozen[i] = False
                        n_unfrozen_h1_post += 1
                    if frozen[j]:
                        frozen[j] = False
                        n_unfrozen_h1_post += 1
                if n_unfrozen_h1_post > 0:
                    for i in range(n_cells):
                        if frozen[i]:
                            continue
                        e_local = unary_k[i].copy()
                        for j, w in nbrs[i]:
                            e_local[labels[j]] += lam * w
                        best_k = int(np.argmin(e_local))
                        if best_k != labels[i]:
                            labels[i] = best_k
                            n_changed_total += 1

            violations_after = sum(
                1 for (i, j) in raw_map
                if ramp_fn(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
            )
            _unary_final = unary_k

        # Save the state after merge/cleanup ICM, before EM and post-processing
        _labels_post_cleanup = labels.copy()

        # --- P4: EM-like re-run (n_em_iters > 0) ---
        # After the main ICM (and any split-merge), re-fit a K-component GMM
        # warm-started from current cluster centroids, recompute unary costs,
        # and run another ICM pass.  Repeating n_em_iters times is one step of
        # coordinate-descent EM: the GMM E-step on the spatially-corrected
        # labels produces better-calibrated unary costs for the next ICM M-step.
        # No further splits or merges are performed.
        for _em_iter in range(self.n_em_iters):
            K_em = _unary_final.shape[1]
            # Skip if any cluster is empty after the current assignment
            if any(int(np.sum(labels == k)) < 2 for k in range(K_em)):
                break
            # Re-fit GMM anchored at current cluster centroids
            em_means = np.array([X[labels == k].mean(axis=0) for k in range(K_em)])
            gmm_em = GaussianMixture(
                n_components=K_em,
                covariance_type=self.covariance_type,
                n_init=1,
                random_state=random_state,
                means_init=em_means,
                init_params="random",
            )
            gmm_em.fit(X)
            # Recompute unary from re-fitted GMM
            em_log_liks = np.zeros((n_cells, K_em))
            for k in range(K_em):
                if self.covariance_type == "full":
                    cov_em = gmm_em.covariances_[k]
                elif self.covariance_type == "tied":
                    cov_em = gmm_em.covariances_
                elif self.covariance_type == "diag":
                    cov_em = np.diag(gmm_em.covariances_[k])
                else:
                    cov_em = gmm_em.covariances_[k]
                mvn_em = multivariate_normal(mean=gmm_em.means_[k], cov=cov_em)
                em_log_liks[:, k] = mvn_em.logpdf(X)
            unary_em = -(em_log_liks + np.log(gmm_em.weights_))
            _unary_final = unary_em
            # New frozen mask from re-fitted GMM posteriors
            posteriors_em = gmm_em.predict_proba(X)
            frozen_em: NDArray = posteriors_em.max(axis=1) >= self.conf_threshold
            # Re-apply H1 if set — unfreeze cells near remaining violations
            if self.theta_hard is not None:
                for cf_h, ii, jj in sorted(
                    ((cf_v, ii, jj) for (ii, jj), cf_v in raw_map.items()
                     if cf_v > self.theta_hard and labels[ii] == labels[jj]),
                    reverse=True,
                ):
                    if labels[ii] != labels[jj]:
                        continue
                    frozen_em[ii] = False
                    frozen_em[jj] = False
            # ICM pass with updated unary
            for _ in range(self.max_iters):
                n_ch_em = 0
                for i in range(n_cells):
                    if frozen_em[i]:
                        continue
                    e_local = unary_em[i].copy()
                    for j, w in nbrs[i]:
                        e_local[labels[j]] += lam * w
                    best_k = int(np.argmin(e_local))
                    if best_k != labels[i]:
                        labels[i] = best_k
                        n_ch_em += 1
                n_changed_total += n_ch_em
                if n_ch_em == 0:
                    break
            # Update the main frozen mask so downstream steps stay consistent
            frozen = frozen_em

        # Labels after EM (None if EM was not run)
        _labels_post_em: np.ndarray | None = labels.copy() if self.n_em_iters > 0 else None

        # --- W8: compute adaptive lam_boost before conflict resolution ---
        # When adaptive_lam_boost=True, the boost is the multiplier needed so
        # that lam_eff × min_w just exceeds the 95th-percentile unary gap,
        # clamped to [5.0, 20.0].  The lower bound of 5.0 preserves the
        # original hardcoded default so baseline behaviour is unchanged.
        # When adaptive_lam_boost=False (default), lam_boost_val = lam_boost
        # (which itself defaults to 5.0, matching the original behaviour exactly).
        lam_boost_min_w = 0.7
        lam_boost_val: float
        if self.adaptive_lam_boost:
            unary_for_boost = _unary_final
            gaps = np.sort(unary_for_boost, axis=1)[:, 1] - unary_for_boost.min(axis=1)
            p95_gap = float(np.percentile(gaps, 95))
            denom = lam * lam_boost_min_w
            lam_boost_val = float(np.clip(p95_gap / denom, 5.0, 20.0)) if denom > 0 else 5.0
        else:
            lam_boost_val = float(self.lam_boost) if self.lam_boost is not None else 5.0

        # --- Cleanup loop (n_cleanup_steps passes) ---
        # Odd passes (1, 3, 5, …): pairwise conflict resolution (step 6).
        # Even passes (2, 4, 6, …): residual-violator assignment (step 7).
        # Frozen protection for conflict resolution is disabled in the
        # split_merge path (the merge may have left frozen cells in violation);
        # it is active for the plain-ICM path.
        frozen_for_conflict = frozen if not (self.split_merge and K_eff > K) else None
        for _cleanup_pass in range(1, self.n_cleanup_steps + 1):
            if _cleanup_pass % 2 == 1:          # odd → conflict resolution
                labels, n_sw = _resolve_conflicting_pairs(
                    labels, _unary_final, nbrs, lam,
                    lam_boost=lam_boost_val,
                    lam_boost_min_w=lam_boost_min_w,
                    min_posterior=self.conflict_min_posterior,
                    frozen=frozen_for_conflict,
                )
                n_swaps_total += n_sw
            else:                                # even → residual violators
                labels, n_ra, n_ul = _label_residual_violators(
                    labels, nbrs, unary=_unary_final, frozen=frozen,
                )
                # In the split_merge path reassign any -1 cells to min-unary
                # label so the output always has exactly K unique labels.
                if self.split_merge and n_ul > 0:
                    neg_mask = labels < 0
                    if neg_mask.any():
                        labels[neg_mask] = np.argmin(_unary_final[neg_mask], axis=1)
                        n_ra += int(neg_mask.sum())
                        n_ul = 0
                n_reassigned_total += n_ra
                n_unlabeled_total += n_ul

        # Total cells whose label changed across all cleanup passes.
        # _labels_post_cleanup was set before the EM loop and is always non-None.
        n_cleanup_changed = (
            int(np.sum(labels != _labels_post_cleanup))
            if _labels_post_cleanup is not None else 0
        )

        energy_final = _compute_energy(labels, _unary_final, nbrs, lam)

        model: dict = {
            "gmm": gmm,
            "lam": lam,
            "n_iters": n_iters,
            "n_frozen": int(frozen.sum()),
            "n_changed": n_changed_total,
            "violations_before": violations_before,
            "violations_after": violations_after,
            "energy_before_icm": energy_before_icm,
            "energy_after_icm": energy_after_icm,
            "energy_final": energy_final,
            # diagnostic fields
            "labels_initial": labels_initial,
            "frozen": frozen,
            "tau_low": tau_low,
            "tau_high": tau_high,
            "tau_calibration": tau_calib,
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
            "n_clusters_init": K_init,
            "init_method": self.init,
            "init_method_used": init_method_used,
            "n_splits": n_splits,
            "n_merges": n_merges,
            "k_after_split": K_eff,
            "labels_gmm_raw": _labels_gmm,
            "labels_post_icm": _labels_post_icm,
            # None when the merge block did not run (K_eff == K).
            # Callers should check for None before using.
            "labels_post_merge": _labels_post_merge,
            "labels_post_cleanup": _labels_post_cleanup,
            "labels_post_recalib": labels_post_recalib,
            "recalib_tau_low": recalib_tau_low,
            "recalib_tau_high": recalib_tau_high,
            "n_cleanup_steps": self.n_cleanup_steps,
            "n_swaps_total": n_swaps_total,
            "n_reassigned_total": n_reassigned_total,
            "n_unlabeled_total": n_unlabeled_total,
            "n_cleanup_changed": n_cleanup_changed,
            "parent_map": dict(_parent_map),
            "merge_history": merge_history,
            "n_unfrozen_merge": n_unfrozen_merge,
            "n_iters_post_merge": n_iters_post_merge,
            "n_unfrozen_h1_post": n_unfrozen_h1_post,
            "lam_boost_used": lam_boost_val,
            "covariance_type": self.covariance_type,
            "kde_unary": self.kde_unary,
            "tau_low_per_k": tau_low_per_k if self.per_cluster_tau else {},
            "n_em_iters": self.n_em_iters,
            "labels_post_em": _labels_post_em,
        }
        return labels, model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, dataset: MosaicDataset) -> ClusteringResult:
        """Fit GMM + ICM and return a ClusteringResult.

        W1: when n_restarts > 1, the coverage map is built once and shared
        across all restarts.  The result with the lowest total MRF energy
        (energy_final in the model dict) is returned.
        """
        # Optionally dilate polygons before coverage-fraction computation (P2).
        # Dilation increases CF for "touching but not overlapping" pairs,
        # giving the signed ramp an attractive signal for genuine same-tile
        # contacts that would otherwise have CF ≈ 0.  The dataset is never
        # modified — dilated_polygons is a temporary copy used only here.
        if self.polygon_dilation is not None and self.polygon_dilation > 0.0:
            dilated_polygons = _dilate_polygons(dataset.polygons, self.polygon_dilation)
        else:
            dilated_polygons = dataset.polygons

        # Build coverage map once — shared across all restarts (W1)
        raw_map = _build_coverage_map(
            dilated_polygons, dataset.centers, self.spatial_radius,
            dataset.clipped, self.exclude_clipped,
            n_workers=self.n_workers,
        )

        best_labels: np.ndarray | None = None
        best_model: dict = {}
        best_energy = np.inf

        n_runs = max(1, self.n_restarts)
        for restart_idx in range(n_runs):
            rs = self.random_state + restart_idx
            labels, model = self._fit_single(dataset, raw_map, rs)
            energy = float(model.get("energy_final", np.inf))
            if energy < best_energy:
                best_energy = energy
                best_labels = labels
                best_model = model

        if n_runs > 1:
            best_model["n_restarts_run"] = n_runs

        assert best_labels is not None
        return ClusteringResult(labels=best_labels, model=best_model)
