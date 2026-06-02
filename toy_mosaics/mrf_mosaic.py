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


def _dilate_polygons(polygons: list[NDArray], fraction: float) -> list[NDArray]:
    """Return a list of polygons each dilated by `fraction` of its own diameter.

    Dilation is applied only for coverage-fraction computation; the dataset
    polygons are never modified.  Diameter is estimated as 2*sqrt(area/pi).
    A round buffer is used (default shapely behaviour) so the dilated polygon
    is always convex and valid.  Falls back to the original polygon on any
    shapely error.

    Typical values: 0.05–0.15 (5–15 % of the cell's own diameter).
    Setting fraction=0 is a no-op.
    """
    if fraction <= 0.0:
        return polygons
    import math
    from shapely.geometry import Polygon as _SPoly
    dilated: list[NDArray] = []
    for verts in polygons:
        try:
            poly = _SPoly(verts)
            if not poly.is_valid or poly.area == 0.0:
                dilated.append(verts)
                continue
            diameter = 2.0 * math.sqrt(poly.area / math.pi)
            buf = poly.buffer(fraction * diameter)
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


def _merge_clusters_to_k_global(
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
    max_cells_for_dist: int | None = 200,
    random_state: int = 42,
    linkage: str = 'complete',
) -> tuple[np.ndarray, list[tuple[int, int, float]], list[int]]:
    """Constrained hierarchical merge via a global dendrogram.

    ``linkage`` controls the scipy linkage method: 'complete' (default,
    conservative — merges minimise the maximum within-group distance),
    'average' (moderate — merges minimise the mean distance; recommended
    for heterogeneous sub-cluster sizes), or 'single' (permissive — merges
    minimise the nearest-neighbour distance).

    Differences from the greedy ``_merge_clusters_to_k``:

    * **Spatial gates are evaluated once on original cell sets** before any
      merging, not on the evolving merged set.  This prevents a wrong early
      merge from corrupting subsequent gate checks.
    * **Global optimum**: a complete-linkage dendrogram is built over ALL valid
      sub-cluster pairs simultaneously; ``fcluster`` then picks the cut at
      K_target that minimises the maximum within-group pairwise feature distance.
    * **Invalid pairs get ∞ distance**.  Under complete linkage this means a
      composite group can only merge with another group if EVERY original
      sub-cluster pair across the two groups is spatially valid.
    * **Scalable**: each sub-cluster is represented by at most
      ``max_cells_for_dist`` randomly sampled cells when computing centroid
      distances.  With K_eff ≤ 20 and ``max_cells_for_dist=200``, the distance
      computation is negligible even for 50 k-cell datasets.

    Falls back to the greedy algorithm if scipy linkage fails or if the
    dendrogram cannot be cut at exactly K_target clusters.
    """
    from collections import defaultdict
    from scipy.cluster.hierarchy import linkage as _slinkage, fcluster as _fcluster
    from scipy.spatial.distance import squareform

    labels = labels.copy()
    active = sorted(np.unique(labels).tolist())
    K_eff = len(active)

    if K_eff <= K_target:
        mapping = {old: new for new, old in enumerate(active)}
        return np.array([mapping[int(l)] for l in labels], dtype=np.int64), [], list(range(K_eff))

    rng = np.random.default_rng(random_state)
    ki = {k: i for i, k in enumerate(active)}  # sub-cluster label → matrix index

    # --- Subsample cells per sub-cluster for distance computation ---
    sub_sample: dict[int, np.ndarray] = {}
    for k in active:
        cells_k = np.where(labels == k)[0]
        if max_cells_for_dist is not None and len(cells_k) > max_cells_for_dist:
            idx = rng.choice(len(cells_k), max_cells_for_dist, replace=False)
            sub_sample[k] = cells_k[idx]
        else:
            sub_sample[k] = cells_k

    # --- Collect CF values for all original cross-cluster cell pairs ---
    pair_all_cfs:    dict[tuple[int, int], list[float]] = {}
    pair_frozen_cfs: dict[tuple[int, int], list[float]] = {}
    for (i, j), cf in raw_map.items():
        ka, kb = int(labels[i]), int(labels[j])
        if ka == kb or ka not in ki or kb not in ki:
            continue
        ia, ib = ki[ka], ki[kb]
        if ia > ib:
            ia, ib = ib, ia
        key = (ia, ib)
        pair_all_cfs.setdefault(key, []).append(cf)
        if frozen is not None and frozen[i] and frozen[j]:
            pair_frozen_cfs.setdefault(key, []).append(cf)

    # --- Build (K_eff × K_eff) distance matrix; ∞ for invalid pairs ---
    n = K_eff
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)

    for ia in range(n):
        for ib in range(ia + 1, n):
            key = (ia, ib)
            all_cfs = pair_all_cfs.get(key, [])
            frz_cfs = pair_frozen_cfs.get(key, [])
            # Gate 1: adjacency
            if len(all_cfs) < min_adj_pairs:
                continue
            # Gate 2: fraction veto
            cfs = frz_cfs if len(frz_cfs) >= min_frozen_pairs else all_cfs
            if len(cfs) >= min_pairs_for_veto:
                if sum(c >= tau_low for c in cfs) / len(cfs) >= veto_frac:
                    continue
            # Valid pair → centroid distance from sampled cells
            ka, kb = active[ia], active[ib]
            mean_a = X[sub_sample[ka]].mean(axis=0)
            mean_b = X[sub_sample[kb]].mean(axis=0)
            D[ia, ib] = D[ib, ia] = float(np.linalg.norm(mean_a - mean_b))

    # --- Dendrogram on the capped distance matrix ---
    finite_vals = D[(D > 0) & ~np.isinf(D)]
    sentinel = float(finite_vals.max() * 1e6) if len(finite_vals) > 0 else 1e6
    D_finite = np.where(np.isinf(D), sentinel, D)
    condensed = squareform(D_finite, checks=False)

    try:
        Z = _slinkage(condensed, method=linkage)
        flat = _fcluster(Z, t=K_target, criterion='maxclust')  # 1-indexed
    except Exception:
        return _merge_clusters_to_k(
            labels, X, raw_map, tau_low, K_target,
            frozen=frozen, min_frozen_pairs=min_frozen_pairs,
            veto_frac=veto_frac, min_adj_pairs=min_adj_pairs,
            min_pairs_for_veto=min_pairs_for_veto,
        )

    n_groups = int(np.max(flat))
    if n_groups != K_target:
        # Dendrogram cut didn't land exactly at K_target → fall back
        return _merge_clusters_to_k(
            labels, X, raw_map, tau_low, K_target,
            frozen=frozen, min_frozen_pairs=min_frozen_pairs,
            veto_frac=veto_frac, min_adj_pairs=min_adj_pairs,
            min_pairs_for_veto=min_pairs_for_veto,
        )

    # --- Apply merges: group sub-clusters by flat-cluster assignment ---
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, grp in enumerate(flat):
        groups[int(grp)].append(active[idx])

    merge_history: list[tuple[int, int, float]] = []
    final_active: list[int] = []
    for grp in sorted(groups):
        members = sorted(groups[grp])
        survivor = members[0]
        final_active.append(survivor)
        for absorbed in members[1:]:
            ia, ib = ki[survivor], ki[absorbed]
            dist_val = float(D[ia, ib])
            if dist_val >= sentinel:
                dist_val = sentinel
            merge_history.append((survivor, absorbed, dist_val))
            labels[labels == absorbed] = survivor

    mapping = {old: new for new, old in enumerate(final_active)}
    return (
        np.array([mapping[int(l)] for l in labels], dtype=np.int64),
        merge_history,
        final_active,
    )


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
    n_em_iters_post_cleanup:
        Number of EM re-fit iterations to run *after* the cleanup passes
        (n_cleanup_steps).  Like n_em_iters, each iteration re-fits the GMM on
        the current labels and runs another ICM pass.  Running EM here
        re-anchors the cleanup's spatial corrections in feature space, reducing
        the risk that the next ICM (or restart) reverts them.  Only beneficial
        when cleanup makes feature-supported moves (i.e. conflict_min_posterior
        > 0); if cleanup bypasses the feature model, post-cleanup EM can
        oscillate with it.  Default 0 (disabled).
    n_tracked_cleanup_iters:
        When > 0, replaces the single-shot steps 11+12 (cleanup + post-cleanup
        EM) with a best-energy tracking loop.  Each of the N iterations runs
        the full n_cleanup_steps alternating passes followed by
        n_em_iters_post_cleanup EM re-fits, then records the total MRF energy.
        The iteration with the lowest energy is returned as the final result —
        oscillations between iterations do not affect the output because the
        tracker keeps a copy of the best state seen.  Adds one
        ``_compute_energy`` call per iteration (negligible cost).  Default 0
        (disabled; runs the original single-shot steps 11+12 instead).
    cleanup_sigma:
        Std. dev. of Gaussian noise added to the unary costs during each
        tracked cleanup pass (Mechanism 3).  The noise is applied only inside
        the cleanup step; the MRF energy used for tracking is always computed
        on the original, noise-free unary — so the tracker is unbiased.  A
        cell whose unary gap between its current and next-best label is smaller
        than σ can be swapped even if the deterministic cleanup would keep it,
        allowing the tracker to find better local optima.  Only effective when
        n_tracked_cleanup_iters > 0.  Default 0.0 (disabled).  Starting point:
        σ ≈ 0.1 × (median second-best unary − best unary across all cells).
    cleanup_unfreeze_frac:
        Fraction of currently frozen cells to randomly unfreeze for each
        tracked cleanup pass (Mechanism 5).  The unfreezing is temporary —
        cells are refrozen before the next tracked iteration and before the EM
        step.  This lets the cleanup occasionally correct feature impostors
        that are frozen with the wrong label and cannot be touched by the
        deterministic pipeline.  Only effective when n_tracked_cleanup_iters
        > 0.  Default 0.0 (disabled).  Starting point: 0.1–0.2 (unfreeze
        10–20 % of frozen cells per pass).
    polygon_dilation:
        If not None and > 0, dilate every polygon by this fraction of its own
        estimated diameter (2*sqrt(area/pi)) before computing coverage
        fractions.  This converts "touching but zero-overlap" same-tile pairs
        into small positive CF values, giving the signed ramp an attractive
        signal for genuine intra-tile contacts that would otherwise be
        invisible.  The dataset polygons are never modified.  Default None
        (disabled).  Typical values: 0.05–0.15 (5–15 % of each cell's diameter).
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
        signed_ramp: bool = True,
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
        leiden_resolution: float | None = 0.5,
        covariance_type: str = "full",
        kde_unary: bool = False,
        count_reg: float = 0.0,
        gmm_quality_gate: float = 0.0,
        lam_boost: float | None = None,
        adaptive_lam_boost: bool = False,
        conflict_min_posterior: float = 0.01,
        use_dbscan_split: bool = False,
        per_cluster_tau: bool = True,
        n_em_iters: int = 1,
        n_cleanup_steps: int = 3,
        n_em_iters_post_cleanup: int = 0,
        n_tracked_cleanup_iters: int = 0,
        cleanup_sigma: float = 0.0,
        cleanup_unfreeze_frac: float = 0.0,
        use_global_merge: bool = True,
        merge_max_cells_for_dist: int | None = 200,
        merge_linkage: str = 'complete',
        remerge_after_cleanup: bool = False,
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
        self.n_em_iters_post_cleanup = n_em_iters_post_cleanup
        self.n_tracked_cleanup_iters = n_tracked_cleanup_iters
        self.cleanup_sigma = cleanup_sigma
        self.cleanup_unfreeze_frac = cleanup_unfreeze_frac
        self.use_global_merge = use_global_merge
        self.merge_max_cells_for_dist = merge_max_cells_for_dist
        self.merge_linkage = merge_linkage
        self.remerge_after_cleanup = remerge_after_cleanup
        self.polygon_dilation = polygon_dilation

    # ------------------------------------------------------------------
    # Step helpers — each encapsulates one logical stage of the pipeline.
    # All use self.* for configuration; mutable state is passed explicitly.
    # ------------------------------------------------------------------

    def _step_gmm_and_unary(
        self, X: NDArray, K_init: int, random_state: int,
    ) -> tuple[GaussianMixture, np.ndarray, NDArray, NDArray, int, str]:
        """GMM fit + unary cost computation. Returns (gmm, labels, posteriors, unary, K_init, init_used)."""
        free_leiden = (
            self.init == "leiden"
            and self.leiden_resolution is not None
            and self.n_clusters_init is None
        )
        means_init, init_used = _init_gmm_means(
            X, K_init, self.init, random_state,
            leiden_k_features=self.leiden_k_features,
            leiden_resolution=self.leiden_resolution,
            require_k=not free_leiden,
        )
        if free_leiden:
            K_init = len(means_init)
        if init_used != self.init:
            init_used = f"{self.init}→{init_used}"
        gmm = GaussianMixture(
            n_components=K_init, covariance_type=self.covariance_type,
            n_init=3, random_state=random_state,
            means_init=means_init, init_params="random",
        )
        gmm.fit(X)
        labels = gmm.predict(X).astype(np.int64)
        posteriors = gmm.predict_proba(X)
        n_cells = len(X)
        if self.kde_unary:
            from scipy.stats import gaussian_kde
            log_liks = np.zeros((n_cells, K_init))
            for k in range(K_init):
                mask = labels == k
                try:
                    if mask.sum() < 2:
                        raise ValueError("too few cells")
                    log_liks[:, k] = gaussian_kde(X[mask].T).logpdf(X.T)
                except Exception:
                    cov = gmm.covariances_[k] if self.covariance_type == "full" else np.diag(gmm.covariances_[k])
                    log_liks[:, k] = multivariate_normal(mean=gmm.means_[k], cov=cov).logpdf(X)
        else:
            log_liks = np.zeros((n_cells, K_init))
            for k in range(K_init):
                cov = (gmm.covariances_[k] if self.covariance_type == "full"
                       else gmm.covariances_ if self.covariance_type == "tied"
                       else np.diag(gmm.covariances_[k]))
                log_liks[:, k] = multivariate_normal(mean=gmm.means_[k], cov=cov).logpdf(X)
        unary = -(log_liks + np.log(gmm.weights_))
        return gmm, labels, posteriors, unary, K_init, init_used

    def _step_split(
        self, labels: np.ndarray, unary: NDArray, centers: NDArray, K: int, K_init: int,
    ) -> tuple[np.ndarray, NDArray, int, int, dict[int, int]]:
        """Optionally split fragmented clusters. Returns (labels, unary, K_eff, n_splits, parent_map)."""
        K_eff, n_splits, parent_map = K_init, 0, {}
        if self.split_merge:
            if K_init <= K:
                labels, K_eff, parent_map = _split_fragmented_clusters(
                    labels, centers, K_init, use_dbscan=self.use_dbscan_split,
                )
                n_splits = K_eff - K_init
                if K_eff > K_init:
                    extra = np.column_stack([unary[:, parent_map[k]] for k in range(K_init, K_eff)])
                    unary = np.hstack([unary, extra])
        return labels, unary, K_eff, n_splits, parent_map

    def _step_calibrate_and_adjacency(
        self, raw_map: dict, frozen: NDArray, labels: np.ndarray, unary: NDArray,
        n_cells: int, K_eff: int,
    ) -> tuple[float, float, float, dict, object, list, dict]:
        """Calibrate tau/lam and build pairwise adjacency.
        Returns (tau_low, tau_high, lam, tau_calib, ramp_fn, nbrs, tau_low_per_k)."""
        tau_low, tau_high = self.tau_low, self.tau_high
        cal_lam: float | None = None
        tau_calib: dict = {}
        if tau_low is None or tau_high is None or self.lam is None:
            cal_low, cal_high, cal_lam, tau_calib = _calibrate_tau(
                raw_map, frozen, labels, unary,
                signed_ramp=self.signed_ramp, lam_alpha=self.lam_alpha,
                tau_low_q=self.tau_low_q, tau_high_q=self.tau_high_q,
                log_ramp=self.log_ramp, log_ramp_alpha=self.log_ramp_alpha,
                gmm_quality_gate=self.gmm_quality_gate, cf_tau_mixture=self.cf_tau_mixture,
            )
            if tau_low is None:  tau_low  = cal_low
            if tau_high is None: tau_high = cal_high
        lam: float = self.lam or cal_lam or 20.0
        tau_calib["lam"] = lam
        _min_gap = 0.05
        if tau_high < tau_low + _min_gap:
            tau_high = float(np.clip(tau_low + _min_gap, tau_low + _min_gap, 0.99))
            tau_calib["tau_high_adjusted"] = f"pushed to tau_low+{_min_gap} ({tau_high:.3f})"
        # Build ramp function
        if self.log_ramp:
            _a = self.log_ramp_alpha
            ramp_fn = (lambda cf, tl, th: _signed_log_ramp(cf, tl, th, _a)) if self.signed_ramp \
                      else (lambda cf, tl, th: _log_ramp(cf, tl, th, _a))
        else:
            ramp_fn = _signed_ramp if self.signed_ramp else _ramp
        # P3: per-cluster tau_low
        tau_low_per_k: dict[int, float] = {}
        if self.per_cluster_tau:
            for k in range(K_eff):
                same_k = [cf for (i, j), cf in raw_map.items()
                          if labels[i] == k and labels[j] == k and frozen[i] and frozen[j]]
                if len(same_k) >= _MIN_CALIB_PAIRS:
                    tau_low_per_k[k] = float(np.clip(np.quantile(same_k, self.tau_low_q), 0.01, 0.49))
        # Build adjacency
        nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]
        for (i, j), cf in raw_map.items():
            tl = tau_low_per_k.get(int(labels[i]), tau_low) \
                 if (self.per_cluster_tau and labels[i] == labels[j]) else tau_low
            w = ramp_fn(cf, tl, tau_high)
            if w != 0.0:
                nbrs[i].append((j, w)); nbrs[j].append((i, w))
        return tau_low, tau_high, lam, tau_calib, ramp_fn, nbrs, tau_low_per_k

    def _step_h1_prepass(
        self, raw_map: dict, labels: np.ndarray, frozen: NDArray,
    ) -> tuple[NDArray, int]:
        """Force-unfreeze cells in near-certain labelling errors. Returns (frozen, n_force_unfrozen)."""
        if self.theta_hard is None:
            return frozen, 0
        n_force_unfrozen = 0
        for cf, i, j in sorted(
            ((cf, i, j) for (i, j), cf in raw_map.items()
             if cf > self.theta_hard and labels[i] == labels[j]),
            reverse=True,
        ):
            if labels[i] != labels[j]:
                continue
            if frozen[i]: frozen[i] = False; n_force_unfrozen += 1
            if frozen[j]: frozen[j] = False; n_force_unfrozen += 1
        return frozen, n_force_unfrozen

    def _step_icm(
        self, labels: np.ndarray, unary: NDArray, nbrs: list, frozen: NDArray,
        lam: float, K_eff: int, n_cells: int,
    ) -> tuple[np.ndarray, int, int, float, float]:
        """One full ICM pass (GS or Jacobi). Returns (labels, n_iters, n_changed, e_before, e_after)."""
        energy_before = _compute_energy(labels, unary, nbrs, lam)
        prev_energy   = energy_before
        n_iters, n_changed_total = 0, 0
        if self.icm_jacobi:
            rows, cols, data = [], [], []
            for i, nl in enumerate(nbrs):
                for j, w in nl: rows.append(i); cols.append(j); data.append(w)
            W_sp = (csr_matrix((np.array(data), (np.array(rows), np.array(cols))),
                               shape=(n_cells, n_cells)) if rows else csr_matrix((n_cells, n_cells)))
            for iteration in range(self.max_iters):
                n_iters = iteration + 1
                one_hot  = (labels[:, None] == np.arange(K_eff)).astype(np.float64)
                new_labels = np.argmin(unary + lam * (W_sp @ one_hot), axis=1).astype(np.int64)
                new_labels[frozen] = labels[frozen]
                n_ch = int(np.sum(new_labels != labels))
                n_changed_total += n_ch; labels = new_labels
                if n_ch == 0: break
                if self.energy_tol > 0.0:
                    curr = _compute_energy(labels, unary, nbrs, lam)
                    if abs(prev_energy - curr) / (abs(prev_energy) + 1e-10) < self.energy_tol: break
                    prev_energy = curr
        else:
            for iteration in range(self.max_iters):
                n_iters = iteration + 1; n_ch = 0
                for i in range(n_cells):
                    if frozen[i]: continue
                    e_local = unary[i].copy()
                    for j, w in nbrs[i]: e_local[labels[j]] += lam * w
                    bk = int(np.argmin(e_local))
                    if bk != labels[i]: labels[i] = bk; n_ch += 1
                n_changed_total += n_ch
                if n_ch == 0: break
                if self.energy_tol > 0.0:
                    curr = _compute_energy(labels, unary, nbrs, lam)
                    if abs(prev_energy - curr) / (abs(prev_energy) + 1e-10) < self.energy_tol: break
                    prev_energy = curr
        return labels, n_iters, n_changed_total, energy_before, _compute_energy(labels, unary, nbrs, lam)

    def _step_recalibrate(
        self, raw_map: dict, labels: np.ndarray, unary: NDArray,
        frozen: NDArray, lam_old: float, n_cells: int,
    ) -> tuple[np.ndarray, list, float, np.ndarray, float | None, float | None, int]:
        """Re-estimate tau/lam from post-ICM labels and run a second ICM pass.
        Returns (labels, nbrs, lam, labels_post_recalib, tau_low, tau_high, n_changed)."""
        rc_low, rc_high, rc_lam, _ = _calibrate_tau(
            raw_map, frozen, labels, unary,
            signed_ramp=self.signed_ramp, lam_alpha=self.lam_alpha,
            tau_low_q=self.tau_low_q, tau_high_q=self.tau_high_q,
            log_ramp=self.log_ramp, log_ramp_alpha=self.log_ramp_alpha,
            gmm_quality_gate=self.gmm_quality_gate, cf_tau_mixture=self.cf_tau_mixture,
        )
        rc_lam_eff = rc_lam if rc_lam is not None else lam_old
        if self.log_ramp:
            _a = self.log_ramp_alpha
            rfn = (lambda cf, tl, th: _signed_log_ramp(cf, tl, th, _a)) if self.signed_ramp \
                  else (lambda cf, tl, th: _log_ramp(cf, tl, th, _a))
        else:
            rfn = _signed_ramp if self.signed_ramp else _ramp
        nbrs_rc: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]
        for (i, j), cf in raw_map.items():
            w = rfn(cf, rc_low, rc_high)
            if w != 0.0: nbrs_rc[i].append((j, w)); nbrs_rc[j].append((i, w))
        labels, _, n_ch, _, _ = self._step_icm(labels, unary, nbrs_rc, frozen, rc_lam_eff, unary.shape[1], n_cells)
        return labels, nbrs_rc, rc_lam_eff, labels.copy(), rc_low, rc_high, n_ch

    def _step_merge_and_refine(
        self, labels: np.ndarray, labels_post_icm: np.ndarray,
        unary: NDArray, X: NDArray, raw_map: dict, nbrs: list,
        frozen: NDArray, lam: float, K: int, K_eff: int,
        tau_low: float, ramp_fn, n_cells: int,
    ) -> tuple[np.ndarray, NDArray, NDArray, np.ndarray, int, dict]:
        """Merge sub-clusters to K and run the cleanup ICM. Returns
        (labels, unary_k, frozen, labels_post_merge, n_merges, extra_stats)."""
        surviving = list(range(K_eff))
        _merge_fn = _merge_clusters_to_k_global if self.use_global_merge else _merge_clusters_to_k
        merge_kwargs: dict = dict(
            frozen=frozen, veto_frac=self.merge_veto_frac,
            min_adj_pairs=self.merge_min_adj_pairs,
            min_pairs_for_veto=self.merge_min_pairs_for_veto,
        )
        if self.use_global_merge:
            merge_kwargs["max_cells_for_dist"] = self.merge_max_cells_for_dist
            merge_kwargs["linkage"] = self.merge_linkage
        labels, merge_history, surviving = _merge_fn(
            labels, X, raw_map, tau_low, K, **merge_kwargs,
        )
        n_merges = K_eff - len(np.unique(labels))
        labels_post_merge = labels.copy()
        # Unfreeze merge-boundary violators
        n_unfrozen_merge = 0
        for (i, j), cf in raw_map.items():
            if labels[i] == labels[j] and labels_post_icm[i] != labels_post_icm[j] and cf >= tau_low:
                if frozen[i]: frozen[i] = False; n_unfrozen_merge += 1
                if frozen[j]: frozen[j] = False; n_unfrozen_merge += 1
        # Build K_final-dimensional mixture unary
        component_map: dict[int, set] = {k: {k} for k in range(K_eff)}
        for a_h, b_h, _ in merge_history:
            if b_h in component_map:
                component_map[a_h] = component_map[a_h] | component_map.pop(b_h)
        K_final = len(surviving)
        unary_k = np.zeros((n_cells, K_final))
        for new_k, old_k in enumerate(surviving):
            comps = sorted(component_map.get(old_k, {old_k}))
            log_mix = -unary[:, comps[0]].copy()
            for c in comps[1:]: log_mix = np.logaddexp(log_mix, -unary[:, c])
            unary_k[:, new_k] = -log_mix
        # Second ICM pass
        labels, _, n_ch, _, _ = self._step_icm(labels, unary_k, nbrs, frozen, lam, K_final, n_cells)
        # H1 fallback after merge
        n_unfrozen_h1_post = 0
        if self.theta_hard is not None:
            frozen, n_unfrozen_h1_post = self._step_h1_prepass(raw_map, labels, frozen)
            if n_unfrozen_h1_post > 0:
                labels, _, n_ch2, _, _ = self._step_icm(labels, unary_k, nbrs, frozen, lam, K_final, n_cells)
                n_ch += n_ch2
        extra = {
            "merge_history": merge_history, "surviving_subclusters": surviving,
            "n_unfrozen_merge": n_unfrozen_merge, "n_unfrozen_h1_post": n_unfrozen_h1_post,
            "n_iters_post_merge": 0,  # tracked inside _step_icm but not exposed; kept for compat
        }
        return labels, unary_k, frozen, labels_post_merge, n_merges, extra, n_ch

    def _step_em_iter(
        self, X: NDArray, labels: np.ndarray, unary_final: NDArray, nbrs: list,
        frozen: NDArray, lam: float, K_em: int, n_cells: int, random_state: int,
    ) -> tuple[np.ndarray, NDArray, NDArray, GaussianMixture]:
        """One EM re-fit iteration. Returns (labels, unary_final, frozen)."""
        if any(int(np.sum(labels == k)) < 2 for k in range(K_em)):
            return labels, unary_final, frozen
        em_means = np.array([X[labels == k].mean(axis=0) for k in range(K_em)])
        gmm_em = GaussianMixture(
            n_components=K_em, covariance_type=self.covariance_type,
            n_init=1, random_state=random_state,
            means_init=em_means, init_params="random",
        )
        gmm_em.fit(X)
        log_liks = np.zeros((n_cells, K_em))
        for k in range(K_em):
            cov = (gmm_em.covariances_[k] if self.covariance_type == "full"
                   else gmm_em.covariances_ if self.covariance_type == "tied"
                   else np.diag(gmm_em.covariances_[k]))
            log_liks[:, k] = multivariate_normal(mean=gmm_em.means_[k], cov=cov).logpdf(X)
        unary_final = -(log_liks + np.log(gmm_em.weights_))
        # ICM runs with NO frozen cells: the EM unary is fresh and all cells
        # should be free to relocate.  Freezing here would reintroduce the
        # normalization-trap problem (cells with near-zero absolute likelihood
        # under all clusters can still get posterior ≥ conf_threshold for the
        # wrong cluster, locking them in).
        labels, _, _, _, _ = self._step_icm(
            labels, unary_final, nbrs,
            np.zeros(n_cells, dtype=bool),   # fully unfrozen
            lam, K_em, n_cells,
        )
        # Compute frozen_em AFTER ICM for use by downstream steps only
        # (cleanup, tracked loop), not for this ICM pass.
        frozen_em: NDArray = gmm_em.predict_proba(X).max(axis=1) >= self.conf_threshold
        return labels, unary_final, frozen_em, gmm_em

    def _step_cleanup(
        self, labels: np.ndarray, unary_final: NDArray, nbrs: list,
        frozen: NDArray, lam: float, lam_boost_val: float, lam_boost_min_w: float,
        labels_pre_cleanup: np.ndarray, K_eff: int, K: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Alternating cleanup passes. Returns (labels, stats).

        When rng is provided (tracked mode only):
          Mechanism 3 — unary noise: Gaussian noise with std=cleanup_sigma is
            added to unary_final for the cleanup passes.  The caller's unary is
            not modified; the tracker compares energies on the original unary.
          Mechanism 5 — stochastic unfreezing: cleanup_unfreeze_frac × 100 %
            of frozen cells are temporarily unfrozen for this call only.  The
            frozen mask is not modified in the calling scope.
        """
        n_swaps_total, n_reassigned_total, n_unlabeled_total = 0, 0, 0

        # Mechanism 5: temporary stochastic unfreezing (copy so caller is unaffected)
        if rng is not None and self.cleanup_unfreeze_frac > 0.0 and frozen is not None:
            frozen = frozen.copy()
            frozen_idx = np.where(frozen)[0]
            if len(frozen_idx) > 0:
                n_unfreeze = max(1, int(self.cleanup_unfreeze_frac * len(frozen_idx)))
                to_unfreeze = rng.choice(frozen_idx, size=min(n_unfreeze, len(frozen_idx)), replace=False)
                frozen[to_unfreeze] = False

        # Mechanism 3: noisy unary (original unary kept intact for energy tracking)
        unary_pass = unary_final
        if rng is not None and self.cleanup_sigma > 0.0:
            unary_pass = unary_final + rng.normal(0.0, self.cleanup_sigma, unary_final.shape)

        frozen_for_conflict = frozen if not (self.split_merge and K_eff > K) else None
        for pass_num in range(1, self.n_cleanup_steps + 1):
            if pass_num % 2 == 1:
                labels, n_sw = _resolve_conflicting_pairs(
                    labels, unary_pass, nbrs, lam,
                    lam_boost=lam_boost_val, lam_boost_min_w=lam_boost_min_w,
                    min_posterior=self.conflict_min_posterior,
                    frozen=frozen_for_conflict,
                )
                n_swaps_total += n_sw
            else:
                labels, n_ra, n_ul = _label_residual_violators(
                    labels, nbrs, unary=unary_pass, frozen=frozen,
                )
                if self.split_merge and n_ul > 0:
                    neg = labels < 0
                    if neg.any():
                        labels[neg] = np.argmin(unary_final[neg], axis=1)
                        n_ra += int(neg.sum()); n_ul = 0
                n_reassigned_total += n_ra; n_unlabeled_total += n_ul
        n_cleanup_changed = int(np.sum(labels != labels_pre_cleanup)) \
                            if labels_pre_cleanup is not None else 0
        return labels, {
            "n_swaps_total": n_swaps_total,
            "n_reassigned_total": n_reassigned_total,
            "n_unlabeled_total": n_unlabeled_total,
            "n_cleanup_changed": n_cleanup_changed,
        }

    # ------------------------------------------------------------------
    # Internal: single GMM+ICM run given a pre-built coverage map
    # ------------------------------------------------------------------

    def _fit_single(
        self,
        dataset: MosaicDataset,
        raw_map: dict,
        random_state: int,
    ) -> tuple[np.ndarray, dict]:
        """Orchestrate one full GMM+ICM pass.  ``raw_map`` is built once by ``fit`` (W1)."""
        X, K, n_cells = dataset.X, self.n_clusters, len(dataset)
        K_init = self.n_clusters_init or K

        # 1. GMM + unary costs
        gmm, labels, posteriors, unary, K_init, init_method_used = \
            self._step_gmm_and_unary(X, K_init, random_state)
        _labels_gmm = labels.copy()

        # 2. Split fragmented clusters (optional)
        labels, unary, K_eff, n_splits, _parent_map = \
            self._step_split(labels, unary, dataset.centers, K, K_init)

        # 3. Freeze + calibrate tau/lam + build adjacency
        frozen: NDArray = posteriors.max(axis=1) >= self.conf_threshold
        tau_low, tau_high, lam, tau_calib, ramp_fn, nbrs, tau_low_per_k = \
            self._step_calibrate_and_adjacency(raw_map, frozen, labels, unary, n_cells, K_eff)

        # 4. Count regulariser — soft cluster-size prior (disabled by default)
        if self.count_reg > 0.0:
            n_exp = n_cells / K_eff
            for k in range(K_eff):
                unary[:, k] += self.count_reg * (float(np.sum(labels == k)) - n_exp) ** 2

        # 5. H1 pre-pass: force-unfreeze cells in near-certain labelling errors
        frozen, n_force_unfrozen = self._step_h1_prepass(raw_map, labels, frozen)
        labels_initial = labels.copy()
        violations_before = sum(
            1 for (i, j) in raw_map
            if ramp_fn(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
        )

        # 6. Main ICM
        labels, n_iters, n_changed_total, energy_before_icm, energy_after_icm = \
            self._step_icm(labels, unary, nbrs, frozen, lam, K_eff, n_cells)
        violations_after = sum(
            1 for (i, j) in raw_map
            if ramp_fn(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
        )
        _labels_post_icm = labels.copy()

        # 7. Iterative recalibration (optional)
        labels_post_recalib, recalib_tau_low, recalib_tau_high = None, None, None
        if self.recalibrate:
            labels, nbrs, lam, labels_post_recalib, recalib_tau_low, recalib_tau_high, n_ch = \
                self._step_recalibrate(raw_map, labels, unary, frozen, lam, n_cells)
            n_changed_total += n_ch

        # 8. Merge sub-clusters back to K + second ICM (optional, only when K_eff > K)
        _unary_final: NDArray = unary
        _labels_post_merge: np.ndarray | None = None
        n_merges, n_unfrozen_merge, n_iters_post_merge, n_unfrozen_h1_post = 0, 0, 0, 0
        merge_history: list[tuple[int, int, float]] = []
        if self.split_merge and K_eff > K:
            labels, _unary_final, frozen, _labels_post_merge, n_merges, merge_extra, n_ch = \
                self._step_merge_and_refine(
                    labels, _labels_post_icm, unary, X, raw_map,
                    nbrs, frozen, lam, K, K_eff, tau_low, ramp_fn, n_cells,
                )
            n_changed_total += n_ch
            merge_history      = merge_extra["merge_history"]
            n_unfrozen_merge   = merge_extra["n_unfrozen_merge"]
            n_unfrozen_h1_post = merge_extra["n_unfrozen_h1_post"]
            violations_after = sum(
                1 for (i, j) in raw_map
                if ramp_fn(raw_map[(i, j)], tau_low, tau_high) > 0 and labels[i] == labels[j]
            )

        _labels_post_cleanup = labels.copy()

        # Capture frozen before EM may replace it with a new GMM's posterior mask.
        # The model dict and tests always reference the post-H1 frozen state.
        frozen_for_model = frozen.copy()

        # 9. EM re-fit iterations (optional — repeatable via n_em_iters)
        # Unfreeze all cells before EM: the old frozen mask was computed from
        # the initial GMM, which may have trapped cells via the normalization
        # trap (a cell far from its Gaussian can still have posterior ≥
        # conf_threshold if it is even further from all other Gaussians).
        # The ICM inside _step_em_iter runs fully unfrozen; frozen_em is
        # recomputed from the new GMM posteriors only for downstream steps.
        if self.n_em_iters > 0:
            frozen = np.zeros(n_cells, dtype=bool)
        _labels_post_em: np.ndarray | None = None
        _gmm_post_em: GaussianMixture | None = None
        for _ in range(self.n_em_iters):
            K_em = _unary_final.shape[1]
            labels, _unary_final, frozen, _gmm_post_em = self._step_em_iter(
                X, labels, _unary_final, nbrs, frozen, lam, K_em, n_cells, random_state,
            )
        if self.n_em_iters > 0:
            _labels_post_em = labels.copy()

        # 10. Adaptive lam_boost for cleanup passes
        lam_boost_min_w = 0.7
        if self.adaptive_lam_boost:
            gaps  = np.sort(_unary_final, axis=1)[:, 1] - _unary_final.min(axis=1)
            denom = lam * lam_boost_min_w
            lam_boost_val = float(np.clip(float(np.percentile(gaps, 95)) / denom, 5.0, 20.0)) \
                            if denom > 0 else 5.0
        else:
            lam_boost_val = float(self.lam_boost) if self.lam_boost is not None else 5.0

        # 11+12. Cleanup passes + post-cleanup EM — tracked or untracked.
        #
        # UNTRACKED (n_tracked_cleanup_iters=0): run steps 11 and 12 once.
        # TRACKED   (n_tracked_cleanup_iters>0): run N cycles of 11+12, keep
        #   the iteration with the lowest MRF energy.  Oscillations between
        #   cycles cannot regress the returned result.

        _labels_post_all_cleanup:    np.ndarray | None = None
        _labels_post_em_post_cleanup: np.ndarray | None = None
        energy_tracked_history: list[float] = []
        best_tracked_iter = 0

        # RNG for stochastic cleanup (Mechanisms 3+5): only created when tracking
        # is active and at least one mechanism is enabled.
        _cleanup_rng: np.random.Generator | None = (
            np.random.default_rng(random_state)
            if self.n_tracked_cleanup_iters > 0
               and (self.cleanup_sigma > 0.0 or self.cleanup_unfreeze_frac > 0.0)
            else None
        )

        def _run_cleanup_and_em(lbl, u_fin, frz, rng=None):
            """One cycle of cleanup passes + post-cleanup EM. Returns (labels, unary, frozen, stats)."""
            lbl, cs = self._step_cleanup(
                lbl, u_fin, nbrs, frz, lam,
                lam_boost_val, lam_boost_min_w, _labels_post_cleanup, K_eff, K,
                rng=rng,
            )
            lbl_after_cleanup = lbl.copy()
            # EM always uses the original frozen mask (frz), not the temporarily
            # unfrozen one that _step_cleanup may have used internally.
            for _ in range(self.n_em_iters_post_cleanup):
                K_em = u_fin.shape[1]
                lbl, u_fin, frz, _ = self._step_em_iter(
                    X, lbl, u_fin, nbrs, frz, lam, K_em, n_cells, random_state,
                )
            return lbl, u_fin, frz, lbl_after_cleanup, cs

        if self.n_tracked_cleanup_iters > 0:
            # Tracked: keep copy of best-energy state across all cycles
            lbl_best    = labels.copy()
            unary_best  = _unary_final.copy()
            frozen_best = frozen.copy()
            E_best      = _compute_energy(labels, _unary_final, nbrs, lam)
            energy_tracked_history = [E_best]
            cleanup_stats: dict = {"n_swaps_total": 0, "n_reassigned_total": 0,
                                   "n_unlabeled_total": 0, "n_cleanup_changed": 0}
            lbl_cur, u_cur, frz_cur = labels.copy(), _unary_final.copy(), frozen.copy()
            _labels_post_all_cleanup_best: np.ndarray = lbl_best.copy()

            for t in range(1, self.n_tracked_cleanup_iters + 1):
                lbl_cur, u_cur, frz_cur, lbl_ac, cs_t = _run_cleanup_and_em(
                    lbl_cur, u_cur, frz_cur, rng=_cleanup_rng,
                )
                E_t = _compute_energy(lbl_cur, u_cur, nbrs, lam)
                energy_tracked_history.append(E_t)
                for k in ("n_swaps_total", "n_reassigned_total",
                          "n_unlabeled_total", "n_cleanup_changed"):
                    cleanup_stats[k] = cleanup_stats.get(k, 0) + cs_t.get(k, 0)
                if E_t < E_best:
                    E_best       = E_t
                    lbl_best     = lbl_cur.copy()
                    unary_best   = u_cur.copy()
                    frozen_best  = frz_cur.copy()
                    best_tracked_iter = t
                    _labels_post_all_cleanup_best = lbl_ac.copy()
                # Fixed-point early stop: energy unchanged → reached a fixed
                # point; further iterations are guaranteed to be identical.
                if t > 1 and abs(energy_tracked_history[-1] - energy_tracked_history[-2]) < 1e-6:
                    break

            labels        = lbl_best
            _unary_final  = unary_best
            frozen        = frozen_best
            _labels_post_all_cleanup    = _labels_post_all_cleanup_best
            _labels_post_em_post_cleanup = lbl_best.copy() if self.n_em_iters_post_cleanup > 0 else None

        else:
            # Untracked: run once
            labels, cleanup_stats = self._step_cleanup(
                labels, _unary_final, nbrs, frozen, lam,
                lam_boost_val, lam_boost_min_w, _labels_post_cleanup, K_eff, K,
            )
            _labels_post_all_cleanup = labels.copy()

            for _ in range(self.n_em_iters_post_cleanup):
                K_em = _unary_final.shape[1]
                labels, _unary_final, frozen, _ = self._step_em_iter(
                    X, labels, _unary_final, nbrs, frozen, lam, K_em, n_cells, random_state,
                )
            if self.n_em_iters_post_cleanup > 0:
                _labels_post_em_post_cleanup = labels.copy()

        # Snapshot before correction — used by figure script to show before/after.
        _labels_pre_remerge_correction = labels.copy()

        # 13. Post-pipeline merge correction (optional)
        # Re-detect spatial disconnectedness in the final K clusters.  Any
        # cluster whose cells have split into ≥ 2 connected components was
        # likely mis-merged earlier; re-split it and re-merge using the global
        # dendrogram on the post-pipeline labels (which are better than the
        # noisy pre-ICM labels used during the original merge).
        _labels_post_remerge_correction: np.ndarray | None = None
        if self.remerge_after_cleanup and self.split_merge:
            re_labels, K_re, re_parent_map = _split_fragmented_clusters(
                labels, dataset.centers, K,
                use_dbscan=self.use_dbscan_split,
            )
            if K_re > K:
                # Expand _unary_final to K_re columns (copy parent column for each new sub)
                extra_cols = np.column_stack([
                    _unary_final[:, re_parent_map[k]] for k in range(K, K_re)
                ])
                unary_re = np.hstack([_unary_final, extra_cols])
                # Re-merge back to K using the global dendrogram
                re_labels, _unary_final, frozen, _, _, _, n_ch_re = \
                    self._step_merge_and_refine(
                        re_labels, re_labels.copy(),
                        unary_re, X, raw_map, nbrs, frozen, lam,
                        K, K_re, tau_low, ramp_fn, n_cells,
                    )
                labels = re_labels
                n_changed_total += n_ch_re
                _labels_post_remerge_correction = labels.copy()
                # One more cleanup pass to resolve any post-correction violations
                labels, cs_re = self._step_cleanup(
                    labels, _unary_final, nbrs, frozen, lam,
                    lam_boost_val, lam_boost_min_w,
                    _labels_post_remerge_correction, K, K,
                )
                for key in cleanup_stats:
                    cleanup_stats[key] = cleanup_stats.get(key, 0) + cs_re.get(key, 0)

        energy_final = _compute_energy(labels, _unary_final, nbrs, lam)

        model: dict = {
            "gmm": gmm, "lam": lam,
            "n_iters": n_iters, "n_frozen": int(frozen.sum()), "n_changed": n_changed_total,
            "violations_before": violations_before, "violations_after": violations_after,
            "energy_before_icm": energy_before_icm, "energy_after_icm": energy_after_icm,
            "energy_final": energy_final,
            "labels_initial": labels_initial, "frozen": frozen_for_model,
            "tau_low": tau_low, "tau_high": tau_high, "tau_calibration": tau_calib,
            "raw_map": raw_map, "spatial_radius": self.spatial_radius,
            "exclude_clipped": self.exclude_clipped, "conf_threshold": self.conf_threshold,
            "signed_ramp": self.signed_ramp, "log_ramp": self.log_ramp,
            "log_ramp_alpha": self.log_ramp_alpha if self.log_ramp else None,
            "theta_hard": self.theta_hard, "n_force_unfrozen": n_force_unfrozen,
            "split_merge": self.split_merge, "n_clusters_init": K_init,
            "init_method": self.init, "init_method_used": init_method_used,
            "n_splits": n_splits, "n_merges": n_merges, "k_after_split": K_eff,
            "labels_gmm_raw": _labels_gmm, "labels_post_icm": _labels_post_icm,
            "labels_post_merge": _labels_post_merge, "labels_post_cleanup": _labels_post_cleanup,
            "labels_post_recalib": labels_post_recalib,
            "recalib_tau_low": recalib_tau_low, "recalib_tau_high": recalib_tau_high,
            "parent_map": dict(_parent_map), "merge_history": merge_history,
            "n_unfrozen_merge": n_unfrozen_merge, "n_iters_post_merge": n_iters_post_merge,
            "n_unfrozen_h1_post": n_unfrozen_h1_post,
            "lam_boost_used": lam_boost_val,
            "covariance_type": self.covariance_type, "kde_unary": self.kde_unary,
            "tau_low_per_k": tau_low_per_k if self.per_cluster_tau else {},
            "n_em_iters": self.n_em_iters, "labels_post_em": _labels_post_em,
            "gmm_post_em": _gmm_post_em,
            "n_cleanup_steps": self.n_cleanup_steps,
            **cleanup_stats,
            "n_em_iters_post_cleanup": self.n_em_iters_post_cleanup,
            "n_tracked_cleanup_iters": self.n_tracked_cleanup_iters,
            "energy_tracked_history": energy_tracked_history,
            "best_tracked_iter": best_tracked_iter,
            "labels_post_all_cleanup": _labels_post_all_cleanup,
            "labels_post_em_post_cleanup": _labels_post_em_post_cleanup,
            "labels_pre_remerge_correction": _labels_pre_remerge_correction,
            "labels_post_remerge_correction": _labels_post_remerge_correction,
            "use_global_merge": self.use_global_merge,
            "remerge_after_cleanup": self.remerge_after_cleanup,
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
