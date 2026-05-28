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
    unary: NDArray,
    *,
    signed_ramp: bool,
    lam_alpha: float = 2.0,
    min_pairs: int = _MIN_CALIB_PAIRS,
) -> tuple[float, float, float | None, dict]:
    """Infer tau_low, tau_high, and lam from the data, without ground-truth labels.

    **tau calibration** (fallback hierarchy for each threshold independently):
      1. Frozen same/diff-label pairs  — high-confidence proxy for true type labels
      2. All same/diff-label pairs     — more data, noisier labels
      3. Hard-coded prior              — last resort

    tau_low  = p99 of same-label CF  (covers 99% of intra-tile contacts)
    tau_high = p25 of diff-label CF  (75% of cross-type pairs at full repulsion)

    **lambda calibration** (using unary gaps and spatial push):
    lam is set so that a cell at the median unary gap flips when it receives
    lam_alpha "average-weight" same-label violating frozen-neighbor contacts.

      median_unary_gap  = median over unfrozen cells of (2nd-best unary − best unary)
      median_push       = median over unfrozen cells-with-violations of
                          Σ_{j∈frozen_same_label_nbrs, CF>tau_low} linear_ramp(CF_ij)
      lam = lam_alpha × median_unary_gap / median_push

    Returns (tau_low, tau_high, lam_calibrated, info_dict).
    lam_calibrated is None when calibration data are insufficient.
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

    # --- Lambda calibration ---
    # lam is set so that a typical unfrozen cell at the median unary gap flips
    # when it has lam_alpha "average-weight" violating frozen same-label contacts:
    #
    #   lam = lam_alpha × median_unary_gap / median_push
    #
    # median_unary_gap: median of (2nd-best unary cost − best unary cost) over
    #                   unfrozen cells — measures how hard it is to flip each cell.
    # median_push:      median pairwise force per unit lam that frozen same-label
    #                   neighbors with CF > tau_low exert on unfrozen cells.
    #                   Uses a linear ramp as a conservative approximation of the
    #                   actual ramp (result is independent of log_ramp/signed_ramp).
    lam_calibrated: float | None = None
    lam_source = "not_calibrated"

    unfrozen_idx = np.where(~frozen)[0]
    if len(unfrozen_idx) >= min_pairs:
        unary_unfrozen = unary[unfrozen_idx]                       # (n_unf, K)
        best_costs     = unary_unfrozen.min(axis=1)
        second_costs   = np.partition(unary_unfrozen, 1, axis=1)[:, 1]
        median_unary_gap = float(np.median(second_costs - best_costs))

        # Pairwise push from frozen same-label neighbors with CF > tau_low.
        # linear_ramp(CF) = (CF - tau_low) / (1 - tau_low), clipped to [0, 1]
        denom = max(1.0 - tau_low, 1e-6)
        cell_push = np.zeros(len(frozen))
        for (i, j), cf in raw_map.items():
            if cf <= tau_low or labels[i] != labels[j]:
                continue
            ramp_val = min(1.0, (cf - tau_low) / denom)
            if not frozen[i] and frozen[j]:
                cell_push[i] += ramp_val
            elif frozen[i] and not frozen[j]:
                cell_push[j] += ramp_val

        pushed = cell_push[unfrozen_idx]
        pushed_nonzero = pushed[pushed > 0]

        # Quality gate: calibration requires frozen same-label pairs to exist in
        # sufficient numbers so we can assess GMM reliability.  Two conditions
        # must both pass before accepting calibrated lam:
        #
        #   1. len(same_frozen) >= min_pairs  — enough frozen data to evaluate
        #   2. frac_frozen_violation <= 0.20  — frozen labels are trustworthy
        #
        # If either fails, the push estimate is unreliable (too few frozen cells,
        # or too many frozen cells already in violation due to a poor GMM fit).
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
) -> tuple[np.ndarray, int, dict[int, int]]:
    """Split clusters whose cells form multiple disconnected spatial groups.

    Connectivity is tested with a k-NN graph built within each cluster's
    own cells, so the check is density-adaptive and independent of any global
    radius.  k_nn=5 is enough to keep well-spread mosaics connected while
    still detecting genuine spatial fragmentation.

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
        k_actual = min(k_nn + 1, n_k)  # +1 because query includes self at index 0
        _, nbr_idx = KDTree(sub_centers).query(sub_centers, k=k_actual)
        # Build undirected k-NN adjacency on local indices 0..n_k-1
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
) -> tuple[np.ndarray, int]:
    """Jointly resolve same-label repulsive cell pairs by 2-cell energy optimization.

    ICM is a 1-cell-at-a-time algorithm and cannot resolve the case where two
    cells compete for the same spatial position: if both are frozen (or if one
    cell's correct assignment depends on the other's), ICM is blind to the joint
    solution.  This step explicitly solves each same-label repulsive pair as a
    small K×K sub-problem.

    For every pair (i, j) where labels[i] == labels[j] and w_ij > 0 (the pair
    is in the repulsive zone — a genuine territorial conflict), find the joint
    label assignment (k_i, k_j) that minimises:

        E_joint(k_i, k_j) = E_base(i, k_i) + E_base(j, k_j)
                           + lam_eff × w_ij × [k_i == k_j]

    where E_base(i, k) = unary[i,k] + lam_eff × Σ_{l≠j, nbrs(i)} w_il × [labels[l]==k]
    counts each (i,j) edge exactly once.  Pairs processed descending by weight.
    Bypasses frozen mask by design.

    **Two-phase strategy:**

    Phase 1 uses the standard ``lam``.  For feature impostors (cells whose
    feature vector falls inside a wrong type's GMM distribution), the unary
    gap can exceed ``lam × w_ij`` even for a clear territorial conflict, so the
    Phase 1 swap criterion is not met.

    Phase 2 re-examines high-weight pairs (w_ij ≥ ``lam_boost_min_w``) with
    ``lam_eff = lam × lam_boost``.  Scaling up both the attractive same-tile
    contacts (CF < τ_low, negative weight) and the repulsive violation contacts
    amplifies the spatial signal relative to the unary.  A cell that genuinely
    belongs to type Y has more CF < τ_low contacts with Y-labeled neighbours
    than with the wrong type, so boosting λ tips the balance in its favour even
    when the feature confidence is misleading.

    Returns (labels, n_swaps_total).
    """
    labels = labels.copy()
    n_cells, K = unary.shape
    n_swaps_total = 0

    def _run_rounds(lam_eff: float, min_w: float) -> int:
        nonlocal labels
        total = 0
        for _ in range(max_rounds):
            # Collect same-label repulsive pairs above the weight threshold.
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

                best_e = curr_e
                best_ki, best_kj = k_curr, k_curr
                for ki in range(K):
                    for kj in range(K):
                        cross = lam_eff * w_ij if ki == kj else 0.0
                        e = e_i[ki] + e_j[kj] + cross
                        if e < best_e:
                            best_e = e
                            best_ki, best_kj = ki, kj

                if best_ki != k_curr or best_kj != k_curr:
                    labels[i] = best_ki
                    labels[j] = best_kj
                    n_this += 1

            total += n_this
            if n_this == 0:
                break
        return total

    # Phase 1: standard lambda — resolves most conflicts via normal energy balance
    n_swaps_total += _run_rounds(lam, min_w=0.0)

    # Phase 2: boosted lambda for high-weight pairs (very clear territorial overlaps)
    # Amplifies same-tile attractive contacts to overcome large unary gaps caused by
    # feature impostors.  Restricted to w ≥ lam_boost_min_w to avoid over-correcting
    # borderline contacts in high-overlap configs.
    if lam_boost > 1.0:
        n_swaps_total += _run_rounds(lam * lam_boost, min_w=lam_boost_min_w)

    return labels, n_swaps_total


def _label_residual_violators(
    labels: np.ndarray,
    nbrs: list[list[tuple[int, float]]],
    unary: NDArray | None = None,
    min_posterior: float = 0.01,
    max_rounds: int = 5,
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
            curr_score = sum(w for j, w in nbrs[i] if w > 0 and labels[j] == labels[i])
            if curr_score == 0:
                continue

            # Violation score for each label k
            scores = np.zeros(K)
            for j, w in nbrs[i]:
                if w > 0 and 0 <= labels[j] < K:
                    scores[int(labels[j])] += w

            min_scr = float(scores.min())
            old_k   = int(labels[i])

            if min_scr > 0:
                # No spatially clean label exists → unlabel
                labels[i] = -1
                n_unlabeled += 1
                n_this += 1
                continue

            # Among zero-violation labels, pick the one with best feature fit
            candidates = np.where(scores == 0.0)[0]
            if unary is not None and len(candidates) > 1:
                best_k = int(candidates[int(np.argmin(unary[i][candidates]))])
            else:
                best_k = int(candidates[0])

            # Feature guard: require at least min_posterior probability
            if unary is not None:
                log_liks = -unary[i].copy()
                log_liks -= log_liks.max()          # numerical stability
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
    ``min_adj_pairs`` spatially adjacent cell pairs in ``raw_map``.  Pairs
    with no spatial neighbourhood have no grounding and are skipped; without
    this gate, non-adjacent different-type clusters close in feature space
    get merged incorrectly.

    Gate 2 — fraction veto: only applied when there are at least
    ``min_pairs_for_veto`` adjacent pairs (so the fraction estimate is
    statistically meaningful).  Blocks the merge if the fraction of adjacent
    (frozen-first) pairs with CF >= tau_low is >= ``veto_frac``.  With fewer
    pairs the fraction is too noisy to trust — a single outlier in 3 pairs
    gives 33 %, well above the 25 % threshold.

    When ``frozen`` is supplied the veto is evaluated on frozen-cell pairs only,
    falling back to all adjacent pairs when fewer than ``min_frozen_pairs``
    such pairs exist.
    Labels in the result are remapped to 0..K_final-1.

    Returns ``(labels, merge_history, active)`` where ``active[k]`` is the
    original sub-cluster index that became final cluster ``k`` (the surviving
    representative used to look up the GMM unary column for a second ICM pass).
    """
    labels = labels.copy()
    active = sorted(np.unique(labels).tolist())
    merge_history: list[tuple[int, int, float]] = []

    while len(active) > K_target:
        means = {k: X[labels == k].mean(axis=0) for k in active}

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
                dist = float(np.linalg.norm(means[a] - means[b]))
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
        # Leiden gave wrong cluster count — fall through to k-means
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
    n_clusters_init:
        Number of GMM components for initialisation.  When larger than
        ``n_clusters`` the GMM is deliberately over-specified; ICM then refines
        the boundaries at that finer resolution, and the merge step collapses
        the result back to ``n_clusters``.  Requires ``split_merge=True`` to
        activate the merge step.  ``None`` (default) uses ``n_clusters``.
    init:
        Method used to compute the starting means for the GMM.  Options:
        ``"kmeans"`` (default) — k-means centroids, matches sklearn's default
        GMM init but made explicit; ``"leiden"`` — Leiden community detection
        on a feature k-NN graph, better for uneven densities; ``"agglomerative"``
        — Ward hierarchical clustering, deterministic and good for elongated
        clusters; ``"random"`` — randomly sampled data points, mainly for
        ablation.  If Leiden does not produce exactly ``n_clusters_init``
        communities, it falls back to k-means and records ``"leiden→kmeans"``
        in the model dict under ``init_method_used``.
    leiden_k_features:
        k-NN neighbourhood size for the feature graph used by the Leiden init.
        Only used when ``init="leiden"``.  Default 15.
    leiden_resolution:
        Leiden resolution parameter.  ``None`` (default) auto-calibrates via
        binary search to target ``n_clusters_init`` communities.  Only used
        when ``init="leiden"``.
    """

    def __init__(
        self,
        n_clusters: int,
        spatial_radius: float = 20.0,
        lam: float | None = None,
        lam_alpha: float = 2.0,
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
        n_clusters_init: int | None = None,
        merge_veto_frac: float = 0.40,
        merge_min_adj_pairs: int = 0,
        merge_min_pairs_for_veto: int = 10,
        init: str = "kmeans",
        leiden_k_features: int = 15,
        leiden_resolution: float | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.spatial_radius = spatial_radius
        self.lam = lam
        self.lam_alpha = lam_alpha
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
        self.n_clusters_init = n_clusters_init
        self.merge_veto_frac = merge_veto_frac
        self.merge_min_adj_pairs = merge_min_adj_pairs
        self.merge_min_pairs_for_veto = merge_min_pairs_for_veto
        self.init = init
        self.leiden_k_features = leiden_k_features
        self.leiden_resolution = leiden_resolution

    def fit(self, dataset: MosaicDataset) -> ClusteringResult:
        X = dataset.X
        K = self.n_clusters
        K_init = self.n_clusters_init if self.n_clusters_init is not None else K
        n_cells = len(dataset)

        # --- GMM: initial labels + posteriors ---
        # When leiden_resolution is explicit and n_clusters_init is unset,
        # accept whatever Leiden returns rather than constraining to K.
        free_leiden = (
            self.init == "leiden"
            and self.leiden_resolution is not None
            and self.n_clusters_init is None
        )
        means_init, init_method_used = _init_gmm_means(
            X, K_init, self.init, self.random_state,
            leiden_k_features=self.leiden_k_features,
            leiden_resolution=self.leiden_resolution,
            require_k=not free_leiden,
        )
        if free_leiden:
            K_init = len(means_init)
        if init_method_used != self.init:
            init_method_used = f"{self.init}→{init_method_used}"
        gmm = GaussianMixture(
            n_components=K_init,
            covariance_type="full",
            n_init=3,
            random_state=self.random_state,
            means_init=means_init,
            init_params="random",
        )
        gmm.fit(X)
        labels = gmm.predict(X).astype(np.int64)
        posteriors = gmm.predict_proba(X)  # (n_cells, K_init) — used for freezing

        # Unary cost = negative unnormalized log-posterior
        #   = -[log P(X_i | cluster k) + log pi_k]
        # This is equivalent to log P(cluster k | X_i) up to the per-cell
        # normalizing constant (which cancels in argmin).  Including log pi_k
        # ensures ICM at lam=0 reproduces the GMM's initial assignment exactly,
        # preventing false flips for cells where mixing-weight differences matter.
        log_liks = np.zeros((n_cells, K_init))
        for k in range(K_init):
            mvn = multivariate_normal(mean=gmm.means_[k], cov=gmm.covariances_[k])
            log_liks[:, k] = mvn.logpdf(X)
        unary = -(log_liks + np.log(gmm.weights_))  # (n_cells, K_init)

        # --- Spatial pairwise weights (coverage fraction + ramp) ---
        raw_map = _build_coverage_map(
            dataset.polygons, dataset.centers, self.spatial_radius,
            dataset.clipped, self.exclude_clipped,
            n_workers=self.n_workers,
        )

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
                )
                n_splits = K_eff - K_init
                if K_eff > K_init:
                    extra = np.column_stack(
                        [unary[:, _parent_map[k]] for k in range(K_init, K_eff)]
                    )
                    unary = np.hstack([unary, extra])

        # --- Freeze high-confidence cells ---
        # Computed here (before ramp) so _calibrate_tau can use frozen status.
        frozen: NDArray = posteriors.max(axis=1) >= self.conf_threshold

        # --- Calibrate tau thresholds and lambda ---
        # Use the frozen-cell CF distribution as a proxy for true same/diff-type.
        # Individual thresholds / lam can be fixed by passing explicit values;
        # None means calibrate from data with a graceful fallback hierarchy.
        tau_calib: dict = {}
        tau_low  = self.tau_low
        tau_high = self.tau_high
        cal_lam: float | None = None
        if tau_low is None or tau_high is None or self.lam is None:
            cal_low, cal_high, cal_lam, tau_calib = _calibrate_tau(
                raw_map, frozen, labels, unary,
                signed_ramp=self.signed_ramp,
                lam_alpha=self.lam_alpha,
            )
            if tau_low is None:
                tau_low = cal_low
            if tau_high is None:
                tau_high = cal_high

        # Resolve effective lambda: explicit > calibrated > hard fallback
        lam: float = (
            self.lam if self.lam is not None
            else cal_lam if cal_lam is not None
            else 20.0
        )
        tau_calib["lam"] = lam

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
        # for genuinely same-type pairs; at least one cell is wrong.
        #
        # Both cells are unfrozen (not just the less-confident one).  The
        # "less-confident" heuristic fails for feature impostors: a cell whose
        # features fall squarely inside the wrong type's distribution gets frozen
        # with HIGHER confidence than its correctly-assigned same-type neighbour,
        # so the old heuristic would unfreeze the correct cell and leave the
        # impostor locked.  Unfreezing both lets the spatial pairwise term — which
        # correctly sees the high-CF violation — dominate in ICM and flip the
        # impostor, while the correct neighbour's low-CF same-tile contacts keep
        # it stable.
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
                    continue
                if frozen[i]:
                    frozen[i] = False
                    n_force_unfrozen += 1
                if frozen[j]:
                    frozen[j] = False
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
                pairwise = lam * (W @ one_hot)
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
                        e_local[labels[j]] += lam * w
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

        _labels_post_icm = labels.copy()

        # --- Merge sub-clusters back to n_clusters (Option A) ---
        merge_history: list[tuple[int, int, float]] = []
        surviving_subclusters: list[int] = list(range(K_eff))
        _labels_post_merge = _labels_post_icm  # overwritten below if merge occurs
        n_unfrozen_merge = 0
        n_iters_post_merge = 0
        n_unfrozen_h1_post = 0
        # unary in the final K-dimensional label space (updated inside merge block)
        _unary_final: NDArray = unary
        _labels_post_cleanup = _labels_post_icm  # overwritten after cleanup
        _labels_post_conflict = _labels_post_icm  # overwritten after conflict resolution
        n_swaps = 0
        n_reassigned = 0
        n_unlabeled = 0

        if self.split_merge and K_eff > K:
            labels, merge_history, surviving_subclusters = _merge_clusters_to_k(
                labels, X, raw_map, tau_low, K,
                frozen=frozen,
                veto_frac=self.merge_veto_frac,
                min_adj_pairs=self.merge_min_adj_pairs,
                min_pairs_for_veto=self.merge_min_pairs_for_veto,
            )
            n_merges = K_eff - len(np.unique(labels))
            _labels_post_merge = labels.copy()  # before second ICM cleanup

            # --- Post-merge cleanup (M1): unfreeze merge-boundary violators ---
            # Cells that were in different sub-clusters before merging but are now
            # same-labeled with CF >= tau_low are new violations created by the merge.
            # The frozen mask is stale for these cells (calibrated on K_eff sub-cluster
            # posteriors, not on K merged-cluster posteriors).  Unfreeze them so the
            # second ICM pass can correct them.
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

            # Build K-dimensional mixture unary for the second ICM.
            # For each final cluster k, pool ALL original sub-cluster GMM components
            # that were merged into k.  Using only the surviving representative's
            # column gives wrong unary scores for cells from absorbed sub-clusters
            # (e.g. both crescent arms of a moon share the same type but are far apart
            # in GMM-component space; absorbed-arm cells would score poorly under the
            # surviving arm's component).
            #
            # component_map[orig_label] = set of all original sub-cluster indices
            # (0..K_eff-1) that were folded into that cluster by the merge sequence.
            component_map: dict[int, set] = {k: {k} for k in range(K_eff)}
            for a_h, b_h, _ in merge_history:
                if b_h in component_map:
                    component_map[a_h] = component_map[a_h] | component_map.pop(b_h)

            # unary[:, c] = -(log p(X | component_c) + log w_c)
            # → -unary[:, c] = log-unnormalized-likelihood for component c
            # mixture unary for cluster k = -log sum_{c in components(k)} exp(-unary[:, c])
            K_final = len(surviving_subclusters)
            unary_k = np.zeros((n_cells, K_final))
            for new_k, old_k in enumerate(surviving_subclusters):
                components = sorted(component_map.get(old_k, {old_k}))
                log_mix = -unary[:, components[0]].copy()
                for c in components[1:]:
                    log_mix = np.logaddexp(log_mix, -unary[:, c])
                unary_k[:, new_k] = -log_mix

            # --- Second ICM pass (sequential, starts near-convergence) ---
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

            # --- H1 fallback: unfreeze remaining CF > theta_hard violations ---
            # Mirrors the pre-ICM H1 pass (§13c) applied to the post-merge labels.
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
            _unary_final = unary_k  # K-dimensional mixture unary for conflict step

        # --- Pairwise conflict resolution (step 6) ---
        # Jointly resolve same-label repulsive pairs that ICM could not correct
        # (e.g. feature impostors competing for the same spatial position).
        # Bypasses the frozen mask by design — see _resolve_conflicting_pairs.
        _labels_post_cleanup = labels.copy()
        labels, n_swaps = _resolve_conflicting_pairs(labels, _unary_final, nbrs, lam)

        # --- Residual violator assignment (step 7) ---
        # Any cell still in a same-label repulsive contact after conflict resolution
        # is re-assigned purely by spatial compatibility (minimum violation score).
        # If no label gives a clean fit, the cell is labelled -1 (unlabeled).
        _labels_post_conflict = labels.copy()
        labels, n_reassigned, n_unlabeled = _label_residual_violators(
            labels, nbrs, unary=_unary_final,
        )

        return ClusteringResult(
            labels=labels,
            model={
                "gmm": gmm,
                "lam": lam,
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
                "n_clusters_init": K_init,
                "init_method": self.init,
                "init_method_used": init_method_used,
                "n_splits": n_splits,
                "n_merges": n_merges,
                "k_after_split": K_eff,
                "labels_gmm_raw": _labels_gmm,
                "labels_post_icm": _labels_post_icm,
                "labels_post_merge": _labels_post_merge,
                "labels_post_cleanup": _labels_post_cleanup,
                "labels_post_conflict": _labels_post_conflict,
                "n_swaps": n_swaps,
                "n_reassigned": n_reassigned,
                "n_unlabeled": n_unlabeled,
                "parent_map": dict(_parent_map),
                "merge_history": merge_history,
                "n_unfrozen_merge": n_unfrozen_merge,
                "n_iters_post_merge": n_iters_post_merge,
                "n_unfrozen_h1_post": n_unfrozen_h1_post,
            },
        )
