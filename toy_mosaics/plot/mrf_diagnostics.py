"""Diagnostic visualizations for MRFMosaicStrategy results."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Polygon
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score

from toy_mosaics._result import ClusteringResult
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.mrf_mosaic import _build_coverage_map, _ramp

_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_labels(labels: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Remap labels to best match y_true via the Hungarian algorithm."""
    n = int(max(labels.max(), y_true.max())) + 1
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, labels):
        conf[t, p] += 1
    _, col = linear_sum_assignment(-conf)
    mapping = {int(col[i]): i for i in range(len(col))}
    return np.array([mapping.get(int(x), int(x)) for x in labels], dtype=int)


def _build_nbrs(
    dataset: MosaicDataset,
    tau_low: float,
    tau_high: float,
    spatial_radius: float,
    exclude_clipped: bool,
) -> list[list[tuple[int, float]]]:
    n = len(dataset)
    raw_map = _build_coverage_map(
        dataset.polygons, dataset.centers, spatial_radius,
        dataset.clipped, exclude_clipped,
    )
    nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), cf in raw_map.items():
        w = _ramp(cf, tau_low, tau_high)
        if w > 0.0:
            nbrs[i].append((j, w))
            nbrs[j].append((i, w))
    return nbrs


def _conflict_scores(labels: np.ndarray, nbrs: list) -> np.ndarray:
    c = np.zeros(len(labels))
    for i, neighbors in enumerate(nbrs):
        for j, w in neighbors:
            if labels[j] == labels[i]:
                c[i] += w
    return c


def _energy_gaps(
    gmm,
    X: np.ndarray,
    labels: np.ndarray,
    lam: float,
    nbrs: list,
) -> np.ndarray:
    K = gmm.n_components
    n = len(X)
    log_liks = np.zeros((n, K))
    for k in range(K):
        mvn = multivariate_normal(mean=gmm.means_[k], cov=gmm.covariances_[k])
        log_liks[:, k] = mvn.logpdf(X)
    unary = -(log_liks + np.log(gmm.weights_))
    gaps = np.zeros(n)
    for i, neighbors in enumerate(nbrs):
        e = unary[i].copy()
        for j, w in neighbors:
            e[labels[j]] += lam * w
        se = np.sort(e)
        if K >= 2:
            gaps[i] = se[1] - se[0]
    return gaps


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_mrf_diagnostics(
    dataset: MosaicDataset,
    result: ClusteringResult,
    *,
    ground_truth: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Six-panel diagnostic figure for MRFMosaicStrategy results.

    Parameters
    ----------
    dataset:
        The dataset the strategy was fitted on.
    result:
        Output of MRFMosaicStrategy.fit(dataset).
    ground_truth:
        True cluster labels, shape (n_cells,).  When provided, errors are
        highlighted and the neighbourhood bar chart is populated.

    Returns
    -------
    fig, axes:
        Matplotlib figure and 1-D array of 6 Axes (2×3 grid, row-major).

    Panel layout
    ------------
    [0] Conflict–confidence scatter   [1] Spatial energy-gap map     [2] Feature scatter + conflict
    [3] Neighbour composition bars    [4] ICM label changes           [5] Run summary
    """
    model = result.model
    gmm = model["gmm"]
    lam = float(model["lam"])
    tau_low = float(model.get("tau_low", 0.10))
    tau_high = float(model.get("tau_high", 0.40))
    spatial_radius = float(model.get("spatial_radius", 20.0))
    exclude_clipped = bool(model.get("exclude_clipped", True))
    conf_threshold = float(model.get("conf_threshold", 0.90))
    frozen_stored: np.ndarray | None = model.get("frozen")
    labels_initial: np.ndarray | None = model.get("labels_initial")

    labels = result.labels
    X = dataset.X
    n = len(dataset)
    K = gmm.n_components

    # --- shared intermediates ---
    nbrs = _build_nbrs(dataset, tau_low, tau_high, spatial_radius, exclude_clipped)
    conflict = _conflict_scores(labels, nbrs)
    gaps = _energy_gaps(gmm, X, labels, lam, nbrs)
    posteriors = gmm.predict_proba(X)
    max_post = posteriors.max(axis=1)

    frozen = frozen_stored if frozen_stored is not None else (max_post >= conf_threshold)

    labels_al = _align_labels(labels, ground_truth) if ground_truth is not None else labels
    labels_init_al = (
        _align_labels(labels_initial, ground_truth)
        if (ground_truth is not None and labels_initial is not None)
        else labels_initial
    )

    fig, axes2d = plt.subplots(2, 3, figsize=(18, 12))
    ax = axes2d.flatten()

    # ── [0] Conflict–confidence scatter ──────────────────────────────────────
    if ground_truth is not None:
        correct = labels_al == ground_truth
        for is_frz, marker, sz in [(True, "s", 60), (False, "o", 25)]:
            m = frozen == is_frz
            fl = "frozen" if is_frz else "unfrozen"
            ax[0].scatter(
                max_post[m & correct], conflict[m & correct],
                marker=marker, c="#2e7d32", alpha=0.45, s=sz, label=f"correct ({fl})",
            )
            ax[0].scatter(
                max_post[m & ~correct], conflict[m & ~correct],
                marker=marker, c="#c62828", alpha=0.9, s=sz * 2, zorder=5,
                label=f"error ({fl})",
            )
    else:
        sc0 = ax[0].scatter(max_post, conflict, c=conflict, cmap="YlOrRd", alpha=0.7, s=25)
        fig.colorbar(sc0, ax=ax[0], label="Conflict score")

    ax[0].axvline(conf_threshold, color="navy", ls="--", lw=1.0,
                  label=f"conf_threshold={conf_threshold}")
    ymax = max(float(conflict.max()) * 1.05, 0.05)
    ax[0].set_ylim(-0.02 * ymax, ymax)
    ax[0].text(0.02, 0.97, "Type-B\n(no signal)", transform=ax[0].transAxes,
               va="top", ha="left", fontsize=7, color="#888888")
    ax[0].text(0.98, 0.97, "ICM\nsweet spot", transform=ax[0].transAxes,
               va="top", ha="right", fontsize=7, color="#2a6496")
    ax[0].set_xlabel("GMM max-posterior")
    ax[0].set_ylabel("Spatial conflict score")
    ax[0].set_title("Conflict–confidence scatter")
    ax[0].legend(fontsize=7, loc="upper left")

    # ── [1] Spatial energy-gap map ───────────────────────────────────────────
    patches1 = [Polygon(poly, closed=True) for poly in dataset.polygons]
    coll1 = PatchCollection(patches1, cmap="RdYlGn", alpha=0.85,
                            edgecolors="black", linewidths=0.3)
    coll1.set_array(gaps)
    ax[1].add_collection(coll1)
    fig.colorbar(coll1, ax=ax[1], label="Energy gap (larger = more stable)")
    if ground_truth is not None:
        errs1 = np.where(labels_al != ground_truth)[0]
        ax[1].scatter(dataset.centers[errs1, 0], dataset.centers[errs1, 1],
                      c="black", s=60, marker="x", linewidths=1.5, zorder=6)
    ax[1].set_xlim(-10, 110)
    ax[1].set_ylim(-10, 110)
    ax[1].set_aspect("equal")
    ax[1].set_title("Spatial energy-gap map  (× = remaining error)")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    # ── [2] Feature scatter + conflict overlay ────────────────────────────────
    sc2 = ax[2].scatter(X[:, 0], X[:, 1], c=conflict, cmap="YlOrRd", s=20, alpha=0.8, zorder=3)
    fig.colorbar(sc2, ax=ax[2], label="Conflict score")
    if X.shape[1] == 2:
        x0, x1 = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
        y0, y1 = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
        xx, yy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y0, y1, 120))
        grid_proba = gmm.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = grid_proba.reshape(120, 120, K)
        for k in range(K):
            ax[2].contour(xx, yy, Z[:, :, k], levels=[0.5],
                          colors=["navy"], linewidths=0.8, alpha=0.6)
    if ground_truth is not None:
        errs2 = np.where(labels_al != ground_truth)[0]
        ax[2].scatter(X[errs2, 0], X[errs2, 1], facecolors="none",
                      edgecolors="black", s=100, linewidths=1.5, zorder=5)
    ax[2].set_title("Feature space + conflict  (navy = GMM decision boundary)")
    ax[2].set_xlabel("Feature 1")
    ax[2].set_ylabel("Feature 2")

    # ── [3] Neighbourhood composition bars ───────────────────────────────────
    if ground_truth is not None:
        errs3 = np.where(labels_al != ground_truth)[0].tolist()
        if not errs3:
            ax[3].text(0.5, 0.5, "No errors — all cells correct",
                       transform=ax[3].transAxes, ha="center", va="center", fontsize=11)
        else:
            top_errs = sorted(errs3, key=lambda i: conflict[i], reverse=True)[:8]
            type_colors = [_COLORS[k % 10] for k in range(K)]
            bh = 0.30
            ypos = np.arange(len(top_errs), dtype=float)

            for bi, ci in enumerate(top_errs):
                nbr_cells = [j for j, _ in nbrs[ci]]
                for row_off, lbl_src in [(-bh / 2, labels_al), (bh / 2, labels_init_al)]:
                    if lbl_src is None:
                        continue
                    counts = np.zeros(K)
                    for j in nbr_cells:
                        k = int(lbl_src[j])
                        if 0 <= k < K:
                            counts[k] += 1
                    total = max(counts.sum(), 1.0)
                    left = 0.0
                    for k in range(K):
                        frac = counts[k] / total
                        if frac > 0:
                            alpha = 0.9 if row_off < 0 else 0.35
                            ax[3].barh(ypos[bi] + row_off, frac, bh, left=left,
                                       color=type_colors[k], alpha=alpha,
                                       label=f"type {k}" if (bi == 0 and row_off < 0) else "")
                            left += frac

            ax[3].set_yticks(ypos)
            ax[3].set_yticklabels(
                [f"cell {ci}  (true:{int(ground_truth[ci])}, pred:{int(labels_al[ci])})"
                 for ci in top_errs],
                fontsize=7,
            )
            ax[3].set_xlim(0, 1)
            ax[3].set_xlabel("Fraction of spatial neighbours by type")
            legend_handles = (
                [Patch(color=type_colors[k]) for k in range(K)]
                + [Patch(color="grey", alpha=0.9), Patch(color="grey", alpha=0.35)]
            )
            legend_labels = (
                [f"type {k}" for k in range(K)]
                + ["after ICM", "before ICM"]
            )
            ax[3].legend(legend_handles, legend_labels, fontsize=7, loc="lower right")
        ax[3].set_title(
            "Neighbour type composition\n(dark=after ICM, light=before; top errors by conflict)"
        )
    else:
        ax[3].text(0.5, 0.5, "No ground truth provided\n(pass ground_truth= to enable)",
                   transform=ax[3].transAxes, ha="center", va="center",
                   fontsize=11, color="grey")
        ax[3].set_title("Neighbour composition (needs ground_truth)")

    # ── [4] ICM label changes in feature space ────────────────────────────────
    if labels_initial is not None:
        ax[4].scatter(X[:, 0], X[:, 1], c="#cccccc", s=10, zorder=1, alpha=0.5)
        changed = np.where(labels != labels_initial)[0]
        for ci in changed:
            old_k = int(labels_initial[ci])
            new_k = int(labels[ci])
            ax[4].scatter(X[ci, 0], X[ci, 1],
                          c=[_COLORS[old_k % 10]], s=60, zorder=4)
            ax[4].scatter(X[ci, 0], X[ci, 1],
                          facecolors="none", edgecolors=_COLORS[new_k % 10],
                          s=130, linewidths=2, zorder=5)
        for k in range(K):
            ax[4].scatter(*gmm.means_[k], marker="*", c=[_COLORS[k % 10]],
                          s=220, zorder=6, edgecolors="black", linewidths=0.5)
        if ground_truth is not None:
            errs4 = np.where(labels_al != ground_truth)[0]
            ax[4].scatter(X[errs4, 0], X[errs4, 1],
                          facecolors="none", edgecolors="black",
                          s=200, linewidths=1.5, zorder=7, marker="D")
        ax[4].set_title(
            f"ICM label changes: {len(changed)} cells moved\n"
            "(fill=old label, ring=new; ★=cluster mean; ◇=remaining error)"
        )
    else:
        ax[4].text(0.5, 0.5, "labels_initial not in model dict",
                   transform=ax[4].transAxes, ha="center", va="center",
                   fontsize=11, color="grey")
        ax[4].set_title("ICM label changes")
    ax[4].set_xlabel("Feature 1")
    ax[4].set_ylabel("Feature 2")

    # ── [5] Run summary ───────────────────────────────────────────────────────
    ax[5].axis("off")
    lines = ["MRF Run Summary", "─" * 28]
    if ground_truth is not None:
        ari = adjusted_rand_score(ground_truth, labels_al)
        n_err = int((labels_al != ground_truth).sum())
        lines += [f"ARI:       {ari:.4f}", f"errors:    {n_err} / {n}", ""]
    lines += [
        f"violations: {model.get('violations_before', '?')} → {model.get('violations_after', '?')}",
        f"n_changed:  {model.get('n_changed', '?')}",
        f"n_iters:    {model.get('n_iters', '?')}",
        f"n_frozen:   {model.get('n_frozen', '?')}",
        "",
        "Parameters:",
        f"  lam:       {lam}",
        f"  tau_low:   {tau_low}",
        f"  tau_high:  {tau_high}",
        f"  radius:    {spatial_radius}",
        f"  conf_thr:  {conf_threshold}",
    ]
    ax[5].text(
        0.05, 0.95, "\n".join(lines), transform=ax[5].transAxes,
        va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.6),
    )
    ax[5].set_title("Run summary")

    fig.tight_layout()
    return fig, ax
