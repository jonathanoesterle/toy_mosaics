"""Shared figure helpers for generate_*.py scripts."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.optimize import linear_sum_assignment

from toy_mosaics.dataset import MosaicDataset

_TAB10 = plt.cm.tab10(np.linspace(0, 1, 10))
_TAB20 = plt.cm.tab20(np.linspace(0, 1, 20, endpoint=False))


def relabel(labels: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Remap labels to best match y_true via Hungarian algorithm.

    Cells with label -1 (unlabeled) are kept as -1 and excluded from the
    confusion matrix so they don't distort the permutation.
    """
    labels = np.asarray(labels)
    valid = labels >= 0
    if not valid.any():
        return labels.copy()
    n = int(max(labels[valid].max(), y_true[valid].max())) + 1
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true[valid], labels[valid]):
        conf[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {int(col_ind[i]): int(row_ind[i]) for i in range(len(row_ind))}
    return np.array(
        [-1 if lbl < 0 else mapping.get(int(lbl), int(lbl)) for lbl in labels],
        dtype=int,
    )


def _label_colors(
    unique_labels: list[int], parent_map: dict[int, int]
) -> dict[int, np.ndarray]:
    colors: dict[int, np.ndarray] = {}
    parent_sub_count: dict[int, int] = {}
    for lbl in sorted(unique_labels):
        if lbl in parent_map:
            parent = parent_map[lbl]
            sub_idx = parent_sub_count.get(parent, 0)
            parent_sub_count[parent] = sub_idx + 1
            colors[lbl] = _TAB20[(2 * parent + 1 + sub_idx * 2) % 20]
        else:
            colors[lbl] = _TAB20[(2 * lbl) % 20]
    return colors


_UNLABELED_COLOR = np.array([0.65, 0.65, 0.65, 1.0])  # gray for -1 cells


def _gmm_ellipse_2d(gmm, k: int, pca=None):
    """Return (mean_2d, cov_2d) for component k, projected via PCA if given."""
    mean = gmm.means_[k]
    ct = gmm.covariance_type
    if ct == "full":
        cov = gmm.covariances_[k]
    elif ct == "tied":
        cov = gmm.covariances_
    elif ct == "diag":
        cov = np.diag(gmm.covariances_[k])
    else:  # spherical
        cov = np.eye(len(mean)) * float(gmm.covariances_[k])

    if pca is not None:
        V = pca.components_          # (2, n_dims)
        mean_2d = V @ (mean - pca.mean_)
        cov_2d = V @ cov @ V.T
    else:
        mean_2d = mean[:2]
        cov_2d = cov[:2, :2]
    return mean_2d, cov_2d


def plot_mosaic_step(
    dataset: MosaicDataset,
    labels: np.ndarray,
    title: str,
    parent_map: dict[int, int] | None = None,
    gt_labels: np.ndarray | None = None,
    bounds: tuple = (0, 100, 0, 100),
    gmm=None,
) -> plt.Figure:
    """Feature scatter on the left, one spatial panel per cluster on the right.

    When parent_map is provided, tab20 paired shades show parent–child
    relationships; otherwise tab10 is used.  When gt_labels is provided and K
    matches, errors are marked with red X (spatial) and open circles (features).

    Cells with label -1 are shown in gray in the feature scatter and get their
    own "unlabeled" spatial panel at the right end.

    When gmm is provided (a fitted sklearn GaussianMixture), the feature scatter
    is annotated with one gray star per component (at the mean) surrounded by a
    light-gray filled 1-σ ellipse derived from the component covariance.
    """
    pm = parent_map or {}
    labels = np.asarray(labels)
    has_unlabeled = np.any(labels < 0)
    named = sorted(lbl for lbl in np.unique(labels).tolist() if lbl >= 0)
    colors = _label_colors(named, pm) if pm else {lbl: _TAB10[int(lbl) % 10] for lbl in named}

    polygons_arr = np.array(dataset.polygons, dtype=object)
    # Show error marks whenever GT is provided.  relabel() already permutes labels
    # to best match gt_labels via Hungarian assignment, so labels != gt_labels is
    # a valid mismatch signal regardless of K.  The -1 (unlabeled) cluster is
    # plotted in its own panel but excluded from the error computation via the
    # `labels >= 0` guard below.
    show_errors = gt_labels is not None

    # Reduce to 2D for the feature scatter when X has more than 2 dimensions.
    if dataset.X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2).fit(dataset.X)
        X_2d = pca.transform(dataset.X)
        feat_xlabel, feat_ylabel = "PC1", "PC2"
    else:
        pca = None
        X_2d = dataset.X
        feat_xlabel, feat_ylabel = "Feature 1", "Feature 2"

    x_min, x_max, y_min, y_max = bounds
    mx, my = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1

    n_panels = 1 + len(named) + (1 if has_unlabeled else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 4))
    axes = np.atleast_1d(axes)
    ax_f = axes[0]

    # Feature scatter — labeled clusters
    for lbl in named:
        mask = labels == lbl
        leg = f"{lbl} (<-{pm[lbl]})" if lbl in pm else str(lbl)
        ax_f.scatter(X_2d[mask, 0], X_2d[mask, 1],
                     color=colors[lbl], label=leg, s=20, alpha=0.85)
    # Unlabeled cells in gray
    if has_unlabeled:
        mask_u = labels < 0
        ax_f.scatter(X_2d[mask_u, 0], X_2d[mask_u, 1],
                     color=_UNLABELED_COLOR, label="-1 (unlabeled)", s=20, alpha=0.85, marker="x")
    if show_errors:
        errors = (labels != gt_labels) & (labels >= 0)
        if errors.any():
            ax_f.scatter(X_2d[errors, 0], X_2d[errors, 1],
                         facecolors="none", edgecolors="black", s=70, linewidths=1.5, zorder=5)
    # GMM component ellipses + stars
    if gmm is not None and dataset.X.shape[1] >= 2:
        from matplotlib.patches import Ellipse as _Ellipse
        for k in range(gmm.n_components):
            try:
                mean, cov = _gmm_ellipse_2d(gmm, k, pca=pca)
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 0.0)
                angle = float(np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1])))
                w = 2.0 * float(np.sqrt(eigvals[-1]))   # 1-σ diameter major axis
                h = 2.0 * float(np.sqrt(eigvals[-2]))   # 1-σ diameter minor axis
                ax_f.add_patch(_Ellipse(
                    xy=mean, width=w, height=h, angle=angle,
                    facecolor="lightgray", alpha=0.40,
                    edgecolor="gray", linewidth=1.2, zorder=1,
                ))
            except Exception:
                pass
            mean_star = pca.transform(gmm.means_[k:k+1])[0] if pca is not None else gmm.means_[k, :2]
            ax_f.scatter(*mean_star, color="gray", marker="*",
                         s=220, linewidths=0.7, edgecolors="dimgray", zorder=6)
    ax_f.set_xlabel(feat_xlabel); ax_f.set_ylabel(feat_ylabel)
    ax_f.set_title("Feature space")
    ax_f.legend(loc="best", fontsize=7, title="cluster")

    # Spatial panels — one per named label
    for idx, lbl in enumerate(named):
        ax = axes[1 + idx]
        mask = labels == lbl
        fg = [Polygon(poly, closed=True) for poly in polygons_arr[mask]]
        if fg:
            ax.add_collection(PatchCollection(
                fg, facecolors=colors[lbl], edgecolors="black", linewidths=0.8, alpha=0.8,
            ))
        n_err = 0
        if show_errors:
            err_mask = mask & (labels != gt_labels)
            n_err = int(err_mask.sum())
            if n_err:
                ax.scatter(dataset.centers[err_mask, 0], dataset.centers[err_mask, 1],
                           c="red", marker="x", s=50, zorder=6, linewidths=1.5)
        ax.set_xlim(x_min - mx, x_max + mx); ax.set_ylim(y_min - my, y_max + my)
        ax.set_aspect("equal")
        lbl_str = f"{lbl} (<-{pm[lbl]})" if lbl in pm else str(lbl)
        err_str = f"  ({n_err} err)" if n_err else ""
        ax.set_title(f"{lbl_str}\n{mask.sum()} cells{err_str}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # Extra panel for unlabeled cells
    if has_unlabeled:
        ax = axes[-1]
        mask_u = labels < 0
        fg = [Polygon(poly, closed=True) for poly in polygons_arr[mask_u]]
        if fg:
            ax.add_collection(PatchCollection(
                fg, facecolors=_UNLABELED_COLOR, edgecolors="black", linewidths=0.8, alpha=0.8,
            ))
        ax.set_xlim(x_min - mx, x_max + mx); ax.set_ylim(y_min - my, y_max + my)
        ax.set_aspect("equal")
        ax.set_title(f"-1 (unlabeled)\n{mask_u.sum()} cells", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
