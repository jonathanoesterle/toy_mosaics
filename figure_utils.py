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
    """Remap labels to best match y_true via Hungarian algorithm."""
    n = int(max(labels.max(), y_true.max())) + 1
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, labels):
        conf[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {int(col_ind[i]): int(row_ind[i]) for i in range(len(row_ind))}
    return np.array([mapping.get(int(lbl), int(lbl)) for lbl in labels], dtype=int)


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


def plot_mosaic_step(
    dataset: MosaicDataset,
    labels: np.ndarray,
    title: str,
    parent_map: dict[int, int] | None = None,
    gt_labels: np.ndarray | None = None,
    bounds: tuple = (0, 100, 0, 100),
) -> plt.Figure:
    """Feature scatter on the left, one spatial panel per cluster on the right.

    When parent_map is provided, tab20 paired shades show parent–child
    relationships; otherwise tab10 is used.  When gt_labels is provided and K
    matches, errors are marked with red X (spatial) and open circles (features).
    """
    pm = parent_map or {}
    unique = sorted(np.unique(labels).tolist())
    colors = _label_colors(unique, pm) if pm else {lbl: _TAB10[int(lbl) % 10] for lbl in unique}

    polygons_arr = np.array(dataset.polygons, dtype=object)
    same_k_as_gt = gt_labels is not None and len(unique) == len(np.unique(gt_labels))

    x_min, x_max, y_min, y_max = bounds
    mx, my = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1

    n_panels = 1 + len(unique)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 4))
    axes = np.atleast_1d(axes)
    ax_f = axes[0]

    # Feature scatter
    for lbl in unique:
        mask = labels == lbl
        leg = f"{lbl} (<-{pm[lbl]})" if lbl in pm else str(lbl)
        ax_f.scatter(dataset.X[mask, 0], dataset.X[mask, 1],
                     color=colors[lbl], label=leg, s=20, alpha=0.85)
    if same_k_as_gt:
        errors = labels != gt_labels
        if errors.any():
            ax_f.scatter(dataset.X[errors, 0], dataset.X[errors, 1],
                         facecolors="none", edgecolors="black", s=70, linewidths=1.5, zorder=5)
    ax_f.set_xlabel("Feature 1"); ax_f.set_ylabel("Feature 2")
    ax_f.set_title("Feature space")
    ax_f.legend(loc="best", fontsize=7, title="cluster")

    # Spatial panels
    for idx, lbl in enumerate(unique):
        ax = axes[1 + idx]
        mask = labels == lbl

        fg = [Polygon(poly, closed=True) for poly in polygons_arr[mask]]
        if fg:
            ax.add_collection(PatchCollection(
                fg, facecolors=colors[lbl], edgecolors="black", linewidths=0.8, alpha=0.8,
            ))
        n_err = 0
        if same_k_as_gt:
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

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
