import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.stats import mode


def plot_mosaics(groups, polygons, centers, mode='basic', highlight=None,
                 iou_matrix=None, iou_threshold=0.05, ncols=4):
    """
    Plot individual mosaics with multiple visualization modes.

    Parameters
    ----------
    groups : array-like
        Group assignments for each polygon
    polygons : array-like
        Polygon coordinates for each cell
    centers : array-like
        Center coordinates for each cell
    mode : str, default 'basic'
        Plotting mode: 'basic', 'violations', 'iou'
    highlight : array-like, optional
        Boolean mask to highlight specific polygons (for 'basic' mode)
    iou_matrix : array-like, optional
        IoU matrix (required for 'violations' and 'iou' modes)
    iou_threshold : float, default 0.05
        Threshold for violation detection (for 'violations' mode)
    ncols : int, default 4
        Number of columns in grid layout

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    u_groups = np.unique(groups)
    n_mosaics = u_groups.size
    ncols = np.min([ncols, n_mosaics])
    nrows = int(np.ceil(n_mosaics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_mosaics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, group in enumerate(u_groups):
        ax = axes[idx]
        gi = groups == group
        polygons_i = polygons[gi]
        centers_i = centers[gi]

        if mode == 'basic':
            _plot_basic_group(ax, polygons_i, centers_i, colors[idx], highlight[gi] if highlight is not None else None)
        elif mode == 'violations':
            if iou_matrix is None:
                raise ValueError("iou_matrix required for 'violations' mode")
            violation_mask = _get_violation_mask(gi, iou_matrix, iou_threshold)
            _plot_basic_group(ax, polygons_i, centers_i, colors[idx], violation_mask)
        elif mode == 'iou':
            if iou_matrix is None:
                raise ValueError("iou_matrix required for 'iou' mode")
            max_ious = _get_max_iou_per_cell(gi, iou_matrix)
            _plot_iou_group(ax, polygons_i, centers_i, max_ious)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'basic', 'violations', 'iou'")

        ax.set_xlim(-10, 110)
        ax.set_ylim(-10, 110)
        ax.set_aspect('equal')
        ax.set_title(f'Mosaic {idx}\n{len(polygons_i)} cells', fontsize=10)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')

    # Hide unused axes
    for idx in range(n_mosaics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig, axes


def _plot_basic_group(ax, polygons, centers, color, highlight):
    """Plot a single group with optional highlighting."""
    if highlight is not None:
        highlight_patches = [Polygon(poly, closed=True) for i, poly in enumerate(polygons) if highlight[i]]
        other_patches = [Polygon(poly, closed=True) for i, poly in enumerate(polygons) if not highlight[i]]
    else:
        highlight_patches = []
        other_patches = [Polygon(poly, closed=True) for poly in polygons]

    # Plot regular cells
    if other_patches:
        collection = PatchCollection(
            other_patches,
            facecolors=color,
            edgecolors='black',
            linewidths=1.5,
            alpha=0.7
        )
        ax.add_collection(collection)

    # Plot highlighted cells
    if highlight_patches:
        highlighted_collection = PatchCollection(
            highlight_patches,
            facecolors=color,
            edgecolors='red',
            linewidths=2.0,
            alpha=0.5
        )
        ax.add_collection(highlighted_collection)

    ax.scatter(centers[:, 0], centers[:, 1], c='darkred', s=10, zorder=5, alpha=0.6)


def _plot_iou_group(ax, polygons, centers, max_ious):
    """Plot a single group with IoU color-coding."""
    patches = [Polygon(poly, closed=True) for poly in polygons]

    collection = PatchCollection(
        patches,
        edgecolors='black',
        linewidths=1.5,
        alpha=0.7,
        cmap='turbo',
    )
    collection.set_array(max_ious)
    collection.set_clim(0.0, 1.0)
    ax.add_collection(collection)

    # Add colorbar
    cbar = plt.colorbar(collection, ax=ax)
    cbar.set_label('Max IoU')

    ax.scatter(centers[:, 0], centers[:, 1], c='darkred', s=10, zorder=5, alpha=0.6)


def _get_violation_mask(group_mask, iou_matrix, iou_threshold):
    """Get mask of cells that violate IoU threshold within their group."""
    indices = np.where(group_mask)[0]

    # Extract submatrix for this group
    if isinstance(iou_matrix, np.ndarray):
        sub_iou = iou_matrix[np.ix_(group_mask, group_mask)]
    else:  # Sparse matrix
        sub_iou = iou_matrix[group_mask, :][:, group_mask].toarray()

    # Find max IoU for each cell (excluding self)
    np.fill_diagonal(sub_iou, 0)
    max_iou_per_cell = sub_iou.max(axis=1)

    # Create violation mask for the full array
    violation_mask = max_iou_per_cell > iou_threshold
    return violation_mask


def _get_max_iou_per_cell(group_mask, iou_matrix):
    """Get maximum IoU value for each cell within their group."""
    # Extract submatrix for this group
    if isinstance(iou_matrix, np.ndarray):
        sub_iou = iou_matrix[np.ix_(group_mask, group_mask)]
    else:  # Sparse matrix
        sub_iou = iou_matrix[group_mask, :][:, group_mask].toarray()

    # Find max IoU for each cell (excluding self)
    np.fill_diagonal(sub_iou, 0)
    max_iou_per_cell = sub_iou.max(axis=1)

    return max_iou_per_cell


def plot_blobs(X, y, y_true=None, assume_same_names=False):
    """Plot 2D feature blobs colored by class."""

    y_plot = y.copy()

    # Align predicted labels to true labels if needed
    if not assume_same_names and y_true is not None:
        label_mapping = {}
        for label in np.unique(y):
            true_labels = y_true[y == label]
            if len(true_labels):
                label_mapping[label] = mode(true_labels, keepdims=False).mode
            else:
                label_mapping[label] = label

        y_plot = np.vectorize(label_mapping.get)(y)

    plt.figure()

    for label in np.unique(y_plot):
        mask = y_plot == label
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            label=f"Class {label}"
        )

    # Highlight misclassified points if true labels are provided
    if y_true is not None:
        misclassified = y_plot != y_true
        plt.scatter(
            X[misclassified, 0],
            X[misclassified, 1],
            facecolors='none',
            edgecolors='k',
            label='Misclassified'
        )


    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Simulated Feature Blobs")
    plt.show()
