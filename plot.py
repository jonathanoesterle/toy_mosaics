import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Polygon as mpl_polygon


def plot_mosaics(groups, polygons, centers, highlight):
    """Plot all generated mosaics."""
    u_groups = np.unique(groups)
    n_mosaics = u_groups.size
    cols = min(3, n_mosaics)
    rows = (n_mosaics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_mosaics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.Set3(np.linspace(0, 1, n_mosaics))

    for idx, group in enumerate(u_groups):
        ax = axes[idx]

        gi = groups == group
        polygons_i = polygons[gi]
        centers_i = centers[gi]

        if highlight is not None:
            highlight_i = highlight[gi]
            highlight_patches = [Polygon(poly, closed=True) for i, poly in enumerate(polygons_i) if highlight_i[i]]
            other_patches = [Polygon(poly, closed=True) for i, poly in enumerate(polygons_i) if not highlight_i[i]]
        else:
            highlight_patches = []
            other_patches = [Polygon(poly, closed=True) for poly in polygons_i]

        # Plot unclipped cells
        if other_patches:
            collection = PatchCollection(
                other_patches,
                facecolors=colors[idx],
                edgecolors='black',
                linewidths=1.5,
                alpha=0.7
            )
            ax.add_collection(collection)

        # Plot clipped cells with different styling
        if highlight_patches:
            clipped_collection = PatchCollection(
                highlight_patches,
                facecolors=colors[idx],
                edgecolors='red',
                linewidths=2.0,
                alpha=0.5
            )
            ax.add_collection(clipped_collection)

        ax.scatter(centers_i[:, 0], centers_i[:, 1], c='darkred', s=10, zorder=5, alpha=0.6)

        ax.set_xlim(-10, 110)
        ax.set_ylim(-10, 110)
        ax.set_aspect('equal')

        ax.set_title(
            f'Mosaic {idx + 1}\n{len(polygons_i)} cells',
            fontsize=10
        )
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')

    for idx in range(n_mosaics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

def plot_spatial_mosaics(adata, hulls, cluster_key='mosaic_groups', ax=None, figsize=(10, 10)):
    """Plots all cell hulls colored by their assigned cluster."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    clusters = adata.obs[cluster_key].unique()
    # Use scanpy's default colors if available
    if f"{cluster_key}_colors" in adata.uns:
        colors = adata.uns[f"{cluster_key}_colors"]
        color_map = {c: colors[i] for i, c in enumerate(adata.obs[cluster_key].cat.categories)}
    else:
        import seaborn as sns
        palette = sns.color_palette("husl", len(clusters))
        color_map = {c: palette[i] for i, c in enumerate(clusters)}

    patches = []
    facecolors = []

    for i, hull in enumerate(hulls):
        cluster = adata.obs[cluster_key].iloc[i]
        poly = mpl_polygon(hull, closed=True)
        patches.append(poly)
        facecolors.append(color_map[cluster])

    p = PatchCollection(patches, facecolors=facecolors, alpha=0.6, edgecolors='white', linewidths=0.5)
    ax.add_collection(p)

    # Auto-scale axis
    all_coords = np.concatenate(hulls)
    ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
    ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())
    ax.set_aspect('equal')
    ax.set_title(f"RGC Mosaic Tiling: {cluster_key}")
    return ax


def plot_mosaic_violations(adata, hulls, iou_matrix, cluster_key='mosaic_groups', iou_threshold=0.05):
    """Highlights spatial overlaps (violations) within clusters."""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot background hulls in light gray
    patches = [mpl_polygon(h, closed=True) for h in hulls]
    p = PatchCollection(patches, facecolor='lightgray', alpha=0.3, edgecolor='none')
    ax.add_collection(p)

    # Calculate centroids for drawing connection lines
    centroids = np.array([np.mean(h, axis=0) for h in hulls])

    labels = adata.obs[cluster_key].values
    unique_labels = np.unique(labels)

    violation_count = 0
    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0]

        # Submatrix of IoU for this cluster
        if isinstance(iou_matrix, np.ndarray):
            sub_iou = iou_matrix[np.ix_(mask, mask)]
        else:  # Sparse
            sub_iou = iou_matrix[mask, :][:, mask].toarray()

        # Find pairs above threshold
        rows, cols = np.where(np.triu(sub_iou, k=1) > iou_threshold)

        for r, c in zip(rows, cols):
            idx1, idx2 = indices[r], indices[c]
            # Draw line between violating centroids
            p1, p2 = centroids[idx1], centroids[idx2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', alpha=0.8, lw=1.5)
            violation_count += 1

    ax.set_aspect('equal')
    ax.set_title(f"Mosaic Violations (IoU > {iou_threshold})\nTotal Violations: {violation_count}")
    return fig


def plot_individual_mosaics(adata, hulls, cluster_key='mosaic_groups', ncols=4):
    """Plots each cluster in its own subplot to check for tiling regularity."""
    clusters = sorted(adata.obs[cluster_key].unique())
    nrows = int(np.ceil(len(clusters) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    centroids = np.array([np.mean(h, axis=0) for h in hulls])
    all_coords = np.concatenate(hulls)

    for i, cluster in enumerate(clusters):
        ax = axes[i]
        mask = adata.obs[cluster_key] == cluster

        # Plot hulls for this cluster
        cluster_hulls = [hulls[j] for j in np.where(mask)[0]]
        patches = [mpl_polygon(h, closed=True) for h in cluster_hulls]
        p = PatchCollection(patches, facecolor='C0', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.add_collection(p)

        ax.set_title(f"Cluster {cluster}\n(n={mask.sum()})")
        ax.set_aspect('equal')
        ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
        ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


def plot_blobs(X, y):
    u_ys = np.unique(y)

    plt.figure()
    for i, yi in enumerate(u_ys):
        plt.scatter(X[y == yi, 0], X[y == yi, 1], label=f'Class {yi}')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Simulated Feature Blobs')
    plt.show()
