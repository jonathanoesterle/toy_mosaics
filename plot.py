import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def plot_mosaics(mosaics_data, mean_diameters):
    """Plot all generated mosaics with clipped cells highlighted."""
    n_mosaics = len(mosaics_data)
    cols = min(3, n_mosaics)
    rows = (n_mosaics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_mosaics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.Set3(np.linspace(0, 1, n_mosaics))

    for idx, (polygons, centers, is_clipped) in enumerate(mosaics_data):
        ax = axes[idx]

        # Separate clipped and unclipped cells
        clipped_patches = []
        unclipped_patches = []

        for i, poly in enumerate(polygons):
            patch = Polygon(poly, closed=True)
            if is_clipped[i]:
                clipped_patches.append(patch)
            else:
                unclipped_patches.append(patch)

        # Plot unclipped cells
        if unclipped_patches:
            collection = PatchCollection(
                unclipped_patches,
                facecolors=colors[idx],
                edgecolors='black',
                linewidths=1.5,
                alpha=0.7
            )
            ax.add_collection(collection)

        # Plot clipped cells with different styling
        if clipped_patches:
            clipped_collection = PatchCollection(
                clipped_patches,
                facecolors=colors[idx],
                edgecolors='red',
                linewidths=2.0,
                alpha=0.5
            )
            ax.add_collection(clipped_collection)

        ax.scatter(centers[:, 0], centers[:, 1], c='darkred', s=10, zorder=5, alpha=0.6)

        ax.set_xlim(-10, 110)
        ax.set_ylim(-10, 110)
        ax.set_aspect('equal')

        n_clipped = sum(is_clipped)
        ax.set_title(
            f'Mosaic {idx + 1}\n{len(polygons)} cells, Ø={mean_diameters[idx]:.1f}\n'
            f'({n_clipped} clipped)',
            fontsize=10
        )
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')

    for idx in range(n_mosaics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()