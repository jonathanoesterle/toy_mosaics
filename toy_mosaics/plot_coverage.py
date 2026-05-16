"""Plotting utilities for cell_mosaics.

This module contains helpers to visualize convex hulls and coverage maps.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Polygon


def plot_polygon(ax: Axes, points: np.ndarray, facecolor: str = 'gray', edgecolor: str = 'gray',
                 edge_kws=None, face_kws=None, facealpha: float = 0.1) -> None:
    """Plot a filled polygon (e.g., a convex hull) onto an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    points : np.ndarray
        Array of shape (N, 2) with polygon vertices (x, y).
    facecolor : str, optional
        Fill color for the polygon. Default 'gray'.
    edgecolor : str, optional
        Line color for the polygon outline. Default 'gray'.
    edge_kws : dict, optional
        Additional keyword arguments for the edge line. Default None.
    face_kws : dict, optional
        Additional keyword arguments for the face fill. Default None.
    facealpha : float, optional
        Alpha transparency for the fill. Default 0.1.
    """
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    edge_kws = {} if edge_kws is None else edge_kws
    face_kws = {} if face_kws is None else face_kws

    ax.plot(points[:, 0], points[:, 1], color=edgecolor, **edge_kws)
    if facealpha > 0:
        polygon = Polygon(points, closed=True, facecolor=facecolor, alpha=facealpha, edgecolor='none', **face_kws)
        ax.add_patch(polygon)


def plot_coverage(
        coverage_count: np.ndarray,
        cell_outlines: Iterable[np.ndarray] | None = None,
        extent: tuple[float, float, float, float] | None = None,
        ax: Axes | None = None,
        colormap: str = 'viridis',
        figsize: tuple[int, int] = (10, 8),
        show_max_coverage: bool = True,
        alpha: float = 1.0,
        interpolation: str = 'nearest',
        edgecolor='k',
        edge_kws=None,
) -> tuple[Figure | None, Axes, AxesImage]:
    """Visualize a 2D coverage grid as an image with colorbar.

    Parameters
    ----------
    coverage_count : np.ndarray
        2D array with integer coverage counts.
    cell_outlines : Iterable[np.ndarray] or None, optional
        If provided, overlay these cell_outlines on the coverage map.
    extent : tuple or None, optional
        (xmin, xmax, ymin, ymax) extent to annotate axes; if None, uses array
        indices. Default None.
    ax : matplotlib.axes.Axes or None, optional
        If provided, draw on this axes; otherwise, create a new figure and axes.
    colormap : str, optional
        Matplotlib colormap name. Default 'viridis'.
    figsize : tuple[int, int], optional
        Figure size if a new figure is created. Default (10, 8).
    show_max_coverage : bool, optional
        Whether to include maximum coverage in the title. Default True.
    alpha : float, optional
        Alpha transparency for the image. Default 1.0.
    interpolation : str, optional
        Interpolation method for imshow. Default 'bilinear'.
    edgecolor : str, optional
        Color for cell outline edges. Default 'k' (black).
    edge_kws : dict, optional
        Additional keyword arguments for cell outline edges. Default None.

    Returns
    -------
    (fig, ax, im)
        fig may be None if ax was provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # Set up discrete colormap for integer data
    min_val = int(np.min(coverage_count))
    max_val = int(np.max(coverage_count))
    n_levels = max_val - min_val + 1

    # Create discrete colormap with clear boundaries
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm

    # Get colors from the specified colormap
    base_cmap = cm.get_cmap(colormap)
    colors = base_cmap(np.linspace(0, 1, n_levels))
    discrete_cmap = ListedColormap(colors)

    im = ax.imshow(
        coverage_count,
        extent=extent,
        origin='lower',
        cmap=discrete_cmap,
        alpha=alpha,
        interpolation=interpolation,
        vmin=min_val - 0.5,
        vmax=max_val + 0.5
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coverage Count (# of overlapping cells)', rotation=270, labelpad=20)

    # Set discrete ticks on colorbar
    cbar.set_ticks(np.arange(min_val, max_val + 1))

    if cell_outlines is not None:
        for poly in cell_outlines:
            plot_polygon(ax, poly, edgecolor=edgecolor, facealpha=0, edge_kws=edge_kws)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if show_max_coverage:
        ax.set_title(f'Coverage Density Map (Max Coverage: {max_val})')
    else:
        ax.set_title('Coverage Density Map')
    ax.set_aspect('equal')

    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    return fig, ax, im
