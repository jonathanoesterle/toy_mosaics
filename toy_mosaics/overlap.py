import numpy as np
from shapely.geometry import Polygon


def compute_iou_matrix(hulls):
    """
    Compute pairwise IoU matrix from convex hulls.

    Parameters
    ----------
    hulls : list of array-like
        List of convex hull vertices for each cell.
        Each hull should be shape (n_vertices, 2) for 2D coordinates.

    Returns
    -------
    iou_matrix : array, shape (n_cells, n_cells)
        Symmetric matrix of pairwise IoU values
    """
    n_cells = len(hulls)
    iou_matrix = np.zeros((n_cells, n_cells))

    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            iou = compute_hull_iou(hulls[i], hulls[j])
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou

    return iou_matrix


def compute_hull_iou(hull1, hull2):
    """
    Compute IoU between two convex hulls.

    This is a placeholder - you should replace with your actual implementation
    or use a library like shapely for accurate polygon intersection.

    Parameters
    ----------
    hull1, hull2 : array-like, shape (n_vertices, 2)
        Vertices of convex hulls

    Returns
    -------
    iou : float
        Intersection over union [0, 1]
    """

    try:
        poly1 = Polygon(hull1)
        poly2 = Polygon(hull2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        if union == 0:
            return 0.0

        return intersection / union
    except Exception:
        # Handle degenerate cases
        return 0.0
