import numpy as np


def compute_polygon_centers(polygons):
    """
    Compute the centroid of each polygon.

    Parameters
    ----------
    polygons : array-like, shape (n_polygons, n_vertices, 2)
        List of polygons with their vertices.

    Returns
    -------
    centers : array, shape (n_polygons, 2)
        Centroid coordinates for each polygon.
    """
    centers = []
    for poly in polygons:
        poly = np.asarray(poly)
        if poly.ndim != 2 or poly.shape[0] < 3:
            centers.append(np.array([np.nan, np.nan]))
            continue
        x_coords = poly[:, 0]
        y_coords = poly[:, 1]
        area = 0.0
        cx = 0.0
        cy = 0.0
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            cross = x_coords[i] * y_coords[j] - x_coords[j] * y_coords[i]
            area += cross
            cx += (x_coords[i] + x_coords[j]) * cross
            cy += (y_coords[i] + y_coords[j]) * cross
        area *= 0.5
        if area == 0:
            centers.append(np.array([np.nan, np.nan]))
            continue
        cx /= (6.0 * area)
        cy /= (6.0 * area)
        centers.append(np.array([cx, cy]))
    return np.array(centers)