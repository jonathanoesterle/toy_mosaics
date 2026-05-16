import numpy as np
from scipy.spatial import Voronoi


def generate_poisson_disk_samples(width, height, min_dist, max_attempts=30):
    """Generate spatially distributed points using Poisson disk sampling."""
    points = []
    active = []

    # Start with random point
    first = np.array([np.random.uniform(0, width), np.random.uniform(0, height)])
    points.append(first)
    active.append(0)

    while active:
        idx = np.random.choice(active)
        found = False

        for _ in range(max_attempts):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(min_dist, 2 * min_dist)
            candidate = points[idx] + radius * np.array([np.cos(angle), np.sin(angle)])

            if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
                if all(np.linalg.norm(candidate - p) >= min_dist for p in points):
                    points.append(candidate)
                    active.append(len(points) - 1)
                    found = True
                    break

        if not found:
            active.remove(idx)

    return np.array(points)


def add_position_noise(points, noise_level):
    """Add Gaussian noise to point positions."""
    noise = np.random.normal(0, noise_level, points.shape)
    return points + noise


def compute_voronoi_polygons(points, bounds):
    """Compute bounded Voronoi polygons for given points."""
    x_min, x_max, y_min, y_max = bounds

    # Add mirror points outside bounds to close edge regions
    mirror_points = []
    margin = (x_max - x_min) * 0.5

    # Mirror on all four sides
    for p in points:
        mirror_points.append([p[0], y_max + margin])  # Top
        mirror_points.append([p[0], y_min - margin])  # Bottom
        mirror_points.append([x_max + margin, p[1]])  # Right
        mirror_points.append([x_min - margin, p[1]])  # Left

    all_points = np.vstack([points, mirror_points])
    vor = Voronoi(all_points)

    polygons = []

    for point_idx in range(len(points)):
        region_idx = vor.point_region[point_idx]
        vertex_indices = vor.regions[region_idx]

        if len(vertex_indices) == 0:
            polygons.append(None)
            continue

        # Get vertices, filtering out the infinite vertex marker
        vertices = []
        for v_idx in vertex_indices:
            if v_idx != -1:
                vertices.append(vor.vertices[v_idx])

        if len(vertices) < 3:
            polygons.append(None)
            continue

        polygon = np.array(vertices)
        clipped = clip_polygon_to_bounds(polygon, bounds)

        # Add boundary corners if cell touches edges
        clipped = close_boundary_polygon(clipped, points[point_idx], bounds)

        polygons.append(clipped)

    return polygons


def clip_polygon_to_bounds(polygon, bounds):
    """Clip polygon vertices to rectangular bounds."""
    x_min, x_max, y_min, y_max = bounds
    clipped = polygon.copy()

    clipped[:, 0] = np.clip(clipped[:, 0], x_min, x_max)
    clipped[:, 1] = np.clip(clipped[:, 1], y_min, y_max)

    return clipped


def clip_polygon_sutherland_hodgman(polygon, bounds):
    """
    Clip a polygon to an axis-aligned rectangular bounding box
    using the Sutherland–Hodgman algorithm.

    Parameters
    ----------
    polygon : (N,2) ndarray
        Polygon vertices (ordered, clockwise or counterclockwise)
    bounds : tuple
        (x_min, x_max, y_min, y_max)

    Returns
    -------
    clipped_polygon : (M,2) ndarray or None
        Clipped polygon or None if fully outside
    """
    x_min, x_max, y_min, y_max = bounds

    def clip_edge(poly, inside_fn, intersect_fn):
        if poly is None or len(poly) == 0:
            return None

        output = []
        prev = poly[-1]
        prev_inside = inside_fn(prev)

        for curr in poly:
            curr_inside = inside_fn(curr)

            if curr_inside:
                if not prev_inside:
                    output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif prev_inside:
                output.append(intersect_fn(prev, curr))

            prev = curr
            prev_inside = curr_inside

        return np.array(output) if len(output) > 0 else None

    # Left edge (x >= x_min)
    polygon = clip_edge(
        polygon,
        lambda p: p[0] >= x_min,
        lambda p1, p2: p1 + (x_min - p1[0]) / (p2[0] - p1[0]) * (p2 - p1)
    )

    # Right edge (x <= x_max)
    polygon = clip_edge(
        polygon,
        lambda p: p[0] <= x_max,
        lambda p1, p2: p1 + (x_max - p1[0]) / (p2[0] - p1[0]) * (p2 - p1)
    )

    # Bottom edge (y >= y_min)
    polygon = clip_edge(
        polygon,
        lambda p: p[1] >= y_min,
        lambda p1, p2: p1 + (y_min - p1[1]) / (p2[1] - p1[1]) * (p2 - p1)
    )

    # Top edge (y <= y_max)
    polygon = clip_edge(
        polygon,
        lambda p: p[1] <= y_max,
        lambda p1, p2: p1 + (y_max - p1[1]) / (p2[1] - p1[1]) * (p2 - p1)
    )

    if polygon is None or len(polygon) < 3:
        return None

    return polygon


def close_boundary_polygon(polygon, center, bounds):
    """
    Ensure polygons at boundaries properly include boundary edges.
    Adds boundary corners when a cell touches multiple edges.
    """
    x_min, x_max, y_min, y_max = bounds

    # Check which boundaries the polygon touches
    touches_left = np.any(np.abs(polygon[:, 0] - x_min) < 0.01)
    touches_right = np.any(np.abs(polygon[:, 0] - x_max) < 0.01)
    touches_bottom = np.any(np.abs(polygon[:, 1] - y_min) < 0.01)
    touches_top = np.any(np.abs(polygon[:, 1] - y_max) < 0.01)

    # Add corners if touching adjacent edges
    corners_to_add = []

    if touches_left and touches_bottom:
        corners_to_add.append([x_min, y_min])
    if touches_left and touches_top:
        corners_to_add.append([x_min, y_max])
    if touches_right and touches_bottom:
        corners_to_add.append([x_max, y_min])
    if touches_right and touches_top:
        corners_to_add.append([x_max, y_max])

    if corners_to_add:
        polygon = np.vstack([polygon, corners_to_add])
        # Sort points by angle from center to maintain convex hull
        angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        polygon = polygon[sorted_indices]

    return polygon


def scale_polygon_to_diameter(polygon, center, target_diameter):
    """Scale polygon to achieve target equivalent circular diameter."""
    current_area = compute_polygon_area(polygon)
    current_diameter = 2 * np.sqrt(current_area / np.pi)

    if current_diameter == 0:
        return polygon

    scale_factor = target_diameter / current_diameter
    scaled = center + (polygon - center) * scale_factor

    return scaled


def compute_polygon_area(polygon):
    """Compute area of polygon using shoelace formula."""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def remove_random_cells(polygons, centers, is_clipped, n_remove):
    """
    Randomly remove n_remove cells from the mosaic.

    Removal is uniform over cells and preserves alignment between
    polygons, centers, and is_clipped.

    Parameters
    ----------
    polygons : list
        List of cell polygons
    centers : ndarray (N, 2)
        Cell centers
    is_clipped : list of bool
        Clipping flags
    n_remove : int
        Number of cells to remove

    Returns
    -------
    polygons_out, centers_out, is_clipped_out
    """
    n_cells = len(polygons)

    if n_cells == 0 or n_remove <= 0:
        return polygons, centers, is_clipped

    if n_remove >= n_cells:
        # Remove all cells
        return [], np.empty((0, 2)), []

    keep_indices = np.random.choice(
        n_cells,
        size=n_cells - n_remove,
        replace=False
    )

    keep_indices = np.sort(keep_indices)

    polygons_out = [polygons[i] for i in keep_indices]
    centers_out = centers[keep_indices]
    is_clipped_out = [is_clipped[i] for i in keep_indices]

    return polygons_out, centers_out, is_clipped_out


def voronoi_equivalent_diameter(polygon):
    """Equivalent circular diameter of a Voronoi polygon."""
    area = compute_polygon_area(polygon)
    return 2 * np.sqrt(area / np.pi)


def apply_overlap_factor(polygon, center, overlap_factor):
    """Scale polygon by overlap factor to allow cell overlap."""
    return center + (polygon - center) * overlap_factor


def generate_mosaic(
        mean_diameter,
        n_missing,
        center_noise,
        diameter_noise,
        overlap_factor=1.0,
        width=100,
        height=100,
):
    """
    Generate a single RGC mosaic using buffered-domain sampling
    to eliminate boundary artifacts.

    Cells are generated on an expanded domain, then only cells whose
    centers lie inside the original bounds are kept.
    """
    # ------------------------------------------------------------
    # 1. Parameters
    # ------------------------------------------------------------
    min_dist = mean_diameter * 0.8
    buffer = 3 * mean_diameter * overlap_factor

    # Expanded domain
    expanded_bounds = (
        -buffer,
        width + buffer,
        -buffer,
        height + buffer
    )

    expanded_width = width + 2 * buffer
    expanded_height = height + 2 * buffer

    # ------------------------------------------------------------
    # 2. Generate centers on expanded domain
    # ------------------------------------------------------------
    centers = generate_poisson_disk_samples(
        expanded_width,
        expanded_height,
        min_dist
    )

    # Shift centers so expanded domain is centered correctly
    centers[:, 0] -= buffer
    centers[:, 1] -= buffer

    centers = add_position_noise(centers, center_noise)

    # ------------------------------------------------------------
    # 3. Voronoi tessellation on expanded domain
    # ------------------------------------------------------------
    polygons = compute_voronoi_polygons(
        centers,
        expanded_bounds
    )

    # ------------------------------------------------------------
    # 4. Scale polygons to target diameter + overlap
    # ------------------------------------------------------------
    scaled_polygons = []
    for i, poly in enumerate(polygons):
        if poly is None:
            scaled_polygons.append(None)
            continue

        # --- Voronoi-consistent diameter ---
        base_diameter = voronoi_equivalent_diameter(poly)

        # Optional biological size noise
        diameter = base_diameter * np.random.normal(
            1.0, diameter_noise
        )
        diameter = max(diameter, base_diameter * 0.5)

        # Scale to Voronoi-consistent size
        scaled = scale_polygon_to_diameter(
            poly, centers[i], diameter
        )

        # Apply overlap / gap control
        scaled = apply_overlap_factor(
            scaled, centers[i], overlap_factor
        )

        scaled_polygons.append(scaled)

    # ------------------------------------------------------------
    # 5. Keep only cells whose centers lie inside original bounds
    # ------------------------------------------------------------
    final_polygons = []
    final_centers = []
    final_clipped = []

    final_bounds = (0, width, 0, height)

    for poly, c in zip(scaled_polygons, centers):
        if poly is None:
            continue

        if 0 <= c[0] <= width and 0 <= c[1] <= height:
            polys_clipped = clip_polygon_sutherland_hodgman(
                poly, final_bounds
            )
            if polys_clipped is not None and len(polys_clipped) >= 3:
                final_polygons.append(polys_clipped)
                final_centers.append(c)
                final_clipped.append(polys_clipped.shape[0] != poly.shape[0])

    final_centers = np.array(final_centers)

    # ------------------------------------------------------------
    # 6. Remove missing cells AFTER cropping
    # ------------------------------------------------------------
    final_polygons, final_centers, final_clipped = remove_random_cells(
        polygons=final_polygons,
        centers=final_centers,
        is_clipped=final_clipped,
        n_remove=n_missing,
    )

    return final_polygons, final_centers, final_clipped


def simulate_rgc_mosaics(
        n_mosaics,
        mean_diameters,
        center_noise,
        diameter_noise,
        n_missing_list,
        overlap_factors=None,
        verbose=False,
):
    """
    Simulate RGC mosaics.

    Parameters:
    -----------
    n_mosaics : int
        Number of mosaics to generate
    mean_diameters : list of float
        Mean cell diameter for each mosaic
    center_noise : float
        Standard deviation of Gaussian noise for cell positions
    diameter_noise : float
        Standard deviation of Gaussian noise for cell diameters (as fraction of mean)
    n_missing_list : list of int
        Number of missing cells for each mosaic
    overlap_factors : list of float or None
        Cell size scaling factor for each mosaic. 1.0=touching, >1.0=overlapping, <1.0=gaps
        If None, defaults to 1.0 for all mosaics
    verbose : bool
        Whether to print progress information
    """
    if overlap_factors is None:
        overlap_factors = [1.0] * n_mosaics

    if len(mean_diameters) != n_mosaics or len(n_missing_list) != n_mosaics:
        raise ValueError("Length of mean_diameters and n_missing_list must equal n_mosaics")

    if len(overlap_factors) != n_mosaics:
        raise ValueError("Length of overlap_factors must equal n_mosaics")

    polygons = []
    centers = []
    clipped = []
    groups = []

    for i in range(n_mosaics):
        if verbose:
            print(f"Generating mosaic {i + 1}/{n_mosaics}...")
        polygons_i, centers_i, clipped_i = generate_mosaic(
            center_noise=center_noise,
            diameter_noise=diameter_noise,
            mean_diameter=mean_diameters[i],
            n_missing=n_missing_list[i],
            overlap_factor=overlap_factors[i]
        )
        polygons += polygons_i
        centers.append(centers_i)
        clipped.append(clipped_i)
        groups.append(np.ones(len(polygons_i), dtype=int) * i)

        if verbose:
            print(f"  Generated {len(groups)} cells")

    polygons = np.array(polygons, dtype=object)
    centers = np.vstack(centers)
    clipped = np.concatenate(clipped)
    groups = np.concatenate(groups)

    return groups, polygons, centers, clipped


if __name__ == "__main__":
    np.random.seed(300)

    # Parameters
    n_mosaics = 3
    center_noise = 0.  # Position noise
    diameter_noise = 0.

    mean_diameters = [10., 15.0, 20.0]  # Mean diameter for each mosaic
    n_missing_list = [2, 2, 0]  # Missing cells per mosaic
    overlap_factors = [1., 1.2, 1.5]  # No overlap, 15% overlap, 10% gaps

    # Generate mosaics
    groups, polygons, centers, clipped = simulate_rgc_mosaics(
        n_mosaics=n_mosaics,
        mean_diameters=mean_diameters,
        center_noise=center_noise,
        diameter_noise=diameter_noise,
        n_missing_list=n_missing_list,
        overlap_factors=overlap_factors,
    )

    plot_mosaics(groups, polygons, centers, clipped)