import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix


def apply_mosaic_constraints(
        adata, iou_matrix, penalty_weight=1.0,
        iou_threshold=0.0, mode='subtract', copy=True):
    """
    Adjusts the morphological similarity graph by penalizing spatial overlaps.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with neighbors already computed.
    iou_matrix : np.ndarray or csr_matrix
        The precomputed pairwise IoU matrix (n_cells × n_cells).
    penalty_weight : float, default=1.0
        How much to penalize overlapping edges.
        mode='subtract': subtract penalty_weight * IoU from connectivity
        mode='multiply': multiply connectivity by exp(-penalty_weight * IoU)
    iou_threshold : float, default=0.0
        Only penalize IoU values above this threshold.
    mode : {'subtract', 'multiply'}, default='subtract'
        How to apply the penalty:
        - 'subtract': conn = max(0, conn - penalty_weight * IoU)
        - 'multiply': conn = conn * exp(-penalty_weight * IoU)
    copy : bool, default=True
        Whether to return a copy or modify in place.

    Returns:
    --------
    adata : AnnData
        Modified AnnData with 'mosaic_connectivities' in obsp.
    """
    if 'neighbors' not in adata.uns:
        raise ValueError("Please run sc.pp.neighbors(adata) first.")

    ad = adata.copy() if copy else adata

    # Get the morphological connectivities (similarity graph)
    conn = ad.obsp['connectivities'].copy()

    # Ensure iou_matrix is sparse
    if not isinstance(iou_matrix, csr_matrix):
        iou_sparse = csr_matrix(iou_matrix)
    else:
        iou_sparse = iou_matrix.copy()

    # Apply threshold - only penalize significant overlaps
    if iou_threshold > 0:
        iou_sparse.data[iou_sparse.data < iou_threshold] = 0
        iou_sparse.eliminate_zeros()

    # Only apply penalty where edges exist in the morphological graph
    penalty_matrix = iou_sparse.multiply(conn > 0)

    if mode == 'subtract':
        # Subtract penalty: new_conn = max(0, conn - penalty_weight * IoU)
        conn = conn - penalty_matrix.multiply(penalty_weight)
        conn.data[conn.data < 0] = 0
        conn.eliminate_zeros()

    elif mode == 'multiply':
        # Exponential penalty: new_conn = conn * exp(-penalty_weight * IoU)
        # This preserves more of the original connectivity structure
        penalty_matrix.data = np.exp(-penalty_weight * penalty_matrix.data)
        conn = conn.multiply(penalty_matrix)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'subtract' or 'multiply'.")

    # Store the modified connectivity
    ad.obsp['mosaic_connectivities'] = conn

    # Also update distances if they exist
    if 'distances' in ad.obsp:
        # For distances, we ADD penalty (higher overlap = more distant)
        dist = ad.obsp['distances'].copy()
        if mode == 'subtract':
            dist = dist + penalty_matrix.multiply(penalty_weight)
        elif mode == 'multiply':
            # For distances, use inverse: dist = dist * exp(penalty_weight * IoU)
            penalty_matrix_dist = iou_sparse.multiply(dist > 0)
            penalty_matrix_dist.data = np.exp(penalty_weight * penalty_matrix_dist.data)
            dist = dist.multiply(penalty_matrix_dist)
        ad.obsp['mosaic_distances'] = dist

    return ad


def compare_clustering_mosaic_quality(
        adata, iou_matrix,
        standard_key='standard_groups',
        mosaic_key='mosaic_groups',
        iou_threshold=0.0):
    """
    Compare mosaic quality between standard and mosaic-aware clustering.

    Returns a DataFrame with statistics for each clustering approach.
    """
    import pandas as pd

    results = []

    for key, name in [(standard_key, 'Standard'), (mosaic_key, 'Mosaic-aware')]:
        if key not in adata.obs:
            continue

        labels = adata.obs[key].values
        unique_labels = np.unique(labels)

        cluster_stats = []
        for label in unique_labels:
            mask = labels == label
            n_cells = np.sum(mask)

            if n_cells < 2:
                continue

            # Extract IoU submatrix for this cluster
            cluster_iou = iou_matrix[np.ix_(mask, mask)]

            # Get upper triangle (exclude diagonal)
            upper_tri = np.triu(cluster_iou, k=1)
            ious = upper_tri[upper_tri > 0]

            if len(ious) == 0:
                continue

            cluster_stats.append({
                'method': name,
                'cluster': label,
                'n_cells': n_cells,
                'mean_iou': np.mean(ious),
                'median_iou': np.median(ious),
                'max_iou': np.max(ious),
                'std_iou': np.std(ious),
                'violations': np.sum(ious > iou_threshold) if iou_threshold > 0 else np.nan
            })

        results.extend(cluster_stats)

    return pd.DataFrame(results)


def find_leiden_resolution(
        adata, target_clusters, adjacency_key='mosaic_connectivities', key_added='leiden_mosaic',
        start_res=1.0, tolerance=0, max_iter=20, verbose=True
):
    """
    Finds the Leiden resolution that results in the target number of clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    target_clusters : int
        Desired number of clusters
    adjacency_key : str, optional
        Key in adata.obsp for the adjacency matrix (default: 'mosaic_connectivities')
    key_added : str, optional
        Key in adata.obs to store the resulting clustering (default: 'leiden_mosaic')
    start_res : float, optional
        Initial resolution to start search (default: 1.0)
    tolerance : int, optional
        Acceptable deviation from target_clusters (default: 0)
    max_iter : int, optional
        Maximum number of iterations (default: 20)
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    float
        Best resolution found

    Raises
    ------
    KeyError
        If adjacency_key not found in adata.obsp
    ValueError
        If target_clusters is not positive
        :param key_added:
    """
    # Input validation
    if target_clusters <= 0:
        raise ValueError(f"target_clusters must be positive, got {target_clusters}")

    if adjacency_key not in adata.obsp:
        raise KeyError(f"Adjacency key '{adjacency_key}' not found in adata.obsp. "
                       f"Available keys: {list(adata.obsp.keys())}")

    res_min, res_max = 0.0, 10.0
    res = start_res
    best_res = res
    best_diff = float('inf')

    if verbose:
        print(f"Searching for resolution to get {target_clusters} clusters (±{tolerance})...")

    for i in range(max_iter):
        sc.tl.leiden(adata, resolution=res, adjacency=adata.obsp[adjacency_key],
                     key_added='temp_leiden', flavor="igraph", n_iterations=2)
        n_found = len(adata.obs['temp_leiden'].unique())
        diff = abs(n_found - target_clusters)

        if verbose:
            print(f"Iteration {i + 1}/{max_iter}: Res = {res:.4f}, Clusters = {n_found} "
                  f"(diff = {n_found - target_clusters:+d})")

        # Track best result
        if diff < best_diff:
            best_diff = diff
            best_res = res

        # Check if target is met within tolerance
        if diff <= tolerance:
            if verbose:
                print(f"✓ Found solution within tolerance at resolution {res:.4f}")
            break

        # Binary search adjustment
        if n_found < target_clusters:
            res_min = res
        else:
            res_max = res

        # Check if search space is exhausted
        if abs(res_max - res_min) < 1e-6:
            if verbose:
                print(f"⚠ Search space exhausted. Using best found: "
                      f"res={best_res:.4f}, clusters={n_found}")
            res = best_res
            break

        res = (res_min + res_max) / 2
    else:
        # Loop completed without break (max_iter reached)
        if verbose:
            print(f"⚠ Max iterations reached. Using best found: "
                  f"res={best_res:.4f} with {best_diff} cluster difference")
        res = best_res
        # Run one final clustering with best resolution
        sc.tl.leiden(adata, resolution=best_res, adjacency=adata.obsp[adjacency_key],
                     key_added='temp_leiden', flavor="igraph", n_iterations=2)

    # Store result and clean up
    adata.obs[key_added] = adata.obs['temp_leiden'].copy()
    del adata.obs['temp_leiden']

    return res
