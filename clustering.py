import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix


def apply_mosaic_constraints(
        adata, iou_matrix,
        attract_threshold=0.15,
        repel_threshold=0.4,
        attract_weight=0.5,
        repel_weight=2.0,
        mode='subtract',
        copy=True
):
    """
    Adjusts the morphological similarity graph by modulating edges based on spatial overlap.

    The key insight: small overlaps (e.g., touching cells) indicate likely grouping,
    while large overlaps (same spatial location) indicate impossible/spurious connections.

    Three zones:
    - 0 < IoU ≤ attract_threshold: attraction (boost connectivity) - touching cells
    - attract_threshold < IoU < repel_threshold: neutral (no modification)
    - IoU ≥ repel_threshold: repulsion (penalize connectivity) - too much overlap

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with neighbors already computed.
    iou_matrix : np.ndarray or csr_matrix
        The precomputed pairwise IoU matrix (n_cells × n_cells).
    attract_threshold : float, default=0.15
        IoU below or equal to this value receives attraction boost (e.g., ≤15% overlap = touching).
    repel_threshold : float, default=0.4
        IoU above or equal to this value receives repulsion penalty (e.g., ≥40% overlap = too much).
    attract_weight : float, default=0.5
        Strength of attraction for small overlaps.
        mode='subtract': add attract_weight * IoU to connectivity
        mode='multiply': multiply connectivity by (1 + attract_weight * IoU)
    repel_weight : float, default=2.0
        Strength of repulsion for large overlaps.
        mode='subtract': subtract repel_weight * IoU from connectivity
        mode='multiply': multiply connectivity by exp(-repel_weight * IoU)
    mode : {'subtract', 'multiply'}, default='subtract'
        How to apply the modulation:
        - 'subtract': conn = conn ± weight * IoU
        - 'multiply': attraction uses (1 + weight*IoU), repulsion uses exp(-weight*IoU)
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

    # Only consider edges that exist in the morphological graph
    iou_sparse = iou_sparse.multiply(conn > 0)

    # Split into three zones (corrected logic)
    attract_mask = (iou_sparse.data > 0) & (iou_sparse.data <= attract_threshold)
    repel_mask = iou_sparse.data >= repel_threshold
    # neutral_mask implicitly: attract_threshold < IoU < repel_threshold

    # Create separate matrices for attraction and repulsion
    attract_matrix = iou_sparse.copy()
    attract_matrix.data = attract_matrix.data * attract_mask
    attract_matrix.eliminate_zeros()

    repel_matrix = iou_sparse.copy()
    repel_matrix.data = repel_matrix.data * repel_mask
    repel_matrix.eliminate_zeros()

    if mode == 'subtract':
        # Attraction: boost connectivity for touching cells
        if attract_matrix.nnz > 0:
            conn = conn + attract_matrix.multiply(attract_weight)

        # Repulsion: penalize connectivity for overlapping cells
        if repel_matrix.nnz > 0:
            conn = conn - repel_matrix.multiply(repel_weight)

    elif mode == 'multiply':
        # Attraction: use (1 + attract_weight * IoU) instead of exp()
        # This keeps values reasonable (e.g., 1.0 to 1.1) instead of exploding
        if attract_matrix.nnz > 0:
            attract_factor = attract_matrix.copy()
            attract_factor.data = 1.0 + attract_weight * attract_factor.data
            # Need to handle sparse multiplication carefully
            # Only multiply where attract_matrix has values
            conn_dense = conn.toarray()
            attract_dense = attract_factor.toarray()
            mask = attract_dense > 0
            conn_dense[mask] *= attract_dense[mask]
            conn = csr_matrix(conn_dense)

        # Repulsion: multiply by exp(-repel_weight * IoU) < 1
        if repel_matrix.nnz > 0:
            repel_factor = repel_matrix.copy()
            repel_factor.data = np.exp(-repel_weight * repel_factor.data)
            conn_dense = conn.toarray()
            repel_dense = repel_factor.toarray()
            mask = repel_dense > 0
            conn_dense[mask] *= repel_dense[mask]
            conn = csr_matrix(conn_dense)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'subtract' or 'multiply'.")

    # Normalize matrix to be between 0 and 1
    v_max = conn.data.max() if conn.nnz > 0 else 1.0
    vmin = conn.data.min() if conn.nnz > 0 else 0.0

    if v_max > vmin:
        conn.data = (conn.data - vmin) / (v_max - vmin)
        conn.eliminate_zeros()

    # Store the modified connectivity
    ad.obsp['mosaic_connectivities'] = conn

    # Update neighbors dictionary
    if 'neighbors' in ad.uns:
        ad.uns['neighbors']['connectivities_key'] = 'mosaic_connectivities'

    # Also update distances if they exist (inverse logic)
    if 'distances' in ad.obsp:
        dist = ad.obsp['distances'].copy()

        if mode == 'subtract':
            # Attraction: reduce distance for touching cells
            if attract_matrix.nnz > 0:
                dist = dist - attract_matrix.multiply(attract_weight)
                dist.data[dist.data < 0] = 1e-6  # Prevent negative distances

            # Repulsion: increase distance for overlapping cells
            if repel_matrix.nnz > 0:
                dist = dist + repel_matrix.multiply(repel_weight)

        elif mode == 'multiply':
            # Attraction: multiply by (1 - attract_weight * IoU) but keep > 0
            if attract_matrix.nnz > 0:
                attract_factor = attract_matrix.copy()
                attract_factor.data = np.maximum(0.1, 1.0 - attract_weight * attract_factor.data)
                dist_dense = dist.toarray()
                attract_dense = attract_factor.toarray()
                mask = attract_dense > 0
                dist_dense[mask] *= attract_dense[mask]
                dist = csr_matrix(dist_dense)

            # Repulsion: multiply by exp(+repel_weight * IoU) > 1
            if repel_matrix.nnz > 0:
                repel_factor = repel_matrix.copy()
                repel_factor.data = np.exp(repel_weight * repel_factor.data)
                dist_dense = dist.toarray()
                repel_dense = repel_factor.toarray()
                mask = repel_dense > 0
                dist_dense[mask] *= repel_dense[mask]
                dist = csr_matrix(dist_dense)

        ad.obsp['mosaic_distances'] = dist

    return ad

def compare_clustering_mosaic_quality(
        adata, iou_matrix,
        standard_key='leiden',
        mosaic_key='leiden_mosaic',
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
        sc.tl.leiden(
            adata, resolution=best_res, adjacency=adata.obsp[adjacency_key],
            key_added='temp_leiden', flavor="igraph", n_iterations=2)

    # Store result and clean up
    adata.obs[key_added] = adata.obs['temp_leiden'].copy()
    del adata.obs['temp_leiden']

    return res
