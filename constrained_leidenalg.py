import numpy as np
import igraph as ig
import leidenalg
import networkx as nx
from scipy import sparse
from scipy.spatial import Delaunay, KDTree
from sklearn.neighbors import NearestNeighbors


class MosaicLeidenPartition(leidenalg.RBConfigurationVertexPartition):
    """
    Optimized Leiden partition that enforces spatial repulsion within clusters.
    This creates a 'tiling' effect where members of a cluster avoid each other.
    """

    def __init__(self, graph, positions, overlap_dist=None, overlap_penalty=1000.0, **kwargs):
        super().__init__(graph, **kwargs)
        self.positions = np.array(positions)
        self.overlap_penalty = overlap_penalty

        # 1. Pre-calculate which pairs are "too close" spatially
        # This turns the O(N^2) search into a fast sparse lookup
        tree = KDTree(self.positions)

        if overlap_dist is None:
            # Auto-estimate: use 1.5x the average NN distance
            dists, _ = tree.query(self.positions, k=2)
            self.overlap_dist = np.mean(dists[:, 1]) * 1.5
        else:
            self.overlap_dist = overlap_dist

        # Build a sparse adjacency matrix of "Forbidden Spatial Pairs"
        pairs = tree.query_pairs(r=self.overlap_dist)
        rows, cols = zip(*pairs) if pairs else ([], [])
        self.forbidden_matrix = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(positions), len(positions))
        )

    def quality(self, resolution_parameter=1.0):
        # Base modularity quality
        q = super().quality(resolution_parameter)

        # Calculate overlaps efficiently using the membership vector
        membership = np.array(self.membership)
        penalty_count = 0

        # We only care about forbidden pairs that share the same membership
        # Iterate through the sparse matrix rows (more efficient than N^2)
        rows, cols = self.forbidden_matrix.nonzero()
        overlap_mask = membership[rows] == membership[cols]
        penalty_count = np.sum(overlap_mask)

        return q - (penalty_count * self.overlap_penalty)

def build_spatial_feature_graph(positions, features, sigma=None, max_distance=None, verbose=False):
    """
    Build graph with edges between spatial neighbors, weighted by feature similarity.

    Parameters
    ----------
    positions : np.ndarray, shape (n_cells, 2)
        Spatial positions (in microns)
    features : np.ndarray, shape (n_cells, n_features)
        Feature matrix
    sigma : float, optional
        Bandwidth for Gaussian kernel. If None, uses median feature distance
    max_distance : float, optional
        Maximum spatial distance for edges. If None, uses Delaunay triangulation

    Returns
    -------
    graph : igraph.Graph
        Graph with weighted edges
    """
    n = len(positions)

    if sigma is None:
        from sklearn.metrics import pairwise_distances
        n_sample = min(200, n)
        sample_indices = np.random.choice(n, n_sample, replace=False)
        dists = pairwise_distances(features[sample_indices])
        sigma = np.median(dists[dists > 0])

    # Build spatial edges
    if max_distance is None:
        tri = Delaunay(positions)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edges.add((simplex[i], simplex[j]))
    else:
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(positions[i] - positions[j]) < max_distance:
                    edges.add((i, j))

    # Create graph and compute weights
    G = ig.Graph(n=n)
    edge_list = list(edges)
    weights = []

    for u, v in edge_list:
        feat_dist = np.linalg.norm(features[u] - features[v])
        weight = np.exp(-feat_dist ** 2 / (2 * sigma ** 2))
        weights.append(weight)

    G.add_edges(edge_list)
    G.es['weight'] = weights

    if verbose:
        print(f"Built graph: {n} nodes, {len(edge_list)} edges")
        print(f"Feature similarity sigma: {sigma:.3f}")

    return G


def build_feature_knn_graph(features, k=15):
    """
    Build graph based on FEATURE similarity, not spatial proximity.
    This allows similar types to find each other across the 'mosaic'.
    """
    n = len(features)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(features)
    distances, indices = nbrs.kneighbors(features)

    edges = []
    weights = []
    sigma = np.median(distances)

    for i in range(n):
        for j_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
            edges.append((i, j_idx))
            weights.append(np.exp(-dist ** 2 / (2 * sigma ** 2)))

    G = ig.Graph(n=n, edges=edges, edge_attrs={'weight': weights})
    return G

def mosaic_leiden_clustering(graph, positions,
                             resolution_parameter=1.0,
                             overlap_dist=None,
                             overlap_penalty=1000.0,
                             n_iterations=2,
                             seed=None,
                             verbose=False):
    """
    Run Leiden algorithm with mosaic tiling constraints.

    Parameters
    ----------
    graph : igraph.Graph
        Graph with edges representing feature similarity
    positions : np.ndarray, shape (n_cells, 2)
        Spatial coordinates of cells (in microns)
    resolution_parameter : float, default=1.0
        Higher values -> more clusters
    overlap_dist : float, default=None
        Distance threshold for considering two cells as overlapping
    overlap_penalty : float, default=1000.0
        Penalty per overlapping pair
    n_iterations : int, default=2
        Leiden iterations
    seed : int, optional
        Random seed

    Returns
    -------
    clusters : np.ndarray
        Cluster assignments
    partition : MosaicLeidenPartition
        Partition object
    """
    if verbose:
        print(f"Mosaic Leiden Clustering")
        print(f"  Resolution: {resolution_parameter}")
        print(f"  Overlap distance: {overlap_dist}")
        print(f"  Overlap penalty: {overlap_penalty}")

    partition = leidenalg.find_partition(
        graph,
        partition_type=MosaicLeidenPartition,
        weights='weight' if 'weight' in graph.es.attributes() else None,
        n_iterations=n_iterations,
        seed=seed,
        resolution_parameter=resolution_parameter,
        positions=positions,
        overlap_dist=overlap_dist,
        overlap_penalty=overlap_penalty
    )

    n_clusters = len(set(partition.membership))
    if verbose:
        print(f"  Result: {n_clusters} clusters")

    return np.array(partition.membership), partition


def find_optimal_resolution(graph, positions, res_start=1., res_min=0.1, res_max=3.0, n_trials=10,
                            target_clusters=(35, 45), verbose=False, **kwargs):
    """
    Sweep resolution parameter to find value that gives target number of clusters.
    """
    if verbose:
        print(f"Searching for resolution to get {target_clusters[0]}-{target_clusters[1]} clusters...")

    results = []

    res_i = res_start
    for i in range(n_trials):
        clusters, _ = mosaic_leiden_clustering(
            graph, positions,
            resolution_parameter=res_i,
            verbose=False,
            **kwargs
        )
        n_clusters = len(np.unique(clusters))
        results.append((res_i, n_clusters))
        if verbose:
            print(f"  Resolution {res_i:.2f} -> {n_clusters} clusters")

        if target_clusters[0] <= n_clusters <= target_clusters[1]:
            if verbose:
                print(f"\n✓ Found good resolution: {res_i:.2f} gives {n_clusters} clusters")
            return res_i
        elif n_clusters < target_clusters[0]:
            res_min = res_i
            res_i = (res_i + res_max) / 2
        else:
            res_max = res_i
            res_i = (res_min + res_i) / 2

    # If not found, pick closest
    results = np.array(results)
    target_mid = (target_clusters[0] + target_clusters[1]) / 2
    idx = np.argmin(np.abs(results[:, 1] - target_mid))
    best_res = results[idx, 0]
    if verbose:
        print(f"\nClosest match: resolution {best_res:.2f} gives {int(results[idx, 1])} clusters")
    return best_res


def post_process_enforce_contiguity(clusters, positions, min_cluster_size=3):
    """
    Post-processing to split non-contiguous clusters.

    Parameters
    ----------
    clusters : np.ndarray
        Cluster assignments
    positions : np.ndarray
        Spatial positions
    min_cluster_size : int
        Minimum size for a cluster to keep

    Returns
    -------
    new_clusters : np.ndarray
        Updated cluster assignments with contiguous clusters
    """
    tri = Delaunay(positions)
    spatial_graph = nx.Graph()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                spatial_graph.add_edge(simplex[i], simplex[j])

    new_clusters = clusters.copy()
    next_cluster_id = clusters.max() + 1

    for cluster_id in np.unique(clusters):
        cluster_cells = np.where(clusters == cluster_id)[0]

        subgraph = spatial_graph.subgraph(cluster_cells)
        components = list(nx.connected_components(subgraph))

        if len(components) > 1:
            largest = max(components, key=len)

            for comp in components:
                if comp != largest and len(comp) >= min_cluster_size:
                    for cell in comp:
                        new_clusters[cell] = next_cluster_id
                    next_cluster_id += 1

    return new_clusters


def analyze_mosaic_quality(clusters, positions, territory_radius=None):
    """
    Analyze the quality of the mosaic tiling.

    Returns
    -------
    dict with per-cluster metrics:
        - n_cells: number of cells
        - diameter: spatial spread
        - n_overlaps: overlapping pairs
        - overlap_ratio: overlaps per cell
    """
    if territory_radius is None:
        nbrs = NearestNeighbors(n_neighbors=7).fit(positions)
        distances, _ = nbrs.kneighbors(positions)
        radii = {i: np.median(distances[i, 1:]) * 1.5
                 for i in range(len(positions))}
    elif isinstance(territory_radius, dict):
        radii = territory_radius
    else:
        radii = {i: territory_radius for i in range(len(positions))}

    results = {}

    for cluster_id in np.unique(clusters):
        indices = np.where(clusters == cluster_id)[0]
        cluster_pos = positions[indices]

        centroid = cluster_pos.mean(axis=0)
        max_dist = np.max(np.linalg.norm(cluster_pos - centroid, axis=1))

        n_overlaps = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                dist = np.linalg.norm(cluster_pos[i] - cluster_pos[j])
                r_sum = radii[idx_i] + radii[idx_j]
                if dist < 2.0 * r_sum:
                    n_overlaps += 1

        results[cluster_id] = {
            'n_cells': len(indices),
            'diameter': max_dist * 2,
            'n_overlaps': n_overlaps,
            'overlap_ratio': n_overlaps / len(indices) if len(indices) > 0 else 0
        }

    return results


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    n_cells = 2000

    # Generate synthetic data
    positions = np.random.rand(n_cells, 2) * 1000  # 1000 x 1000 micron area

    n_features = 50
    n_types = 40
    true_types = np.random.randint(0, n_types, n_cells)
    features = np.random.randn(n_cells, n_features)

    for i in range(n_types):
        mask = true_types == i
        if np.any(mask):
            features[mask] += np.random.randn(n_features) * 2

    # Build graph
    G = build_feature_knn_graph(features, k=5)

    # Find optimal resolution
    optimal_res = find_optimal_resolution(
        G, positions,
        target_clusters=(35, 45),
        res_start=1.0,
        res_min=0.1,
        res_max=3.0,
        n_trials=15,
        overlap_dist=None,
        overlap_penalty=1000.0,
        n_iterations=10,
        seed=42,
        verbose=True
    )

    # Run clustering
    clusters, partition = mosaic_leiden_clustering(
        G, positions,
        resolution_parameter=optimal_res,
        overlap_dist=None,
        overlap_penalty=1000.0,
        n_iterations=10,
        seed=42,
        verbose=True
    )

    print(f"\nFinal result: {len(np.unique(clusters))} clusters")
    cluster_sizes = np.bincount(clusters)
    print(f"Cluster sizes: min={cluster_sizes[cluster_sizes > 0].min()}, "
          f"max={cluster_sizes.max()}, "
          f"mean={cluster_sizes[cluster_sizes > 0].mean():.1f}")

    # Optional: enforce contiguity
    clusters_clean = post_process_enforce_contiguity(clusters, positions)
    print(f"After contiguity enforcement: {len(np.unique(clusters_clean))} clusters")

    # Analyze quality
    quality = analyze_mosaic_quality(clusters_clean, positions)
    total_overlaps = sum(q['n_overlaps'] for q in quality.values())
    print(f"Total overlapping pairs: {total_overlaps}")