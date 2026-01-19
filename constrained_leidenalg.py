import leidenalg
import pandas as pd
from scipy.spatial import KDTree


class SpatialLeidenPartition(leidenalg.RBConfigurationVertexPartition):
    """
    Custom Leiden partition that prevents merging clusters with overlapping territories.

    This modifies the quality function to heavily penalize configurations where
    clusters have overlapping territories, preventing such merges during optimization.
    """

    def __init__(self, graph, positions, territory_radius=None,
                 overlap_threshold=0.5, overlap_penalty=1000.0,
                 require_adjacency=False, resolution_parameter=1.0,
                 initial_membership=None, weights=None):
        """
        Parameters
        ----------
        graph : igraph.Graph
            Graph with edges weighted by feature similarity
        positions : np.ndarray, shape (n_cells, 2)
            Spatial positions of cells (in microns)
        territory_radius : float or dict, optional
            Territory radius for each cell. If None, estimated from data.
        overlap_threshold : float, default=0.5
            Cells closer than overlap_threshold * (r1 + r2) are considered overlapping
        overlap_penalty : float, default=1000.0
            Penalty added to quality for each overlapping cluster pair
        require_adjacency : bool, default=False
            If True, also penalizes non-adjacent clusters
        initial_membership : list, optional
            Initial cluster assignments
        weights : str or list, optional
            Edge weights attribute name or list of weights
        """
        super().__init__(graph, initial_membership=initial_membership,
                         weights=weights, resolution_parameter=resolution_parameter)
        self.positions = np.array(positions)
        self.overlap_threshold = overlap_threshold
        self.overlap_penalty = overlap_penalty
        self.require_adjacency = require_adjacency

        # Estimate territory sizes if not provided
        if territory_radius is None:
            self.territory_radius = self._estimate_territory_radii()
        elif isinstance(territory_radius, dict):
            self.territory_radius = territory_radius
        else:
            # Single radius for all cells
            self.territory_radius = {i: territory_radius for i in range(len(positions))}

        # Build spatial adjacency
        self._build_spatial_adjacency()

        # Cache for overlap checks
        self._overlap_cache = {}

    def _estimate_territory_radii(self, k=6):
        """
        Estimate territory radius for each cell based on local density.
        """
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(self.positions))).fit(self.positions)
        distances, _ = nbrs.kneighbors(self.positions)

        radii = {}
        for i in range(len(self.positions)):
            if len(distances[i]) > 1:
                radii[i] = np.median(distances[i, 1:]) * 1.5
            else:
                radii[i] = 50.0  # default

        return radii

    def _build_spatial_adjacency(self):
        """Build spatial adjacency using Delaunay triangulation."""
        if len(self.positions) < 4:
            # Not enough points for Delaunay
            self.spatial_edges = set()
            self.spatial_neighbors = defaultdict(set)
            return

        tri = Delaunay(self.positions)

        self.spatial_edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = simplex[i], simplex[j]
                    self.spatial_edges.add((min(u, v), max(u, v)))

        self.spatial_neighbors = defaultdict(set)
        for u, v in self.spatial_edges:
            self.spatial_neighbors[u].add(v)
            self.spatial_neighbors[v].add(u)

    def _clusters_overlap_in_space(self, cluster1_vertices, cluster2_vertices):
        """
        Check if two clusters have overlapping territories.

        Returns True if overlap detected (should be penalized).
        """
        # Create cache key
        v1_tuple = tuple(sorted(cluster1_vertices[:5]))  # Sample for cache
        v2_tuple = tuple(sorted(cluster2_vertices[:5]))
        cache_key = (v1_tuple, v2_tuple)

        if cache_key in self._overlap_cache:
            return self._overlap_cache[cache_key]

        # Check for overlaps (optimized - early exit)
        for v1 in cluster1_vertices:
            pos1 = self.positions[v1]
            r1 = self.territory_radius.get(v1, 50.0)

            for v2 in cluster2_vertices:
                pos2 = self.positions[v2]
                r2 = self.territory_radius.get(v2, 50.0)

                distance = np.linalg.norm(pos1 - pos2)
                overlap_distance = self.overlap_threshold * (r1 + r2)

                if distance < overlap_distance:
                    self._overlap_cache[cache_key] = True
                    return True

        self._overlap_cache[cache_key] = False
        return False

    def _clusters_are_adjacent(self, cluster1_vertices, cluster2_vertices):
        """Check if clusters share spatial edges."""
        cluster2_set = set(cluster2_vertices)

        for v1 in cluster1_vertices:
            if any(v2 in cluster2_set for v2 in self.spatial_neighbors[v1]):
                return True

        return False

    def quality(self, resolution_parameter=1.0):
        """
        Override quality function to include spatial penalties.

        Quality = base_quality - overlap_penalty * n_overlapping_pairs
        """
        # Get base quality from parent class with resolution parameter
        base_quality = super().quality(resolution_parameter)

        # Calculate spatial penalties
        penalty = 0.0

        # Get all unique clusters
        clusters = {}
        for v in range(len(self.membership)):
            c = self.membership[v]
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(v)

        cluster_ids = list(clusters.keys())

        # Check all pairs of clusters for violations
        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i + 1:]:
                vertices1 = clusters[c1]
                vertices2 = clusters[c2]

                # Penalty for overlapping territories
                if self._clusters_overlap_in_space(vertices1, vertices2):
                    penalty += self.overlap_penalty

                # Optional: penalty for non-adjacent clusters (if required)
                if self.require_adjacency:
                    if not self._clusters_are_adjacent(vertices1, vertices2):
                        penalty += self.overlap_penalty * 0.1

        return base_quality - penalty


import numpy as np
import igraph as ig
import leidenalg
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


class MosaicLeidenPartition(leidenalg.RBConfigurationVertexPartition):
    """
    Penalizes clusters if their own members are physically too close (overlapping).
    This forces clusters to spread out into a mosaic/tiling pattern.

    The key insight: cells of the SAME type should NOT overlap in space.
    """

    def __init__(self, graph, positions, territory_radius=None,
                 overlap_threshold=2.0, overlap_penalty=1000.0, **kwargs):
        """
        Parameters
        ----------
        graph : igraph.Graph
            Graph with feature similarity edges
        positions : np.ndarray, shape (n_cells, 2)
            Spatial positions in microns
        territory_radius : float or dict, optional
            Territory radius for cells. If None, auto-estimated from density.
            Can be dict mapping cell_id -> radius for varying sizes.
        overlap_threshold : float, default=2.0
            Cells closer than overlap_threshold * (r1 + r2) are overlapping.
            - 2.0 = territories just touching (no overlap)
            - 1.0 = centers must be > sum of radii apart (50% overlap allowed)
            - 0.5 = very strict, no overlap at all
        overlap_penalty : float, default=1000.0
            Penalty per overlapping pair within a cluster
        """
        super().__init__(graph, **kwargs)
        self.positions = np.array(positions)
        self.overlap_threshold = overlap_threshold
        self.overlap_penalty = overlap_penalty

        # Estimate or store territory radii
        if territory_radius is None:
            self.territory_radius = self._estimate_territory_radii()
        elif isinstance(territory_radius, dict):
            self.territory_radius = territory_radius
        else:
            # Single radius for all
            self.territory_radius = {i: territory_radius for i in range(len(positions))}

    def _estimate_territory_radii(self, k=6):
        """
        Estimate territory radius for each cell based on local density.
        Uses median distance to k nearest neighbors.
        """
        if len(self.positions) < k + 1:
            # Not enough cells, use default
            return {i: 50.0 for i in range(len(self.positions))}

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(self.positions)
        distances, _ = nbrs.kneighbors(self.positions)

        radii = {}
        for i in range(len(self.positions)):
            # Median distance to neighbors (excluding self)
            radii[i] = np.median(distances[i, 1:]) * 1.5

        return radii

    def quality(self, resolution_parameter=1.0):
        """
        Quality = base_quality - overlap_penalty * n_overlapping_pairs

        Base quality rewards feature similarity (standard Leiden).
        Penalty punishes spatial overlap within clusters (mosaic constraint).
        """
        # 1. Get standard Leiden quality (feature-based clustering)
        q = super().quality(resolution_parameter)

        # 2. Calculate spatial overlap penalty
        # Penalize if cells in the SAME cluster are too close
        penalty = 0.0

        # Group nodes by cluster
        membership = np.array(self.membership)

        for cluster_id in np.unique(membership):
            indices = np.where(membership == cluster_id)[0]
            if len(indices) < 2:
                continue

            # Check pairwise distances within this cluster
            cluster_pos = self.positions[indices]

            # Count overlaps using radius-based check
            # Two cells overlap if distance < overlap_threshold * (r1 + r2)
            n_overlaps = 0

            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_i = indices[i]
                    idx_j = indices[j]

                    # Get territory radii
                    r_i = self.territory_radius.get(idx_i, 50.0)
                    r_j = self.territory_radius.get(idx_j, 50.0)

                    # Calculate distance
                    dist = np.linalg.norm(cluster_pos[i] - cluster_pos[j])

                    # Check overlap
                    overlap_distance = self.overlap_threshold * (r_i + r_j)

                    if dist < overlap_distance:
                        n_overlaps += 1

            penalty += n_overlaps * self.overlap_penalty

        return q - penalty


def build_feature_knn_graph(features, k=15):
    """Connects cells that are similar in features, regardless of location."""
    nn = NearestNeighbors(n_neighbors=k).fit(features)
    adj = nn.kneighbors_graph(features).toarray()

    # Create igraph from adjacency
    sources, targets = np.where(adj > 0)
    edges = list(zip(sources, targets))
    g = ig.Graph(n=len(features), edges=edges)
    return g


def spatially_constrained_leiden(
        graph, positions,
        partition_type=MosaicLeidenPartition,
        resolution_parameter=1.0,
        territory_radius=None,
        overlap_threshold=0.5,
        overlap_penalty=1000.0,
        require_adjacency=False,
        n_iterations=2,
        seed=None,
        verbose=False):
    """
    Run Leiden algorithm with spatial mosaic constraints.

    Prevents merging clusters with overlapping territories by adding
    penalties to the quality function.

    Parameters
    ----------
    graph : igraph.Graph
        Graph with edges representing feature similarity
    positions : np.ndarray, shape (n_cells, 2)
        Spatial coordinates of cells (in microns)
    partition_type : leidenalg.Partition class, default=MosaicLeidenPartition
        Partition class to use for Leiden optimization
    resolution_parameter : float, default=1.0
        Resolution parameter for Leiden algorithm. Higher -> more clusters
    territory_radius : float or dict, optional
        Territory radius for cells. If None, estimated from local density
    overlap_threshold : float, default=0.5
        Cells closer than overlap_threshold * (r1+r2) are overlapping
        Lower values = stricter separation requirement
    overlap_penalty : float, default=1000.0
        Penalty for overlapping clusters. Higher = stricter enforcement
    require_adjacency : bool, default=False
        If True, also penalizes non-adjacent clusters
    n_iterations : int, default=2
        Number of iterations (Leiden paper suggests 2 is usually enough)
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        If True, print progress messages

    Returns
    -------
    clusters : np.ndarray
        Cluster assignments for each cell
    partition : SpatialLeidenPartition
        Full partition object with additional info
    """
    if verbose:
        print(f"Starting spatially-constrained Leiden clustering...")
        print(f"  Resolution parameter: {resolution_parameter}")
        print(f"  Overlap threshold: {overlap_threshold}")
        print(f"  Overlap penalty: {overlap_penalty}")
        print(f"  Require adjacency: {require_adjacency}")

    # Initialize partition with spatial constraints
    initial_partition = partition_type(
        graph,
        positions,
        territory_radius=territory_radius,
        overlap_threshold=overlap_threshold,
        overlap_penalty=overlap_penalty,
        require_adjacency=require_adjacency,
        weights='weight' if 'weight' in graph.es.attributes() else None
    )

    # Use leidenalg's find_partition which properly handles resolution
    if verbose:
        print(f"  Running optimization...")

    # This is the correct way to use resolution_parameter
    partition = leidenalg.find_partition(
        graph,
        partition_type=partition_type,
        initial_membership=initial_partition.membership,
        weights='weight' if 'weight' in graph.es.attributes() else None,
        n_iterations=n_iterations,
        seed=seed,
        # Pass custom parameters to partition class
        positions=positions,
        territory_radius=territory_radius,
        overlap_threshold=overlap_threshold,
        overlap_penalty=overlap_penalty,
        require_adjacency=require_adjacency,
        # Resolution parameter for the quality function
        resolution_parameter=resolution_parameter
    )

    final_clusters = len(set(partition.membership))
    if verbose:
        print(f"Optimization complete:")
        print(f"  Final clusters: {final_clusters}")

    return np.array(partition.membership), partition


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
    verbose : bool, default=False
        If True, print progress messages
    Returns
    -------
    graph : igraph.Graph
        Graph with weighted edges
    """
    n = len(positions)

    # Estimate sigma if not provided
    if sigma is None:
        from sklearn.metrics import pairwise_distances
        n_sample = min(200, n)
        sample_indices = np.random.choice(n, n_sample, replace=False)
        dists = pairwise_distances(features[sample_indices])
        sigma = np.median(dists[dists > 0])

    # Build spatial edges
    if max_distance is None:
        # Use Delaunay triangulation
        tri = Delaunay(positions)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edges.add((simplex[i], simplex[j]))
    else:
        # Use distance threshold
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


def evaluate_mosaic_quality(positions, clusters):
    """
    Calculates the average Regularity Index across all mosaics.
    Higher is better.
    """
    unique_clusters = np.unique(clusters)
    regularity_indices = []

    for c in unique_clusters:
        idx = np.where(clusters == c)[0]
        if len(idx) < 10:  # Skip tiny clusters
            continue

        pos = positions[idx]
        tree = KDTree(pos)
        # Get distance to the nearest neighbor (excluding self)
        dist, _ = tree.query(pos, k=2)
        nn_distances = dist[:, 1]

        ri = np.mean(nn_distances) / np.std(nn_distances)
        regularity_indices.append(ri)

    return np.mean(regularity_indices) if regularity_indices else 0


def mosaic_leiden_clustering(graph, positions,
                             resolution_parameter=1.0,
                             territory_radius=None,
                             overlap_threshold=2.0,
                             overlap_penalty=1000.0,
                             n_iterations=2,
                             seed=None):
    """
    Run Leiden algorithm with mosaic tiling constraints.

    Prevents cells of the same type from having overlapping territories.

    Parameters
    ----------
    graph : igraph.Graph
        Graph with edges representing feature similarity
    positions : np.ndarray, shape (n_cells, 2)
        Spatial coordinates of cells (in microns)
    resolution_parameter : float, default=1.0
        Higher values -> more clusters
    territory_radius : float or dict, optional
        Territory radius. If None, estimated from local density.
    overlap_threshold : float, default=2.0
        Cells closer than overlap_threshold * (r1+r2) are overlapping.
        - 2.0 = just touching (recommended for retinal mosaics)
        - 1.0 = 50% overlap allowed
        - 0.5 = very strict, no overlap
    overlap_penalty : float, default=1000.0
        Penalty per overlapping pair. Higher = stricter enforcement.
    n_iterations : int, default=2
        Leiden iterations (2 is usually sufficient)
    seed : int, optional
        Random seed

    Returns
    -------
    clusters : np.ndarray
        Cluster assignments
    partition : MosaicLeidenPartition
        Partition object
    """
    print(f"Mosaic Leiden Clustering")
    print(f"  Resolution: {resolution_parameter}")
    print(f"  Overlap threshold: {overlap_threshold}")
    print(f"  Overlap penalty: {overlap_penalty}")

    # Run Leiden with custom partition class
    partition = leidenalg.find_partition(
        graph,
        partition_type=MosaicLeidenPartition,
        weights='weight' if 'weight' in graph.es.attributes() else None,
        n_iterations=n_iterations,
        seed=seed,
        resolution_parameter=resolution_parameter,
        # Custom parameters for MosaicLeidenPartition
        positions=positions,
        territory_radius=territory_radius,
        overlap_threshold=overlap_threshold,
        overlap_penalty=overlap_penalty
    )

    n_clusters = len(set(partition.membership))
    print(f"  Result: {n_clusters} clusters")

    return np.array(partition.membership), partition


def analyze_mosaic_quality(clusters, positions, territory_radius=None):
    """
    Analyze the quality of the mosaic tiling.

    Returns
    -------
    dict with:
        - coverage_factor: mean cells per spatial unit
        - overlap_pairs: number of overlapping pairs per cluster
        - spatial_spread: average cluster diameter
    """
    from sklearn.neighbors import NearestNeighbors

    # Estimate radii if not provided
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

        # Spatial spread
        centroid = cluster_pos.mean(axis=0)
        max_dist = np.max(np.linalg.norm(cluster_pos - centroid, axis=1))

        # Count overlaps
        n_overlaps = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                dist = np.linalg.norm(cluster_pos[i] - cluster_pos[j])
                r_sum = radii[idx_i] + radii[idx_j]
                if dist < 2.0 * r_sum:  # threshold of 2.0
                    n_overlaps += 1

        results[cluster_id] = {
            'n_cells': len(indices),
            'diameter': max_dist * 2,
            'n_overlaps': n_overlaps,
            'overlap_ratio': n_overlaps / len(indices) if len(indices) > 0 else 0
        }

    return results

def tune_mosaic_parameters(
        graph, positions,
        partition_type=MosaicLeidenPartition,
        radius_range=[5, 10, 15, 20],
        res_range=[0.8, 1.0, 1.2, 1.5],
):
    """
    Grid search to find the best spatial constraints.
    """
    results = []

    print(f"{'Radius':<10} | {'Res':<10} | {'Clusters':<10} | {'Regularity (RI)':<15}")
    print("-" * 55)

    for r in radius_range:
        for res in res_range:
            # Run the Leiden optimization with these params
            # Note: Using the MosaicLeidenPartition class from previous step
            clusters, _ = spatially_constrained_leiden(
                graph, positions,
                partition_type=partition_type,
                territory_radius=r,
                resolution_parameter=res,
                overlap_penalty=2000.0,  # Keep this high
                n_iterations=5
            )

            num_clusters = len(np.unique(clusters))
            ri_score = evaluate_mosaic_quality(positions, clusters)

            results.append({
                'radius': r,
                'resolution': res,
                'n_clusters': num_clusters,
                'ri': ri_score
            })

            print(f"{r:<10} | {res:<10} | {num_clusters:<10} | {ri_score:<15.3f}")

    return pd.DataFrame(results)


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
    import networkx as nx
    from scipy.spatial import Delaunay

    # Build spatial graph
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

        # Find connected components
        subgraph = spatial_graph.subgraph(cluster_cells)
        components = list(nx.connected_components(subgraph))

        if len(components) > 1:
            # Keep largest component with original ID
            largest = max(components, key=len)

            # Assign new IDs to other components
            for comp in components:
                if comp != largest and len(comp) >= min_cluster_size:
                    for cell in comp:
                        new_clusters[cell] = next_cluster_id
                    next_cluster_id += 1

    return new_clusters


def find_optimal_resolution(graph, positions, resolutions, target_clusters=(35, 45), verbose=False, **kwargs):
    """
    Sweep resolution parameter to find value that gives target number of clusters.
    """
    if verbose:
        print(f"Searching for resolution to get {target_clusters[0]}-{target_clusters[1]} clusters...")

    results = []
    for res in resolutions:
        clusters, _ = spatially_constrained_leiden(
            graph, positions,
            resolution_parameter=res,
            verbose=verbose,
            **kwargs
        )
        n_clusters = len(np.unique(clusters))
        results.append((res, n_clusters))
        if verbose:
            print(f"  Resolution {res:.2f} -> {n_clusters} clusters")

        if target_clusters[0] <= n_clusters <= target_clusters[1]:
            if verbose:
                print(f"\n✓ Found good resolution: {res:.2f} gives {n_clusters} clusters")
            return res

    # If not found, pick closest
    results = np.array(results)
    target_mid = (target_clusters[0] + target_clusters[1]) / 2
    idx = np.argmin(np.abs(results[:, 1] - target_mid))
    best_res = results[idx, 0]
    if verbose:
        print(f"\nClosest match: resolution {best_res:.2f} gives {int(results[idx, 1])} clusters")
    return best_res


# Example usage
if __name__ == "__main__":
    # Simulate retinal mosaic data
    np.random.seed(42)
    n_cells = 2000

    # Generate positions (uniform with some structure)
    positions = np.random.rand(n_cells, 2) * 1000  # 1000 x 1000 micron area

    # Generate features (50 features, with cluster structure)
    n_features = 50
    n_types = 40
    true_types = np.random.randint(0, n_types, n_cells)
    features = np.random.randn(n_cells, n_features)

    # Add cluster structure to features
    for i in range(n_types):
        mask = true_types == i
        if np.any(mask):
            features[mask] += np.random.randn(n_features) * 2

    # Build graph
    G = build_spatial_feature_graph(positions, features)

    opt_kws = dict(
        overlap_threshold=0.5,
        overlap_penalty=1000.0,
        require_adjacency=False,
        n_iterations=10,
        seed=42
    )

    optimal_res = find_optimal_resolution(
        G, positions, target_clusters=(35, 45),
        resolutions=np.linspace(0.5, 3.0, 15), **opt_kws)

    # Run spatially-constrained Leiden
    clusters, partition = spatially_constrained_leiden(
        G,
        positions,
        **opt_kws,
        resolution_parameter=optimal_res,
    )

    print(f"\nFinal result: {len(np.unique(clusters))} clusters")
    cluster_sizes = np.bincount(clusters)
    print(f"Cluster sizes: min={cluster_sizes[cluster_sizes > 0].min()}, "
          f"max={cluster_sizes.max()}, "
          f"mean={cluster_sizes[cluster_sizes > 0].mean():.1f}")

    # Optional: post-process to ensure contiguity
    clusters_clean = post_process_enforce_contiguity(clusters, positions)
    print(f"After contiguity enforcement: {len(np.unique(clusters_clean))} clusters")
