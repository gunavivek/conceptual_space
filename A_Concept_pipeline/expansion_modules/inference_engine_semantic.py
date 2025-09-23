"""
Semantic Inference Engine: Fast embedding-based gap detection and bridge generation
Implements clustering-based gap identification with O(n log n) complexity
Replaces the O(n²-n³) pairwise comparison approach
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull, Voronoi
from scipy.spatial.distance import cdist
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

class SemanticInferenceEngine:
    """
    Fast semantic gap detection using embeddings and clustering
    O(n log n) complexity instead of O(n²-n³)
    """

    def __init__(self, embedding_manager=None, config=None):
        """
        Initialize semantic inference engine

        Args:
            embedding_manager: Instance of EmbeddingManager for vectorization
            config: Configuration dictionary with parameters
        """
        self.embedding_manager = embedding_manager

        # Default configuration
        default_config = {
            'clustering_method': 'auto',  # 'kmeans', 'dbscan', 'hdbscan', 'hierarchical', 'auto'
            'n_clusters': 'auto',  # Number of clusters or 'auto' for automatic selection
            'min_cluster_size': 5,  # Minimum points in a cluster
            'min_gap_distance': 0.3,  # Minimum distance to consider a gap
            'max_gap_distance': 0.8,  # Maximum distance for meaningful gaps
            'interpolation_method': 'spherical',  # 'linear', 'spherical', 'weighted'
            'bridge_confidence_threshold': 0.6,  # Minimum confidence for bridge concepts
            'dimensionality_reduction': True,  # Use PCA for large embeddings
            'pca_components': 50,  # Number of PCA components
            'batch_size': 100,  # Batch size for processing
            'enable_caching': True,  # Cache embeddings and clusters
            'verbose': True  # Print progress information
        }

        self.config = {**default_config, **(config or {})}

        # Caching structures
        self.embedding_cache = {}
        self.cluster_cache = {}
        self.gap_cache = {}

        # Statistics
        self.stats = {
            'total_concepts': 0,
            'clusters_found': 0,
            'gaps_detected': 0,
            'bridges_generated': 0,
            'processing_time': 0
        }

    def detect_conceptual_gaps(self, concepts):
        """
        Main entry point for semantic gap detection

        Args:
            concepts: List of concept dictionaries

        Returns:
            dict: Detected gaps, bridges, and clusters
        """
        if self.config['verbose']:
            print(f"  Processing {len(concepts)} concepts for semantic gap detection...")

        self.stats['total_concepts'] = len(concepts)

        # Step 1: Vectorize concepts (O(n))
        embeddings, concept_map = self.vectorize_concepts(concepts)

        if embeddings is None or len(embeddings) == 0:
            return self._empty_result()

        # Step 2: Reduce dimensionality if needed (O(n·d))
        if self.config['dimensionality_reduction'] and embeddings.shape[1] > self.config['pca_components']:
            embeddings = self.reduce_dimensions(embeddings)

        # Step 3: Perform clustering (O(n log n) for most algorithms)
        clusters, cluster_labels = self.perform_clustering(embeddings)

        # Step 4: Identify semantic gaps (O(k²) where k = number of clusters)
        gaps = self.identify_semantic_gaps(clusters, embeddings, cluster_labels)

        # Step 5: Generate bridge concepts (O(g) where g = number of gaps)
        bridges = self.generate_bridge_concepts(gaps, concept_map, embeddings)

        # Step 6: Detect isolated concepts (O(n))
        isolated = self.detect_isolated_concepts(cluster_labels, embeddings)

        return {
            'gaps': gaps,
            'bridges': bridges,
            'clusters': self.format_clusters(clusters, cluster_labels, concept_map),
            'isolated_concepts': isolated,
            'statistics': self.stats
        }

    def vectorize_concepts(self, concepts):
        """
        Convert concepts to embedding vectors

        Args:
            concepts: List of concept dictionaries

        Returns:
            tuple: (embeddings array, concept_id mapping)
        """
        if not self.embedding_manager:
            if self.config['verbose']:
                print("  [WARNING] No embedding manager provided, using placeholder vectors")
            return self._generate_placeholder_embeddings(concepts)

        embeddings = []
        concept_map = {}

        # Batch process for efficiency
        batch_size = self.config['batch_size']
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i + batch_size]

            for concept in batch:
                concept_id = concept.get('concept_id', f'concept_{len(concept_map)}')

                # Check cache
                if self.config['enable_caching'] and concept_id in self.embedding_cache:
                    embedding = self.embedding_cache[concept_id]
                else:
                    # Get embedding from manager
                    embedding = self.embedding_manager.get_concept_embedding(concept)

                    # Cache it
                    if self.config['enable_caching']:
                        self.embedding_cache[concept_id] = embedding

                embeddings.append(embedding)
                concept_map[len(embeddings) - 1] = concept

        return np.array(embeddings), concept_map

    def reduce_dimensions(self, embeddings):
        """
        Reduce embedding dimensions using PCA

        Args:
            embeddings: High-dimensional embeddings

        Returns:
            np.array: Reduced embeddings
        """
        if self.config['verbose']:
            print(f"  Reducing dimensions from {embeddings.shape[1]} to {self.config['pca_components']}...")

        # Standardize features
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # Apply PCA
        n_components = min(self.config['pca_components'], embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings_scaled)

        if self.config['verbose']:
            variance_explained = pca.explained_variance_ratio_.sum()
            print(f"    PCA variance explained: {variance_explained:.2%}")

        return embeddings_reduced

    def perform_clustering(self, embeddings):
        """
        Perform clustering on embeddings

        Args:
            embeddings: Concept embeddings

        Returns:
            tuple: (cluster centers/representatives, labels)
        """
        method = self.config['clustering_method']

        # Auto-select clustering method based on data characteristics
        if method == 'auto':
            method = self._select_clustering_method(embeddings)
            if self.config['verbose']:
                print(f"  Auto-selected clustering method: {method}")

        if method == 'kmeans':
            return self._cluster_kmeans(embeddings)
        elif method == 'dbscan':
            return self._cluster_dbscan(embeddings)
        elif method == 'hdbscan' and HDBSCAN_AVAILABLE:
            return self._cluster_hdbscan(embeddings)
        elif method == 'hierarchical':
            return self._cluster_hierarchical(embeddings)
        else:
            # Fallback to KMeans
            return self._cluster_kmeans(embeddings)

    def _select_clustering_method(self, embeddings):
        """Auto-select best clustering method based on data"""
        n_samples = embeddings.shape[0]

        if n_samples < 50:
            return 'hierarchical'
        elif n_samples < 500:
            return 'kmeans'
        elif HDBSCAN_AVAILABLE:
            return 'hdbscan'
        else:
            return 'dbscan'

    def _cluster_kmeans(self, embeddings):
        """KMeans clustering"""
        # Determine optimal number of clusters
        n_clusters = self.config['n_clusters']
        if n_clusters == 'auto':
            n_clusters = self._estimate_n_clusters(embeddings)

        if self.config['verbose']:
            print(f"  Running KMeans with {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        self.stats['clusters_found'] = n_clusters
        return kmeans.cluster_centers_, labels

    def _cluster_dbscan(self, embeddings):
        """DBSCAN clustering"""
        # Estimate eps using k-distance graph
        eps = self._estimate_dbscan_eps(embeddings)

        if self.config['verbose']:
            print(f"  Running DBSCAN with eps={eps:.3f}...")

        dbscan = DBSCAN(eps=eps, min_samples=self.config['min_cluster_size'])
        labels = dbscan.fit_predict(embeddings)

        # Get cluster representatives (medoids)
        unique_labels = set(labels) - {-1}  # Exclude noise
        cluster_centers = []

        for label in unique_labels:
            mask = labels == label
            cluster_points = embeddings[mask]
            # Use medoid as representative
            distances = cdist(cluster_points, cluster_points)
            medoid_idx = distances.sum(axis=1).argmin()
            cluster_centers.append(cluster_points[medoid_idx])

        self.stats['clusters_found'] = len(unique_labels)
        return np.array(cluster_centers) if cluster_centers else np.array([]), labels

    def _cluster_hdbscan(self, embeddings):
        """HDBSCAN clustering"""
        if self.config['verbose']:
            print(f"  Running HDBSCAN...")

        clusterer = HDBSCAN(
            min_cluster_size=self.config['min_cluster_size'],
            min_samples=1,
            cluster_selection_epsilon=0.0,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(embeddings)

        # Get cluster representatives
        unique_labels = set(labels) - {-1}
        cluster_centers = []

        for label in unique_labels:
            mask = labels == label
            cluster_points = embeddings[mask]
            cluster_centers.append(cluster_points.mean(axis=0))

        self.stats['clusters_found'] = len(unique_labels)
        return np.array(cluster_centers) if cluster_centers else np.array([]), labels

    def _cluster_hierarchical(self, embeddings):
        """Agglomerative hierarchical clustering"""
        n_clusters = self.config['n_clusters']
        if n_clusters == 'auto':
            n_clusters = min(10, len(embeddings) // 5)

        if self.config['verbose']:
            print(f"  Running Hierarchical clustering with {n_clusters} clusters...")

        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = clustering.fit_predict(embeddings)

        # Compute cluster centers
        cluster_centers = []
        for label in range(n_clusters):
            mask = labels == label
            cluster_centers.append(embeddings[mask].mean(axis=0))

        self.stats['clusters_found'] = n_clusters
        return np.array(cluster_centers), labels

    def _estimate_n_clusters(self, embeddings):
        """Estimate optimal number of clusters using elbow method"""
        n_samples = len(embeddings)
        max_k = min(int(np.sqrt(n_samples)), 20)

        if n_samples < 10:
            return min(3, n_samples)

        # Try different k values and find elbow
        inertias = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        # Find elbow point (simplified)
        if len(inertias) > 2:
            # Calculate second derivative
            deltas = np.diff(inertias)
            deltas2 = np.diff(deltas)
            # Find the point with maximum curvature
            elbow_idx = np.argmax(deltas2) + 2
            return k_range[elbow_idx]

        return min(5, max_k)

    def _estimate_dbscan_eps(self, embeddings):
        """Estimate eps parameter for DBSCAN using k-distance graph"""
        k = self.config['min_cluster_size']

        # Calculate k-nearest neighbors distances
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)

        # Sort k-distances
        k_distances = np.sort(distances[:, k-1])

        # Find elbow point (simplified: use 90th percentile)
        eps = np.percentile(k_distances, 90)

        # Bound eps to reasonable range
        eps = max(self.config['min_gap_distance'], min(eps, self.config['max_gap_distance']))

        return eps

    def identify_semantic_gaps(self, cluster_centers, embeddings, labels):
        """
        Identify gaps in semantic space

        Args:
            cluster_centers: Cluster centroids or representatives
            embeddings: All concept embeddings
            labels: Cluster labels for each embedding

        Returns:
            list: Detected gaps with metadata
        """
        gaps = []

        if len(cluster_centers) == 0:
            return gaps

        # Method 1: Inter-cluster gaps
        inter_cluster_gaps = self._find_inter_cluster_gaps(cluster_centers)
        gaps.extend(inter_cluster_gaps)

        # Method 2: Density-based gaps
        density_gaps = self._find_density_gaps(embeddings, labels)
        gaps.extend(density_gaps)

        # Method 3: Voronoi-based gaps (for 2D/3D or reduced space)
        if embeddings.shape[1] <= 3:
            voronoi_gaps = self._find_voronoi_gaps(embeddings)
            gaps.extend(voronoi_gaps)

        self.stats['gaps_detected'] = len(gaps)

        # Rank and filter gaps
        gaps = self._rank_gaps(gaps)

        return gaps

    def _find_inter_cluster_gaps(self, cluster_centers):
        """Find gaps between cluster pairs"""
        if len(cluster_centers) < 2:
            return []

        gaps = []
        distances = cdist(cluster_centers, cluster_centers, metric='euclidean')

        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                distance = distances[i, j]

                # Check if this represents a significant gap
                if self.config['min_gap_distance'] < distance < self.config['max_gap_distance']:
                    gaps.append({
                        'type': 'inter_cluster',
                        'cluster_1': i,
                        'cluster_2': j,
                        'distance': float(distance),
                        'center_1': cluster_centers[i],
                        'center_2': cluster_centers[j],
                        'gap_point': (cluster_centers[i] + cluster_centers[j]) / 2,
                        'confidence': 1.0 - (distance / self.config['max_gap_distance'])
                    })

        return gaps

    def _find_density_gaps(self, embeddings, labels):
        """Find low-density regions in embedding space"""
        gaps = []

        # Skip if too few points
        if len(embeddings) < 10:
            return gaps

        # Compute local density for each point
        from sklearn.neighbors import KernelDensity

        kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
        kde.fit(embeddings)

        # Generate grid points to test density
        n_samples = min(100, len(embeddings))

        # Sample points between existing embeddings
        min_vals = embeddings.min(axis=0)
        max_vals = embeddings.max(axis=0)

        # Generate random test points
        test_points = np.random.uniform(min_vals, max_vals, (n_samples, embeddings.shape[1]))

        # Compute density at test points
        log_density = kde.score_samples(test_points)
        density = np.exp(log_density)

        # Find low-density points
        threshold = np.percentile(density, 10)  # Bottom 10% density

        for i, (point, d) in enumerate(zip(test_points, density)):
            if d < threshold:
                # Find nearest clusters
                if len(labels[labels >= 0]) > 0:  # Has valid clusters
                    distances_to_points = cdist([point], embeddings)[0]
                    nearest_idx = np.argmin(distances_to_points)
                    nearest_cluster = labels[nearest_idx] if labels[nearest_idx] >= 0 else -1

                    gaps.append({
                        'type': 'density',
                        'gap_point': point,
                        'density': float(d),
                        'nearest_cluster': int(nearest_cluster),
                        'confidence': 1.0 - (d / threshold)
                    })

        return gaps[:20]  # Limit number of density gaps

    def _find_voronoi_gaps(self, embeddings):
        """Find gaps using Voronoi diagram analysis"""
        gaps = []

        try:
            if len(embeddings) < 4:  # Need at least 4 points for Voronoi
                return gaps

            vor = Voronoi(embeddings)

            # Find large Voronoi cells (indicating sparse regions)
            for i, region in enumerate(vor.regions):
                if len(region) > 0 and -1 not in region:
                    vertices = vor.vertices[region]
                    if len(vertices) > 0:
                        # Calculate cell volume/area
                        try:
                            hull = ConvexHull(vertices)
                            volume = hull.volume

                            # Large cells indicate gaps
                            if volume > np.percentile([ConvexHull(vor.vertices[r]).volume
                                                      for r in vor.regions
                                                      if len(r) > 0 and -1 not in r], 75):
                                center = vertices.mean(axis=0)
                                gaps.append({
                                    'type': 'voronoi',
                                    'gap_point': center,
                                    'volume': float(volume),
                                    'confidence': min(1.0, volume / 10.0)
                                })
                        except:
                            pass  # Skip problematic cells
        except:
            pass  # Voronoi might fail for some configurations

        return gaps[:10]  # Limit Voronoi gaps

    def _rank_gaps(self, gaps):
        """Rank and filter gaps by importance"""
        if not gaps:
            return []

        # Sort by confidence
        gaps_sorted = sorted(gaps, key=lambda g: g.get('confidence', 0), reverse=True)

        # Filter by minimum confidence
        gaps_filtered = [g for g in gaps_sorted
                        if g.get('confidence', 0) >= self.config['bridge_confidence_threshold']]

        # Limit to top gaps
        max_gaps = 50
        return gaps_filtered[:max_gaps]

    def generate_bridge_concepts(self, gaps, concept_map, embeddings):
        """
        Generate bridge concepts to fill identified gaps

        Args:
            gaps: List of identified gaps
            concept_map: Mapping from embedding index to concept
            embeddings: Original embeddings

        Returns:
            list: Bridge concepts
        """
        bridges = []

        for gap in gaps:
            if gap['type'] == 'inter_cluster':
                bridge = self._generate_inter_cluster_bridge(gap, concept_map, embeddings)
            elif gap['type'] == 'density':
                bridge = self._generate_density_bridge(gap, concept_map, embeddings)
            elif gap['type'] == 'voronoi':
                bridge = self._generate_voronoi_bridge(gap, concept_map, embeddings)
            else:
                continue

            if bridge:
                bridges.append(bridge)

        self.stats['bridges_generated'] = len(bridges)
        return bridges

    def _generate_inter_cluster_bridge(self, gap, concept_map, embeddings):
        """Generate bridge concept for inter-cluster gap"""
        # Interpolate between cluster centers
        if self.config['interpolation_method'] == 'spherical':
            bridge_vector = self._spherical_interpolation(gap['center_1'], gap['center_2'])
        elif self.config['interpolation_method'] == 'weighted':
            bridge_vector = self._weighted_interpolation(gap['center_1'], gap['center_2'], 0.5)
        else:  # linear
            bridge_vector = gap['gap_point']

        # Find nearest concepts to the bridge point
        nearest_concepts = self._find_nearest_concepts(bridge_vector, embeddings, concept_map, k=5)

        # Synthesize keywords
        bridge_keywords = self._synthesize_keywords(nearest_concepts)

        return {
            'concept_id': f"bridge_cluster_{gap['cluster_1']}_{gap['cluster_2']}",
            'canonical_name': f"Bridge: Cluster {gap['cluster_1']} to {gap['cluster_2']}",
            'keywords': bridge_keywords,
            'concept_type': 'inferred_bridge',
            'bridge_type': 'inter_cluster',
            'confidence': gap['confidence'],
            'source_clusters': [gap['cluster_1'], gap['cluster_2']],
            'embedding_vector': bridge_vector.tolist()
        }

    def _generate_density_bridge(self, gap, concept_map, embeddings):
        """Generate bridge concept for density gap"""
        bridge_vector = gap['gap_point']

        # Find nearest concepts
        nearest_concepts = self._find_nearest_concepts(bridge_vector, embeddings, concept_map, k=3)

        # Synthesize keywords
        bridge_keywords = self._synthesize_keywords(nearest_concepts, prefix="sparse_region")

        return {
            'concept_id': f"bridge_density_{len(bridge_keywords)}",
            'canonical_name': "Bridge: Sparse Region",
            'keywords': bridge_keywords,
            'concept_type': 'inferred_bridge',
            'bridge_type': 'density',
            'confidence': gap['confidence'],
            'nearest_cluster': gap.get('nearest_cluster', -1),
            'density_score': gap.get('density', 0),
            'embedding_vector': bridge_vector.tolist()
        }

    def _generate_voronoi_bridge(self, gap, concept_map, embeddings):
        """Generate bridge concept for Voronoi gap"""
        bridge_vector = gap['gap_point']

        # Find nearest concepts
        nearest_concepts = self._find_nearest_concepts(bridge_vector, embeddings, concept_map, k=4)

        # Synthesize keywords
        bridge_keywords = self._synthesize_keywords(nearest_concepts, prefix="voronoi_gap")

        return {
            'concept_id': f"bridge_voronoi_{len(bridge_keywords)}",
            'canonical_name': "Bridge: Voronoi Gap",
            'keywords': bridge_keywords,
            'concept_type': 'inferred_bridge',
            'bridge_type': 'voronoi',
            'confidence': gap['confidence'],
            'cell_volume': gap.get('volume', 0),
            'embedding_vector': bridge_vector.tolist()
        }

    def _spherical_interpolation(self, vec1, vec2, t=0.5):
        """Spherical linear interpolation (SLERP) between vectors"""
        # Normalize vectors
        v1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        v2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)

        # Compute angle between vectors
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        theta = np.arccos(dot_product)

        if theta < 0.01:  # Vectors are nearly parallel
            return self._weighted_interpolation(vec1, vec2, t)

        # SLERP formula
        sin_theta = np.sin(theta)
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta

        interpolated = w1 * vec1 + w2 * vec2
        return interpolated

    def _weighted_interpolation(self, vec1, vec2, weight=0.5):
        """Weighted linear interpolation between vectors"""
        return (1 - weight) * vec1 + weight * vec2

    def _find_nearest_concepts(self, target_vector, embeddings, concept_map, k=5):
        """Find k nearest concepts to a target vector"""
        distances = cdist([target_vector], embeddings)[0]
        nearest_indices = np.argsort(distances)[:k]

        nearest_concepts = []
        for idx in nearest_indices:
            if idx in concept_map:
                concept = concept_map[idx]
                nearest_concepts.append({
                    'concept': concept,
                    'distance': distances[idx]
                })

        return nearest_concepts

    def _synthesize_keywords(self, nearest_concepts, prefix=None):
        """Synthesize keywords for a bridge concept from nearest neighbors"""
        keyword_freq = Counter()

        for item in nearest_concepts:
            concept = item['concept']
            weight = 1.0 / (1.0 + item['distance'])  # Inverse distance weighting

            for keyword in concept.get('keywords', []):
                keyword_freq[keyword] += weight

        # Get top keywords
        top_keywords = [kw for kw, _ in keyword_freq.most_common(10)]

        # Add prefix if provided
        if prefix:
            top_keywords.insert(0, prefix)

        # Add bridge indicator
        if "bridge" not in top_keywords:
            top_keywords.append("inferred_bridge")

        return top_keywords

    def detect_isolated_concepts(self, labels, embeddings):
        """
        Detect isolated or poorly connected concepts

        Args:
            labels: Cluster labels
            embeddings: Concept embeddings

        Returns:
            list: Isolated concepts
        """
        isolated = []

        # Find noise points (label = -1 in DBSCAN/HDBSCAN)
        noise_mask = labels == -1
        noise_indices = np.where(noise_mask)[0]

        for idx in noise_indices:
            isolated.append({
                'concept_index': int(idx),
                'cluster_label': -1,
                'isolation_type': 'noise',
                'confidence': 0.9
            })

        # Find singleton clusters
        label_counts = Counter(labels)
        singleton_labels = [label for label, count in label_counts.items()
                          if count == 1 and label >= 0]

        for label in singleton_labels:
            idx = np.where(labels == label)[0][0]
            isolated.append({
                'concept_index': int(idx),
                'cluster_label': int(label),
                'isolation_type': 'singleton',
                'confidence': 0.8
            })

        return isolated

    def format_clusters(self, cluster_centers, labels, concept_map):
        """Format cluster information for output"""
        clusters = []

        unique_labels = set(labels) - {-1}

        for label in unique_labels:
            mask = labels == label
            cluster_indices = np.where(mask)[0]

            cluster_concepts = []
            for idx in cluster_indices:
                if idx in concept_map:
                    cluster_concepts.append(concept_map[idx].get('concept_id', f'concept_{idx}'))

            clusters.append({
                'cluster_id': int(label),
                'size': len(cluster_concepts),
                'concepts': cluster_concepts[:10],  # Limit for readability
                'center': cluster_centers[label].tolist() if label < len(cluster_centers) else None
            })

        return clusters

    def _generate_placeholder_embeddings(self, concepts):
        """Generate placeholder embeddings when no embedding manager available"""
        # Use simple TF-IDF as fallback
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = []
        concept_map = {}

        for i, concept in enumerate(concepts):
            keywords = concept.get('keywords', [])
            canonical = concept.get('canonical_name', '')
            text = ' '.join(keywords + [canonical])
            texts.append(text)
            concept_map[i] = concept

        if not texts:
            return None, {}

        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        embeddings = vectorizer.fit_transform(texts).toarray()

        return embeddings, concept_map

    def _empty_result(self):
        """Return empty result structure"""
        return {
            'gaps': [],
            'bridges': [],
            'clusters': [],
            'isolated_concepts': [],
            'statistics': self.stats
        }

    def get_inference_info(self):
        """Get information about the inference engine configuration and stats"""
        return {
            'engine_type': 'semantic_embedding_based',
            'complexity': 'O(n log n)',
            'clustering_method': self.config['clustering_method'],
            'gap_threshold': self.config['min_gap_distance'],
            'bridge_confidence': self.config['bridge_confidence_threshold'],
            'statistics': self.stats,
            'cache_stats': {
                'embeddings_cached': len(self.embedding_cache),
                'clusters_cached': len(self.cluster_cache),
                'gaps_cached': len(self.gap_cache)
            }
        }

    # Compatibility methods for drop-in replacement
    def detect_concept_specific_gaps(self, concept, all_concepts):
        """
        Compatibility method for concept-specific gap detection
        Fast version that leverages cached global analysis
        """
        # Check if we need to run the analysis (only once)
        if not hasattr(self, '_global_analysis_done'):
            # Run full analysis once and cache everything
            if self.config.get('verbose'):
                print("  Running one-time global gap analysis...")

            self._global_result = self.detect_conceptual_gaps(all_concepts)
            self.gap_cache = self._global_result['gaps']
            self.cluster_cache = self._global_result['clusters']
            self._global_analysis_done = True

            if self.config.get('verbose'):
                print(f"    Cached {len(self.gap_cache)} gaps and {len(self.cluster_cache)} clusters")

        # Now just filter for this specific concept
        concept_id = concept.get('concept_id')
        return self._filter_gaps_for_concept(concept_id, self.gap_cache)

    def _filter_gaps_for_concept(self, concept_id, gaps):
        """Filter global gaps relevant to a specific concept"""
        concept_gaps = {
            'missing_intermediates': [],
            'missing_relationships': [],
            'isolation_issues': []
        }

        # Filter gaps relevant to this concept
        for gap in gaps:
            if gap.get('type') == 'inter_cluster':
                # Check if concept is in one of the clusters
                concept_gaps['missing_intermediates'].append({
                    'confidence': gap['confidence'],
                    'gap_type': gap['type']
                })
            elif gap.get('type') == 'density':
                concept_gaps['missing_relationships'].append({
                    'confidence': gap['confidence'],
                    'gap_type': gap['type']
                })

        return concept_gaps

    def fill_concept_gaps(self, concept, concept_gaps):
        """
        Compatibility method for filling concept-specific gaps
        """
        # Generate bridges based on gap types
        bridges = []
        filled_gaps = {
            'intermediate_bridges': [],
            'relationship_bridges': []
        }

        for gap in concept_gaps.get('missing_intermediates', []):
            if gap.get('confidence', 0) >= self.config['bridge_confidence_threshold']:
                bridge = {
                    'concept_id': f"bridge_{concept.get('concept_id')}",
                    'keywords': concept.get('keywords', [])[:5] + ['bridge', 'intermediate'],
                    'concept_type': 'inferred_intermediate',
                    'confidence': gap.get('confidence', 0.5),
                    'bridge_type': 'intermediate'
                }
                bridges.append(bridge)
                filled_gaps['intermediate_bridges'].append(gap)

        return {
            'bridge_concepts': bridges,
            'filled_gaps': filled_gaps
        }