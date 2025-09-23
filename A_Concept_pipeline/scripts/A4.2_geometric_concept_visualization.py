#!/usr/bin/env python3
"""
A4.2: Geometric Concept Space Visualization Module
Generates comprehensive visualizations from A4 geometric concept space data
Including interactive 3D plots, network graphs, and analytical visualizations
"""

import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from datetime import datetime
import colorsys


class A4_2_GeometricConceptVisualizer:
    """
    Visualization engine for A4 geometric concept spaces
    Creates interactive and static visualizations for concept analysis
    """

    def __init__(self, base_path: str = None):
        """Initialize the visualizer with paths and settings"""
        if base_path is None:
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = Path(base_path)

        self.outputs_dir = self.base_path / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)

        # Visualization settings
        self.color_palette = self._generate_color_palette()
        self.figure_dpi = 150
        self.interactive_height = 800
        self.interactive_width = 1200

        # Data holders
        self.geometric_space = None
        self.concept_centroids = None
        self.concept_embeddings = None
        self.concept_names = []
        self.document_ids = []

    def _generate_color_palette(self, n_colors: int = 20) -> List[str]:
        """Generate a visually distinct color palette"""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            # Use high saturation and moderate lightness for visibility
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors

    def load_geometric_space(self, json_path: Optional[str] = None) -> bool:
        """Load the A4 geometric concept space data"""
        if json_path is None:
            json_path = self.outputs_dir / "A4_geometric_concept_space.json"
        else:
            json_path = Path(json_path)

        if not json_path.exists():
            print(f"Error: Cannot find {json_path}")
            return False

        print(f"Loading geometric space from {json_path.name}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract first document's geometric space (they all share the same space)
        self.document_ids = list(data.keys())
        first_doc = self.document_ids[0] if self.document_ids else None

        if not first_doc:
            print("Error: No documents found in geometric space data")
            return False

        self.geometric_space = data[first_doc]['geometric_concept_space']
        self.concept_centroids = self.geometric_space.get('concept_centroids', {})

        # Extract embeddings and names
        self.concept_embeddings = []
        self.concept_names = []

        for concept_id, centroid_data in self.concept_centroids.items():
            if 'centroid_coordinates' in centroid_data:
                self.concept_embeddings.append(centroid_data['centroid_coordinates'])
                # Use canonical name if available, otherwise concept_id
                name = centroid_data.get('canonical_name', concept_id)
                self.concept_names.append(name[:30])  # Truncate long names

        self.concept_embeddings = np.array(self.concept_embeddings)

        print(f"Loaded {len(self.concept_names)} concepts with {self.concept_embeddings.shape[1]} dimensions")
        print(f"Documents in space: {len(self.document_ids)}")

        return True

    def create_pca_visualization(self, n_components: int = 3) -> Tuple[go.Figure, go.Figure]:
        """Create PCA dimensionality reduction visualizations (2D and 3D)"""
        print("\nGenerating PCA visualizations...")

        if len(self.concept_embeddings) < 3:
            print("Warning: Not enough concepts for PCA visualization")
            return None, None

        # Perform PCA
        pca = PCA(n_components=min(n_components, len(self.concept_embeddings)))
        reduced_embeddings = pca.fit_transform(self.concept_embeddings)

        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        total_var_3d = sum(explained_var[:3]) if len(explained_var) >= 3 else sum(explained_var)
        total_var_2d = sum(explained_var[:2]) if len(explained_var) >= 2 else sum(explained_var)

        # Create 2D visualization
        fig_2d = go.Figure()

        # Add scatter plot
        fig_2d.add_trace(go.Scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            mode='markers+text',
            marker=dict(
                size=10,
                color=list(range(len(self.concept_names))),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Concept Index")
            ),
            text=self.concept_names,
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
        ))

        fig_2d.update_layout(
            title=f'A4.2: Concept Space - PCA 2D Projection<br><sub>Explained Variance: {total_var_2d:.1%}</sub>',
            xaxis_title=f'PC1 ({explained_var[0]:.1%} variance)',
            yaxis_title=f'PC2 ({explained_var[1]:.1%} variance)' if len(explained_var) > 1 else 'PC2',
            height=self.interactive_height,
            width=self.interactive_width,
            hovermode='closest'
        )

        # Create 3D visualization if we have enough dimensions
        fig_3d = None
        if reduced_embeddings.shape[1] >= 3:
            fig_3d = go.Figure()

            fig_3d.add_trace(go.Scatter3d(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                z=reduced_embeddings[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=list(range(len(self.concept_names))),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Concept Index")
                ),
                text=self.concept_names,
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>'
            ))

            fig_3d.update_layout(
                title=f'A4.2: Concept Space - PCA 3D Projection<br><sub>Explained Variance: {total_var_3d:.1%}</sub>',
                scene=dict(
                    xaxis_title=f'PC1 ({explained_var[0]:.1%})',
                    yaxis_title=f'PC2 ({explained_var[1]:.1%})' if len(explained_var) > 1 else 'PC2',
                    zaxis_title=f'PC3 ({explained_var[2]:.1%})' if len(explained_var) > 2 else 'PC3'
                ),
                height=self.interactive_height,
                width=self.interactive_width
            )

        return fig_2d, fig_3d

    def create_tsne_visualization(self, perplexity: int = 30) -> Tuple[go.Figure, go.Figure]:
        """Create t-SNE dimensionality reduction visualizations (2D and 3D)"""
        print("\nGenerating t-SNE visualizations...")

        if len(self.concept_embeddings) < 3:
            print("Warning: Not enough concepts for t-SNE visualization")
            return None, None

        # Adjust perplexity if needed
        perplexity = min(perplexity, len(self.concept_embeddings) - 1)

        # 2D t-SNE
        tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_2d = tsne_2d.fit_transform(self.concept_embeddings)

        # Create 2D visualization
        fig_2d = go.Figure()

        fig_2d.add_trace(go.Scatter(
            x=reduced_2d[:, 0],
            y=reduced_2d[:, 1],
            mode='markers+text',
            marker=dict(
                size=10,
                color=list(range(len(self.concept_names))),
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Concept Index")
            ),
            text=self.concept_names,
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>t-SNE1: %{x:.3f}<br>t-SNE2: %{y:.3f}<extra></extra>'
        ))

        fig_2d.update_layout(
            title='A4.2: Concept Space - t-SNE 2D Projection<br><sub>Non-linear dimensionality reduction</sub>',
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            height=self.interactive_height,
            width=self.interactive_width,
            hovermode='closest'
        )

        # 3D t-SNE
        tsne_3d = TSNE(n_components=3, perplexity=perplexity, random_state=42)
        reduced_3d = tsne_3d.fit_transform(self.concept_embeddings)

        fig_3d = go.Figure()

        fig_3d.add_trace(go.Scatter3d(
            x=reduced_3d[:, 0],
            y=reduced_3d[:, 1],
            z=reduced_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=list(range(len(self.concept_names))),
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Concept Index")
            ),
            text=self.concept_names,
            hovertemplate='<b>%{text}</b><br>t-SNE1: %{x:.3f}<br>t-SNE2: %{y:.3f}<br>t-SNE3: %{z:.3f}<extra></extra>'
        ))

        fig_3d.update_layout(
            title='A4.2: Concept Space - t-SNE 3D Projection<br><sub>Non-linear dimensionality reduction</sub>',
            scene=dict(
                xaxis_title='t-SNE Dimension 1',
                yaxis_title='t-SNE Dimension 2',
                zaxis_title='t-SNE Dimension 3'
            ),
            height=self.interactive_height,
            width=self.interactive_width
        )

        return fig_2d, fig_3d

    def create_concept_similarity_heatmap(self, top_n: int = 50):
        """Create a heatmap showing concept similarities"""
        print("\nGenerating concept similarity heatmap...")

        # Limit to top_n concepts for readability
        n_concepts = min(top_n, len(self.concept_embeddings))
        subset_embeddings = self.concept_embeddings[:n_concepts]
        subset_names = self.concept_names[:n_concepts]

        # Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(subset_embeddings)

        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarities,
            x=subset_names,
            y=subset_names,
            colorscale='RdBu',
            zmid=0,
            text=similarities.round(2),
            texttemplate='%{text}',
            textfont={"size": 8},
            hovertemplate='%{x} vs %{y}<br>Similarity: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'A4.2: Concept Similarity Matrix<br><sub>Top {n_concepts} concepts - Cosine similarity</sub>',
            xaxis_title='Concepts',
            yaxis_title='Concepts',
            height=900,
            width=1000,
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8))
        )

        return fig

    def create_concept_network_graph(self, similarity_threshold: float = 0.5):
        """Create network graph showing concept relationships"""
        print("\nGenerating concept network graph...")

        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(self.concept_embeddings)

        # Create network graph
        G = nx.Graph()

        # Add nodes
        for i, name in enumerate(self.concept_names):
            G.add_node(i, label=name)

        # Add edges for similar concepts
        for i in range(len(similarities)):
            for j in range(i+1, len(similarities)):
                if similarities[i][j] > similarity_threshold:
                    G.add_edge(i, j, weight=similarities[i][j])

        # Use spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Create Plotly figure
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color='#888'),
            hoverinfo='none', mode='lines'
        )

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)

        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            ),
            text=[], textposition="top center"
        )

        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)

        # Color by degree
        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(f'{self.concept_names[node]}<br>Connections: {len(adjacencies[1])}')

        node_trace['marker']['color'] = node_adjacencies
        node_trace['text'] = self.concept_names
        node_trace['hovertext'] = node_text

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=f'A4.2: Concept Relationship Network<br><sub>Similarity threshold: {similarity_threshold}</sub>',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=self.interactive_height,
            width=self.interactive_width
        )

        return fig

    def create_concept_clustering_visualization(self, n_clusters: int = 5):
        """Visualize concept clusters using K-means"""
        print("\nGenerating concept clustering visualization...")

        from sklearn.cluster import KMeans

        # Perform clustering
        n_clusters = min(n_clusters, len(self.concept_embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.concept_embeddings)

        # Reduce to 3D for visualization
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(self.concept_embeddings)

        # Create 3D scatter plot with clusters
        fig = go.Figure()

        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_points = reduced[mask]
            cluster_names = [self.concept_names[i] for i in range(len(self.concept_names)) if mask[i]]

            fig.add_trace(go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode='markers+text',
                name=f'Cluster {cluster_id + 1}',
                marker=dict(size=8),
                text=cluster_names,
                hovertemplate='<b>%{text}</b><br>Cluster: ' + f'{cluster_id + 1}' + '<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ))

        # Add cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        fig.add_trace(go.Scatter3d(
            x=centers_pca[:, 0],
            y=centers_pca[:, 1],
            z=centers_pca[:, 2],
            mode='markers',
            name='Cluster Centers',
            marker=dict(
                size=15,
                color='black',
                symbol='diamond'
            ),
            hovertemplate='Cluster Center<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'A4.2: Concept Clustering Analysis<br><sub>K-means with {n_clusters} clusters</sub>',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            height=self.interactive_height,
            width=self.interactive_width
        )

        return fig

    def create_convex_hull_visualization(self):
        """Create visualization of concept space convex hulls"""
        print("\nGenerating convex hull visualization...")

        # Reduce to 2D for convex hull visualization
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.concept_embeddings)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot points
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1],
                           c=range(len(self.concept_names)),
                           cmap='viridis', s=100, alpha=0.7)

        # Try to compute convex hull
        if len(reduced) >= 3:
            try:
                hull = ConvexHull(reduced)

                # Plot hull
                for simplex in hull.simplices:
                    ax.plot(reduced[simplex, 0], reduced[simplex, 1], 'r-', alpha=0.3)

                # Fill hull
                hull_points = reduced[hull.vertices]
                ax.fill(hull_points[:, 0], hull_points[:, 1],
                       alpha=0.1, color='red', label='Convex Hull')
            except:
                print("Could not compute convex hull")

        # Add labels for some points
        for i in range(min(20, len(self.concept_names))):
            ax.annotate(self.concept_names[i],
                       xy=(reduced[i, 0], reduced[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)

        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_title('A4.2: Concept Space Convex Hull\nPCA-reduced 2D projection')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Concept Index')

        if len(reduced) >= 3:
            ax.legend()

        plt.tight_layout()
        return fig

    def create_concept_distance_distribution(self):
        """Create distribution plots of concept distances"""
        print("\nGenerating distance distribution plots...")

        # Calculate pairwise distances
        distances = pdist(self.concept_embeddings, metric='euclidean')
        cosine_distances = pdist(self.concept_embeddings, metric='cosine')

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Euclidean Distance Distribution',
                          'Cosine Distance Distribution',
                          'Euclidean Distance Matrix',
                          'Distance Statistics'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'heatmap'}, {'type': 'table'}]]
        )

        # Euclidean histogram
        fig.add_trace(
            go.Histogram(x=distances, nbinsx=50, name='Euclidean',
                        marker_color='blue', opacity=0.7),
            row=1, col=1
        )

        # Cosine histogram
        fig.add_trace(
            go.Histogram(x=cosine_distances, nbinsx=50, name='Cosine',
                        marker_color='green', opacity=0.7),
            row=1, col=2
        )

        # Distance matrix heatmap (subset for visibility)
        n_subset = min(30, len(self.concept_embeddings))
        dist_matrix_subset = squareform(distances)[:n_subset, :n_subset]

        fig.add_trace(
            go.Heatmap(z=dist_matrix_subset, colorscale='Viridis',
                      x=self.concept_names[:n_subset],
                      y=self.concept_names[:n_subset]),
            row=2, col=1
        )

        # Statistics table
        stats_data = {
            'Metric': ['Euclidean', 'Euclidean', 'Euclidean', 'Cosine', 'Cosine', 'Cosine'],
            'Statistic': ['Mean', 'Std', 'Median', 'Mean', 'Std', 'Median'],
            'Value': [
                f'{np.mean(distances):.3f}',
                f'{np.std(distances):.3f}',
                f'{np.median(distances):.3f}',
                f'{np.mean(cosine_distances):.3f}',
                f'{np.std(cosine_distances):.3f}',
                f'{np.median(cosine_distances):.3f}'
            ]
        }

        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Statistic', 'Value'],
                           fill_color='lightgrey'),
                cells=dict(values=[stats_data['Metric'],
                                 stats_data['Statistic'],
                                 stats_data['Value']])
            ),
            row=2, col=2
        )

        fig.update_layout(
            title='A4.2: Concept Space Distance Analysis',
            height=900,
            width=1400,
            showlegend=False
        )

        return fig

    def save_all_visualizations(self):
        """Generate and save all visualizations"""
        print("\n" + "="*70)
        print("A4.2: GENERATING ALL VISUALIZATIONS")
        print("="*70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # PCA visualizations
        pca_2d, pca_3d = self.create_pca_visualization()
        if pca_2d:
            pca_2d_path = self.outputs_dir / f"A4.2_concept_space_pca_2d_{timestamp}.html"
            pca_2d.write_html(str(pca_2d_path))
            print(f"[OK] Saved PCA 2D to {pca_2d_path.name}")

        if pca_3d:
            pca_3d_path = self.outputs_dir / f"A4.2_concept_space_pca_3d_{timestamp}.html"
            pca_3d.write_html(str(pca_3d_path))
            print(f"[OK] Saved PCA 3D to {pca_3d_path.name}")

        # t-SNE visualizations
        tsne_2d, tsne_3d = self.create_tsne_visualization()
        if tsne_2d:
            tsne_2d_path = self.outputs_dir / f"A4.2_concept_space_tsne_2d_{timestamp}.html"
            tsne_2d.write_html(str(tsne_2d_path))
            print(f"[OK] Saved t-SNE 2D to {tsne_2d_path.name}")

        if tsne_3d:
            tsne_3d_path = self.outputs_dir / f"A4.2_concept_space_tsne_3d_{timestamp}.html"
            tsne_3d.write_html(str(tsne_3d_path))
            print(f"[OK] Saved t-SNE 3D to {tsne_3d_path.name}")

        # Similarity heatmap
        heatmap = self.create_concept_similarity_heatmap()
        heatmap_path = self.outputs_dir / f"A4.2_concept_similarity_heatmap_{timestamp}.html"
        heatmap.write_html(str(heatmap_path))
        print(f"[OK] Saved similarity heatmap to {heatmap_path.name}")

        # Network graph
        network = self.create_concept_network_graph()
        network_path = self.outputs_dir / f"A4.2_concept_network_{timestamp}.html"
        network.write_html(str(network_path))
        print(f"[OK] Saved network graph to {network_path.name}")

        # Clustering visualization
        clustering = self.create_concept_clustering_visualization()
        clustering_path = self.outputs_dir / f"A4.2_concept_clusters_{timestamp}.html"
        clustering.write_html(str(clustering_path))
        print(f"[OK] Saved clustering visualization to {clustering_path.name}")

        # Convex hull
        convex_hull_fig = self.create_convex_hull_visualization()
        convex_hull_path = self.outputs_dir / f"A4.2_convex_hull_{timestamp}.png"
        convex_hull_fig.savefig(str(convex_hull_path), dpi=self.figure_dpi, bbox_inches='tight')
        plt.close(convex_hull_fig)
        print(f"[OK] Saved convex hull to {convex_hull_path.name}")

        # Distance analysis
        distance_fig = self.create_concept_distance_distribution()
        distance_path = self.outputs_dir / f"A4.2_distance_analysis_{timestamp}.html"
        distance_fig.write_html(str(distance_path))
        print(f"[OK] Saved distance analysis to {distance_path.name}")

        # Create summary dashboard
        self.create_summary_dashboard(timestamp)

        print("\n" + "="*70)
        print("A4.2: VISUALIZATION GENERATION COMPLETE!")
        print("="*70)
        print(f"\nTotal visualizations created: 9")
        print(f"Output directory: {self.outputs_dir}")

        return timestamp

    def create_summary_dashboard(self, timestamp: str):
        """Create an interactive summary dashboard combining key visualizations"""
        print("\nGenerating summary dashboard...")

        # Create subplot figure with multiple visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Concept Distribution (PCA)',
                          'Concept Clusters',
                          'Top Concept Similarities',
                          'Distance Statistics'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # PCA 2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.concept_embeddings)

        fig.add_trace(
            go.Scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                mode='markers',
                marker=dict(size=8, color=range(len(self.concept_names)), colorscale='Viridis'),
                text=self.concept_names,
                hovertemplate='%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Clustering
        from sklearn.cluster import KMeans
        n_clusters = min(5, len(self.concept_embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.concept_embeddings)

        fig.add_trace(
            go.Scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                mode='markers',
                marker=dict(size=8, color=cluster_labels, colorscale='Plasma'),
                text=self.concept_names,
                hovertemplate='%{text}<br>Cluster: %{marker.color}<extra></extra>'
            ),
            row=1, col=2
        )

        # Top similarities heatmap
        from sklearn.metrics.pairwise import cosine_similarity
        n_top = min(15, len(self.concept_embeddings))
        similarities = cosine_similarity(self.concept_embeddings[:n_top])

        fig.add_trace(
            go.Heatmap(
                z=similarities,
                x=self.concept_names[:n_top],
                y=self.concept_names[:n_top],
                colorscale='RdBu',
                zmid=0,
                showscale=False
            ),
            row=2, col=1
        )

        # Distance statistics bar chart
        distances = pdist(self.concept_embeddings, metric='euclidean')
        bins, edges = np.histogram(distances, bins=20)

        fig.add_trace(
            go.Bar(
                x=edges[:-1],
                y=bins,
                marker_color='lightblue',
                hovertemplate='Distance: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title='A4.2: Geometric Concept Space - Summary Dashboard',
            height=900,
            width=1400,
            showlegend=False
        )

        # Update axes labels
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="PC1", row=1, col=2)
        fig.update_yaxes(title_text="PC2", row=1, col=2)
        fig.update_xaxes(title_text="", row=2, col=1, tickfont=dict(size=8))
        fig.update_yaxes(title_text="", row=2, col=1, tickfont=dict(size=8))
        fig.update_xaxes(title_text="Distance", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)

        # Save dashboard
        dashboard_path = self.outputs_dir / f"A4.2_summary_dashboard_{timestamp}.html"
        fig.write_html(str(dashboard_path))
        print(f"[OK] Saved summary dashboard to {dashboard_path.name}")

        return fig


def main():
    """Main execution function"""
    print("="*70)
    print("A4.2: GEOMETRIC CONCEPT SPACE VISUALIZATION MODULE")
    print("="*70)

    # Initialize visualizer
    visualizer = A4_2_GeometricConceptVisualizer()

    # Load geometric space data
    if not visualizer.load_geometric_space():
        print("Failed to load geometric space data. Exiting.")
        return

    # Generate all visualizations
    timestamp = visualizer.save_all_visualizations()

    # Print summary statistics
    print("\n" + "="*70)
    print("VISUALIZATION SUMMARY STATISTICS:")
    print("="*70)
    print(f"Total concepts visualized: {len(visualizer.concept_names)}")
    print(f"Embedding dimensions: {visualizer.concept_embeddings.shape[1]}")
    print(f"Documents in space: {len(visualizer.document_ids)}")

    if visualizer.geometric_space:
        metadata = visualizer.geometric_space.get('document_metadata', {})
        print(f"Total chunks mapped: {metadata.get('total_mapped_chunks', 0)}")
        print(f"Average chunks per concept: {metadata.get('average_chunk_per_concept', 0):.2f}")

    print(f"\nAll visualizations saved with timestamp: {timestamp}")
    print("Visualization generation complete!")


if __name__ == "__main__":
    main()