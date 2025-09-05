#!/usr/bin/env python3
"""
Conceptual Space Visualization System
Interactive 3D visualization of concepts as convex balls in semantic space
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class ConceptualSpaceVisualizer:
    """
    Creates interactive visualizations of concepts as convex balls in semantic space
    """
    
    def __init__(self, concepts_file="A_Concept_pipeline/outputs/A2.4_core_concepts.json"):
        self.concepts_file = Path(concepts_file)
        self.concepts = None
        self.concept_df = None
        self.embeddings = None
        self.coordinates_3d = None
        self.overlaps = None
        self.graph = None
        
        # Color scheme for business domains
        self.domain_colors = {
            'Financial': '#2E86AB',      # Blue
            'Operational': '#A23B72',     # Purple
            'Tax': '#F18F01',            # Orange
            'Accounting': '#C73E1D',      # Red
            'Other': '#6C6C6C'            # Gray
        }
        
    def load_and_process_concepts(self):
        """Load concepts and extract features"""
        with open(self.concepts_file, 'r') as f:
            data = json.load(f)
        
        self.concepts = data.get("core_concepts", [])
        
        # Build concept dataframe
        concept_data = []
        for concept in self.concepts:
            # Collect all keywords
            all_keywords = set()
            for instance in concept.get("document_instances", []):
                all_keywords.update(instance.get("keywords", []))
            
            # Categorize by domain
            domain = self._categorize_domain(concept["canonical_name"])
            
            concept_data.append({
                "concept_id": concept["concept_id"],
                "name": concept["canonical_name"],
                "importance": concept["importance_score"],
                "doc_count": concept["document_count"],
                "coverage": concept["coverage_ratio"],
                "keywords": list(all_keywords),
                "keyword_text": " ".join(all_keywords),
                "keyword_count": len(all_keywords),
                "domain": domain,
                "color": self.domain_colors[domain],
                "radius": min(len(all_keywords) / 15.0, 1.0) * concept["importance_score"],
                "documents": concept.get("related_documents", [])
            })
        
        self.concept_df = pd.DataFrame(concept_data)
        print(f"Loaded {len(self.concepts)} concepts")
        
    def _categorize_domain(self, name):
        """Categorize concept by business domain"""
        name_lower = name.lower()
        
        if any(term in name_lower for term in ['income', 'revenue', 'balance', 'receivable', 'contract', 'deferred']):
            return 'Financial'
        elif any(term in name_lower for term in ['operation', 'inventory', 'valuation', 'process']):
            return 'Operational'
        elif any(term in name_lower for term in ['tax', 'twdv']):
            return 'Tax'
        elif any(term in name_lower for term in ['nbv', 'net', 'book', 'unearned']):
            return 'Accounting'
        else:
            return 'Other'
    
    def compute_semantic_embeddings(self):
        """Compute semantic embeddings for concepts"""
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform(self.concept_df["keyword_text"])
        
        # Compute similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # 3D coordinates using t-SNE for better separation
        tsne_3d = TSNE(n_components=3, perplexity=min(5, len(self.concepts)-1), 
                       random_state=42, n_iter=1000)
        self.coordinates_3d = tsne_3d.fit_transform(tfidf_matrix.toarray())
        
        # Also compute PCA for comparison
        pca = PCA(n_components=min(3, len(self.concepts)-1))
        self.pca_coords = pca.fit_transform(tfidf_matrix.toarray())
        
        # Store embeddings
        self.embeddings = tfidf_matrix.toarray()
        self.similarities = similarities
        
        print(f"Computed embeddings: {self.embeddings.shape}")
        
    def detect_overlaps(self):
        """Detect overlapping concept balls"""
        overlaps = []
        n = len(self.concept_df)
        
        for i in range(n):
            for j in range(i+1, n):
                # Calculate distance between centers
                dist = np.linalg.norm(self.coordinates_3d[i] - self.coordinates_3d[j])
                
                # Get radii
                r1 = self.concept_df.iloc[i]["radius"]
                r2 = self.concept_df.iloc[j]["radius"]
                
                # Check for overlap
                if dist < (r1 + r2):
                    overlap_strength = max(0, 1 - dist / (r1 + r2))
                    
                    overlaps.append({
                        "source": i,
                        "target": j,
                        "distance": dist,
                        "overlap": overlap_strength,
                        "similarity": self.similarities[i, j]
                    })
        
        self.overlaps = pd.DataFrame(overlaps)
        print(f"Detected {len(overlaps)} overlapping concept pairs")
        
    def create_3d_visualization(self):
        """Create interactive 3D visualization of conceptual space"""
        fig = go.Figure()
        
        # Normalize coordinates for better visualization
        coords = self.coordinates_3d
        coords_normalized = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-10)
        
        # 1. Add edges for overlapping concepts
        if not self.overlaps.empty:
            for _, edge in self.overlaps.iterrows():
                if edge["overlap"] > 0.3:  # Only show significant overlaps
                    i, j = int(edge["source"]), int(edge["target"])
                    
                    fig.add_trace(go.Scatter3d(
                        x=[coords_normalized[i, 0], coords_normalized[j, 0]],
                        y=[coords_normalized[i, 1], coords_normalized[j, 1]],
                        z=[coords_normalized[i, 2], coords_normalized[j, 2]],
                        mode='lines',
                        line=dict(
                            color='rgba(150, 150, 150, 0.3)',
                            width=edge["overlap"] * 5
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # 2. Add concept balls (spheres)
        for idx, row in self.concept_df.iterrows():
            # Create sphere mesh for each concept
            radius = row["radius"] * 0.5  # Scale for visualization
            center = coords_normalized[idx]
            
            # Generate sphere points
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            
            # Add transparent sphere
            fig.add_trace(go.Surface(
                x=x_sphere,
                y=y_sphere,
                z=z_sphere,
                showscale=False,
                opacity=0.3,
                surfacecolor=[[row["importance"]] * x_sphere.shape[1]] * x_sphere.shape[0],
                colorscale=[[0, row["color"]], [1, row["color"]]],
                showlegend=False,
                hoverinfo='skip',
                name=row["concept_id"]
            ))
            
            # Add center point
            hover_text = (
                f"<b>{row['concept_id']}: {row['name']}</b><br>"
                f"Importance: {row['importance']:.3f}<br>"
                f"Keywords: {row['keyword_count']}<br>"
                f"Documents: {row['doc_count']}<br>"
                f"Coverage: {row['coverage']:.1%}<br>"
                f"Domain: {row['domain']}<br>"
                f"Top Keywords: {', '.join(row['keywords'][:5])}"
            )
            
            fig.add_trace(go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode='markers+text',
                marker=dict(
                    size=10 + row["importance"] * 20,
                    color=row["color"],
                    line=dict(color='white', width=2),
                    symbol='circle'
                ),
                text=row["concept_id"],
                textposition="top center",
                textfont=dict(size=10, color=row["color"]),
                showlegend=False,
                hovertext=hover_text,
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Conceptual Space: 3D Interactive Visualization",
                'font': {'size': 20}
            },
            scene=dict(
                xaxis=dict(title="Semantic Dimension 1", showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                yaxis=dict(title="Semantic Dimension 2", showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                zaxis=dict(title="Semantic Dimension 3", showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                bgcolor='rgba(240,240,240,0.9)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            height=800,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Add domain legend
        for domain, color in self.domain_colors.items():
            if domain in self.concept_df['domain'].values:
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    showlegend=True,
                    name=domain
                ))
        
        return fig
    
    def create_network_graph(self):
        """Create 2D network graph visualization"""
        # Build network
        G = nx.Graph()
        
        # Add nodes
        for idx, row in self.concept_df.iterrows():
            G.add_node(idx, **row.to_dict())
        
        # Add edges based on overlaps
        if not self.overlaps.empty:
            for _, edge in self.overlaps.iterrows():
                if edge["overlap"] > 0.3:
                    G.add_edge(int(edge["source"]), int(edge["target"]), 
                              weight=edge["overlap"], similarity=edge["similarity"])
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=edge[2]['weight'] * 3, color='rgba(150,150,150,0.5)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add nodes
        for node_id, node_data in G.nodes(data=True):
            x, y = pos[node_id]
            
            hover_text = (
                f"<b>{node_data['concept_id']}: {node_data['name']}</b><br>"
                f"Importance: {node_data['importance']:.3f}<br>"
                f"Keywords: {node_data['keyword_count']}<br>"
                f"Domain: {node_data['domain']}"
            )
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=20 + node_data['importance'] * 40,
                    color=node_data['color'],
                    line=dict(color='white', width=2)
                ),
                text=node_data['concept_id'],
                textposition="top center",
                showlegend=False,
                hovertext=hover_text,
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title="Concept Network Graph",
            showlegend=False,
            hovermode='closest',
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(240,240,240,0.9)'
        )
        
        return fig
    
    def create_overlap_heatmap(self):
        """Create heatmap of concept overlaps/similarities"""
        # Create similarity matrix
        n = len(self.concept_df)
        overlap_matrix = np.zeros((n, n))
        
        for _, row in self.overlaps.iterrows():
            i, j = int(row["source"]), int(row["target"])
            overlap_matrix[i, j] = row["similarity"]
            overlap_matrix[j, i] = row["similarity"]
        
        # Add diagonal
        np.fill_diagonal(overlap_matrix, 1.0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=overlap_matrix,
            x=self.concept_df["concept_id"].tolist(),
            y=self.concept_df["concept_id"].tolist(),
            colorscale='Viridis',
            text=[[f"{val:.2f}" for val in row] for row in overlap_matrix],
            texttemplate="%{text}",
            textfont={"size": 8},
            colorbar=dict(title="Similarity"),
            hovertemplate="Concept 1: %{y}<br>Concept 2: %{x}<br>Similarity: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Concept Similarity Matrix",
            height=600,
            xaxis=dict(title="Concepts", tickangle=45),
            yaxis=dict(title="Concepts"),
        )
        
        return fig
    
    def create_importance_bubble_chart(self):
        """Create bubble chart showing concept importance and coverage"""
        fig = px.scatter(
            self.concept_df,
            x="coverage",
            y="importance",
            size="keyword_count",
            color="domain",
            hover_data=["concept_id", "name", "doc_count"],
            labels={
                "coverage": "Document Coverage",
                "importance": "Importance Score",
                "keyword_count": "Keywords"
            },
            title="Concept Importance vs Coverage",
            color_discrete_map=self.domain_colors
        )
        
        # Add concept labels
        for _, row in self.concept_df.iterrows():
            fig.add_annotation(
                x=row["coverage"],
                y=row["importance"],
                text=row["concept_id"],
                showarrow=False,
                font=dict(size=8)
            )
        
        fig.update_layout(
            height=600,
            xaxis=dict(tickformat=".0%"),
            hovermode='closest'
        )
        
        return fig
    
    def generate_dashboard(self):
        """Generate complete HTML dashboard with all visualizations"""
        # Create all visualizations
        fig_3d = self.create_3d_visualization()
        fig_network = self.create_network_graph()
        fig_heatmap = self.create_overlap_heatmap()
        fig_bubble = self.create_importance_bubble_chart()
        
        # Create HTML dashboard
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conceptual Space Visualization</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                .container {{
                    max-width: 1600px;
                    margin: 0 auto;
                }}
                .viz-section {{
                    background: white;
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }}
                .stats {{
                    background: #e8f4f8;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                }}
                .stat-card {{
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2E86AB;
                }}
                .stat-label {{
                    font-size: 14px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌐 Conceptual Space Visualization</h1>
                
                <div class="stats">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{total_concepts}</div>
                            <div class="stat-label">Total Concepts</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{total_overlaps}</div>
                            <div class="stat-label">Concept Overlaps</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{avg_importance:.3f}</div>
                            <div class="stat-label">Avg Importance</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{total_keywords}</div>
                            <div class="stat-label">Total Keywords</div>
                        </div>
                    </div>
                </div>
                
                <div class="viz-section">
                    <h2>3D Conceptual Space</h2>
                    <p>Interactive 3D visualization of concepts as convex balls. Size represents importance, 
                    transparency shows overlap regions, and edges connect related concepts.</p>
                    <div id="viz3d"></div>
                </div>
                
                <div class="grid">
                    <div class="viz-section">
                        <h2>Concept Network</h2>
                        <p>2D network showing concept relationships</p>
                        <div id="vizNetwork"></div>
                    </div>
                    
                    <div class="viz-section">
                        <h2>Importance vs Coverage</h2>
                        <p>Bubble chart of concept characteristics</p>
                        <div id="vizBubble"></div>
                    </div>
                </div>
                
                <div class="viz-section">
                    <h2>Similarity Matrix</h2>
                    <p>Heatmap showing semantic similarity between all concept pairs</p>
                    <div id="vizHeatmap"></div>
                </div>
            </div>
            
            <script>
                Plotly.newPlot('viz3d', {viz3d_json});
                Plotly.newPlot('vizNetwork', {vizNetwork_json});
                Plotly.newPlot('vizBubble', {vizBubble_json});
                Plotly.newPlot('vizHeatmap', {vizHeatmap_json});
            </script>
        </body>
        </html>
        """
        
        # Calculate statistics
        stats = {
            'total_concepts': len(self.concept_df),
            'total_overlaps': len(self.overlaps),
            'avg_importance': self.concept_df['importance'].mean(),
            'total_keywords': self.concept_df['keyword_count'].sum()
        }
        
        # Fill template
        html_content = html_template.format(
            **stats,
            viz3d_json=fig_3d.to_json(),
            vizNetwork_json=fig_network.to_json(),
            vizBubble_json=fig_bubble.to_json(),
            vizHeatmap_json=fig_heatmap.to_json()
        )
        
        # Save HTML
        output_file = Path("conceptual_space_visualization.html")
        output_file.write_text(html_content, encoding='utf-8')
        
        print(f"\nDashboard saved to: {output_file.absolute()}")
        
        return output_file
    
    def run_complete_visualization(self):
        """Execute complete visualization pipeline"""
        print("\n" + "="*80)
        print("CONCEPTUAL SPACE VISUALIZATION SYSTEM")
        print("="*80)
        
        # Step 1: Load data
        print("\n1. Loading concepts...")
        self.load_and_process_concepts()
        
        # Step 2: Compute embeddings
        print("\n2. Computing semantic embeddings...")
        self.compute_semantic_embeddings()
        
        # Step 3: Detect overlaps
        print("\n3. Detecting concept overlaps...")
        self.detect_overlaps()
        
        # Step 4: Generate visualizations
        print("\n4. Generating visualizations...")
        dashboard_path = self.generate_dashboard()
        
        # Summary
        print("\n" + "="*80)
        print("VISUALIZATION SUMMARY")
        print("="*80)
        print(f"Total Concepts: {len(self.concept_df)}")
        print(f"Semantic Dimensions: {self.embeddings.shape[1]}")
        print(f"Detected Overlaps: {len(self.overlaps)}")
        print(f"Domain Distribution:")
        for domain, count in self.concept_df['domain'].value_counts().items():
            print(f"  - {domain}: {count} concepts")
        
        print(f"\nOpen the dashboard in your browser:")
        print(f"   file:///{dashboard_path.absolute()}")
        
        return dashboard_path

if __name__ == "__main__":
    visualizer = ConceptualSpaceVisualizer()
    dashboard_path = visualizer.run_complete_visualization()