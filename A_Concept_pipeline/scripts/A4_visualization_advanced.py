"""
A4 Geometric Concept Space - Advanced Interactive Visualization
Advanced visualization using t-SNE, UMAP, and interactive plots with convex ball boundaries
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import os

# Optional: Install with pip install umap-learn
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

def load_a4_output(file_path: str = None) -> dict:
    """Load A4 geometric concept space output"""
    if file_path is None:
        file_path = "../outputs/A4_geometric_concept_space.json"

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_visualization_dataset(a4_data: dict) -> pd.DataFrame:
    """Create comprehensive dataset for visualization"""

    # Extract document data
    doc_key = list(a4_data.keys())[0]
    geometric_space = a4_data[doc_key]['geometric_concept_space']

    concept_centroids = geometric_space['concept_centroids']
    convex_balls = geometric_space['convex_balls']

    visualization_data = []

    # Add concept centroids
    for concept_id, centroid_data in concept_centroids.items():
        coords = centroid_data['centroid_coordinates']
        concept_source = centroid_data.get('concept_source', 'unknown')

        visualization_data.append({
            'id': concept_id,
            'type': 'centroid',
            'coordinates': coords,
            'label': centroid_data['canonical_name'],
            'concept_source': concept_source,
            'importance': centroid_data.get('concept_metadata', {}).get('importance_score', 0.5),
            'membership_strength': 1.0,  # Centroids have full membership
            'radius': convex_balls.get(concept_id, {}).get('radius', 1.0),
            'chunk_count': len(convex_balls.get(concept_id, {}).get('member_chunks', [])),
            'size': 300,  # Large markers for centroids
            'color_group': concept_source
        })

    # Add chunk coordinates
    for concept_id, ball_data in convex_balls.items():
        member_chunks = ball_data.get('member_chunks', [])

        for chunk in member_chunks:
            chunk_coords = chunk.get('chunk_coordinates', [])
            if chunk_coords:
                chunk_id = chunk['chunk_id']
                membership = chunk.get('membership_strength', 0)

                visualization_data.append({
                    'id': chunk_id,
                    'type': 'chunk',
                    'coordinates': chunk_coords,
                    'label': f"{chunk_id} (→{concept_id})",
                    'concept_source': f"member_of_{concept_id}",
                    'importance': membership,
                    'membership_strength': membership,
                    'radius': 0.0,  # Chunks don't have radius
                    'chunk_count': 0,
                    'size': max(50, membership * 150),  # Size based on membership strength
                    'color_group': 'chunks',
                    'parent_concept': concept_id
                })

    return pd.DataFrame(visualization_data)

def apply_dimensionality_reduction(df: pd.DataFrame, method: str = 'tsne', perplexity: int = 30) -> pd.DataFrame:
    """Apply dimensionality reduction to coordinates"""

    # Extract coordinate matrix
    coordinates = np.array(df['coordinates'].tolist())
    print(f"Applying {method.upper()} to {coordinates.shape[0]} points in {coordinates.shape[1]}D space...")

    if method == 'pca':
        reducer = PCA(n_components=2)
        coords_2d = reducer.fit_transform(coordinates)
        explained_var = reducer.explained_variance_ratio_.sum()
        print(f"PCA explains {explained_var:.1%} of variance")

    elif method == 'tsne':
        # Adjust perplexity based on data size
        perplexity = min(perplexity, len(coordinates) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        coords_2d = reducer.fit_transform(coordinates)
        print(f"t-SNE completed with perplexity={perplexity}")

    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(coordinates) - 1))
        coords_2d = reducer.fit_transform(coordinates)
        print("UMAP completed")

    else:
        print(f"Method {method} not available, falling back to PCA")
        reducer = PCA(n_components=2)
        coords_2d = reducer.fit_transform(coordinates)

    # Add 2D coordinates to dataframe
    df = df.copy()
    df['x'] = coords_2d[:, 0]
    df['y'] = coords_2d[:, 1]

    return df

def create_interactive_plotly_visualization(df: pd.DataFrame, method: str = 'tsne'):
    """Create interactive Plotly visualization"""

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'A4 Concept Space ({method.upper()})', 'Membership Strength Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Define colors
    color_map = {
        'A2.4_core': 'red',
        'A2.5_surrounding': 'blue',
        'chunks': 'green',
        'unknown': 'gray'
    }

    # Plot 1: Main concept space visualization
    for color_group, group_df in df.groupby('color_group'):
        color = color_map.get(color_group, 'gray')

        # Different markers for centroids vs chunks
        marker_symbol = 'circle' if group_df['type'].iloc[0] == 'centroid' else 'triangle-up'

        fig.add_trace(
            go.Scatter(
                x=group_df['x'],
                y=group_df['y'],
                mode='markers+text',
                marker=dict(
                    size=group_df['size'] / 10,  # Scale down for Plotly
                    color=color,
                    symbol=marker_symbol,
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                text=group_df['label'],
                textposition='top center',
                textfont=dict(size=8),
                name=color_group.replace('_', ' ').title(),
                hovertemplate=
                    '<b>%{text}</b><br>' +
                    'Type: %{customdata[0]}<br>' +
                    'Importance: %{customdata[1]:.3f}<br>' +
                    'Membership: %{customdata[2]:.3f}<br>' +
                    'Chunks: %{customdata[3]}<br>' +
                    '<extra></extra>',
                customdata=np.column_stack((
                    group_df['type'],
                    group_df['importance'],
                    group_df['membership_strength'],
                    group_df['chunk_count']
                ))
            ),
            row=1, col=1
        )

    # Plot 2: Membership strength analysis
    chunk_df = df[df['type'] == 'chunk']
    if not chunk_df.empty:
        fig.add_trace(
            go.Scatter(
                x=chunk_df['x'],
                y=chunk_df['y'],
                mode='markers',
                marker=dict(
                    size=chunk_df['membership_strength'] * 20,
                    color=chunk_df['membership_strength'],
                    colorscale='Viridis',
                    colorbar=dict(title='Membership Strength'),
                    showscale=True,
                    opacity=0.7
                ),
                text=chunk_df['label'],
                name='Chunk Membership',
                hovertemplate=
                    '<b>%{text}</b><br>' +
                    'Membership: %{marker.color:.3f}<br>' +
                    '<extra></extra>'
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title=f'A4 Geometric Concept Space Visualization ({method.upper()})',
        height=600,
        showlegend=True,
        template='plotly_white'
    )

    # Update axes
    fig.update_xaxes(title_text=f'{method.upper()} Component 1', row=1, col=1)
    fig.update_yaxes(title_text=f'{method.upper()} Component 2', row=1, col=1)
    fig.update_xaxes(title_text=f'{method.upper()} Component 1', row=1, col=2)
    fig.update_yaxes(title_text=f'{method.upper()} Component 2', row=1, col=2)

    return fig

def create_3d_visualization(df: pd.DataFrame, method: str = 'pca'):
    """Create 3D visualization"""

    # Apply 3D dimensionality reduction
    coordinates = np.array(df['coordinates'].tolist())

    if method == 'pca':
        reducer = PCA(n_components=3)
        coords_3d = reducer.fit_transform(coordinates)
        explained_var = reducer.explained_variance_ratio_.sum()
        print(f"3D PCA explains {explained_var:.1%} of variance")

    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=min(15, len(coordinates) - 1))
        coords_3d = reducer.fit_transform(coordinates)
        print("3D UMAP completed")

    else:
        print(f"3D {method} not available, using PCA")
        reducer = PCA(n_components=3)
        coords_3d = reducer.fit_transform(coordinates)

    # Create 3D plot
    fig = go.Figure()

    color_map = {
        'A2.4_core': 'red',
        'A2.5_surrounding': 'blue',
        'chunks': 'green',
        'unknown': 'gray'
    }

    for color_group, group_df in df.groupby('color_group'):
        color = color_map.get(color_group, 'gray')
        group_coords = coords_3d[df['color_group'] == color_group]

        marker_symbol = 'circle' if group_df['type'].iloc[0] == 'centroid' else 'diamond'

        fig.add_trace(
            go.Scatter3d(
                x=group_coords[:, 0],
                y=group_coords[:, 1],
                z=group_coords[:, 2],
                mode='markers+text',
                marker=dict(
                    size=group_df['size'] / 20,
                    color=color,
                    symbol=marker_symbol,
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                text=group_df['label'],
                textposition='top center',
                name=color_group.replace('_', ' ').title(),
                hovertemplate=
                    '<b>%{text}</b><br>' +
                    'Type: %{customdata[0]}<br>' +
                    'Importance: %{customdata[1]:.3f}<br>' +
                    'Membership: %{customdata[2]:.3f}<br>' +
                    '<extra></extra>',
                customdata=np.column_stack((
                    group_df['type'],
                    group_df['importance'],
                    group_df['membership_strength']
                ))
            )
        )

    fig.update_layout(
        title=f'A4 Concept Space - 3D {method.upper()} Projection',
        scene=dict(
            xaxis_title=f'{method.upper()} Component 1',
            yaxis_title=f'{method.upper()} Component 2',
            zaxis_title=f'{method.upper()} Component 3'
        ),
        height=700
    )

    return fig

def main():
    """Main advanced visualization function"""
    print("A4 Geometric Concept Space - Advanced Visualization")
    print("="*55)

    # Load A4 output
    try:
        a4_data = load_a4_output()
        print("[OK] Loaded A4 geometric concept space data")
    except FileNotFoundError:
        print("Error: A4_geometric_concept_space.json not found")
        print("Please run A4_geometric_concept_space.py first")
        return

    # Create visualization dataset
    df = create_visualization_dataset(a4_data)
    print(f"[OK] Created visualization dataset with {len(df)} points")
    print(f"  - Concept centroids: {len(df[df['type'] == 'centroid'])}")
    print(f"  - Chunk coordinates: {len(df[df['type'] == 'chunk'])}")

    # Create multiple visualizations
    methods = ['pca', 'tsne']
    if UMAP_AVAILABLE:
        methods.append('umap')

    for method in methods:
        print(f"\n{method.upper()} Visualization:")
        print("-" * 30)

        # Apply dimensionality reduction
        df_reduced = apply_dimensionality_reduction(df.copy(), method=method)

        # Create interactive 2D visualization
        fig_2d = create_interactive_plotly_visualization(df_reduced, method=method)

        # Save to HTML file instead of showing
        html_path = f"../outputs/A4_concept_space_{method}_2d.html"
        fig_2d.write_html(html_path)
        print(f"[OK] Saved 2D {method.upper()} visualization to {html_path}")

        # Create 3D visualization
        print("Creating 3D visualization...")
        fig_3d = create_3d_visualization(df, method=method)

        # Save 3D visualization to HTML file
        html_3d_path = f"../outputs/A4_concept_space_{method}_3d.html"
        fig_3d.write_html(html_3d_path)
        print(f"[OK] Saved 3D {method.upper()} visualization to {html_3d_path}")

    print("\nAdvanced visualization complete!")
    print("\nInterpretation Guide:")
    print("- Red circles = A2.4 core concepts (document-central)")
    print("- Blue circles = A2.5 surrounding concepts (contextual)")
    print("- Green triangles = Chunks with membership connections")
    print("- Size = Importance/membership strength")
    print("- Proximity = Geometric similarity in 384D space")
    print("- Interactive: Hover for details, zoom/pan to explore")

if __name__ == "__main__":
    main()