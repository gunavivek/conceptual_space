"""
A4 Geometric Concept Space - Enhanced Convex Ball Visualization
Focused visualization showing clear convex ball boundaries and chunk memberships
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import os
from matplotlib.patches import Circle
import matplotlib.patches as patches

def load_a4_output(file_path: str = None) -> dict:
    """Load A4 geometric concept space output"""
    if file_path is None:
        file_path = "../outputs/A4_geometric_concept_space.json"

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_individual_concept_visualizations(a4_data: dict):
    """Create separate visualization for each concept's convex ball"""

    # Extract document data
    doc_key = list(a4_data.keys())[0]
    geometric_space = a4_data[doc_key]['geometric_concept_space']
    concept_centroids = geometric_space['concept_centroids']
    convex_balls = geometric_space['convex_balls']

    # Create individual plots for each concept
    concepts_with_chunks = [cid for cid, ball in convex_balls.items() if ball.get('member_chunks')]

    # Calculate grid size for subplots
    n_concepts = len(concepts_with_chunks)
    cols = min(4, n_concepts)
    rows = (n_concepts + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

    # Ensure axes is always a 2D array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, concept_id in enumerate(concepts_with_chunks):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Get concept data
        centroid_data = concept_centroids[concept_id]
        ball_data = convex_balls[concept_id]

        centroid_coords = np.array(centroid_data['centroid_coordinates'])
        radius = ball_data['radius']
        member_chunks = ball_data.get('member_chunks', [])

        # Collect all coordinates for this concept
        all_coords = [centroid_coords]
        labels = [f"Centroid\n{concept_id}"]
        colors = ['red']
        sizes = [200]

        for chunk in member_chunks:
            chunk_coords = chunk.get('chunk_coordinates', [])
            if chunk_coords:
                all_coords.append(np.array(chunk_coords))
                membership = chunk.get('membership_strength', 0)
                labels.append(f"{chunk['chunk_id']}\n({membership:.2f})")
                colors.append('green')
                sizes.append(100)

        # Apply PCA to reduce to 2D
        if len(all_coords) > 1:
            coords_matrix = np.array(all_coords)
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(coords_matrix)

            # Plot centroid
            ax.scatter(coords_2d[0, 0], coords_2d[0, 1], c='red', s=200, marker='o',
                      alpha=0.8, edgecolors='black', label='Centroid')

            # Plot chunks
            for i in range(1, len(coords_2d)):
                ax.scatter(coords_2d[i, 0], coords_2d[i, 1], c='green', s=100, marker='^',
                          alpha=0.7, edgecolors='black')
                ax.annotate(labels[i], (coords_2d[i, 0], coords_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            # Draw convex ball boundary (approximate circle in 2D)
            # Calculate radius in 2D space as proportion of data spread
            data_spread = np.max(coords_2d, axis=0) - np.min(coords_2d, axis=0)
            circle_radius = 0.3 * np.mean(data_spread)

            circle = Circle((coords_2d[0, 0], coords_2d[0, 1]), circle_radius,
                          fill=False, color='red', linestyle='--', alpha=0.6, linewidth=2)
            ax.add_patch(circle)

            # Draw lines from centroid to chunks
            for i in range(1, len(coords_2d)):
                ax.plot([coords_2d[0, 0], coords_2d[i, 0]],
                       [coords_2d[0, 1], coords_2d[i, 1]],
                       'k--', alpha=0.4, linewidth=1)

        ax.set_title(f"{concept_id}\n({len(member_chunks)} chunks, R={radius:.3f})", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # Hide unused subplots
    for idx in range(n_concepts, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.suptitle('A4 Concept Space - Individual Convex Balls\n(Dashed circles show approximate boundaries)',
                fontsize=14, y=0.98)

    return fig

def create_focused_membership_view(a4_data: dict, selected_concepts: list = None):
    """Create focused view showing only selected concepts and their memberships"""

    # Extract document data
    doc_key = list(a4_data.keys())[0]
    geometric_space = a4_data[doc_key]['geometric_concept_space']
    concept_centroids = geometric_space['concept_centroids']
    convex_balls = geometric_space['convex_balls']

    # If no concepts selected, use top 6 concepts by chunk count
    if selected_concepts is None:
        concept_chunk_counts = [(cid, len(ball.get('member_chunks', [])))
                               for cid, ball in convex_balls.items()]
        concept_chunk_counts.sort(key=lambda x: x[1], reverse=True)
        selected_concepts = [cid for cid, count in concept_chunk_counts[:6] if count > 0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, concept_id in enumerate(selected_concepts[:6]):
        ax = axes[idx]

        # Get concept data
        centroid_data = concept_centroids[concept_id]
        ball_data = convex_balls[concept_id]

        centroid_coords = np.array(centroid_data['centroid_coordinates'])
        member_chunks = ball_data.get('member_chunks', [])

        if not member_chunks:
            ax.text(0.5, 0.5, f"{concept_id}\nNo chunks",
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Collect coordinates
        all_coords = [centroid_coords]
        membership_strengths = [1.0]  # Centroid has full membership

        for chunk in member_chunks:
            chunk_coords = chunk.get('chunk_coordinates', [])
            if chunk_coords:
                all_coords.append(np.array(chunk_coords))
                membership_strengths.append(chunk.get('membership_strength', 0))

        # Apply PCA
        coords_matrix = np.array(all_coords)
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(coords_matrix)

        # Plot centroid
        ax.scatter(coords_2d[0, 0], coords_2d[0, 1], c='red', s=300, marker='*',
                  alpha=0.9, edgecolors='black', label='Centroid', zorder=5)

        # Plot chunks with size based on membership strength
        for i in range(1, len(coords_2d)):
            membership = membership_strengths[i]
            size = 50 + membership * 150  # Size based on membership
            alpha = 0.4 + membership * 0.6  # Transparency based on membership

            ax.scatter(coords_2d[i, 0], coords_2d[i, 1],
                      c='green', s=size, marker='o', alpha=alpha,
                      edgecolors='darkgreen', linewidth=1, zorder=3)

            # Draw connection line with thickness based on membership
            line_width = 0.5 + membership * 2
            ax.plot([coords_2d[0, 0], coords_2d[i, 0]],
                   [coords_2d[0, 1], coords_2d[i, 1]],
                   'gray', alpha=0.6, linewidth=line_width, zorder=1)

            # Add membership strength label
            ax.annotate(f"{membership:.2f}",
                       (coords_2d[i, 0], coords_2d[i, 1]),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=8, fontweight='bold')

        ax.set_title(f"{concept_id}\n{len(member_chunks)} chunks", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # Hide unused subplots
    for idx in range(len(selected_concepts), 6):
        if idx < len(axes):
            axes[idx].set_visible(False)

    plt.tight_layout()
    plt.suptitle('A4 Focused Membership View\nCircle size & opacity = Membership strength, Line thickness = Connection strength',
                fontsize=14, y=0.98)

    return fig

def create_interactive_convex_ball_explorer(a4_data: dict):
    """Create interactive Plotly visualization with concept selection"""

    # Extract document data
    doc_key = list(a4_data.keys())[0]
    geometric_space = a4_data[doc_key]['geometric_concept_space']
    concept_centroids = geometric_space['concept_centroids']
    convex_balls = geometric_space['convex_balls']

    # Create dropdown options for concept selection
    concepts_with_chunks = [(cid, len(ball.get('member_chunks', [])))
                           for cid, ball in convex_balls.items()
                           if ball.get('member_chunks')]
    concepts_with_chunks.sort(key=lambda x: x[1], reverse=True)

    # Create traces for each concept (initially only show first one)
    fig = go.Figure()

    for concept_idx, (concept_id, chunk_count) in enumerate(concepts_with_chunks):
        centroid_data = concept_centroids[concept_id]
        ball_data = convex_balls[concept_id]

        centroid_coords = np.array(centroid_data['centroid_coordinates'])
        member_chunks = ball_data.get('member_chunks', [])

        # Collect coordinates for this concept
        all_coords = [centroid_coords]
        labels = [f"Centroid: {concept_id}"]
        membership_strengths = [1.0]

        for chunk in member_chunks:
            chunk_coords = chunk.get('chunk_coordinates', [])
            if chunk_coords:
                all_coords.append(np.array(chunk_coords))
                membership = chunk.get('membership_strength', 0)
                labels.append(f"Chunk: {chunk['chunk_id']} (Membership: {membership:.3f})")
                membership_strengths.append(membership)

        # Apply PCA
        coords_matrix = np.array(all_coords)
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(coords_matrix)

        # Add centroid trace
        fig.add_trace(
            go.Scatter(
                x=[coords_2d[0, 0]],
                y=[coords_2d[0, 1]],
                mode='markers+text',
                marker=dict(size=20, color='red', symbol='star'),
                text=[concept_id],
                textposition='top center',
                name=f'{concept_id} Centroid',
                visible=(concept_idx == 0),  # Only first concept visible initially
                hovertemplate=f'<b>{concept_id}</b><br>Type: Centroid<br>Chunks: {len(member_chunks)}<extra></extra>'
            )
        )

        # Add chunk traces
        if len(coords_2d) > 1:
            chunk_x = coords_2d[1:, 0]
            chunk_y = coords_2d[1:, 1]
            chunk_labels = labels[1:]
            chunk_memberships = membership_strengths[1:]

            fig.add_trace(
                go.Scatter(
                    x=chunk_x,
                    y=chunk_y,
                    mode='markers+text',
                    marker=dict(
                        size=[10 + m*20 for m in chunk_memberships],
                        color=chunk_memberships,
                        colorscale='Greens',
                        showscale=True,
                        colorbar=dict(title='Membership Strength'),
                        symbol='circle'
                    ),
                    text=[f"M:{m:.2f}" for m in chunk_memberships],
                    textposition='top center',
                    name=f'{concept_id} Chunks',
                    visible=(concept_idx == 0),
                    hovertemplate='<b>%{text}</b><br>Membership: %{marker.color:.3f}<extra></extra>'
                )
            )

        # Add connection lines
        if len(coords_2d) > 1:
            for i in range(1, len(coords_2d)):
                fig.add_trace(
                    go.Scatter(
                        x=[coords_2d[0, 0], coords_2d[i, 0]],
                        y=[coords_2d[0, 1], coords_2d[i, 1]],
                        mode='lines',
                        line=dict(color='gray', width=1+membership_strengths[i]*3),
                        showlegend=False,
                        visible=(concept_idx == 0),
                        hoverinfo='skip'
                    )
                )

    # Create dropdown for concept selection
    dropdown_buttons = []
    for concept_idx, (concept_id, chunk_count) in enumerate(concepts_with_chunks):
        # Calculate which traces to show for this concept
        traces_per_concept = 2 + len(convex_balls[concept_id].get('member_chunks', []))  # centroid + chunks + lines

        visibility = [False] * len(fig.data)
        start_idx = concept_idx * traces_per_concept
        end_idx = start_idx + traces_per_concept
        for i in range(start_idx, min(end_idx, len(visibility))):
            visibility[i] = True

        dropdown_buttons.append(
            dict(
                label=f"{concept_id} ({chunk_count} chunks)",
                method="update",
                args=[{"visible": visibility}]
            )
        )

    fig.update_layout(
        title="A4 Interactive Convex Ball Explorer<br><sub>Select concept to view its convex ball and chunk memberships</sub>",
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.0,
                yanchor="top"
            )
        ],
        height=600,
        showlegend=True
    )

    return fig

def main():
    """Main enhanced visualization function"""
    print("A4 Enhanced Convex Ball Visualization")
    print("="*40)

    # Load A4 output
    try:
        a4_data = load_a4_output()
        print("[OK] Loaded A4 geometric concept space data")
    except FileNotFoundError:
        print("Error: A4_geometric_concept_space.json not found")
        return

    print("\n1. Creating individual concept visualizations...")
    fig1 = create_individual_concept_visualizations(a4_data)
    output_path1 = "../outputs/A4_individual_convex_balls.png"
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved individual convex balls to {output_path1}")
    plt.close(fig1)

    print("\n2. Creating focused membership view...")
    fig2 = create_focused_membership_view(a4_data)
    output_path2 = "../outputs/A4_focused_membership.png"
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved focused membership view to {output_path2}")
    plt.close(fig2)

    print("\n3. Creating interactive convex ball explorer...")
    fig3 = create_interactive_convex_ball_explorer(a4_data)
    output_path3 = "../outputs/A4_interactive_convex_balls.html"
    fig3.write_html(output_path3)
    print(f"[OK] Saved interactive explorer to {output_path3}")

    print("\nEnhanced visualization complete!")
    print("\nVisualization Features:")
    print("- Individual convex balls: Shows each concept's boundary and chunks separately")
    print("- Focused membership: Top 6 concepts with clear membership strength indicators")
    print("- Interactive explorer: Select individual concepts to explore their convex balls")
    print("- Clear membership connections: Lines thickness = membership strength")
    print("- Circle size = membership strength, transparency = membership strength")

if __name__ == "__main__":
    main()