"""
A4 Geometric Concept Space - Simple 2D Visualization
Quick visualization of concept centroids and chunk memberships using PCA
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def load_a4_output(file_path: str = None) -> dict:
    """Load A4 geometric concept space output"""
    if file_path is None:
        file_path = "../outputs/A4_geometric_concept_space.json"

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_coordinates_and_labels(a4_data: dict):
    """Extract all coordinates and create labels for visualization"""

    coordinates = []
    labels = []
    types = []
    colors = []

    # Extract document data (assuming single document for now)
    doc_key = list(a4_data.keys())[0]
    geometric_space = a4_data[doc_key]['geometric_concept_space']

    concept_centroids = geometric_space['concept_centroids']
    convex_balls = geometric_space['convex_balls']

    # 1. Add concept centroids
    for concept_id, centroid_data in concept_centroids.items():
        coords = centroid_data['centroid_coordinates']
        concept_source = centroid_data.get('concept_source', 'unknown')

        coordinates.append(coords)
        labels.append(f"{concept_id}\n({centroid_data['canonical_name']})")
        types.append('centroid')

        # Color by concept source
        if concept_source == 'A2.4_core':
            colors.append('red')  # Core concepts
        elif concept_source == 'A2.5_surrounding':
            colors.append('blue')  # Surrounding concepts
        else:
            colors.append('gray')  # Unknown

    # 2. Add chunk coordinates
    for concept_id, ball_data in convex_balls.items():
        member_chunks = ball_data.get('member_chunks', [])

        for chunk in member_chunks:
            chunk_coords = chunk.get('chunk_coordinates', [])
            if chunk_coords:
                coordinates.append(chunk_coords)

                chunk_id = chunk['chunk_id']
                membership = chunk.get('membership_strength', 0)
                labels.append(f"{chunk_id}\n(membership: {membership:.2f})")
                types.append('chunk')
                colors.append('green')  # All chunks in green

    return np.array(coordinates), labels, types, colors

def create_simple_2d_visualization(coordinates: np.ndarray, labels: list, types: list, colors: list):
    """Create simple 2D visualization using PCA"""

    print(f"Visualizing {len(coordinates)} points in 384D -> 2D space")

    # Apply PCA to reduce 384D -> 2D
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(coordinates)

    # Create plot
    plt.figure(figsize=(14, 10))

    # Plot concept centroids
    centroid_mask = np.array(types) == 'centroid'
    chunk_mask = np.array(types) == 'chunk'

    # Plot centroids (larger markers)
    centroid_coords = coords_2d[centroid_mask]
    centroid_colors = np.array(colors)[centroid_mask]
    centroid_labels = np.array(labels)[centroid_mask]

    for i, (coord, color, label) in enumerate(zip(centroid_coords, centroid_colors, centroid_labels)):
        plt.scatter(coord[0], coord[1], c=color, s=200, marker='o', alpha=0.8, edgecolors='black')
        plt.annotate(label, (coord[0], coord[1]), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Plot chunks (smaller markers)
    chunk_coords = coords_2d[chunk_mask]
    chunk_colors = np.array(colors)[chunk_mask]
    chunk_labels = np.array(labels)[chunk_mask]

    for i, (coord, color, label) in enumerate(zip(chunk_coords, chunk_colors, chunk_labels)):
        plt.scatter(coord[0], coord[1], c=color, s=50, marker='^', alpha=0.6)
        plt.annotate(label, (coord[0], coord[1]), xytext=(3, 3), textcoords='offset points',
                    fontsize=6, ha='left', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.5))

    # Add legend
    plt.scatter([], [], c='red', s=200, marker='o', label='A2.4 Core Concepts', alpha=0.8)
    plt.scatter([], [], c='blue', s=200, marker='o', label='A2.5 Surrounding Concepts', alpha=0.8)
    plt.scatter([], [], c='green', s=50, marker='^', label='Chunks', alpha=0.6)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('A4 Geometric Concept Space - 2D PCA Projection\nConcept Centroids and Chunk Memberships')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return pca, coords_2d

def create_membership_network_view(a4_data: dict):
    """Create network-style visualization showing membership connections"""

    # Extract document data
    doc_key = list(a4_data.keys())[0]
    geometric_space = a4_data[doc_key]['geometric_concept_space']
    convex_balls = geometric_space['convex_balls']

    plt.figure(figsize=(12, 8))

    # Create concept-chunk membership network
    concept_positions = {}
    chunk_positions = {}

    # Position concepts in a circle
    concepts_with_chunks = [cid for cid, ball in convex_balls.items() if ball.get('member_chunks')]
    n_concepts = len(concepts_with_chunks)

    for i, concept_id in enumerate(concepts_with_chunks):
        angle = 2 * np.pi * i / n_concepts
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle)
        concept_positions[concept_id] = (x, y)

        # Plot concept
        plt.scatter(x, y, c='red', s=300, marker='o', alpha=0.8, edgecolors='black')
        plt.annotate(concept_id, (x, y), ha='center', va='center', fontweight='bold', fontsize=10)

    # Position chunks and draw connections
    chunk_counter = 0
    for concept_id, ball_data in convex_balls.items():
        if concept_id not in concept_positions:
            continue

        member_chunks = ball_data.get('member_chunks', [])
        concept_pos = concept_positions[concept_id]

        for i, chunk in enumerate(member_chunks):
            chunk_id = chunk['chunk_id']
            membership = chunk.get('membership_strength', 0)

            # Position chunk near its concept
            angle_offset = (i - len(member_chunks)/2) * 0.3
            chunk_angle = np.arctan2(concept_pos[1], concept_pos[0]) + angle_offset
            chunk_x = concept_pos[0] + 0.5 * np.cos(chunk_angle)
            chunk_y = concept_pos[1] + 0.5 * np.sin(chunk_angle)

            # Plot chunk
            plt.scatter(chunk_x, chunk_y, c='green', s=100, marker='^', alpha=0.6)
            plt.annotate(f"{chunk_id}\n({membership:.2f})", (chunk_x, chunk_y),
                        xytext=(5, 5), textcoords='offset points', fontsize=6, ha='left')

            # Draw connection line (thickness based on membership strength)
            line_width = max(0.5, membership * 3)
            plt.plot([concept_pos[0], chunk_x], [concept_pos[1], chunk_y],
                    'k-', alpha=0.4, linewidth=line_width)

    plt.title('A4 Concept-Chunk Membership Network\nLine thickness = Membership strength')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def main():
    """Main visualization function"""
    print("A4 Geometric Concept Space - Simple Visualization")
    print("="*50)

    # Load A4 output
    try:
        a4_data = load_a4_output()
        print("[OK] Loaded A4 geometric concept space data")
    except FileNotFoundError:
        print("Error: A4_geometric_concept_space.json not found")
        print("Please run A4_geometric_concept_space.py first")
        return

    # Extract coordinates and labels
    coordinates, labels, types, colors = extract_coordinates_and_labels(a4_data)
    print(f"[OK] Extracted {len(coordinates)} points for visualization")
    print(f"  - Concept centroids: {sum(1 for t in types if t == 'centroid')}")
    print(f"  - Chunk coordinates: {sum(1 for t in types if t == 'chunk')}")

    # Create visualizations
    print("\n1. Creating 2D PCA projection...")
    pca, coords_2d = create_simple_2d_visualization(coordinates, labels, types, colors)

    print(f"[OK] PCA explains {pca.explained_variance_ratio_.sum():.1%} of variance")

    # Save plot instead of showing it
    output_path = "../outputs/A4_concept_space_2d.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved 2D visualization to {output_path}")
    plt.close()

    print("\n2. Creating membership network view...")
    create_membership_network_view(a4_data)

    # Save network plot instead of showing it
    network_path = "../outputs/A4_membership_network.png"
    plt.savefig(network_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved membership network to {network_path}")
    plt.close()

    print("\nVisualization complete!")
    print("\nInterpretation:")
    print("- Red circles = A2.4 core concepts")
    print("- Blue circles = A2.5 surrounding concepts")
    print("- Green triangles = Chunks")
    print("- Proximity indicates geometric similarity in 384D space")

if __name__ == "__main__":
    main()