"""
A4 Convex Ball Boundary Analysis
Detailed analysis and visualization of convex ball boundaries and chunk distributions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import pandas as pd

def load_a4_output(file_path: str = None) -> dict:
    """Load A4 geometric concept space output"""
    if file_path is None:
        file_path = "../outputs/A4_geometric_concept_space.json"

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_convex_ball_statistics(a4_data: dict):
    """Analyze convex ball statistics and create summary"""

    # Extract document data
    doc_key = list(a4_data.keys())[0]
    geometric_space = a4_data[doc_key]['geometric_concept_space']
    convex_balls = geometric_space['convex_balls']

    # Collect statistics
    stats = []
    for concept_id, ball_data in convex_balls.items():
        member_chunks = ball_data.get('member_chunks', [])
        if member_chunks:
            memberships = [chunk.get('membership_strength', 0) for chunk in member_chunks]
            stats.append({
                'concept_id': concept_id,
                'chunk_count': len(member_chunks),
                'radius': ball_data.get('radius', 0),
                'avg_membership': np.mean(memberships),
                'min_membership': np.min(memberships),
                'max_membership': np.max(memberships),
                'membership_std': np.std(memberships)
            })

    # Create summary DataFrame
    df = pd.DataFrame(stats)
    df = df.sort_values('chunk_count', ascending=False)

    print("\nConvex Ball Statistics Summary:")
    print("=" * 50)
    print(f"{'Concept ID':<15} {'Chunks':<7} {'Radius':<8} {'Avg Memb':<10} {'Min Memb':<10} {'Max Memb':<10}")
    print("-" * 70)

    for _, row in df.head(10).iterrows():
        print(f"{row['concept_id']:<15} {row['chunk_count']:<7} {row['radius']:<8.3f} {row['avg_membership']:<10.3f} {row['min_membership']:<10.3f} {row['max_membership']:<10.3f}")

    return df

def visualize_convex_hull_boundaries(a4_data: dict, top_n_concepts: int = 4):
    """Visualize actual convex hull boundaries for top concepts"""

    # Extract document data
    doc_key = list(a4_data.keys())[0]
    geometric_space = a4_data[doc_key]['geometric_concept_space']
    concept_centroids = geometric_space['concept_centroids']
    convex_balls = geometric_space['convex_balls']

    # Get top concepts by chunk count
    concept_chunk_counts = [(cid, len(ball.get('member_chunks', [])))
                           for cid, ball in convex_balls.items()]
    concept_chunk_counts.sort(key=lambda x: x[1], reverse=True)
    top_concepts = [cid for cid, count in concept_chunk_counts[:top_n_concepts] if count >= 3]  # Need at least 3 points for convex hull

    # Create subplots
    cols = 2
    rows = (len(top_concepts) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))

    if len(top_concepts) == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, concept_id in enumerate(top_concepts):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        # Get concept data
        centroid_data = concept_centroids[concept_id]
        ball_data = convex_balls[concept_id]

        centroid_coords = np.array(centroid_data['centroid_coordinates'])
        member_chunks = ball_data.get('member_chunks', [])

        # Collect all coordinates including centroid
        all_coords = [centroid_coords]
        membership_strengths = [1.0]  # Centroid has full membership
        labels = ['Centroid']

        for chunk in member_chunks:
            chunk_coords = chunk.get('chunk_coordinates', [])
            if chunk_coords:
                all_coords.append(np.array(chunk_coords))
                membership = chunk.get('membership_strength', 0)
                membership_strengths.append(membership)
                labels.append(f"C{len(labels)-1}({membership:.2f})")

        if len(all_coords) < 3:
            ax.text(0.5, 0.5, f"{concept_id}\nNot enough points\nfor convex hull",
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Apply PCA to reduce to 2D
        coords_matrix = np.array(all_coords)
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(coords_matrix)

        # Calculate and plot convex hull
        try:
            hull = ConvexHull(coords_2d)

            # Plot convex hull boundary
            for simplex in hull.simplices:
                ax.plot(coords_2d[simplex, 0], coords_2d[simplex, 1], 'r-', alpha=0.7, linewidth=2)

            # Fill convex hull area
            hull_points = coords_2d[hull.vertices]
            ax.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.2, color='red', label='Convex Hull')

        except Exception as e:
            print(f"Warning: Could not compute convex hull for {concept_id}: {e}")

        # Plot points
        # Centroid
        ax.scatter(coords_2d[0, 0], coords_2d[0, 1], c='red', s=300, marker='*',
                  alpha=0.9, edgecolors='black', label='Centroid', zorder=5)

        # Chunks with size and color based on membership
        for i in range(1, len(coords_2d)):
            membership = membership_strengths[i]
            size = 80 + membership * 120
            alpha = 0.5 + membership * 0.5

            ax.scatter(coords_2d[i, 0], coords_2d[i, 1],
                      c='green', s=size, marker='o', alpha=alpha,
                      edgecolors='darkgreen', linewidth=1, zorder=3)

            # Add labels
            ax.annotate(labels[i], (coords_2d[i, 0], coords_2d[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Draw radius circle (approximate in 2D)
        radius_2d = ball_data.get('radius', 1.0) * 0.1  # Scale down for visualization
        circle = plt.Circle((coords_2d[0, 0], coords_2d[0, 1]), radius_2d,
                           fill=False, color='blue', linestyle='--', alpha=0.6, linewidth=2)
        ax.add_patch(circle)

        ax.set_title(f"{concept_id}\n{len(member_chunks)} chunks, R={ball_data.get('radius', 0):.3f}",
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(len(top_concepts), rows * cols):
        row = idx // cols
        col = idx % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        elif len(top_concepts) > 1:
            axes[col].set_visible(False)

    plt.tight_layout()
    plt.suptitle('Convex Hull Boundaries vs Computed Radius\nRed boundary = Actual convex hull, Blue dashed = Computed radius',
                fontsize=14, y=0.98)

    return fig

def create_membership_distribution_analysis(a4_data: dict):
    """Analyze and visualize membership strength distributions"""

    # Extract document data
    doc_key = list(a4_data.keys())[0]
    geometric_space = a4_data[doc_key]['geometric_concept_space']
    convex_balls = geometric_space['convex_balls']

    # Collect all membership data
    all_memberships = []
    concept_memberships = {}

    for concept_id, ball_data in convex_balls.items():
        member_chunks = ball_data.get('member_chunks', [])
        if member_chunks:
            memberships = [chunk.get('membership_strength', 0) for chunk in member_chunks]
            all_memberships.extend(memberships)
            concept_memberships[concept_id] = memberships

    # Create analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Overall membership distribution
    ax1.hist(all_memberships, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(all_memberships), color='red', linestyle='--',
               label=f'Mean: {np.mean(all_memberships):.3f}')
    ax1.axvline(np.median(all_memberships), color='green', linestyle='--',
               label=f'Median: {np.median(all_memberships):.3f}')
    ax1.set_xlabel('Membership Strength')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Membership Strength Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot by concept (top 10 concepts)
    top_concepts = sorted(concept_memberships.items(),
                         key=lambda x: len(x[1]), reverse=True)[:10]

    concept_names = [f"{cid}\n({len(membs)})" for cid, membs in top_concepts]
    membership_data = [membs for _, membs in top_concepts]

    bp = ax2.boxplot(membership_data, labels=concept_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    ax2.set_ylabel('Membership Strength')
    ax2.set_title('Membership Distribution by Concept (Top 10)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Membership strength vs chunk count
    chunk_counts = []
    avg_memberships = []
    concept_ids = []

    for concept_id, ball_data in convex_balls.items():
        member_chunks = ball_data.get('member_chunks', [])
        if member_chunks:
            memberships = [chunk.get('membership_strength', 0) for chunk in member_chunks]
            chunk_counts.append(len(member_chunks))
            avg_memberships.append(np.mean(memberships))
            concept_ids.append(concept_id)

    scatter = ax3.scatter(chunk_counts, avg_memberships, alpha=0.7, s=100, c='coral')

    # Add correlation coefficient
    correlation = np.corrcoef(chunk_counts, avg_memberships)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))

    ax3.set_xlabel('Number of Chunks')
    ax3.set_ylabel('Average Membership Strength')
    ax3.set_title('Chunk Count vs Average Membership Strength')
    ax3.grid(True, alpha=0.3)

    # 4. Membership threshold analysis
    thresholds = np.arange(0, 1.01, 0.05)
    chunk_counts_by_threshold = []

    for threshold in thresholds:
        count = sum(1 for m in all_memberships if m >= threshold)
        chunk_counts_by_threshold.append(count)

    ax4.plot(thresholds, chunk_counts_by_threshold, 'b-', linewidth=2, marker='o')
    ax4.axhline(len(all_memberships) * 0.5, color='red', linestyle='--',
               label='50% of chunks')
    ax4.axhline(len(all_memberships) * 0.1, color='orange', linestyle='--',
               label='10% of chunks')
    ax4.set_xlabel('Membership Threshold')
    ax4.set_ylabel('Number of Chunks Above Threshold')
    ax4.set_title('Chunk Count by Membership Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig

def main():
    """Main convex boundary analysis function"""
    print("A4 Convex Ball Boundary Analysis")
    print("=" * 35)

    # Load A4 output
    try:
        a4_data = load_a4_output()
        print("[OK] Loaded A4 geometric concept space data")
    except FileNotFoundError:
        print("Error: A4_geometric_concept_space.json not found")
        return

    # 1. Analyze convex ball statistics
    stats_df = analyze_convex_ball_statistics(a4_data)

    # 2. Visualize convex hull boundaries
    print("\n1. Creating convex hull boundary visualization...")
    fig1 = visualize_convex_hull_boundaries(a4_data)
    output_path1 = "../outputs/A4_convex_hull_boundaries.png"
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved convex hull boundaries to {output_path1}")
    plt.close(fig1)

    # 3. Create membership distribution analysis
    print("\n2. Creating membership distribution analysis...")
    fig2 = create_membership_distribution_analysis(a4_data)
    output_path2 = "../outputs/A4_membership_analysis.png"
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved membership analysis to {output_path2}")
    plt.close(fig2)

    # 4. Save statistics to CSV
    stats_path = "../outputs/A4_convex_ball_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[OK] Saved statistics to {stats_path}")

    print("\nBoundary analysis complete!")
    print("\nKey Insights:")
    print("- Red boundaries show actual convex hulls of chunk distributions")
    print("- Blue dashed circles show computed radius boundaries")
    print("- Membership analysis shows distribution patterns and thresholds")
    print("- Statistics CSV provides detailed quantitative analysis")

if __name__ == "__main__":
    main()