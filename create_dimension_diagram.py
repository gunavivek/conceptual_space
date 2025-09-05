#!/usr/bin/env python3
"""
Create visual diagram explaining the Financial Dimensions and Elliptical Disks
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

def create_dimension_explanation_diagram():
    """Create comprehensive diagram explaining the 3D financial space"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # 1. Financial Dimension Axes Explanation
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Financial Dimensions in Semantic Space", fontsize=16, fontweight='bold')
    
    # Create coordinate system
    ax1.arrow(0, 0, 1, 0, head_width=0.05, head_length=0.05, fc='blue', ec='blue', linewidth=2)
    ax1.arrow(0, 0, 0, 1, head_width=0.05, head_length=0.05, fc='red', ec='red', linewidth=2)
    ax1.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.05, fc='green', ec='green', linewidth=2)
    
    # Label axes
    ax1.text(1.1, 0, 'Financial Dim 1\n(Contract ↔ Revenue)', fontsize=12, ha='left', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.text(0, 1.1, 'Financial Dim 2\n(Balance ↔ Operations)', fontsize=12, ha='center', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax1.text(0.6, 0.6, 'Financial Dim 3\n(Recognition Timing)', fontsize=12, ha='left', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax1.set_xlim(-0.2, 1.5)
    ax1.set_ylim(-0.2, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # 2. Concept Positions
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Concept Positions in Financial Space", fontsize=14, fontweight='bold')
    
    # finqa_test_96 concept positions (from analysis)
    concepts = [
        {'name': 'Contract\nBalances', 'pos': [-0.416, 0.663], 'color': '#2E86AB', 'size': 8},
        {'name': 'Revenue\nUnearned', 'pos': [0.833, 0.0], 'color': '#A23B72', 'size': 3}, 
        {'name': 'Receivable\nBalance', 'pos': [-0.416, -0.663], 'color': '#F18F01', 'size': 2}
    ]
    
    for concept in concepts:
        # Plot concept center
        ax2.scatter(concept['pos'][0], concept['pos'][1], 
                   s=concept['size']*50, c=concept['color'], 
                   alpha=0.8, edgecolors='black', linewidth=2)
        
        # Add concept name
        ax2.text(concept['pos'][0], concept['pos'][1] + 0.1, concept['name'], 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add elliptical uncertainty region
        ellipse = Ellipse(concept['pos'], width=0.3, height=0.2, 
                         alpha=0.2, facecolor=concept['color'])
        ax2.add_patch(ellipse)
    
    ax2.set_xlabel('Financial Dim 1 (Contract ← → Revenue)', fontsize=10)
    ax2.set_ylabel('Financial Dim 2 (Balance ← → Operations)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 1.2)
    ax2.set_ylim(-1, 1)
    
    # 3. Elliptical Disk Explanation
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("Elliptical Disk Interpretation", fontsize=14, fontweight='bold')
    
    # Draw example ellipse with annotations
    center = [0, 0]
    ellipse_main = Ellipse(center, width=0.8, height=0.4, alpha=0.3, facecolor='blue')
    ax3.add_patch(ellipse_main)
    
    # Center point
    ax3.scatter(0, 0, s=200, c='blue', marker='o', edgecolors='black', linewidth=2)
    ax3.text(0, 0, 'Concept\nCentroid', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Ellipse annotations
    ax3.arrow(0, 0, 0.4, 0, head_width=0.02, head_length=0.03, fc='red', ec='red')
    ax3.text(0.45, 0, 'Major axis:\nMax semantic\nvariation', ha='left', va='center', fontsize=9)
    
    ax3.arrow(0, 0, 0, 0.2, head_width=0.02, head_length=0.03, fc='orange', ec='orange')
    ax3.text(0, 0.25, 'Minor axis:\nMin semantic\nvariation', ha='center', va='bottom', fontsize=9)
    
    # Boundary annotation
    ax3.annotate('Uncertainty\nBoundary', xy=(0.3, 0.15), xytext=(0.6, 0.3),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, ha='center', color='green', fontweight='bold')
    
    ax3.set_xlim(-0.6, 0.8)
    ax3.set_ylim(-0.4, 0.5)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # 4. Mathematical Explanation
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    explanation_text = """
MATHEMATICAL INTERPRETATION OF FINANCIAL DIMENSIONS:

• DIMENSION REDUCTION: 30 TF-IDF features → 3 Principal Components (PCA)
    - Financial Dim 1 (54.2% variance): Contract/Revenue semantic axis  
    - Financial Dim 2 (45.8% variance): Balance/Operations semantic axis
    - Financial Dim 3 (0.0% variance): Recognition timing (minimal variation)

• ELLIPTICAL DISKS represent:
    1. SEMANTIC UNCERTAINTY: Covariance of keywords in reduced space
    2. CONFIDENCE REGIONS: Where related terms might exist around core concept  
    3. CONCEPT BOUNDARIES: Limits of semantic interpretation

• CONVEX BALLS (spheres) represent:
    1. CONCEPT CENTROIDS: Mean position in semantic space
    2. SEMANTIC DENSITY: Size ∝ importance score × keyword count
    3. INFLUENCE ZONES: Radius ∝ document coverage × semantic diversity

• POSITIONING LOGIC:
    - Closer concepts = More semantically similar
    - Axis position = Weighted combination of keyword frequencies
    - Ellipse orientation = Direction of maximum keyword variance
    """
    
    ax4.text(0.05, 0.95, explanation_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('financial_dimensions_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("Creating Financial Dimensions explanation diagram...")
    fig = create_dimension_explanation_diagram()
    print("Diagram saved as: financial_dimensions_explanation.png")