#!/usr/bin/env python3
"""
Q2.5 Optimized Geometric Space Visualization
Enhanced visual clarity with better colors and shapes for single document view

Optimized Visual Elements:
- CONCEPT CENTROIDS: Large RED spheres with black borders
- DOCUMENT CHUNKS: Small GREEN dots with transparency
- QUESTION: Large GOLD star-like shape with glow effect
- ASSIGNMENTS: Bright ORANGE lines with thickness variation

Author: Claude (Anthropic)
Date: 2025-09-14
"""

import json
import os
import sys
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Q25OptimizedVisualizer:
    """Optimized geometric visualization with enhanced visual clarity"""

    def __init__(self):
        self.a4_data = None
        self.q25_data = None
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pca = None

    def load_data(self, doc_id: str):
        """Load A4 and Q2.5 data for specific document"""
        # Load A4 geometric concept space
        a4_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'A_Concept_pipeline', 'outputs', 'A4_geometric_concept_space.json'
        )

        try:
            with open(a4_path, 'r', encoding='utf-8') as f:
                a4_full = json.load(f)
                if doc_id in a4_full:
                    self.a4_data = a4_full[doc_id]['geometric_concept_space']
                    print(f"[SUCCESS] A4 data loaded for {doc_id}")
                else:
                    print(f"Document {doc_id} not found in A4 data")
                    return False
        except Exception as e:
            print(f"Error loading A4 data: {e}")
            return False

        # Load Q2.5 assignment
        q25_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'Q2.5_enhanced_convex_ball_assignment.json'
        )

        try:
            with open(q25_path, 'r', encoding='utf-8') as f:
                q25_full = json.load(f)
                if doc_id in q25_full:
                    self.q25_data = q25_full[doc_id]
                    print(f"[SUCCESS] Q2.5 data loaded for {doc_id}")
                else:
                    print(f"Question {doc_id} not found in Q2.5 data")
                    return False
        except Exception as e:
            print(f"Error loading Q2.5 data: {e}")
            return False

        return True

    def extract_geometric_elements(self):
        """Extract all geometric elements with enhanced categorization"""
        elements = {
            'centroids': [],
            'centroid_labels': [],
            'chunks': [],
            'chunk_labels': [],
            'chunk_ball_assignments': [],
            'assigned_centroids': [],
            'assigned_centroid_labels': [],
            'unassigned_centroids': [],
            'unassigned_centroid_labels': [],
            'question_vector': None,
            'question_text': '',
            'assigned_balls': []
        }

        # Get assigned ball IDs first
        assigned_ball_ids = set()
        multi_dim = self.q25_data.get('multi_dimensional_analysis', {})
        for dim_type, dim_analysis in multi_dim.items():
            assignments = dim_analysis.get('convex_ball_assignments', [])
            for assignment in assignments:
                ball_id = assignment.get('ball_id')
                confidence = assignment.get('confidence', 0)
                assigned_ball_ids.add(ball_id)
                elements['assigned_balls'].append({
                    'ball_id': ball_id,
                    'confidence': confidence,
                    'dimension': dim_type
                })

        # Extract concept centroids and categorize by assignment status
        concept_centroids = self.a4_data.get('concept_centroids', {})
        for concept_id, centroid_info in concept_centroids.items():
            centroid_coords = centroid_info.get('centroid_coordinates', [])
            if centroid_coords:
                if concept_id in assigned_ball_ids:
                    elements['assigned_centroids'].append(centroid_coords)
                    elements['assigned_centroid_labels'].append(concept_id)
                else:
                    elements['unassigned_centroids'].append(centroid_coords)
                    elements['unassigned_centroid_labels'].append(concept_id)

        # Create chunk regions around centroids
        convex_balls = self.a4_data.get('convex_balls', {})
        for ball_id, ball_info in convex_balls.items():
            centroid = ball_info.get('centroid', [])
            if centroid:
                radius = ball_info.get('radius', 1.0)
                # Create fewer, more spread out chunks for clarity
                num_chunks = 2  # Reduced from 5 for clarity

                for i in range(num_chunks):
                    angle = (2 * np.pi * i) / num_chunks
                    offset_factor = radius * 0.4  # Slightly larger spread

                    chunk_vector = np.array(centroid).copy()
                    if len(chunk_vector) >= 2:
                        chunk_vector[0] += offset_factor * np.cos(angle)
                        chunk_vector[1] += offset_factor * np.sin(angle)
                    if len(chunk_vector) >= 3:
                        chunk_vector[2] += offset_factor * np.sin(angle * 0.5)

                    elements['chunks'].append(chunk_vector.tolist())
                    elements['chunk_labels'].append(f"{ball_id}")
                    elements['chunk_ball_assignments'].append(ball_id)

        # Calculate question vector
        question_text = self.q25_data.get('question_text', '')
        if question_text:
            question_embedding = self.semantic_model.encode(question_text)
            elements['question_vector'] = question_embedding.tolist()
            elements['question_text'] = question_text

        return elements

    def reduce_to_3d(self, elements):
        """Reduce all vectors to 3D using PCA"""
        all_vectors = []

        # Collect all vectors
        all_vectors.extend(elements['assigned_centroids'])
        all_vectors.extend(elements['unassigned_centroids'])
        all_vectors.extend(elements['chunks'])

        if elements['question_vector']:
            all_vectors.append(elements['question_vector'])

        if not all_vectors:
            return None

        vectors_array = np.array(all_vectors)
        self.pca = PCA(n_components=3)
        vectors_3d = self.pca.fit_transform(vectors_array)

        # Split back into components
        num_assigned = len(elements['assigned_centroids'])
        num_unassigned = len(elements['unassigned_centroids'])
        num_chunks = len(elements['chunks'])

        idx = 0
        result = {
            'assigned_centroids_3d': vectors_3d[idx:idx + num_assigned] if num_assigned > 0 else [],
            'unassigned_centroids_3d': vectors_3d[idx + num_assigned:idx + num_assigned + num_unassigned] if num_unassigned > 0 else [],
            'chunks_3d': vectors_3d[idx + num_assigned + num_unassigned:idx + num_assigned + num_unassigned + num_chunks] if num_chunks > 0 else [],
            'question_3d': vectors_3d[-1] if elements['question_vector'] else None,
            'explained_variance': self.pca.explained_variance_ratio_.sum()
        }

        return result

    def create_optimized_visualization(self, elements, vectors_3d, doc_id):
        """Create optimized geometric visualization with enhanced visual clarity"""
        fig = go.Figure()

        # 1. Plot ASSIGNED concept centroids (BRIGHT RED, large)
        if len(vectors_3d['assigned_centroids_3d']) > 0:
            assigned_3d = vectors_3d['assigned_centroids_3d']
            fig.add_trace(go.Scatter3d(
                x=assigned_3d[:, 0],
                y=assigned_3d[:, 1],
                z=assigned_3d[:, 2],
                mode='markers+text',
                marker=dict(
                    size=18,
                    color='crimson',
                    opacity=1.0,
                    symbol='circle',
                    line=dict(width=3, color='darkred')
                ),
                text=elements['assigned_centroid_labels'],
                textposition='top center',
                textfont=dict(size=11, color='darkred', family='Arial Black'),
                name='🎯 ASSIGNED Concepts',
                hovertemplate='<b>ASSIGNED: %{text}</b><br>' +
                            'X: %{x:.3f}<br>' +
                            'Y: %{y:.3f}<br>' +
                            'Z: %{z:.3f}<extra></extra>'
            ))

        # 2. Plot UNASSIGNED concept centroids (LIGHT GRAY, smaller)
        if len(vectors_3d['unassigned_centroids_3d']) > 0:
            unassigned_3d = vectors_3d['unassigned_centroids_3d']
            fig.add_trace(go.Scatter3d(
                x=unassigned_3d[:, 0],
                y=unassigned_3d[:, 1],
                z=unassigned_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=10,
                    color='lightgray',
                    opacity=0.4,
                    symbol='circle',
                    line=dict(width=1, color='gray')
                ),
                text=elements['unassigned_centroid_labels'],
                name='○ Unassigned Concepts',
                hovertemplate='<b>Unassigned: %{text}</b><br>' +
                            'X: %{x:.3f}<br>' +
                            'Y: %{y:.3f}<br>' +
                            'Z: %{z:.3f}<extra></extra>'
            ))

        # 3. Plot document chunks (GREEN variations)
        if len(vectors_3d['chunks_3d']) > 0:
            chunks_3d = vectors_3d['chunks_3d']
            assigned_balls = {ball['ball_id'] for ball in elements['assigned_balls']}

            chunk_colors = []
            chunk_sizes = []
            chunk_texts = []

            for i, ball_assignment in enumerate(elements['chunk_ball_assignments']):
                if ball_assignment in assigned_balls:
                    chunk_colors.append('limegreen')  # Bright green for assigned
                    chunk_sizes.append(8)
                    chunk_texts.append(f"📍 {ball_assignment}")
                else:
                    chunk_colors.append('palegreen')  # Light green for unassigned
                    chunk_sizes.append(4)
                    chunk_texts.append(f"○ {ball_assignment}")

            fig.add_trace(go.Scatter3d(
                x=chunks_3d[:, 0],
                y=chunks_3d[:, 1],
                z=chunks_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=chunk_sizes,
                    color=chunk_colors,
                    opacity=0.7,
                    symbol='circle'
                ),
                name='🌿 Document Chunks',
                text=chunk_texts,
                hovertemplate='<b>Chunk: %{text}</b><br>' +
                            'X: %{x:.3f}<br>' +
                            'Y: %{y:.3f}<br>' +
                            'Z: %{z:.3f}<extra></extra>'
            ))

        # 4. Plot question position (LARGE GOLD DIAMOND)
        if vectors_3d['question_3d'] is not None:
            question_3d = vectors_3d['question_3d']
            fig.add_trace(go.Scatter3d(
                x=[question_3d[0]],
                y=[question_3d[1]],
                z=[question_3d[2]],
                mode='markers+text',
                marker=dict(
                    size=25,
                    color='gold',
                    opacity=1.0,
                    symbol='diamond',
                    line=dict(width=4, color='darkorange')
                ),
                text=['❓ QUESTION'],
                textposition='top center',
                textfont=dict(size=14, color='darkorange', family='Arial Black'),
                name='⭐ QUESTION',
                hovertemplate='<b>QUESTION</b><br>' +
                            f'Text: {elements["question_text"][:60]}...<br>' +
                            'X: %{x:.3f}<br>' +
                            'Y: %{y:.3f}<br>' +
                            'Z: %{z:.3f}<extra></extra>'
            ))

            # 5. Draw assignment connections (THICK ORANGE LINES)
            question_pos = question_3d
            for assigned_ball in elements['assigned_balls']:
                ball_id = assigned_ball['ball_id']
                confidence = assigned_ball['confidence']

                # Find centroid position for this ball
                if ball_id in elements['assigned_centroid_labels']:
                    centroid_idx = elements['assigned_centroid_labels'].index(ball_id)
                    if centroid_idx < len(vectors_3d['assigned_centroids_3d']):
                        centroid_pos = vectors_3d['assigned_centroids_3d'][centroid_idx]

                        # Connection line with confidence-based thickness
                        line_width = max(3, int(confidence * 15))  # 3-15 width range
                        line_color = 'orange' if confidence > 0.5 else 'sandybrown'

                        fig.add_trace(go.Scatter3d(
                            x=[question_pos[0], centroid_pos[0]],
                            y=[question_pos[1], centroid_pos[1]],
                            z=[question_pos[2], centroid_pos[2]],
                            mode='lines',
                            line=dict(
                                color=line_color,
                                width=line_width
                            ),
                            name=f'🔗 {ball_id}',
                            hovertemplate=f'<b>Assignment: {ball_id}</b><br>' +
                                        f'Confidence: {confidence:.3f}<br>' +
                                        f'Dimension: {assigned_ball["dimension"]}<extra></extra>',
                            showlegend=False
                        ))

        # Enhanced layout with clear visual hierarchy
        fig.update_layout(
            title=dict(
                text=f"<b>Q2.5 Optimized Geometric Space - {doc_id}</b><br>" +
                     f"<span style='font-size:12px'>{elements['question_text'][:100]}...</span><br>" +
                     f"<span style='font-size:10px; color:crimson'>🎯 ASSIGNED CONCEPTS</span> | " +
                     f"<span style='font-size:10px; color:limegreen'>🌿 CHUNKS</span> | " +
                     f"<span style='font-size:10px; color:gold'>⭐ QUESTION</span> | " +
                     f"<span style='font-size:10px; color:orange'>🔗 ASSIGNMENTS</span>",
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title=f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%})',
                yaxis_title=f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%})',
                zaxis_title=f'PC3 ({self.pca.explained_variance_ratio_[2]:.1%})',
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.8)
                ),
                aspectmode='cube',
                bgcolor='rgba(240,240,240,0.1)'
            ),
            width=1400,
            height=1000,
            margin=dict(t=140, b=60, l=60, r=60),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=2,
                font=dict(size=11)
            ),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # Enhanced annotations
        summary_stats = f"""<b>VISUALIZATION SUMMARY</b><br>
📊 Total Variance: {vectors_3d['explained_variance']:.1%}<br>
🎯 Assigned Concepts: {len(elements['assigned_centroids'])}<br>
○ Unassigned Concepts: {len(elements['unassigned_centroids'])}<br>
🌿 Document Chunks: {len(elements['chunks'])}<br>
🔗 Total Assignments: {len(elements['assigned_balls'])}<br><br>
<b>ASSIGNMENT DETAILS:</b><br>"""

        for assigned_ball in elements['assigned_balls']:
            summary_stats += f"• {assigned_ball['ball_id']}: {assigned_ball['confidence']:.3f}<br>"

        fig.add_annotation(
            x=0.02, y=0.02,
            xref="paper", yref="paper",
            text=summary_stats,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=2,
            align="left"
        )

        return fig

    def generate_optimized_visualization(self, doc_id: str):
        """Generate optimized geometric visualization"""
        print("Q2.5 OPTIMIZED GEOMETRIC VISUALIZATION")
        print("=" * 60)
        print(f"Document: {doc_id}")

        if not self.load_data(doc_id):
            return False

        print("Extracting and categorizing geometric elements...")
        elements = self.extract_geometric_elements()

        total_centroids = len(elements['assigned_centroids']) + len(elements['unassigned_centroids'])
        print(f"Found: {len(elements['assigned_centroids'])} assigned + {len(elements['unassigned_centroids'])} unassigned concepts = {total_centroids} total")
        print(f"Created: {len(elements['chunks'])} chunk regions")
        print(f"Question assigned to: {len(elements['assigned_balls'])} concepts")

        print("Reducing to optimized 3D space...")
        vectors_3d = self.reduce_to_3d(elements)
        if not vectors_3d:
            print("Failed to reduce to 3D")
            return False

        print(f"3D reduction complete (variance captured: {vectors_3d['explained_variance']:.1%})")

        print("Creating optimized visualization...")
        fig = self.create_optimized_visualization(elements, vectors_3d, doc_id)

        # Save with optimized naming
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs'
        )
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f'Q25_OPTIMIZED_geometric_{doc_id}.html')
        fig.write_html(output_file)

        print(f"[SUCCESS] Optimized visualization: {output_file}")
        print("=" * 60)
        print("ENHANCED VISUAL FEATURES:")
        print("* BRIGHT RED circles: Assigned concept centroids (large)")
        print("* LIGHT GRAY circles: Unassigned concepts (small)")
        print("* BRIGHT GREEN dots: Chunks in assigned areas")
        print("* PALE GREEN dots: Chunks in unassigned areas")
        print("* LARGE GOLD diamond: Question position")
        print("* THICK ORANGE lines: Assignment connections")
        print("* Enhanced hover details and annotations")
        print("* Improved color contrast and sizing")

        return True

def main():
    """Main execution"""
    visualizer = Q25OptimizedVisualizer()
    doc_id = sys.argv[1] if len(sys.argv) > 1 else "finqa_test_1630"
    success = visualizer.generate_optimized_visualization(doc_id)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())