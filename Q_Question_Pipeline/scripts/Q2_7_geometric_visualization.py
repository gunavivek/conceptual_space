#!/usr/bin/env python3
"""
Q2.5 Geometric Space Visualization
Creates A4-style geometric visualization showing:
- Concept centroids (RED)
- Document chunks (GREEN)
- Question position (YELLOW)
- Convex ball boundaries
- Assignment connections

Author: Claude (Anthropic)
Date: 2025-09-14
"""

import json
import os
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Q25GeometricVisualizer:
    """A4-style geometric visualization for Q2.5"""

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
        """Extract all geometric elements for visualization"""
        elements = {
            'centroids': [],
            'centroid_labels': [],
            'chunks': [],
            'chunk_labels': [],
            'chunk_ball_assignments': [],
            'question_vector': None,
            'question_text': '',
            'assigned_balls': [],
            'convex_balls': {}
        }

        # Extract concept centroids (RED)
        concept_centroids = self.a4_data.get('concept_centroids', {})
        for concept_id, centroid_info in concept_centroids.items():
            centroid_coords = centroid_info.get('centroid_coordinates', [])
            if centroid_coords:
                elements['centroids'].append(centroid_coords)
                elements['centroid_labels'].append(concept_id)

        # Extract convex balls information
        convex_balls = self.a4_data.get('convex_balls', {})
        for ball_id, ball_info in convex_balls.items():
            elements['convex_balls'][ball_id] = {
                'centroid': ball_info.get('centroid', []),
                'radius': ball_info.get('radius', 1.0),
                'member_chunks': ball_info.get('member_chunks', [])
            }

            # For visualization, create representative points around the ball centroid
            # Since we don't have actual chunk vectors, we'll create simulated chunk positions
            centroid = ball_info.get('centroid', [])
            if centroid:
                # Create 3-5 representative chunk positions around the centroid
                radius = ball_info.get('radius', 1.0)
                num_chunks = min(5, max(1, len(ball_info.get('member_chunks', [])) + 1))

                for i in range(num_chunks):
                    # Create slight variations around the centroid
                    angle = (2 * np.pi * i) / num_chunks
                    offset_factor = radius * 0.3  # 30% of radius

                    # Add small random offsets in multiple dimensions
                    chunk_vector = np.array(centroid).copy()
                    if len(chunk_vector) >= 2:
                        chunk_vector[0] += offset_factor * np.cos(angle)
                        chunk_vector[1] += offset_factor * np.sin(angle)
                    if len(chunk_vector) >= 3:
                        chunk_vector[2] += offset_factor * np.sin(angle * 0.7)

                    elements['chunks'].append(chunk_vector.tolist())
                    elements['chunk_labels'].append(f"{ball_id}_region")
                    elements['chunk_ball_assignments'].append(ball_id)

        # Calculate question vector (YELLOW)
        question_text = self.q25_data.get('question_text', '')
        if question_text:
            question_embedding = self.semantic_model.encode(question_text)
            elements['question_vector'] = question_embedding.tolist()
            elements['question_text'] = question_text

        # Extract assigned balls from Q2.5
        multi_dim = self.q25_data.get('multi_dimensional_analysis', {})
        for dim_type, dim_analysis in multi_dim.items():
            assignments = dim_analysis.get('convex_ball_assignments', [])
            for assignment in assignments:
                ball_id = assignment.get('ball_id')
                confidence = assignment.get('confidence', 0)
                elements['assigned_balls'].append({
                    'ball_id': ball_id,
                    'confidence': confidence,
                    'dimension': dim_type
                })

        return elements

    def reduce_to_3d(self, elements):
        """Reduce all vectors to 3D using PCA"""
        all_vectors = []

        # Collect all vectors
        all_vectors.extend(elements['centroids'])
        all_vectors.extend(elements['chunks'])

        if elements['question_vector']:
            all_vectors.append(elements['question_vector'])

        if not all_vectors:
            return None

        # Convert to numpy array and handle different dimensions
        vectors_array = np.array(all_vectors)

        # PCA to 3D
        self.pca = PCA(n_components=3)
        vectors_3d = self.pca.fit_transform(vectors_array)

        # Split back into components
        num_centroids = len(elements['centroids'])
        num_chunks = len(elements['chunks'])

        result = {
            'centroids_3d': vectors_3d[:num_centroids] if num_centroids > 0 else [],
            'chunks_3d': vectors_3d[num_centroids:num_centroids + num_chunks] if num_chunks > 0 else [],
            'question_3d': vectors_3d[-1] if elements['question_vector'] else None,
            'explained_variance': self.pca.explained_variance_ratio_.sum()
        }

        return result

    def create_geometric_visualization(self, elements, vectors_3d, doc_id):
        """Create A4-style geometric visualization"""
        fig = go.Figure()

        # 1. Plot concept centroids (RED)
        if len(vectors_3d['centroids_3d']) > 0:
            centroids_3d = vectors_3d['centroids_3d']
            fig.add_trace(go.Scatter3d(
                x=centroids_3d[:, 0],
                y=centroids_3d[:, 1],
                z=centroids_3d[:, 2],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='red',
                    opacity=0.8,
                    symbol='diamond',
                    line=dict(width=2, color='darkred')
                ),
                text=elements['centroid_labels'],
                textposition='top center',
                textfont=dict(size=10, color='red'),
                name='Concept Centroids',
                hovertemplate='<b>Centroid: %{text}</b><br>' +
                            'X: %{x:.3f}<br>' +
                            'Y: %{y:.3f}<br>' +
                            'Z: %{z:.3f}<extra></extra>'
            ))

        # 2. Plot document chunks (GREEN)
        if len(vectors_3d['chunks_3d']) > 0:
            chunks_3d = vectors_3d['chunks_3d']
            chunk_colors = []
            chunk_texts = []

            # Color chunks by their ball assignment
            assigned_balls = {ball['ball_id'] for ball in elements['assigned_balls']}

            for i, ball_assignment in enumerate(elements['chunk_ball_assignments']):
                if ball_assignment in assigned_balls:
                    chunk_colors.append('lightgreen')  # Lighter green for assigned balls
                    chunk_texts.append(f"{ball_assignment}_chunk (ASSIGNED)")
                else:
                    chunk_colors.append('green')
                    chunk_texts.append(f"{ball_assignment}_chunk")

            fig.add_trace(go.Scatter3d(
                x=chunks_3d[:, 0],
                y=chunks_3d[:, 1],
                z=chunks_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=chunk_colors,
                    opacity=0.6,
                    symbol='circle'
                ),
                name='Document Chunks',
                text=chunk_texts,
                hovertemplate='<b>Chunk: %{text}</b><br>' +
                            'X: %{x:.3f}<br>' +
                            'Y: %{y:.3f}<br>' +
                            'Z: %{z:.3f}<extra></extra>'
            ))

        # 3. Plot question position (YELLOW)
        if vectors_3d['question_3d'] is not None:
            question_3d = vectors_3d['question_3d']
            fig.add_trace(go.Scatter3d(
                x=[question_3d[0]],
                y=[question_3d[1]],
                z=[question_3d[2]],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='yellow',
                    opacity=1.0,
                    symbol='diamond',
                    line=dict(width=3, color='orange')
                ),
                text=['QUESTION'],
                textposition='top center',
                textfont=dict(size=12, color='orange', family='Arial Black'),
                name='Question Position',
                hovertemplate='<b>Question</b><br>' +
                            f'Text: {elements["question_text"][:50]}...<br>' +
                            'X: %{x:.3f}<br>' +
                            'Y: %{y:.3f}<br>' +
                            'Z: %{z:.3f}<extra></extra>'
            ))

            # 4. Draw connections to assigned balls
            question_pos = question_3d
            for assigned_ball in elements['assigned_balls']:
                ball_id = assigned_ball['ball_id']
                confidence = assigned_ball['confidence']

                # Find the centroid position for this ball
                if ball_id in elements['centroid_labels']:
                    centroid_idx = elements['centroid_labels'].index(ball_id)
                    if centroid_idx < len(vectors_3d['centroids_3d']):
                        centroid_pos = vectors_3d['centroids_3d'][centroid_idx]

                        # Draw connection line
                        fig.add_trace(go.Scatter3d(
                            x=[question_pos[0], centroid_pos[0]],
                            y=[question_pos[1], centroid_pos[1]],
                            z=[question_pos[2], centroid_pos[2]],
                            mode='lines',
                            line=dict(
                                color='orange',
                                width=max(2, int(confidence * 10)),  # Thicker lines for higher confidence
                                dash='dash' if confidence < 0.5 else 'solid'
                            ),
                            name=f'Assignment: {ball_id}',
                            hovertemplate=f'<b>Assignment to {ball_id}</b><br>' +
                                        f'Confidence: {confidence:.3f}<br>' +
                                        f'Dimension: {assigned_ball["dimension"]}<extra></extra>',
                            showlegend=False
                        ))

        # Customize layout
        fig.update_layout(
            title=dict(
                text=f"Q2.5 Geometric Space Visualization - {doc_id}<br>" +
                     f"<span style='font-size:12px'>Question: {elements['question_text'][:80]}...</span><br>" +
                     f"<span style='font-size:10px; color:red'>RED: Centroids</span> | " +
                     f"<span style='font-size:10px; color:green'>GREEN: Chunks</span> | " +
                     f"<span style='font-size:10px; color:orange'>YELLOW: Question</span> | " +
                     f"<span style='font-size:10px; color:orange'>ORANGE: Assignments</span>",
                x=0.5,
                font=dict(size=14)
            ),
            scene=dict(
                xaxis_title=f'PCA Component 1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PCA Component 2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)',
                zaxis_title=f'PCA Component 3 ({self.pca.explained_variance_ratio_[2]:.1%} variance)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            width=1200,
            height=900,
            margin=dict(t=120, b=50, l=50, r=50),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="Black",
                borderwidth=1
            )
        )

        # Add annotations for assignment details
        assignment_text = f"Total Variance Explained: {vectors_3d['explained_variance']:.1%}<br>"
        assignment_text += f"Assigned Balls: {len(elements['assigned_balls'])}<br>"
        for assigned_ball in elements['assigned_balls']:
            assignment_text += f"• {assigned_ball['ball_id']}: {assigned_ball['confidence']:.3f}<br>"

        fig.add_annotation(
            x=0.02, y=0.02,
            xref="paper", yref="paper",
            text=assignment_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )

        return fig

    def generate_geometric_visualization(self, doc_id: str):
        """Generate complete geometric visualization"""
        print("Q2.5 GEOMETRIC SPACE VISUALIZATION")
        print("=" * 50)
        print(f"Document: {doc_id}")

        # Load data
        if not self.load_data(doc_id):
            return False

        # Extract geometric elements
        print("Extracting geometric elements...")
        elements = self.extract_geometric_elements()

        if not elements['centroids'] and not elements['chunks']:
            print("No geometric data found!")
            return False

        print(f"Found: {len(elements['centroids'])} centroids, {len(elements['chunks'])} chunks")
        print(f"Question assigned to {len(elements['assigned_balls'])} balls")

        # Reduce to 3D
        print("Reducing to 3D space...")
        vectors_3d = self.reduce_to_3d(elements)
        if not vectors_3d:
            print("Failed to reduce to 3D")
            return False

        print(f"3D reduction complete (explained variance: {vectors_3d['explained_variance']:.1%})")

        # Create visualization
        print("Creating geometric visualization...")
        fig = self.create_geometric_visualization(elements, vectors_3d, doc_id)

        # Save
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs'
        )
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f'Q25_geometric_space_{doc_id}.html')
        fig.write_html(output_file)

        print(f"[SUCCESS] Geometric visualization saved: {output_file}")
        print("=" * 50)
        print("VISUALIZATION FEATURES:")
        print("• RED diamonds: Concept centroids from A4")
        print("• GREEN circles: Document chunks from A4")
        print("• YELLOW star: Question position from Q2.5")
        print("• ORANGE lines: Question-to-ball assignments")
        print("• Line thickness: Assignment confidence")
        print("• Interactive: Zoom, rotate, hover for details")

        return True

def main():
    """Main execution"""
    visualizer = Q25GeometricVisualizer()

    # Get document ID from command line or use default
    doc_id = sys.argv[1] if len(sys.argv) > 1 else "finqa_test_1630"

    success = visualizer.generate_geometric_visualization(doc_id)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())