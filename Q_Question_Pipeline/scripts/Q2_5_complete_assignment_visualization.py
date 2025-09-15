#!/usr/bin/env python3
"""
Q2.5 Complete Assignment Hierarchy Visualization
Shows the full assignment chain: Chunks → Concepts → Question

Visual Elements:
- CONCEPT CENTROIDS: Red (assigned) / Gray (unassigned)
- DOCUMENT CHUNKS: Green dots with concept connections
- QUESTION: Large gold diamond
- CHUNK→CONCEPT LINES: Thin blue lines showing chunk membership
- CONCEPT→QUESTION LINES: Thick orange lines showing question assignments

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

class Q25CompleteAssignmentVisualizer:
    """Complete assignment hierarchy visualization"""

    def __init__(self):
        self.a4_data = None
        self.q25_data = None
        self.chunk_data = None
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

        # Load actual document chunks
        chunk_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'A_Concept_pipeline', 'outputs', 'A3_raw_chunks_no_dedup.json'
        )

        try:
            with open(chunk_path, 'r', encoding='utf-8') as f:
                all_chunk_data = json.load(f)
                # Filter chunks for this document
                doc_chunks = [chunk for chunk in all_chunk_data['chunks'] if chunk['doc_id'] == doc_id]
                self.chunk_data = doc_chunks
                print(f"[SUCCESS] Found {len(doc_chunks)} actual chunks for {doc_id}")
        except Exception as e:
            print(f"Warning: Could not load chunk data: {e}")
            self.chunk_data = []

        return True

    def extract_complete_assignment_elements(self):
        """Extract all elements with complete assignment relationships"""
        elements = {
            'centroids': [],
            'centroid_labels': [],
            'chunks': [],
            'chunk_labels': [],
            'chunk_to_concept_assignments': [],  # New: chunk->concept relationships
            'assigned_centroids': [],
            'assigned_centroid_labels': [],
            'unassigned_centroids': [],
            'unassigned_centroid_labels': [],
            'question_vector': None,
            'question_text': '',
            'question_to_concept_assignments': []  # question->concept relationships
        }

        # Get question-to-concept assignments first
        assigned_concept_ids = set()
        multi_dim = self.q25_data.get('multi_dimensional_analysis', {})
        for dim_type, dim_analysis in multi_dim.items():
            assignments = dim_analysis.get('convex_ball_assignments', [])
            for assignment in assignments:
                ball_id = assignment.get('ball_id')
                confidence = assignment.get('confidence', 0)
                assigned_concept_ids.add(ball_id)
                elements['question_to_concept_assignments'].append({
                    'concept_id': ball_id,
                    'confidence': confidence,
                    'dimension': dim_type
                })

        # Extract and categorize concept centroids
        concept_centroids = self.a4_data.get('concept_centroids', {})
        for concept_id, centroid_info in concept_centroids.items():
            centroid_coords = centroid_info.get('centroid_coordinates', [])
            if centroid_coords:
                elements['centroids'].append(centroid_coords)
                elements['centroid_labels'].append(concept_id)

                if concept_id in assigned_concept_ids:
                    elements['assigned_centroids'].append(centroid_coords)
                    elements['assigned_centroid_labels'].append(concept_id)
                else:
                    elements['unassigned_centroids'].append(centroid_coords)
                    elements['unassigned_centroid_labels'].append(concept_id)

        # Use real document chunks if available
        if self.chunk_data:
            # Use actual document chunks
            for chunk_index, chunk in enumerate(self.chunk_data):
                chunk_id = chunk.get('chunk_id', f'chunk_{chunk_index}')
                chunk_content = chunk.get('content', '')
                chunk_preview = chunk_content[:50] + '...' if len(chunk_content) > 50 else chunk_content

                # Get chunk's semantic vector
                chunk_embedding = self.semantic_model.encode(chunk_content)
                elements['chunks'].append(chunk_embedding.tolist())

                # Create meaningful chunk label
                elements['chunk_labels'].append(f"{chunk_id}: {chunk_preview}")

                # Find which concept this chunk belongs to based on concept memberships
                concept_memberships = chunk.get('concept_memberships', [])

                # Use the highest scoring concept membership
                if concept_memberships:
                    membership_scores = chunk.get('membership_scores', {})
                    best_concept = max(concept_memberships,
                                     key=lambda c: membership_scores.get(c, 0))

                    # Map concept core_id to actual concept name
                    concept_mapping = self.get_concept_mapping(best_concept)

                    elements['chunk_to_concept_assignments'].append({
                        'chunk_index': chunk_index,
                        'concept_id': concept_mapping,
                        'chunk_label': f"{chunk_id}: {chunk_preview}",
                        'assigned_to_question': concept_mapping in assigned_concept_ids,
                        'chunk_id': chunk_id,
                        'chunk_preview': chunk_preview
                    })
                else:
                    # Fallback: assign to first available concept
                    fallback_concept = list(self.a4_data.get('convex_balls', {}).keys())[0] if self.a4_data.get('convex_balls') else 'unknown'
                    elements['chunk_to_concept_assignments'].append({
                        'chunk_index': chunk_index,
                        'concept_id': fallback_concept,
                        'chunk_label': f"{chunk_id}: {chunk_preview}",
                        'assigned_to_question': fallback_concept in assigned_concept_ids,
                        'chunk_id': chunk_id,
                        'chunk_preview': chunk_preview
                    })
        else:
            # Fallback: Create simulated chunks if no real data available
            convex_balls = self.a4_data.get('convex_balls', {})
            chunk_index = 0

            for concept_id, ball_info in convex_balls.items():
                centroid = ball_info.get('centroid', [])
                if centroid:
                    radius = ball_info.get('radius', 1.0)
                    num_chunks = 2  # Reduced for clarity

                    for i in range(num_chunks):
                        angle = (2 * np.pi * i) / num_chunks
                        offset_factor = radius * 0.35

                        chunk_vector = np.array(centroid).copy()
                        if len(chunk_vector) >= 2:
                            chunk_vector[0] += offset_factor * np.cos(angle)
                            chunk_vector[1] += offset_factor * np.sin(angle)
                        if len(chunk_vector) >= 3:
                            chunk_vector[2] += offset_factor * np.sin(angle * 0.6)

                        elements['chunks'].append(chunk_vector.tolist())
                        elements['chunk_labels'].append(f"simulated_chunk_{chunk_index}")

                        elements['chunk_to_concept_assignments'].append({
                            'chunk_index': chunk_index,
                            'concept_id': concept_id,
                            'chunk_label': f"simulated_chunk_{chunk_index}",
                            'assigned_to_question': concept_id in assigned_concept_ids
                        })

                        chunk_index += 1

        # Calculate question vector
        question_text = self.q25_data.get('question_text', '')
        if question_text:
            question_embedding = self.semantic_model.encode(question_text)
            elements['question_vector'] = question_embedding.tolist()
            elements['question_text'] = question_text

        return elements

    def get_concept_mapping(self, core_id: str) -> str:
        """Map core concept ID to A4 concept name using improved mapping strategy"""
        # Get available A4 concepts
        available_concepts = list(self.a4_data.get('convex_balls', {}).keys())
        if not available_concepts:
            return 'unknown'

        # Load A3 concept mapping for semantic understanding
        try:
            chunk_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'A_Concept_pipeline', 'outputs', 'A3_concept_based_chunks.json'
            )

            with open(chunk_path, 'r', encoding='utf-8') as f:
                a3_data = json.load(f)

            # Get the canonical name for this core_id
            concept_centroids = a3_data.get('concept_centroids', {})
            if core_id in concept_centroids:
                canonical_name = concept_centroids[core_id].get('canonical_name', '').lower()

                # Enhanced keyword matching with financial domain knowledge
                financial_mappings = {
                    'revenue': ['revenue_recognition', 'revenue', 'income_from_sales'],
                    'income': ['net_income', 'income_taxes', 'operating_income'],
                    'tax': ['income_taxes', 'tax_expense'],
                    'compensation': ['incentive_compensation', 'employee_benefits'],
                    'expense': ['operating_expenses', 'cost_of_sales'],
                    'asset': ['total_assets', 'current_assets'],
                    'liability': ['total_liabilities', 'current_liabilities'],
                    'cash': ['cash_flow', 'cash_equivalents'],
                    'equity': ['shareholders_equity', 'retained_earnings']
                }

                # Check for semantic matches using financial keywords
                canonical_words = canonical_name.replace('_', ' ').split()

                for canonical_word in canonical_words:
                    if len(canonical_word) > 3:  # Skip short words
                        for keyword, a4_options in financial_mappings.items():
                            if keyword in canonical_word:
                                # Find matching A4 concept
                                for a4_concept in available_concepts:
                                    if any(option in a4_concept.lower() for option in a4_options):
                                        print(f"[MAPPING] {core_id} -> {a4_concept} (via {keyword})")
                                        return a4_concept

                # Direct string matching as backup
                for a4_concept in available_concepts:
                    a4_concept_lower = a4_concept.lower().replace('_', ' ')
                    canonical_lower = canonical_name.replace('_', ' ')

                    # Check for any word overlap
                    if any(word in a4_concept_lower for word in canonical_lower.split() if len(word) > 3):
                        print(f"[MAPPING] {core_id} -> {a4_concept} (direct match)")
                        return a4_concept

        except Exception as e:
            print(f"Warning: Could not load A3 concept mapping: {e}")

        # Consistent hash-based distribution as fallback
        concept_index = hash(core_id) % len(available_concepts)
        mapped_concept = available_concepts[concept_index]
        print(f"[MAPPING] {core_id} -> {mapped_concept} (hash-based)")
        return mapped_concept

    def get_concept_definition(self, concept_id: str) -> str:
        """Get concept definition text from A4 data"""
        try:
            concept_info = self.a4_data.get('convex_balls', {}).get(concept_id, {})
            definition = concept_info.get('definition', '')
            if definition:
                # Truncate to reasonable length for display
                return definition[:100] + '...' if len(definition) > 100 else definition
            else:
                # Fallback: use concept name as readable definition
                readable_name = concept_id.replace('_', ' ').title()
                return f"Financial concept: {readable_name}"
        except Exception as e:
            return f"Concept: {concept_id.replace('_', ' ').title()}"

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

        vectors_array = np.array(all_vectors)
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

    def create_complete_assignment_visualization(self, elements, vectors_3d, doc_id):
        """Create complete assignment hierarchy visualization"""
        fig = go.Figure()

        # 1. Plot concept centroids (categorized by assignment)
        centroids_3d = vectors_3d['centroids_3d']
        assigned_concept_ids = {assign['concept_id'] for assign in elements['question_to_concept_assignments']}

        assigned_indices = []
        unassigned_indices = []

        for i, concept_id in enumerate(elements['centroid_labels']):
            if concept_id in assigned_concept_ids:
                assigned_indices.append(i)
            else:
                unassigned_indices.append(i)

        # Assigned concepts (RED)
        if assigned_indices:
            assigned_centroids = centroids_3d[assigned_indices]
            assigned_labels = [elements['centroid_labels'][i] for i in assigned_indices]

            fig.add_trace(go.Scatter3d(
                x=assigned_centroids[:, 0],
                y=assigned_centroids[:, 1],
                z=assigned_centroids[:, 2],
                mode='markers+text',
                marker=dict(
                    size=14,
                    color='crimson',
                    opacity=1.0,
                    symbol='square',
                    line=dict(width=3, color='darkred')
                ),
                text=assigned_labels,
                textposition='top center',
                textfont=dict(size=10, color='darkred', family='Arial Black'),
                name='🎯 ASSIGNED Concepts',
                customdata=[self.get_concept_definition(label) for label in assigned_labels],
                hovertemplate='<b>ASSIGNED: %{text}</b><br>Definition: %{customdata}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ))

        # Unassigned concepts (GRAY)
        if unassigned_indices:
            unassigned_centroids = centroids_3d[unassigned_indices]
            unassigned_labels = [elements['centroid_labels'][i] for i in unassigned_indices]

            fig.add_trace(go.Scatter3d(
                x=unassigned_centroids[:, 0],
                y=unassigned_centroids[:, 1],
                z=unassigned_centroids[:, 2],
                mode='markers',
                marker=dict(
                    size=14,
                    color='lightgray',
                    opacity=0.3,
                    symbol='square',
                    line=dict(width=1, color='gray')
                ),
                text=unassigned_labels,
                name='○ Unassigned Concepts',
                customdata=[self.get_concept_definition(label) for label in unassigned_labels],
                hovertemplate='<b>Unassigned: %{text}</b><br>Definition: %{customdata}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ))

        # 2. Plot document chunks with concept assignment categorization
        if len(vectors_3d['chunks_3d']) > 0:
            chunks_3d = vectors_3d['chunks_3d']

            # Categorize chunks by their concept assignment status
            assigned_chunk_indices = []
            unassigned_chunk_indices = []

            for assignment in elements['chunk_to_concept_assignments']:
                if assignment['assigned_to_question']:
                    assigned_chunk_indices.append(assignment['chunk_index'])
                else:
                    unassigned_chunk_indices.append(assignment['chunk_index'])

            # Assigned chunks (BRIGHT GREEN)
            if assigned_chunk_indices:
                assigned_chunks = chunks_3d[assigned_chunk_indices]
                assigned_chunk_texts = [f"📍 {elements['chunk_to_concept_assignments'][i]['chunk_preview']}" for i in assigned_chunk_indices]

                fig.add_trace(go.Scatter3d(
                    x=assigned_chunks[:, 0],
                    y=assigned_chunks[:, 1],
                    z=assigned_chunks[:, 2],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color='limegreen',
                        opacity=0.8,
                        symbol='circle'
                    ),
                    name='🌿 Assigned Chunks',
                    text=assigned_chunk_texts,
                    customdata=[(elements['chunk_to_concept_assignments'][i]['chunk_id'], elements['chunk_to_concept_assignments'][i]['concept_id']) for i in assigned_chunk_indices],
                    hovertemplate='<b>ASSIGNED CHUNK</b><br>ID: %{customdata[0]}<br>Content: %{text}<br>Assigned to: %{customdata[1]}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
                ))

            # Unassigned chunks (PALE GREEN)
            if unassigned_chunk_indices:
                unassigned_chunks = chunks_3d[unassigned_chunk_indices]
                unassigned_chunk_texts = [f"○ {elements['chunk_to_concept_assignments'][i]['chunk_preview']}" for i in unassigned_chunk_indices]

                fig.add_trace(go.Scatter3d(
                    x=unassigned_chunks[:, 0],
                    y=unassigned_chunks[:, 1],
                    z=unassigned_chunks[:, 2],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color='palegreen',
                        opacity=0.4,
                        symbol='circle'
                    ),
                    name='○ Unassigned Chunks',
                    text=unassigned_chunk_texts,
                    customdata=[(elements['chunk_to_concept_assignments'][i]['chunk_id'], elements['chunk_to_concept_assignments'][i]['concept_id']) for i in unassigned_chunk_indices],
                    hovertemplate='<b>UNASSIGNED CHUNK</b><br>ID: %{customdata[0]}<br>Content: %{text}<br>Assigned to: %{customdata[1]}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
                ))

        # 3. Plot question position (LARGE GOLD)
        if vectors_3d['question_3d'] is not None:
            question_3d = vectors_3d['question_3d']
            fig.add_trace(go.Scatter3d(
                x=[question_3d[0]],
                y=[question_3d[1]],
                z=[question_3d[2]],
                mode='markers+text',
                marker=dict(
                    size=14,
                    color='gold',
                    opacity=1.0,
                    symbol='diamond',
                    line=dict(width=4, color='darkorange')
                ),
                text=[f'❓ {elements["question_text"][:50]}...'],
                textposition='top center',
                textfont=dict(size=12, color='darkorange', family='Arial Black'),
                name='⭐ QUESTION',
                hovertemplate=f'<b>QUESTION</b><br>Full Text: {elements["question_text"]}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'
            ))

            question_pos = question_3d

            # 4. Draw CHUNK → CONCEPT assignment lines (THIN BLUE)
            for assignment in elements['chunk_to_concept_assignments']:
                chunk_idx = assignment['chunk_index']
                concept_id = assignment['concept_id']

                if chunk_idx < len(chunks_3d) and concept_id in elements['centroid_labels']:
                    chunk_pos = chunks_3d[chunk_idx]
                    concept_idx = elements['centroid_labels'].index(concept_id)
                    concept_pos = centroids_3d[concept_idx]

                    # Green colors to match chunks (assigned vs unassigned)
                    line_color = 'limegreen' if assignment['assigned_to_question'] else 'palegreen'
                    line_width = 2  # Uniform width for all chunk-concept lines
                    line_opacity = 0.7 if assignment['assigned_to_question'] else 0.3

                    fig.add_trace(go.Scatter3d(
                        x=[chunk_pos[0], concept_pos[0]],
                        y=[chunk_pos[1], concept_pos[1]],
                        z=[chunk_pos[2], concept_pos[2]],
                        mode='lines',
                        line=dict(
                            color=line_color,
                            width=line_width,
                            dash='solid'
                        ),
                        opacity=line_opacity,
                        name=f'🔗 Chunk→{concept_id}',
                        hovertemplate=f'<b>Chunk → {concept_id}</b><br>Assignment: {"Question Related" if assignment["assigned_to_question"] else "Background"}<extra></extra>',
                        showlegend=False
                    ))

            # 5. Draw CONCEPT → QUESTION assignment lines (THICK ORANGE)
            for assignment in elements['question_to_concept_assignments']:
                concept_id = assignment['concept_id']
                confidence = assignment['confidence']

                if concept_id in elements['centroid_labels']:
                    concept_idx = elements['centroid_labels'].index(concept_id)
                    concept_pos = centroids_3d[concept_idx]

                    line_width = 2  # Uniform width for all concept-question lines
                    line_color = 'gold' if confidence > 0.5 else 'goldenrod'

                    fig.add_trace(go.Scatter3d(
                        x=[question_pos[0], concept_pos[0]],
                        y=[question_pos[1], concept_pos[1]],
                        z=[question_pos[2], concept_pos[2]],
                        mode='lines',
                        line=dict(
                            color=line_color,
                            width=line_width
                        ),
                        name=f'🔗 Q→{concept_id}',
                        hovertemplate=f'<b>Question → {concept_id}</b><br>Confidence: {confidence:.3f}<br>Dimension: {assignment["dimension"]}<extra></extra>',
                        showlegend=False
                    ))

        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=f"<b>Q2.5 COMPLETE Assignment Hierarchy - {doc_id}</b><br>" +
                     f"<span style='font-size:12px'>{elements['question_text'][:100]}...</span><br>" +
                     f"<span style='font-size:10px; color:limegreen'>🔗 GREEN: Chunk→Concept</span> | " +
                     f"<span style='font-size:10px; color:gold'>🔗 GOLD: Concept→Question</span> | " +
                     f"<span style='font-size:9px; color:gray'>Line Length = Semantic Distance in 3D Space</span>",
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title=f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%})',
                yaxis_title=f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%})',
                zaxis_title=f'PC3 ({self.pca.explained_variance_ratio_[2]:.1%})',
                camera=dict(eye=dict(x=2.0, y=2.0, z=2.0)),
                aspectmode='cube',
                bgcolor='rgba(240,240,240,0.1)'
            ),
            width=1500,
            height=1100,
            margin=dict(t=150, b=60, l=60, r=60),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=2,
                font=dict(size=11)
            )
        )

        # Add assignment hierarchy explanation
        hierarchy_text = f"""<b>ASSIGNMENT HIERARCHY</b><br>
📊 Total Variance: {vectors_3d['explained_variance']:.1%}<br>
🌿 Total Chunks: {len(elements['chunks'])}<br>
🎯 Total Concepts: {len(elements['centroids'])}<br>
⭐ Question Assignments: {len(elements['question_to_concept_assignments'])}<br><br>
<b>FLOW: Chunks → Concepts → Question</b><br>
1. Green lines: Chunk membership in concepts<br>
2. Gold lines: Question assignments to concepts<br>
3. Line length = Semantic distance in 3D space<br>
4. Uniform line width for visual clarity"""

        fig.add_annotation(
            x=0.02, y=0.02,
            xref="paper", yref="paper",
            text=hierarchy_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=2,
            align="left"
        )

        return fig

    def generate_complete_assignment_visualization(self, doc_id: str):
        """Generate complete assignment hierarchy visualization"""
        print("Q2.5 COMPLETE ASSIGNMENT HIERARCHY VISUALIZATION")
        print("=" * 70)
        print(f"Document: {doc_id}")

        if not self.load_data(doc_id):
            return False

        print("Extracting complete assignment relationships...")
        elements = self.extract_complete_assignment_elements()

        print(f"Found:")
        print(f"  Concepts: {len(elements['centroids'])} total")
        print(f"  Chunks: {len(elements['chunks'])} total")
        print(f"  Chunk->Concept assignments: {len(elements['chunk_to_concept_assignments'])}")
        print(f"  Question->Concept assignments: {len(elements['question_to_concept_assignments'])}")

        print("Reducing to 3D space...")
        vectors_3d = self.reduce_to_3d(elements)
        if not vectors_3d:
            print("Failed to reduce to 3D")
            return False

        print(f"3D reduction complete (variance: {vectors_3d['explained_variance']:.1%})")

        print("Creating complete assignment visualization...")
        fig = self.create_complete_assignment_visualization(elements, vectors_3d, doc_id)

        # Save
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs'
        )
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f'Q25_geometric_space_{doc_id}.html')
        fig.write_html(output_file)

        print(f"[SUCCESS] Complete assignment visualization: {output_file}")
        print("=" * 70)
        print("COMPLETE ASSIGNMENT HIERARCHY FEATURES:")
        print("* BLUE lines: Chunk -> Concept assignments")
        print("* ORANGE lines: Concept -> Question assignments")
        print("* RED concepts: Assigned to question")
        print("* GRAY concepts: Not assigned to question")
        print("* BRIGHT GREEN chunks: In assigned concepts")
        print("* PALE GREEN chunks: In unassigned concepts")
        print("* GOLD question: Final assignment target")
        print("* Complete visual hierarchy: Chunks -> Concepts -> Question")

        return True

def main():
    """Main execution"""
    visualizer = Q25CompleteAssignmentVisualizer()
    doc_id = sys.argv[1] if len(sys.argv) > 1 else "finqa_test_1630"
    success = visualizer.generate_complete_assignment_visualization(doc_id)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())