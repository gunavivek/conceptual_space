#!/usr/bin/env python3
"""
Q2.5 Enhanced Visualization System
Creates comprehensive visualizations showing question assignments to A4 convex balls

Features:
- Interactive 3D visualization of convex balls with question assignments
- Multi-dimensional membership analysis charts
- Confidence score distributions
- Detailed question-to-ball mapping views
- Integration with A4 geometric concept space

Author: Claude (Anthropic)
Date: 2025-09-14
"""

import json
import os
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Q25VisualizationEngine:
    """Enhanced visualization engine for Q2.5 outputs"""

    def __init__(self):
        self.a4_data = None
        self.q25_data = None
        self.combined_data = {}
        self.visualization_cache = {}

    def load_data(self):
        """Load A4 and Q2.5 data for visualization"""
        # Load A4 geometric concept space
        a4_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'A_Concept_pipeline', 'outputs', 'A4_geometric_concept_space.json'
        )

        try:
            with open(a4_path, 'r', encoding='utf-8') as f:
                self.a4_data = json.load(f)
                print("[SUCCESS] A4 geometric concept space loaded")
        except Exception as e:
            print(f"Error loading A4 data: {e}")
            return False

        # Load Q2.5 assignments
        q25_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'Q2.5_enhanced_convex_ball_assignment.json'
        )

        try:
            with open(q25_path, 'r', encoding='utf-8') as f:
                self.q25_data = json.load(f)
                print("[SUCCESS] Q2.5 assignment data loaded")
        except Exception as e:
            print(f"Error loading Q2.5 data: {e}")
            return False

        return True

    def extract_visualization_data(self, doc_id: str) -> Dict:
        """Extract and combine A4 + Q2.5 data for visualization"""
        if doc_id not in self.a4_data:
            print(f"Document {doc_id} not found in A4 data")
            return {}

        # Get A4 concept space
        a4_concept_space = self.a4_data[doc_id]['geometric_concept_space']
        convex_balls = a4_concept_space['convex_balls']
        concept_centroids = a4_concept_space['concept_centroids']

        # Get Q2.5 assignments
        q25_assignment = self.q25_data.get(doc_id, {})

        # Combine data
        visualization_data = {
            'doc_id': doc_id,
            'coordinate_system': a4_concept_space['coordinate_system'],
            'convex_balls': convex_balls,
            'concept_centroids': concept_centroids,
            'question_assignment': q25_assignment,
            'ball_summary': self.create_ball_summary(convex_balls, q25_assignment),
            'question_summary': self.create_question_summary(q25_assignment)
        }

        return visualization_data

    def create_ball_summary(self, convex_balls: Dict, q25_assignment: Dict) -> List[Dict]:
        """Create summary of convex balls with question assignments"""
        ball_summary = []

        for ball_id, ball_info in convex_balls.items():
            summary = {
                'ball_id': ball_id,
                'centroid': ball_info.get('centroid', []),
                'radius': ball_info.get('radius', 1.0),
                'chunk_count': len(ball_info.get('member_chunks', [])),
                'question_assigned': False,
                'assignment_confidence': 0.0,
                'assignment_dimension': None,
                'distance_to_question': None
            }

            # Check if question is assigned to this ball
            multi_dim = q25_assignment.get('multi_dimensional_analysis', {})
            for dim_type, dim_analysis in multi_dim.items():
                assignments = dim_analysis.get('convex_ball_assignments', [])
                for assignment in assignments:
                    if assignment.get('ball_id') == ball_id:
                        summary['question_assigned'] = True
                        summary['assignment_confidence'] = assignment.get('confidence', 0.0)
                        summary['assignment_dimension'] = dim_type
                        summary['distance_to_question'] = assignment.get('distance_to_centroid', 0.0)
                        break

            ball_summary.append(summary)

        return ball_summary

    def create_question_summary(self, q25_assignment: Dict) -> Dict:
        """Create question assignment summary"""
        return {
            'question_id': q25_assignment.get('question_id', 'unknown'),
            'question_text': q25_assignment.get('question_text', ''),
            'overall_confidence': q25_assignment.get('assignment_confidence', 0.0),
            'assigned_balls_count': self.count_assigned_balls(q25_assignment),
            'fusion_strategy': q25_assignment.get('fusion_analysis', {}).get('fusion_strategy', 'unknown'),
            'processing_status': q25_assignment.get('processing_metadata', {}).get('a_pipeline_integration_status', 'unknown')
        }

    def count_assigned_balls(self, q25_assignment: Dict) -> int:
        """Count total balls assigned across all dimensions"""
        total = 0
        multi_dim = q25_assignment.get('multi_dimensional_analysis', {})
        for dim_analysis in multi_dim.values():
            total += len(dim_analysis.get('convex_ball_assignments', []))
        return total

    def create_3d_convex_ball_visualization(self, viz_data: Dict) -> go.Figure:
        """Create 3D visualization of convex balls with question assignments"""
        # Perform PCA to reduce to 3D
        all_centroids = []
        ball_ids = []

        for ball_summary in viz_data['ball_summary']:
            centroid = ball_summary['centroid']
            if len(centroid) > 0:
                all_centroids.append(centroid)
                ball_ids.append(ball_summary['ball_id'])

        if not all_centroids:
            return go.Figure()

        # PCA reduction to 3D
        centroids_array = np.array(all_centroids)
        pca = PCA(n_components=3)
        centroids_3d = pca.fit_transform(centroids_array)

        # Create 3D scatter plot
        fig = go.Figure()

        # Plot convex balls
        for i, ball_summary in enumerate(viz_data['ball_summary']):
            if i >= len(centroids_3d):
                continue

            x, y, z = centroids_3d[i]

            # Color based on question assignment
            color = 'red' if ball_summary['question_assigned'] else 'lightblue'
            size = 15 if ball_summary['question_assigned'] else 8
            opacity = 0.9 if ball_summary['question_assigned'] else 0.6

            # Create hover text
            # Handle None values safely
            confidence_str = f"{ball_summary['assignment_confidence']:.3f}" if ball_summary['assignment_confidence'] else "0.000"
            dimension_str = ball_summary['assignment_dimension'] if ball_summary['assignment_dimension'] else "None"
            distance_str = f"{ball_summary['distance_to_question']:.3f}" if ball_summary['distance_to_question'] else "N/A"

            hover_text = f"""Ball: {ball_summary['ball_id']}<br>Chunks: {ball_summary['chunk_count']}<br>Question Assigned: {ball_summary['question_assigned']}<br>Confidence: {confidence_str}<br>Dimension: {dimension_str}<br>Distance: {distance_str}"""

            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    opacity=opacity,
                    line=dict(width=2, color='darkblue' if ball_summary['question_assigned'] else 'gray')
                ),
                text=[ball_summary['ball_id']],
                textposition='top center',
                hovertext=hover_text,
                hoverinfo='text',
                name='Assigned Ball' if ball_summary['question_assigned'] else 'Convex Ball'
            ))

        # Customize layout
        fig.update_layout(
            title=f"Q2.5 Question Assignment Visualization - {viz_data['doc_id']}",
            scene=dict(
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                zaxis_title='PCA Component 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800
        )

        return fig

    def create_dimensional_analysis_chart(self, viz_data: Dict) -> go.Figure:
        """Create dimensional membership analysis chart"""
        q25_assignment = viz_data['question_assignment']
        multi_dim = q25_assignment.get('multi_dimensional_analysis', {})

        dimensions = []
        balls_assigned = []
        avg_confidence = []
        containment_status = []

        for dim_type, dim_analysis in multi_dim.items():
            stats = dim_analysis.get('membership_statistics', {})
            dimensions.append(dim_type.replace('_dimensional_membership', ''))
            balls_assigned.append(stats.get('total_balls_assigned', 0))
            avg_confidence.append(stats.get('avg_confidence', 0))
            containment_status.append(dim_analysis.get('containment_status', 'none'))

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Balls Assigned by Dimension', 'Average Confidence by Dimension',
                          'Containment Status', 'Dimensional Strength'],
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'pie'}, {'type': 'bar'}]]
        )

        # Balls assigned
        fig.add_trace(
            go.Bar(x=dimensions, y=balls_assigned, name='Balls Assigned'),
            row=1, col=1
        )

        # Average confidence
        fig.add_trace(
            go.Bar(x=dimensions, y=avg_confidence, name='Avg Confidence'),
            row=1, col=2
        )

        # Containment status pie chart
        containment_counts = {status: containment_status.count(status) for status in set(containment_status)}
        fig.add_trace(
            go.Pie(labels=list(containment_counts.keys()), values=list(containment_counts.values()), name="Containment"),
            row=2, col=1
        )

        # Dimensional strength
        fusion_analysis = q25_assignment.get('fusion_analysis', {})
        dim_strengths = fusion_analysis.get('dimensional_strengths', {})
        if dim_strengths:
            strength_dims = list(dim_strengths.keys())
            strength_vals = list(dim_strengths.values())
            fig.add_trace(
                go.Bar(x=strength_dims, y=strength_vals, name='Dimensional Strength'),
                row=2, col=2
            )

        fig.update_layout(
            title=f"Multi-Dimensional Analysis - {viz_data['question_summary']['question_id']}",
            height=800,
            showlegend=False
        )

        return fig

    def create_assignment_confidence_chart(self, viz_data: Dict) -> go.Figure:
        """Create assignment confidence analysis chart"""
        q25_assignment = viz_data['question_assignment']
        multi_dim = q25_assignment.get('multi_dimensional_analysis', {})

        all_assignments = []
        for dim_type, dim_analysis in multi_dim.items():
            assignments = dim_analysis.get('convex_ball_assignments', [])
            for assignment in assignments:
                all_assignments.append({
                    'ball_id': assignment.get('ball_id', 'unknown'),
                    'dimension': dim_type.replace('_dimensional_membership', ''),
                    'confidence': assignment.get('confidence', 0),
                    'distance': assignment.get('distance_to_centroid', 0),
                    'membership_strength': assignment.get('membership_strength', 0),
                    'containment_type': assignment.get('containment_type', 'unknown')
                })

        if not all_assignments:
            return go.Figure().add_annotation(
                text="No assignments found", xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        df = pd.DataFrame(all_assignments)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Confidence by Ball', 'Distance vs Confidence',
                          'Assignment by Dimension', 'Containment Types'],
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'pie'}]]
        )

        # Confidence by ball
        fig.add_trace(
            go.Bar(x=df['ball_id'], y=df['confidence'], name='Confidence'),
            row=1, col=1
        )

        # Distance vs Confidence scatter
        fig.add_trace(
            go.Scatter(
                x=df['distance'], y=df['confidence'],
                mode='markers+text',
                text=df['ball_id'],
                textposition='top center',
                name='Distance vs Confidence'
            ),
            row=1, col=2
        )

        # Assignment by dimension
        dimension_counts = df['dimension'].value_counts()
        fig.add_trace(
            go.Bar(x=dimension_counts.index, y=dimension_counts.values, name='Assignments by Dimension'),
            row=2, col=1
        )

        # Containment types
        containment_counts = df['containment_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=containment_counts.index, values=containment_counts.values, name="Containment Types"),
            row=2, col=2
        )

        fig.update_layout(
            title=f"Assignment Confidence Analysis - {viz_data['question_summary']['question_id']}",
            height=800,
            showlegend=False
        )

        return fig

    def create_question_ball_mapping_table(self, viz_data: Dict) -> pd.DataFrame:
        """Create detailed mapping table"""
        q25_assignment = viz_data['question_assignment']
        multi_dim = q25_assignment.get('multi_dimensional_analysis', {})

        mapping_data = []
        for dim_type, dim_analysis in multi_dim.items():
            assignments = dim_analysis.get('convex_ball_assignments', [])
            for assignment in assignments:
                # Find corresponding ball info
                ball_id = assignment.get('ball_id', 'unknown')
                ball_info = next((b for b in viz_data['ball_summary'] if b['ball_id'] == ball_id), {})

                mapping_data.append({
                    'Ball_ID': ball_id,
                    'Dimension': dim_type.replace('_dimensional_membership', ''),
                    'Confidence': assignment.get('confidence', 0),
                    'Distance_to_Centroid': assignment.get('distance_to_centroid', 0),
                    'Membership_Strength': assignment.get('membership_strength', 0),
                    'Containment_Type': assignment.get('containment_type', 'unknown'),
                    'Fallback_Applied': assignment.get('fallback_applied', False),
                    'Ball_Chunk_Count': ball_info.get('chunk_count', 0),
                    'Ball_Radius': ball_info.get('radius', 0)
                })

        return pd.DataFrame(mapping_data)

    def generate_comprehensive_visualization(self, doc_id: str = None) -> bool:
        """Generate complete Q2.5 visualization suite"""
        print("Q2.5 ENHANCED VISUALIZATION GENERATOR")
        print("=" * 50)

        # Load data
        if not self.load_data():
            return False

        # Auto-detect doc_id if not provided
        if not doc_id:
            available_docs = list(self.q25_data.keys())
            if not available_docs:
                print("No Q2.5 assignments found!")
                return False
            doc_id = available_docs[0]
            print(f"Using document: {doc_id}")

        # Extract visualization data
        viz_data = self.extract_visualization_data(doc_id)
        if not viz_data:
            return False

        print(f"\nGenerating visualizations for {doc_id}...")
        print(f"Question: {viz_data['question_summary']['question_text'][:60]}...")

        # Create output directory
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs'
        )
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 1. 3D Convex Ball Visualization
            print("Creating 3D convex ball visualization...")
            fig_3d = self.create_3d_convex_ball_visualization(viz_data)
            html_3d = os.path.join(output_dir, f'Q25_convex_balls_3d_{doc_id}.html')
            fig_3d.write_html(html_3d)
            print(f"[SUCCESS] 3D visualization saved: {html_3d}")

            # 2. Dimensional Analysis Chart
            print("Creating dimensional analysis chart...")
            fig_dim = self.create_dimensional_analysis_chart(viz_data)
            html_dim = os.path.join(output_dir, f'Q25_dimensional_analysis_{doc_id}.html')
            fig_dim.write_html(html_dim)
            print(f"[SUCCESS] Dimensional analysis saved: {html_dim}")

            # 3. Assignment Confidence Chart
            print("Creating assignment confidence chart...")
            fig_conf = self.create_assignment_confidence_chart(viz_data)
            html_conf = os.path.join(output_dir, f'Q25_assignment_confidence_{doc_id}.html')
            fig_conf.write_html(html_conf)
            print(f"[SUCCESS] Confidence analysis saved: {html_conf}")

            # 4. Detailed Mapping Table
            print("Creating detailed mapping table...")
            mapping_df = self.create_question_ball_mapping_table(viz_data)
            csv_mapping = os.path.join(output_dir, f'Q25_question_ball_mapping_{doc_id}.csv')
            mapping_df.to_csv(csv_mapping, index=False)
            print(f"[SUCCESS] Mapping table saved: {csv_mapping}")

            # 5. Summary Report
            print("Creating summary report...")
            self.create_summary_report(viz_data, output_dir, doc_id)

            print(f"\n" + "=" * 50)
            print("Q2.5 VISUALIZATION COMPLETE!")
            print(f"All files saved to: {output_dir}")
            print("=" * 50)

            return True

        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return False

    def create_summary_report(self, viz_data: Dict, output_dir: str, doc_id: str):
        """Create text summary report"""
        report_path = os.path.join(output_dir, f'Q25_summary_report_{doc_id}.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Q2.5 QUESTION-TO-CONVEX-BALL ASSIGNMENT REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Question Summary
            q_summary = viz_data['question_summary']
            f.write(f"Document ID: {doc_id}\n")
            f.write(f"Question ID: {q_summary['question_id']}\n")
            f.write(f"Question Text: {q_summary['question_text']}\n")
            f.write(f"Overall Confidence: {q_summary['overall_confidence']:.3f}\n")
            f.write(f"Assigned Balls Count: {q_summary['assigned_balls_count']}\n")
            f.write(f"Fusion Strategy: {q_summary['fusion_strategy']}\n")
            f.write(f"Processing Status: {q_summary['processing_status']}\n\n")

            # Ball Assignments
            f.write("CONVEX BALL ASSIGNMENTS:\n")
            f.write("-" * 40 + "\n")

            assigned_balls = [b for b in viz_data['ball_summary'] if b['question_assigned']]
            for ball in assigned_balls:
                f.write(f"Ball: {ball['ball_id']}\n")
                f.write(f"  Confidence: {ball['assignment_confidence']:.3f}\n")
                f.write(f"  Dimension: {ball['assignment_dimension']}\n")
                f.write(f"  Distance: {ball['distance_to_question']:.3f}\n")
                f.write(f"  Chunks: {ball['chunk_count']}\n")
                f.write(f"  Radius: {ball['radius']:.3f}\n\n")

        print(f"[SUCCESS] Summary report saved: {report_path}")

def main():
    """Main execution"""
    visualizer = Q25VisualizationEngine()

    # Allow command line doc_id specification
    doc_id = None
    if len(sys.argv) > 1:
        doc_id = sys.argv[1]

    success = visualizer.generate_comprehensive_visualization(doc_id)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())