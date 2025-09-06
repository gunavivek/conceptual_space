#!/usr/bin/env python3
"""
A32: Convex Ball 3D Visualization
Interactive 3D visualization of concept centroids, convex balls, and chunk memberships from A3 results
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import math


def load_a3_results():
    """Load A3 chunking results"""
    with open('A_Concept_pipeline/outputs/A3_concept_based_chunks.json', 'r') as f:
        return json.load(f)


def extract_visualization_data(a3_data):
    """Extract data needed for 3D visualization"""
    
    # Extract concept centroids
    centroids = {}
    for concept_id, centroid_info in a3_data['concept_centroids'].items():
        centroids[concept_id] = centroid_info
    
    # Extract chunk information
    chunks = []
    for doc_id, doc_chunks in a3_data['document_chunks'].items():
        for chunk in doc_chunks:
            # Create simplified chunk representation
            chunk_info = {
                'chunk_id': chunk['chunk_id'],
                'doc_id': doc_id,
                'primary_centroid': chunk['primary_centroid'],
                'chunk_type': chunk['chunk_type'],
                'memberships': chunk['concept_memberships'],
                'distances': chunk['centroid_distances'],
                'convex_memberships': chunk['convex_ball_memberships'],
                'word_count': chunk['word_count'],
                'text_preview': chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
            }
            chunks.append(chunk_info)
    
    return centroids, chunks


def create_3d_positions(centroids, chunks):
    """Create 3D positions for centroids and chunks using dimensionality reduction"""
    
    # For demonstration, create positions based on concept relationships
    # In a real implementation, this would use actual embedding vectors
    
    concept_positions = {}
    chunk_positions = {}
    
    # Predefined layout for better visualization
    centroid_layouts = {
        'core_1': np.array([0, 0, 0]),         # Deferred income - center
        'core_10': np.array([2, 1, 0]),       # Contract balances 
        'core_11': np.array([1.5, -1, 0]),    # Revenue unearned
        'core_12': np.array([2.5, 0, 1]),     # Receivable balance
        'core_26': np.array([-2, 1, 0]),      # Inventory valuation
        'core_27': np.array([-1.5, -1, 0]),   # Operations the
        'core_43': np.array([0, 2, 1]),       # Operations discontinued
        'core_44': np.array([0, 2.5, 0.5]),   # Discontinued operation
        'core_63': np.array([0, 0, -2]),      # TWDV tax
        'core_64': np.array([0.5, 0, -1.5]),  # NBV net
    }
    
    # Assign concept positions
    for concept_id in centroids.keys():
        if concept_id in centroid_layouts:
            concept_positions[concept_id] = centroid_layouts[concept_id]
        else:
            # Fallback random position
            concept_positions[concept_id] = np.random.normal(0, 1, 3)
    
    # Position chunks near their primary centroids with some offset
    for i, chunk in enumerate(chunks):
        primary_centroid = chunk['primary_centroid']
        base_pos = concept_positions[primary_centroid]
        
        # Add small random offset for chunks
        offset = np.random.normal(0, 0.3, 3)
        
        # If chunk has convex membership, move it closer to centroid
        if any(chunk['convex_memberships'].values()):
            offset *= 0.5  # Closer to centroid
        else:
            offset *= 1.2  # Further from centroid
            
        chunk_positions[chunk['chunk_id']] = base_pos + offset
    
    return concept_positions, chunk_positions


def create_3d_visualization(centroids, chunks, concept_positions, chunk_positions):
    """Create interactive 3D visualization"""
    
    fig = go.Figure()
    
    # Color schemes
    concept_colors = {
        'core_1': 'blue',
        'core_10': 'red', 'core_11': 'red', 'core_12': 'red',  # Financial concepts
        'core_26': 'green', 'core_27': 'green',                # Operational concepts  
        'core_43': 'orange', 'core_44': 'orange',              # Discontinued ops
        'core_63': 'purple', 'core_64': 'purple'               # Tax concepts
    }
    
    chunk_colors = {
        'single_concept': 'rgba(0, 150, 0, 0.8)',
        'multi_concept': 'rgba(150, 0, 150, 0.8)', 
        'overlap_zone': 'rgba(150, 150, 0, 0.8)',
        'weak_association': 'rgba(100, 100, 100, 0.6)'
    }
    
    # 1. Add concept centroids as spheres
    for concept_id, centroid_info in centroids.items():
        pos = concept_positions[concept_id]
        color = concept_colors.get(concept_id, 'gray')
        
        # Centroid point
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers+text',
            marker=dict(size=15, color=color, symbol='diamond'),
            text=[f"{centroid_info['canonical_name']}<br>({concept_id})"],
            textposition="top center",
            name=f"Centroid: {centroid_info['canonical_name']}",
            hovertemplate=f"<b>{centroid_info['canonical_name']}</b><br>" +
                         f"Concept ID: {concept_id}<br>" +
                         f"Domain: {centroid_info['domain']}<br>" +
                         f"Core Terms: {centroid_info['core_terms_count']}<br>" +
                         f"Expanded Terms: {centroid_info['expanded_terms_count']}<br>" +
                         f"Radius: {centroid_info['radius']:.3f}<br>" +
                         "<extra></extra>"
        ))
        
        # 2. Add convex ball (transparent sphere)
        radius = centroid_info['radius'] * 1.2  # Match the multiplier from A3
        
        # Create sphere mesh
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + pos[1] 
        z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.2,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f"Convex Ball: {centroid_info['canonical_name']}",
            hovertemplate=f"<b>Convex Ball</b><br>" +
                         f"Concept: {centroid_info['canonical_name']}<br>" +
                         f"Radius: {radius:.3f}<br>" +
                         "<extra></extra>"
        ))
    
    # 3. Add chunks as points
    for chunk in chunks:
        pos = chunk_positions[chunk['chunk_id']]
        color = chunk_colors.get(chunk['chunk_type'], 'gray')
        
        # Determine if chunk is inside any convex ball
        in_convex = any(chunk['convex_memberships'].values())
        symbol = 'circle' if in_convex else 'x'
        size = 12 if in_convex else 8
        
        # Get membership info for hover
        membership_text = ""
        for concept_id, score in chunk['memberships'].items():
            if score > 0.01:  # Only show significant memberships
                concept_name = centroids[concept_id]['canonical_name']
                membership_text += f"{concept_name}: {score:.3f}<br>"
        
        convex_text = ""
        for concept_id, in_ball in chunk['convex_memberships'].items():
            if in_ball:
                concept_name = centroids[concept_id]['canonical_name']
                convex_text += f"• {concept_name}<br>"
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers+text',
            marker=dict(size=size, color=color, symbol=symbol, line=dict(width=2, color='black')),
            text=[chunk['chunk_id'].split('_')[-1]],  # Show just the chunk number
            textposition="bottom center",
            name=f"Chunk: {chunk['chunk_id']}",
            hovertemplate=f"<b>{chunk['chunk_id']}</b><br>" +
                         f"Document: {chunk['doc_id']}<br>" +
                         f"Type: {chunk['chunk_type']}<br>" +
                         f"Primary Centroid: {chunk['primary_centroid']}<br>" +
                         f"Words: {chunk['word_count']}<br>" +
                         f"<br><b>Memberships:</b><br>{membership_text}" +
                         f"<br><b>In Convex Balls:</b><br>{convex_text or 'None'}" +
                         f"<br><b>Text Preview:</b><br>{chunk['text_preview']}" +
                         "<extra></extra>"
        ))
    
    # 4. Add connecting lines from chunks to their primary centroids
    for chunk in chunks:
        chunk_pos = chunk_positions[chunk['chunk_id']]
        centroid_pos = concept_positions[chunk['primary_centroid']]
        
        # Only show lines for chunks inside convex balls
        if any(chunk['convex_memberships'].values()):
            fig.add_trace(go.Scatter3d(
                x=[chunk_pos[0], centroid_pos[0]],
                y=[chunk_pos[1], centroid_pos[1]], 
                z=[chunk_pos[2], centroid_pos[2]],
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.3)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="A3 Concept-Based Chunking: 3D Convex Ball Visualization",
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title="Semantic Dimension 1",
            yaxis_title="Semantic Dimension 2", 
            zaxis_title="Semantic Dimension 3",
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        width=1200,
        height=800
    )
    
    return fig


def create_analysis_dashboard(centroids, chunks):
    """Create analysis dashboard with multiple views"""
    
    # Prepare data for analysis plots
    chunk_df = []
    for chunk in chunks:
        # Count convex memberships
        convex_count = sum(chunk['convex_memberships'].values())
        membership_count = sum(1 for score in chunk['memberships'].values() if score > 0.05)
        
        chunk_df.append({
            'chunk_id': chunk['chunk_id'],
            'doc_id': chunk['doc_id'],
            'chunk_type': chunk['chunk_type'],
            'convex_count': convex_count,
            'membership_count': membership_count,
            'max_membership': max(chunk['memberships'].values()),
            'min_distance': min(chunk['distances'].values()),
            'primary_centroid': centroids[chunk['primary_centroid']]['canonical_name'],
            'word_count': chunk['word_count']
        })
    
    df = pd.DataFrame(chunk_df)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Chunk Types Distribution", 
            "Convex Ball Memberships",
            "Membership Scores vs Distance",
            "Document-Concept Relationships"
        ],
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Chunk Types Pie Chart
    chunk_type_counts = df['chunk_type'].value_counts()
    fig.add_trace(
        go.Pie(labels=chunk_type_counts.index, values=chunk_type_counts.values, name="Chunk Types"),
        row=1, col=1
    )
    
    # 2. Convex Ball Memberships Bar Chart
    convex_counts = df['convex_count'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=convex_counts.index, y=convex_counts.values, name="Convex Memberships"),
        row=1, col=2
    )
    
    # 3. Membership Score vs Distance Scatter
    fig.add_trace(
        go.Scatter(
            x=df['min_distance'], 
            y=df['max_membership'],
            mode='markers',
            marker=dict(size=8, color=df['word_count'], colorscale='viridis', showscale=True),
            text=df['chunk_id'],
            name="Score vs Distance"
        ),
        row=2, col=1
    )
    
    # 4. Document-Concept Distribution
    doc_concept_counts = df.groupby(['doc_id', 'primary_centroid']).size().reset_index(name='count')
    fig.add_trace(
        go.Bar(
            x=doc_concept_counts['doc_id'], 
            y=doc_concept_counts['count'],
            text=doc_concept_counts['primary_centroid'],
            name="Doc-Concept Distribution"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="A3 Chunking Analysis Dashboard",
        showlegend=False,
        height=800
    )
    
    return fig


def main():
    """Main execution function"""
    print("="*60)
    print("A32: CONVEX BALL 3D VISUALIZATION")
    print("="*60)
    
    # Load data
    print("Loading A3 chunking results...")
    a3_data = load_a3_results()
    
    # Extract visualization data  
    print("Extracting visualization data...")
    centroids, chunks = extract_visualization_data(a3_data)
    
    print(f"Found {len(centroids)} concept centroids")
    print(f"Found {len(chunks)} chunks")
    
    # Create 3D positions
    print("Creating 3D spatial layout...")
    concept_positions, chunk_positions = create_3d_positions(centroids, chunks)
    
    # Create main 3D visualization
    print("Generating 3D convex ball visualization...")
    main_fig = create_3d_visualization(centroids, chunks, concept_positions, chunk_positions)
    
    # Create analysis dashboard
    print("Creating analysis dashboard...")
    dashboard_fig = create_analysis_dashboard(centroids, chunks)
    
    # Save visualizations
    output_dir = Path("A_Concept_pipeline/outputs")
    
    main_output = output_dir / "A3_convex_ball_3d_visualization.html"
    main_fig.write_html(main_output)
    
    dashboard_output = output_dir / "A3_chunking_analysis_dashboard.html"
    dashboard_fig.write_html(dashboard_output)
    
    print(f"\n[SUCCESS] Visualizations saved:")
    print(f"  3D Convex Ball Plot: {main_output}")
    print(f"  Analysis Dashboard: {dashboard_output}")
    
    # Show summary statistics
    print(f"\n[SUMMARY] VISUALIZATION SUMMARY:")
    print(f"  - Concept Centroids: {len(centroids)} (with transparent convex balls)")
    print(f"  - Document Chunks: {len(chunks)} (color-coded by type)")
    print(f"  - Chunks in convex balls: {sum(1 for c in chunks if any(c['convex_memberships'].values()))}")
    print(f"  - Chunks outside all balls: {sum(1 for c in chunks if not any(c['convex_memberships'].values()))}")
    
    chunk_types = {}
    for chunk in chunks:
        chunk_type = chunk['chunk_type']
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    print(f"  - Chunk type breakdown:")
    for chunk_type, count in chunk_types.items():
        print(f"    * {chunk_type}: {count}")
    
    print(f"\n[FEATURES] VISUALIZATION FEATURES:")
    print(f"  - Interactive 3D plot with hover details")
    print(f"  - Concept centroids as diamonds with labels")  
    print(f"  - Transparent spheres showing convex ball boundaries")
    print(f"  - Chunks as circles (in balls) or X's (outside)")
    print(f"  - Connecting lines for chunks inside convex balls")
    print(f"  - Comprehensive hover information")
    
    return main_fig, dashboard_fig


if __name__ == "__main__":
    main_fig, dashboard_fig = main()