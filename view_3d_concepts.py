#!/usr/bin/env python3
"""
Quick 3D Concept Visualization - Opens directly in browser
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import webbrowser

def create_3d_concept_visualization():
    """Create and display 3D conceptual space"""
    
    # Load concepts
    with open("A_Concept_pipeline/outputs/A2.4_core_concepts.json", 'r') as f:
        data = json.load(f)
    
    concepts = data.get("core_concepts", [])
    
    # Process concepts
    concept_data = []
    for concept in concepts:
        all_keywords = set()
        for instance in concept.get("document_instances", []):
            all_keywords.update(instance.get("keywords", []))
        
        concept_data.append({
            "id": concept["concept_id"],
            "name": concept["canonical_name"],
            "importance": concept["importance_score"],
            "text": " ".join(all_keywords),
            "keywords": len(all_keywords),
            "docs": concept["document_count"]
        })
    
    df = pd.DataFrame(concept_data)
    
    # Create embeddings
    vectorizer = TfidfVectorizer(max_features=50)
    tfidf = vectorizer.fit_transform(df["text"])
    
    # Get 3D coordinates
    tsne = TSNE(n_components=3, perplexity=min(5, len(concepts)-1), random_state=42)
    coords = tsne.fit_transform(tfidf.toarray())
    
    # Normalize coordinates
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-10)
    
    # Create 3D plot
    fig = go.Figure()
    
    # Add concept spheres
    for i, row in df.iterrows():
        # Sphere size based on importance
        size = 20 + row["importance"] * 50
        
        # Color based on importance (gradient)
        color_val = row["importance"]
        
        # Hover text
        hover = (
            f"<b>{row['id']}: {row['name']}</b><br>"
            f"Importance: {row['importance']:.3f}<br>"
            f"Keywords: {row['keywords']}<br>"
            f"Documents: {row['docs']}"
        )
        
        # Add point
        fig.add_trace(go.Scatter3d(
            x=[coords[i, 0]],
            y=[coords[i, 1]],
            z=[coords[i, 2]],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color_val,
                colorscale='Viridis',
                cmin=0.3,
                cmax=0.7,
                opacity=0.8,
                line=dict(color='white', width=2),
                showscale=i==0  # Show colorbar only once
            ),
            text=row["id"],
            textposition="top center",
            textfont=dict(size=10),
            hovertext=hover,
            hoverinfo='text',
            showlegend=False
        ))
    
    # Update layout for better visualization
    fig.update_layout(
        title={
            'text': "3D Conceptual Space - Interactive Visualization",
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis=dict(
                title="Dimension 1",
                showgrid=True,
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(240, 240, 240, 0.9)'
            ),
            yaxis=dict(
                title="Dimension 2",
                showgrid=True,
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(240, 240, 240, 0.9)'
            ),
            zaxis=dict(
                title="Dimension 3",
                showgrid=True,
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(240, 240, 240, 0.9)'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            )
        ),
        height=800,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Add instructions
    fig.add_annotation(
        text="Drag to rotate | Scroll to zoom | Hover for details",
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    return fig

if __name__ == "__main__":
    print("Creating 3D visualization...")
    fig = create_3d_concept_visualization()
    
    # Save and open
    output_file = "concept_space_3d.html"
    fig.write_html(output_file)
    print(f"Saved to: {output_file}")
    
    # Automatically open in browser
    webbrowser.open(f"file:///{Path(output_file).absolute()}")
    print("Opening in browser...")