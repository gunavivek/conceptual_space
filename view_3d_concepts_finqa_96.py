#!/usr/bin/env python3
"""
3D Concept Visualization for Document finqa_test_96 Only
Shows concept names instead of IDs
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import webbrowser

def create_3d_concept_visualization_finqa96():
    """Create 3D visualization for concepts from finqa_test_96 only"""
    
    # Load concepts
    with open("A_Concept_pipeline/outputs/A2.4_core_concepts.json", 'r') as f:
        data = json.load(f)
    
    all_concepts = data.get("core_concepts", [])
    
    # Filter for finqa_test_96 only
    concepts = [c for c in all_concepts if 'finqa_test_96' in c.get('related_documents', [])]
    
    if not concepts:
        print("No concepts found for document finqa_test_96!")
        return None
    
    print(f"Found {len(concepts)} concepts for finqa_test_96:")
    for c in concepts:
        print(f"  - {c['concept_id']}: {c['canonical_name']}")
    
    # Process concepts
    concept_data = []
    for concept in concepts:
        # Get keywords specifically from finqa_test_96 document instances
        keywords_96 = []
        for instance in concept.get("document_instances", []):
            if instance.get("doc_id") == "finqa_test_96":
                keywords_96.extend(instance.get("keywords", []))
        
        # If no specific instance found, use all keywords
        if not keywords_96:
            all_keywords = set()
            for instance in concept.get("document_instances", []):
                all_keywords.update(instance.get("keywords", []))
            keywords_96 = list(all_keywords)
        
        concept_data.append({
            "id": concept["concept_id"],
            "name": concept["canonical_name"],
            "display_name": concept["canonical_name"].title(),  # Capitalized for display
            "importance": concept["importance_score"],
            "text": " ".join(keywords_96),
            "keywords": keywords_96,
            "keyword_count": len(keywords_96),
            "docs": concept["document_count"],
            "coverage": concept["coverage_ratio"]
        })
    
    df = pd.DataFrame(concept_data)
    
    # Create embeddings
    if len(concepts) >= 2:
        vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(df["text"])
        
        # Use PCA for better positioning with small number of concepts
        if len(concepts) == 3:
            # Perfect for 3D - use 3 components
            pca = PCA(n_components=3)
            coords = pca.fit_transform(tfidf.toarray())
            
            # Scale coordinates for better visualization
            coords = coords * 2  # Make distances more visible
            
        else:
            # Use t-SNE for other cases
            tsne = TSNE(n_components=min(3, len(concepts)), 
                       perplexity=min(2, len(concepts)-1), 
                       random_state=42, n_iter=500)
            coords = tsne.fit_transform(tfidf.toarray())
            
            # Pad to 3D if needed
            if coords.shape[1] < 3:
                padding = np.zeros((coords.shape[0], 3 - coords.shape[1]))
                coords = np.hstack([coords, padding])
    
    else:
        # Single concept - place at origin
        coords = np.array([[0, 0, 0]])
    
    # Create 3D plot
    fig = go.Figure()
    
    # Define colors for each concept
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Add concept spheres with names
    for i, row in df.iterrows():
        # Sphere size based on importance and keyword count
        base_size = 30
        size = base_size + row["importance"] * 40 + row["keyword_count"] * 2
        
        # Color
        color = colors[i % len(colors)]
        
        # Enhanced hover text
        top_keywords = row["keywords"][:8]  # Show more keywords
        hover = (
            f"<b>{row['display_name']}</b><br>"
            f"Concept ID: {row['id']}<br>"
            f"Importance Score: {row['importance']:.3f}<br>"
            f"Keywords ({row['keyword_count']}): {', '.join(top_keywords)}<br>"
            f"Document Coverage: {row['coverage']:.1%}<br>"
            f"Appears in {row['docs']} document(s)"
        )
        
        # Create semi-transparent sphere around each concept
        if len(concepts) > 1:  # Only create sphere if multiple concepts
            radius = (row["importance"] * 0.5 + row["keyword_count"] * 0.02)
            
            # Generate sphere coordinates
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 10)
            x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + coords[i, 0]
            y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + coords[i, 1]
            z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + coords[i, 2]
            
            # Add transparent sphere (convex ball)
            fig.add_trace(go.Surface(
                x=x_sphere,
                y=y_sphere,
                z=z_sphere,
                showscale=False,
                opacity=0.2,
                surfacecolor=np.full(x_sphere.shape, row["importance"]),
                colorscale=[[0, color], [1, color]],
                showlegend=False,
                hoverinfo='skip',
                name=f"{row['display_name']}_sphere"
            ))
        
        # Add central point with concept name
        fig.add_trace(go.Scatter3d(
            x=[coords[i, 0]],
            y=[coords[i, 1]],
            z=[coords[i, 2]],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                opacity=0.9,
                line=dict(color='white', width=3),
                symbol='circle'
            ),
            text=row["display_name"],  # Show concept name instead of ID
            textposition="top center",
            textfont=dict(size=12, color='black', family='Arial Black'),
            hovertext=hover,
            hoverinfo='text',
            showlegend=True,
            name=row["display_name"]
        ))
    
    # Update layout for finqa_test_96 specific visualization
    fig.update_layout(
        title={
            'text': f"Conceptual Space: Document finqa_test_96<br><sub>Financial Concepts Visualization</sub>",
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis=dict(
                title="Financial Dimension 1",
                showgrid=True,
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(248, 249, 250, 0.9)',
                zeroline=True,
                zerolinecolor='gray'
            ),
            yaxis=dict(
                title="Financial Dimension 2",
                showgrid=True,
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(248, 249, 250, 0.9)',
                zeroline=True,
                zerolinecolor='gray'
            ),
            zaxis=dict(
                title="Financial Dimension 3",
                showgrid=True,
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(248, 249, 250, 0.9)',
                zeroline=True,
                zerolinecolor='gray'
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.8),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube'  # Equal aspect ratio
        ),
        height=800,
        width=1000,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor="black"
        ),
        margin=dict(l=0, r=0, t=80, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Add concept summary annotation
    concept_summary = f"Document finqa_test_96 contains {len(concepts)} financial concepts:<br>"
    for i, row in df.iterrows():
        concept_summary += f"• <b>{row['display_name']}</b> (importance: {row['importance']:.2f})<br>"
    
    fig.add_annotation(
        text=concept_summary,
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        showarrow=False,
        font=dict(size=10, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        align="left"
    )
    
    # Add interaction instructions
    fig.add_annotation(
        text="<b>Interactions:</b> Drag to rotate • Scroll to zoom • Hover for details",
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    return fig

if __name__ == "__main__":
    print("Creating 3D visualization for finqa_test_96...")
    fig = create_3d_concept_visualization_finqa96()
    
    if fig:
        # Save and open
        output_file = "finqa_test_96_concepts_3d.html"
        fig.write_html(output_file)
        print(f"Saved to: {output_file}")
        
        # Display summary
        print("\nDocument finqa_test_96 Concept Summary:")
        print("- contract balances (core_10)")
        print("- revenue unearned (core_11)") 
        print("- receivable balance (core_12)")
        print("\nAll concepts are financial/accounting related.")
        
        # Automatically open in browser
        webbrowser.open(f"file:///{Path(output_file).absolute()}")
        print("Opening in browser...")
    else:
        print("Failed to create visualization.")