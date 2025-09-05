#!/usr/bin/env python3
"""
A2.42: Concept Centroid Analysis and Convex Ball Visualization
Analyzes A2.4 core concepts as centroids in convex balls for conceptual space representation
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import sys
import io

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class ConceptCentroidAnalyzer:
    """
    Analyzes concepts as centroids in a conceptual space where each concept
    represents a convex ball with:
    - Centroid: Mean position of all keywords in semantic space
    - Radius: Variance/spread of keywords around centroid
    - Density: Importance score and keyword frequency
    """
    
    def __init__(self, concepts_file="A_Concept_pipeline/outputs/A2.4_core_concepts.json"):
        self.concepts_file = Path(concepts_file)
        self.concepts = None
        
    def load_concepts(self):
        """Load core concepts from A2.4 output"""
        with open(self.concepts_file, 'r') as f:
            data = json.load(f)
        self.concepts = data.get("core_concepts", [])
        print(f"Loaded {len(self.concepts)} core concepts")
        return self.concepts
    
    def extract_concept_features(self):
        """
        Extract features for each concept to define its convex ball:
        1. Keyword set (defines the ball's content)
        2. Importance score (defines density/weight)
        3. Document coverage (defines influence radius)
        4. Keyword diversity (defines ball variance)
        """
        concept_features = []
        
        for concept in self.concepts:
            # Collect all unique keywords for this concept
            all_keywords = set()
            for instance in concept.get("document_instances", []):
                all_keywords.update(instance.get("keywords", []))
            
            features = {
                "concept_id": concept["concept_id"],
                "canonical_name": concept["canonical_name"],
                "keywords": list(all_keywords),
                "keyword_count": len(all_keywords),
                "importance": concept["importance_score"],
                "doc_coverage": concept["coverage_ratio"],
                "doc_count": concept["document_count"],
                "centroid_text": " ".join(all_keywords),  # For vectorization
                # Ball properties
                "radius_factor": min(len(all_keywords) / 10.0, 2.0),  # Normalized radius
                "density": concept["importance_score"],  # Ball density
                "influence": concept["coverage_ratio"]  # Ball influence
            }
            concept_features.append(features)
        
        return pd.DataFrame(concept_features)
    
    def compute_semantic_vectors(self, concept_df):
        """
        Compute semantic vectors for concepts using TF-IDF on their keywords
        This creates the coordinate system for our convex balls
        """
        # Create TF-IDF vectors from concept keywords
        vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        concept_vectors = vectorizer.fit_transform(concept_df["centroid_text"])
        
        # Store feature names for interpretation
        feature_names = vectorizer.get_feature_names_out()
        
        return concept_vectors, feature_names
    
    def compute_concept_distances(self, concept_vectors):
        """
        Compute pairwise distances between concept centroids
        This defines the spatial relationships in our conceptual space
        """
        # Cosine similarity (1 - similarity = distance)
        similarities = cosine_similarity(concept_vectors)
        distances = 1 - similarities
        
        return distances, similarities
    
    def calculate_convex_hull_properties(self, concept_df, concept_vectors):
        """
        Calculate properties of each concept's convex ball:
        - Center (centroid position)
        - Radius (keyword spread)
        - Volume (importance * keyword_count)
        - Overlap with other balls
        """
        # Reduce dimensions for analysis
        n_components = min(3, concept_vectors.shape[0] - 1)
        if n_components < 3:
            n_components = 2
            
        pca = PCA(n_components=n_components)
        reduced_vectors = pca.fit_transform(concept_vectors.toarray())
        
        ball_properties = []
        
        for idx, row in concept_df.iterrows():
            properties = {
                "concept_id": row["concept_id"],
                "name": row["canonical_name"],
                "center": reduced_vectors[idx],
                "radius": row["radius_factor"] * 0.5,  # Scale for visualization
                "volume": row["importance"] * row["keyword_count"],
                "density": row["density"],
                "keyword_count": row["keyword_count"],
                "doc_coverage": row["doc_coverage"]
            }
            ball_properties.append(properties)
        
        return ball_properties, reduced_vectors, pca.explained_variance_ratio_
    
    def analyze_ball_overlaps(self, ball_properties, threshold=0.5):
        """
        Analyze overlapping convex balls to identify concept relationships
        """
        overlaps = []
        
        for i, ball1 in enumerate(ball_properties):
            for j, ball2 in enumerate(ball_properties[i+1:], i+1):
                # Calculate distance between centers
                center_dist = np.linalg.norm(ball1["center"] - ball2["center"])
                
                # Check if balls overlap (distance < sum of radii)
                if center_dist < (ball1["radius"] + ball2["radius"]):
                    overlap_strength = 1 - (center_dist / (ball1["radius"] + ball2["radius"]))
                    overlaps.append({
                        "concept1": ball1["concept_id"],
                        "concept1_name": ball1["name"],
                        "concept2": ball2["concept_id"],
                        "concept2_name": ball2["name"],
                        "distance": center_dist,
                        "overlap_strength": overlap_strength,
                        "combined_importance": (ball1["density"] + ball2["density"]) / 2
                    })
        
        return pd.DataFrame(overlaps)
    
    def identify_concept_clusters(self, distances, concept_df, threshold=0.5):
        """
        Identify clusters of related concepts based on centroid proximity
        """
        n_concepts = len(distances)
        clusters = []
        visited = set()
        
        for i in range(n_concepts):
            if i in visited:
                continue
                
            cluster = [i]
            visited.add(i)
            
            for j in range(n_concepts):
                if i != j and distances[i][j] < threshold:
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) > 1:
                clusters.append({
                    "indices": cluster,
                    "concepts": [concept_df.iloc[idx]["canonical_name"] for idx in cluster],
                    "concept_ids": [concept_df.iloc[idx]["concept_id"] for idx in cluster],
                    "avg_importance": np.mean([concept_df.iloc[idx]["importance"] for idx in cluster])
                })
        
        return clusters

def analyze_conceptual_space():
    """
    Main analysis function with recommendations
    """
    analyzer = ConceptCentroidAnalyzer()
    
    print("\n" + "="*80)
    print("A2.42 CONCEPT CENTROID ANALYSIS - CONVEX BALL REPRESENTATION")
    print("="*80)
    
    # Load and process concepts
    analyzer.load_concepts()
    concept_df = analyzer.extract_concept_features()
    
    # Compute semantic vectors
    concept_vectors, features = analyzer.compute_semantic_vectors(concept_df)
    
    # Compute distances
    distances, similarities = analyzer.compute_concept_distances(concept_vectors)
    
    # Get convex ball properties
    ball_properties, reduced_vectors, variance_explained = analyzer.calculate_convex_hull_properties(
        concept_df, concept_vectors
    )
    
    # Analyze overlaps
    overlaps = analyzer.analyze_ball_overlaps(ball_properties)
    
    # Identify clusters
    clusters = analyzer.identify_concept_clusters(distances, concept_df, threshold=0.6)
    
    print("\n" + "-"*80)
    print("CONCEPTUAL SPACE METRICS")
    print("-"*80)
    print(f"Total Concepts: {len(analyzer.concepts)}")
    print(f"Semantic Dimensions: {concept_vectors.shape[1]}")
    print(f"Reduced Dimensions: {len(variance_explained)}")
    print(f"Variance Explained: {variance_explained.sum():.2%}")
    print(f"Average Ball Radius: {np.mean([b['radius'] for b in ball_properties]):.3f}")
    print(f"Average Ball Volume: {np.mean([b['volume'] for b in ball_properties]):.3f}")
    
    print("\n" + "-"*80)
    print("CONVEX BALL PROPERTIES (Top 5 by Importance)")
    print("-"*80)
    
    # Sort balls by density (importance)
    sorted_balls = sorted(ball_properties, key=lambda x: x['density'], reverse=True)[:5]
    
    for i, ball in enumerate(sorted_balls, 1):
        print(f"\n{i}. {ball['concept_id']}: {ball['name']}")
        print(f"   - Centroid Position: {ball['center'][:3].round(3)}")
        print(f"   - Radius: {ball['radius']:.3f}")
        print(f"   - Volume: {ball['volume']:.3f}")
        print(f"   - Density (Importance): {ball['density']:.3f}")
        print(f"   - Keywords: {ball['keyword_count']}")
        print(f"   - Document Coverage: {ball['doc_coverage']:.2%}")
    
    if not overlaps.empty:
        print("\n" + "-"*80)
        print("OVERLAPPING CONCEPT BALLS (Semantic Relationships)")
        print("-"*80)
        
        top_overlaps = overlaps.nlargest(5, 'overlap_strength')
        for _, row in top_overlaps.iterrows():
            print(f"\n• {row['concept1']} ({row['concept1_name']}) ↔ {row['concept2']} ({row['concept2_name']})")
            print(f"  - Distance: {row['distance']:.3f}")
            print(f"  - Overlap Strength: {row['overlap_strength']:.2%}")
            print(f"  - Combined Importance: {row['combined_importance']:.3f}")
    
    if clusters:
        print("\n" + "-"*80)
        print("CONCEPT CLUSTERS (Semantic Neighborhoods)")
        print("-"*80)
        
        for i, cluster in enumerate(clusters, 1):
            print(f"\nCluster {i} (Avg Importance: {cluster['avg_importance']:.3f}):")
            for concept_id, concept_name in zip(cluster['concept_ids'], cluster['concepts']):
                print(f"  - {concept_id}: {concept_name}")
    
    print("\n" + "="*80)
    print("RECOMMENDED APPROACH FOR CONCEPT VISUALIZATION")
    print("="*80)
    
    print("""
BEST APPROACH: Hierarchical Convex Ball Representation

1. PRIMARY VISUALIZATION: Force-Directed Graph
   - Concepts as nodes (sized by importance)
   - Edges weighted by semantic similarity
   - Interactive 3D visualization using plotly/d3.js
   - Color coding by business domain

2. MATHEMATICAL REPRESENTATION:
   - Each concept C_i is a convex ball B(c_i, r_i)
   - Center c_i: TF-IDF centroid of keywords
   - Radius r_i: sqrt(keyword_variance) * importance
   - Volume V_i: (4/3)πr_i³ * importance_score
   
3. DISTANCE METRICS:
   - Semantic Distance: 1 - cosine_similarity(c_i, c_j)
   - Ball Overlap: max(0, r_i + r_j - ||c_i - c_j||)
   - Influence Zone: r_influence = r_i * doc_coverage

4. IMPLEMENTATION STRATEGY:
   a) Use PCA/t-SNE for dimensionality reduction
   b) Apply force-directed layout for optimal spacing
   c) Render as interactive 3D scatter with spheres
   d) Add hover tooltips with concept details
   e) Enable filtering by document/domain

5. ANALYSIS FEATURES:
   - Concept density heatmap
   - Overlap detection for related concepts
   - Path finding between concepts
   - Hierarchical clustering visualization
   - Document-concept bipartite graph

6. RECOMMENDED LIBRARIES:
   - Plotly: Interactive 3D visualization
   - NetworkX: Graph analysis
   - scikit-learn: Clustering and dimensionality reduction
   - D3.js: Web-based interactive visualization
    """)
    
    return {
        "concept_df": concept_df,
        "ball_properties": ball_properties,
        "overlaps": overlaps,
        "clusters": clusters,
        "distances": distances,
        "similarities": similarities
    }

if __name__ == "__main__":
    results = analyze_conceptual_space()
    print("\n✓ Analysis complete!")