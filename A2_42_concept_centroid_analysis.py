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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

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
        self.concept_vectors = None
        self.keyword_vectors = None
        
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
                "radius_factor": len(all_keywords) / 10.0,  # Normalized radius
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
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
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
    
    def identify_concept_clusters(self, distances, threshold=0.5):
        """
        Identify clusters of related concepts based on centroid proximity
        Concepts within threshold distance form overlapping convex balls
        """
        n_concepts = len(distances)
        clusters = []
        visited = set()
        
        for i in range(n_concepts):
            if i in visited:
                continue
                
            cluster = [i]
            visited.add(i)
            
            for j in range(i + 1, n_concepts):
                if distances[i][j] < threshold and j not in visited:
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def calculate_convex_hull_properties(self, concept_df, concept_vectors):
        """
        Calculate properties of each concept's convex ball:
        - Center (centroid position)
        - Radius (keyword spread)
        - Volume (importance * keyword_count)
        - Overlap with other balls
        """
        # Reduce dimensions for visualization
        if concept_vectors.shape[0] > 3:
            pca = PCA(n_components=3)
            reduced_vectors = pca.fit_transform(concept_vectors.toarray())
        else:
            reduced_vectors = concept_vectors.toarray()
        
        ball_properties = []
        
        for idx, row in concept_df.iterrows():
            properties = {
                "concept_id": row["concept_id"],
                "name": row["canonical_name"],
                "center": reduced_vectors[idx],
                "radius": row["radius_factor"] * 0.5,  # Scale for visualization
                "volume": row["importance"] * row["keyword_count"],
                "density": row["density"],
                "color_intensity": row["importance"]
            }
            ball_properties.append(properties)
        
        return ball_properties, reduced_vectors
    
    def visualize_concept_space(self, ball_properties, save_path="concept_space.png"):
        """
        Visualize concepts as convex balls in 3D space
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map for importance
        colors = plt.cm.viridis(np.linspace(0.3, 1, len(ball_properties)))
        
        for i, ball in enumerate(ball_properties):
            center = ball["center"]
            radius = ball["radius"]
            
            # Draw sphere (convex ball)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            
            ax.plot_surface(x, y, z, color=colors[i], alpha=0.3)
            
            # Mark centroid
            ax.scatter(center[0], center[1], center[2], 
                      color=colors[i], s=100, marker='o', 
                      edgecolors='black', linewidth=2)
            
            # Label
            ax.text(center[0], center[1], center[2], 
                   ball["concept_id"], fontsize=8)
        
        ax.set_xlabel('Semantic Dimension 1')
        ax.set_ylabel('Semantic Dimension 2')
        ax.set_zlabel('Semantic Dimension 3')
        ax.set_title('Concept Space: Concepts as Convex Balls')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
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
                    overlap_volume = self._calculate_overlap_volume(
                        ball1, ball2, center_dist
                    )
                    overlaps.append({
                        "concept1": ball1["concept_id"],
                        "concept2": ball2["concept_id"],
                        "distance": center_dist,
                        "overlap_volume": overlap_volume,
                        "overlap_ratio": overlap_volume / min(ball1["volume"], ball2["volume"])
                    })
        
        return pd.DataFrame(overlaps)
    
    def _calculate_overlap_volume(self, ball1, ball2, distance):
        """
        Approximate overlap volume between two spheres
        """
        r1, r2 = ball1["radius"], ball2["radius"]
        
        if distance >= r1 + r2:
            return 0
        elif distance <= abs(r1 - r2):
            # One ball contains the other
            return (4/3) * np.pi * min(r1, r2)**3
        else:
            # Partial overlap - use spherical cap formula
            h1 = (r1 + r2 - distance) * (r1 - r2 + distance) / (2 * distance)
            h2 = (r1 + r2 - distance) * (r2 - r1 + distance) / (2 * distance)
            
            vol1 = np.pi * h1**2 * (3*r1 - h1) / 3
            vol2 = np.pi * h2**2 * (3*r2 - h2) / 3
            
            return vol1 + vol2
    
    def generate_analysis_report(self):
        """
        Generate comprehensive analysis of concept space
        """
        print("\n" + "="*80)
        print("CONCEPT SPACE ANALYSIS REPORT")
        print("="*80)
        
        # Load and process concepts
        self.load_concepts()
        concept_df = self.extract_concept_features()
        
        # Compute semantic vectors
        concept_vectors, features = self.compute_semantic_vectors(concept_df)
        
        # Compute distances
        distances, similarities = self.compute_concept_distances(concept_vectors)
        
        # Get convex ball properties
        ball_properties, reduced_vectors = self.calculate_convex_hull_properties(
            concept_df, concept_vectors
        )
        
        # Analyze overlaps
        overlaps = self.analyze_ball_overlaps(ball_properties)
        
        print(f"\n📊 CONCEPT SPACE METRICS:")
        print(f"   • Total Concepts: {len(self.concepts)}")
        print(f"   • Semantic Dimensions: {concept_vectors.shape[1]}")
        print(f"   • Average Ball Radius: {np.mean([b['radius'] for b in ball_properties]):.3f}")
        print(f"   • Average Ball Volume: {np.mean([b['volume'] for b in ball_properties]):.3f}")
        
        print(f"\n🔗 CONCEPT RELATIONSHIPS:")
        if not overlaps.empty:
            print(f"   • Overlapping Concept Pairs: {len(overlaps)}")
            print(f"   • Average Overlap Ratio: {overlaps['overlap_ratio'].mean():.3f}")
            print(f"\n   Top Overlapping Concepts:")
            top_overlaps = overlaps.nlargest(5, 'overlap_ratio')
            for _, row in top_overlaps.iterrows():
                print(f"      {row['concept1']} ↔ {row['concept2']}: {row['overlap_ratio']:.2f}")
        
        # Identify concept clusters
        clusters = self.identify_concept_clusters(distances)
        print(f"\n🎯 CONCEPT CLUSTERS:")
        print(f"   • Number of Clusters: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            cluster_names = [concept_df.iloc[idx]["canonical_name"] for idx in cluster]
            print(f"   • Cluster {i+1}: {', '.join(cluster_names)}")
        
        return {
            "concept_df": concept_df,
            "ball_properties": ball_properties,
            "overlaps": overlaps,
            "clusters": clusters,
            "distances": distances
        }

def main():
    """
    Main execution for concept centroid analysis
    """
    analyzer = ConceptCentroidAnalyzer()
    
    # Generate analysis
    results = analyzer.generate_analysis_report()
    
    # Visualize concept space
    print("\n📈 Generating visualizations...")
    analyzer.visualize_concept_space(results["ball_properties"])
    
    print("\n✅ Analysis complete!")
    
    return results

if __name__ == "__main__":
    results = main()