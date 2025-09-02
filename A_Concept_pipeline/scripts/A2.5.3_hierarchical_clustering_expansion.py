#!/usr/bin/env python3
"""
A2.5.3: Hierarchical Clustering Expansion Strategy
Expands concepts using hierarchical clustering to find concept relationships at different levels
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import math

def calculate_concept_distance(concept1, concept2):
    """
    Calculate distance between two concepts for clustering
    
    Args:
        concept1: First concept
        concept2: Second concept
        
    Returns:
        float: Distance (lower = more similar)
    """
    # Extract features for distance calculation
    kw1 = set(concept1.get("primary_keywords", []))
    kw2 = set(concept2.get("primary_keywords", []))
    
    # Jaccard distance (1 - similarity)
    if not kw1 or not kw2:
        jaccard_dist = 1.0
    else:
        intersection = len(kw1 & kw2)
        union = len(kw1 | kw2)
        jaccard_dist = 1 - (intersection / union)
    
    # Domain distance
    domain1 = concept1.get("domain", "general")
    domain2 = concept2.get("domain", "general")
    domain_dist = 0.0 if domain1 == domain2 else 0.3
    
    # Importance distance
    imp1 = concept1.get("importance_score", 0.5)
    imp2 = concept2.get("importance_score", 0.5)
    imp_dist = abs(imp1 - imp2) * 0.2
    
    # Document coverage distance
    docs1 = set(concept1.get("related_documents", []))
    docs2 = set(concept2.get("related_documents", []))
    if docs1 or docs2:
        doc_overlap = len(docs1 & docs2) / len(docs1 | docs2) if docs1 | docs2 else 0
        doc_dist = (1 - doc_overlap) * 0.3
    else:
        doc_dist = 0.5
    
    return jaccard_dist * 0.4 + domain_dist + imp_dist + doc_dist

def build_distance_matrix(concepts):
    """
    Build distance matrix for hierarchical clustering
    
    Args:
        concepts: List of concepts
        
    Returns:
        numpy.ndarray: Distance matrix
    """
    n = len(concepts)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = calculate_concept_distance(concepts[i], concepts[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    
    return distance_matrix

def hierarchical_clustering(concepts, distance_matrix, max_clusters=5):
    """
    Perform hierarchical clustering on concepts
    
    Args:
        concepts: List of concepts
        distance_matrix: Distance matrix
        max_clusters: Maximum number of clusters
        
    Returns:
        dict: Clustering results
    """
    n = len(concepts)
    clusters = [{i} for i in range(n)]  # Start with each concept as its own cluster
    cluster_history = []
    
    # Agglomerative clustering
    while len(clusters) > max_clusters:
        # Find closest pair of clusters
        min_dist = float('inf')
        merge_i, merge_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Calculate average linkage distance
                total_dist = 0
                count = 0
                for ci in clusters[i]:
                    for cj in clusters[j]:
                        total_dist += distance_matrix[ci][cj]
                        count += 1
                
                avg_dist = total_dist / count if count > 0 else float('inf')
                
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    merge_i, merge_j = i, j
        
        # Merge clusters
        if merge_i != -1 and merge_j != -1:
            merged_cluster = clusters[merge_i] | clusters[merge_j]
            cluster_history.append({
                "merged_clusters": [merge_i, merge_j],
                "distance": min_dist,
                "cluster_size": len(merged_cluster)
            })
            
            # Remove old clusters and add merged one
            new_clusters = []
            for i, cluster in enumerate(clusters):
                if i != merge_i and i != merge_j:
                    new_clusters.append(cluster)
            new_clusters.append(merged_cluster)
            clusters = new_clusters
        else:
            break
    
    return {
        "final_clusters": clusters,
        "merge_history": cluster_history,
        "num_clusters": len(clusters)
    }

def extract_cluster_characteristics(cluster_indices, concepts):
    """
    Extract characteristics of a cluster
    
    Args:
        cluster_indices: Indices of concepts in cluster
        concepts: All concepts
        
    Returns:
        dict: Cluster characteristics
    """
    cluster_concepts = [concepts[i] for i in cluster_indices]
    
    # Aggregate keywords
    all_keywords = []
    for concept in cluster_concepts:
        all_keywords.extend(concept.get("primary_keywords", []))
    
    keyword_freq = Counter(all_keywords)
    representative_keywords = [kw for kw, count in keyword_freq.most_common(8)]
    
    # Aggregate domains
    domains = [concept.get("domain", "general") for concept in cluster_concepts]
    dominant_domain = Counter(domains).most_common(1)[0][0]
    
    # Aggregate documents
    all_docs = set()
    for concept in cluster_concepts:
        all_docs.update(concept.get("related_documents", []))
    
    # Calculate importance
    avg_importance = sum(c.get("importance_score", 0) for c in cluster_concepts) / len(cluster_concepts)
    
    return {
        "concept_ids": [concepts[i].get("concept_id") for i in cluster_indices],
        "concept_count": len(cluster_concepts),
        "representative_keywords": representative_keywords,
        "dominant_domain": dominant_domain,
        "document_coverage": len(all_docs),
        "average_importance": avg_importance,
        "cluster_coherence": len(set(representative_keywords)) / max(len(all_keywords), 1)
    }

def expand_concept_hierarchically(concept, clustering_results, concepts):
    """
    Expand a concept using hierarchical clustering results
    
    Args:
        concept: Target concept
        clustering_results: Clustering results
        concepts: All concepts
        
    Returns:
        dict: Hierarchical expansion
    """
    concept_id = concept.get("concept_id")
    concept_idx = None
    
    # Find concept index
    for i, c in enumerate(concepts):
        if c.get("concept_id") == concept_id:
            concept_idx = i
            break
    
    if concept_idx is None:
        return {"error": "Concept not found"}
    
    # Find which cluster this concept belongs to
    target_cluster = None
    for cluster in clustering_results["final_clusters"]:
        if concept_idx in cluster:
            target_cluster = cluster
            break
    
    if not target_cluster:
        return {"error": "Concept not in any cluster"}
    
    # Get cluster characteristics
    cluster_char = extract_cluster_characteristics(target_cluster, concepts)
    
    # Find related concepts in same cluster
    related_concepts = []
    for idx in target_cluster:
        if idx != concept_idx:
            related_concepts.append({
                "concept": concepts[idx],
                "relationship": "same_cluster",
                "cluster_position": "peer"
            })
    
    # Calculate expansion strength
    original_keywords = set(concept.get("primary_keywords", []))
    expanded_keywords = set(cluster_char["representative_keywords"])
    expansion_ratio = len(expanded_keywords) / max(len(original_keywords), 1)
    
    return {
        "original_concept": concept,
        "cluster_info": cluster_char,
        "related_concepts": related_concepts,
        "expanded_keywords": list(expanded_keywords),
        "expansion_metrics": {
            "cluster_size": len(target_cluster),
            "expansion_ratio": expansion_ratio,
            "cluster_coherence": cluster_char["cluster_coherence"],
            "hierarchical_level": "peer"  # All concepts at same level in this cluster
        }
    }

def process_hierarchical_expansion(core_concepts):
    """
    Process hierarchical clustering expansion
    
    Args:
        core_concepts: List of core concepts
        
    Returns:
        dict: Hierarchical expansion results
    """
    if len(core_concepts) < 2:
        return {
            "strategy": "hierarchical_clustering",
            "error": "Need at least 2 concepts for clustering",
            "statistics": {"concepts_processed": 0}
        }
    
    # Build distance matrix
    distance_matrix = build_distance_matrix(core_concepts)
    
    # Perform hierarchical clustering
    clustering_results = hierarchical_clustering(core_concepts, distance_matrix)
    
    # Expand each concept
    expansions = []
    for concept in core_concepts:
        expansion = expand_concept_hierarchically(concept, clustering_results, core_concepts)
        if "error" not in expansion:
            expansions.append(expansion)
    
    # Extract cluster summaries
    cluster_summaries = []
    for i, cluster in enumerate(clustering_results["final_clusters"]):
        cluster_char = extract_cluster_characteristics(cluster, core_concepts)
        cluster_summaries.append({
            "cluster_id": f"cluster_{i+1}",
            **cluster_char
        })
    
    # Calculate statistics
    avg_cluster_size = sum(len(cluster) for cluster in clustering_results["final_clusters"]) / max(len(clustering_results["final_clusters"]), 1)
    
    return {
        "strategy": "hierarchical_clustering",
        "clustering_results": clustering_results,
        "cluster_summaries": cluster_summaries,
        "expansions": expansions,
        "statistics": {
            "concepts_processed": len(expansions),
            "num_clusters": clustering_results["num_clusters"],
            "average_cluster_size": avg_cluster_size,
            "merge_steps": len(clustering_results["merge_history"]),
            "high_coherence_clusters": len([c for c in cluster_summaries if c["cluster_coherence"] > 0.6])
        }
    }

def load_input(input_path="outputs/A2.4_core_concepts.json"):
    """Load core concepts from A2.4"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """Main execution"""
    print("="*60)
    print("A2.5.3: Hierarchical Clustering Expansion Strategy")
    print("="*60)
    
    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()
        core_concepts = input_data.get("core_concepts", [])
        
        # Process hierarchical expansion
        print(f"Processing hierarchical clustering for {len(core_concepts)} concepts...")
        expansion_results = process_hierarchical_expansion(core_concepts)
        
        if "error" in expansion_results:
            print(f"Error: {expansion_results['error']}")
            return
        
        # Display results
        stats = expansion_results["statistics"]
        print(f"\nHierarchical Clustering Results:")
        print(f"  Concepts Processed: {stats['concepts_processed']}")
        print(f"  Number of Clusters: {stats['num_clusters']}")
        print(f"  Average Cluster Size: {stats['average_cluster_size']:.1f}")
        print(f"  Merge Steps: {stats['merge_steps']}")
        print(f"  High Coherence Clusters: {stats['high_coherence_clusters']}")
        
        # Show cluster summaries
        print(f"\nCluster Summaries:")
        for cluster in expansion_results["cluster_summaries"]:
            print(f"  {cluster['cluster_id']}: {cluster['concept_count']} concepts")
            print(f"    Domain: {cluster['dominant_domain']}")
            print(f"    Keywords: {', '.join(cluster['representative_keywords'][:5])}")
            print(f"    Coherence: {cluster['cluster_coherence']:.3f}")
        
        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "hierarchical_clustering",
            "results": expansion_results,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        output_path = Path(__file__).parent.parent / "outputs/A2.5.3_hierarchical_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.3 Hierarchical Clustering Expansion completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.5.3: {str(e)}")
        raise

if __name__ == "__main__":
    main()