#!/usr/bin/env python3
"""
A2.5.3: Hierarchical Clustering Expansion Strategy
Expands concepts using agglomerative clustering of concept vectors to enable cross-cluster term sharing
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def create_concept_vectors(concepts):
    """
    Create TF-IDF vectors for all concepts

    Args:
        concepts: List of concepts

    Returns:
        tuple: (vectorizer, concept_vectors, concept_texts)
    """
    # Create text representations of concepts
    concept_texts = []
    for concept in concepts:
        keywords = concept.get("keywords", [])
        # Combine keywords into a single text
        text = " ".join(keywords)
        concept_texts.append(text)

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2)
    )

    try:
        concept_vectors = vectorizer.fit_transform(concept_texts)
        return vectorizer, concept_vectors, concept_texts
    except ValueError:
        # Handle case where no features can be extracted
        return None, None, concept_texts

def perform_agglomerative_clustering(concept_vectors, n_clusters=None):
    """
    Perform agglomerative clustering on concept vectors

    Args:
        concept_vectors: TF-IDF vectors of concepts
        n_clusters: Number of clusters (auto-determined if None)

    Returns:
        numpy.array: Cluster labels for each concept
    """
    if concept_vectors is None:
        return np.array([0] * concept_vectors.shape[0] if concept_vectors else [])

    # Auto-determine number of clusters if not specified
    n_concepts = concept_vectors.shape[0]
    if n_clusters is None:
        # Use approximately sqrt(n) clusters, min 3, max 20
        n_clusters = max(3, min(20, int(np.sqrt(n_concepts))))

    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )

    # Convert sparse matrix to dense for clustering
    dense_vectors = concept_vectors.toarray()
    cluster_labels = clustering.fit_predict(dense_vectors)

    return cluster_labels

def find_cluster_expansions(concept, concepts, cluster_labels, max_expansions=5):
    """
    Find expansion terms from concepts in the same cluster

    Args:
        concept: Target concept to expand
        concepts: All concepts
        cluster_labels: Cluster assignments
        max_expansions: Maximum expansion terms

    Returns:
        list: Expansion terms with metadata
    """
    # Find the concept's index and cluster
    concept_id = concept.get("concept_id", "")
    concept_index = None

    for i, c in enumerate(concepts):
        if c.get("concept_id") == concept_id:
            concept_index = i
            break

    if concept_index is None:
        return []

    concept_cluster = cluster_labels[concept_index]
    original_keywords = set(kw.lower() for kw in concept.get("keywords", []))

    # Find other concepts in the same cluster
    cluster_concepts = []
    for i, cluster_label in enumerate(cluster_labels):
        if cluster_label == concept_cluster and i != concept_index:
            cluster_concepts.append(concepts[i])

    if not cluster_concepts:
        return []

    # Collect expansion candidates from cluster members
    expansion_candidates = []
    term_frequency = Counter()

    for cluster_concept in cluster_concepts:
        cluster_keywords = cluster_concept.get("keywords", [])
        for keyword in cluster_keywords:
            if keyword.lower() not in original_keywords:
                term_frequency[keyword] += 1
                expansion_candidates.append({
                    "term": keyword,
                    "source_concept_id": cluster_concept.get("concept_id"),
                    "cluster_frequency": term_frequency[keyword],
                    "cluster_id": concept_cluster
                })

    # Sort by cluster frequency (how many cluster members have this term)
    expansion_candidates.sort(key=lambda x: x["cluster_frequency"], reverse=True)

    # Remove duplicates while preserving order
    seen_terms = set()
    unique_candidates = []
    for candidate in expansion_candidates:
        if candidate["term"] not in seen_terms:
            seen_terms.add(candidate["term"])
            unique_candidates.append(candidate)

    return unique_candidates[:max_expansions]

def expand_concept_with_clustering(concept, concepts, cluster_labels, max_expansions=5):
    """
    Expand a concept using hierarchical clustering

    Args:
        concept: Target concept to expand
        concepts: All concepts
        cluster_labels: Cluster assignments
        max_expansions: Maximum expansion terms

    Returns:
        dict: Expanded concept with clustering-based terms
    """
    # Get cluster expansions
    cluster_expansions = find_cluster_expansions(concept, concepts, cluster_labels, max_expansions)

    # Add expansion terms to keywords
    original_keywords = concept.get("keywords", [])
    expanded_keywords = original_keywords.copy()

    for expansion in cluster_expansions:
        expanded_keywords.append(expansion["term"])

    # Find concept's cluster info
    concept_id = concept.get("concept_id", "")
    concept_cluster = None
    cluster_size = 0

    for i, c in enumerate(concepts):
        if c.get("concept_id") == concept_id:
            if i < len(cluster_labels):
                concept_cluster = cluster_labels[i]
                cluster_size = sum(1 for label in cluster_labels if label == concept_cluster)
            break

    # Create expanded concept
    expanded_concept = concept.copy()
    expanded_concept["keywords"] = expanded_keywords
    expanded_concept["expansion_metadata"] = {
        "strategy": "hierarchical_clustering",
        "original_keyword_count": len(original_keywords),
        "expanded_keyword_count": len(expanded_keywords),
        "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
        "cluster_id": concept_cluster,
        "cluster_size": cluster_size,
        "cluster_expansions": cluster_expansions
    }

    return expanded_concept

def process_hierarchical_clustering_expansion(core_concepts):
    """
    Process hierarchical clustering expansion for all concepts

    Args:
        core_concepts: List of core concepts from A2.4

    Returns:
        dict: Hierarchical clustering expansion results
    """
    # Create concept vectors
    vectorizer, concept_vectors, concept_texts = create_concept_vectors(core_concepts)

    if concept_vectors is None:
        # Fallback if vectorization fails
        print("Warning: Could not create concept vectors, using minimal clustering")
        cluster_labels = np.array([i % 5 for i in range(len(core_concepts))])  # Simple grouping
    else:
        # Perform clustering
        cluster_labels = perform_agglomerative_clustering(concept_vectors)

    # Expand concepts
    expanded_concepts = []
    expansion_stats = {
        "total_concepts": len(core_concepts),
        "concepts_expanded": 0,
        "total_original_keywords": 0,
        "total_expanded_keywords": 0,
        "expansion_ratios": [],
        "cluster_distribution": Counter(),
        "total_clusters": len(set(cluster_labels)) if len(cluster_labels) > 0 else 0
    }

    for concept in core_concepts:
        # Expand the concept
        expanded_concept = expand_concept_with_clustering(concept, core_concepts, cluster_labels)
        expanded_concepts.append(expanded_concept)

        # Update statistics
        metadata = expanded_concept["expansion_metadata"]
        expansion_stats["total_original_keywords"] += metadata["original_keyword_count"]
        expansion_stats["total_expanded_keywords"] += metadata["expanded_keyword_count"]
        expansion_stats["expansion_ratios"].append(metadata["expansion_ratio"])

        if metadata["cluster_id"] is not None:
            expansion_stats["cluster_distribution"][metadata["cluster_id"]] += 1

        if metadata["expansion_ratio"] > 1.0:
            expansion_stats["concepts_expanded"] += 1

    # Calculate overall statistics
    expansion_stats["average_expansion_ratio"] = sum(expansion_stats["expansion_ratios"]) / len(expansion_stats["expansion_ratios"]) if expansion_stats["expansion_ratios"] else 0
    expansion_stats["expansion_coverage"] = expansion_stats["concepts_expanded"] / expansion_stats["total_concepts"]
    expansion_stats["average_cluster_size"] = len(core_concepts) / max(expansion_stats["total_clusters"], 1)

    return {
        "strategy": "hierarchical_clustering",
        "expansions": expanded_concepts,
        "statistics": expansion_stats,
        "clustering_metadata": {
            "total_clusters": expansion_stats["total_clusters"],
            "cluster_distribution": dict(expansion_stats["cluster_distribution"]),
            "vectorization_successful": concept_vectors is not None
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

        # Extract core concepts from document structure
        core_concepts = []
        for doc in input_data.get("documents", []):
            core_concepts.extend(doc.get("core_concepts", []))

        print(f"Processing hierarchical clustering expansion for {len(core_concepts)} concepts...")

        # Process hierarchical clustering expansion
        expansion_results = process_hierarchical_clustering_expansion(core_concepts)

        # Display results
        stats = expansion_results["statistics"]
        clustering_meta = expansion_results["clustering_metadata"]

        print(f"\nHierarchical Clustering Expansion Results:")
        print(f"  Concepts Processed: {stats['total_concepts']}")
        print(f"  Concepts Expanded: {stats['concepts_expanded']}")
        print(f"  Expansion Coverage: {stats['expansion_coverage']:.1%}")
        print(f"  Original Keywords: {stats['total_original_keywords']}")
        print(f"  Expanded Keywords: {stats['total_expanded_keywords']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")

        print(f"\nClustering Information:")
        print(f"  Total Clusters: {clustering_meta['total_clusters']}")
        print(f"  Average Cluster Size: {stats['average_cluster_size']:.1f}")
        print(f"  Vectorization Successful: {clustering_meta['vectorization_successful']}")

        # Show sample expansions
        print(f"\nSample Clustering Expansions:")
        for i, concept in enumerate(expansion_results["expansions"][:3], 1):
            metadata = concept["expansion_metadata"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Cluster ID: {metadata['cluster_id']}")
            print(f"     Cluster Size: {metadata['cluster_size']}")
            print(f"     Expansion: {metadata['original_keyword_count']} -> {metadata['expanded_keyword_count']} keywords")
            print(f"     Ratio: {metadata['expansion_ratio']:.2f}")

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