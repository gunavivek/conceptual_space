#!/usr/bin/env python3
"""
A2.5.5: Contextual Embedding Expansion Strategy
Expands concepts using dense vector similarity in embedding space to capture deeper semantic relationships
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding_vectors(concepts):
    """
    Create dense embedding vectors for concepts using TF-IDF as proxy for embeddings

    Args:
        concepts: List of concepts

    Returns:
        tuple: (vectors, concept_texts)
    """
    # Create text representations
    concept_texts = []
    for concept in concepts:
        keywords = concept.get("keywords", [])
        # Enhanced text representation
        text = " ".join(keywords)
        # Add concept metadata for richer context
        if concept.get("canonical_name"):
            text += " " + concept["canonical_name"]
        concept_texts.append(text)

    # Create dense vectors using TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 3),  # Include trigrams for richer context
        min_df=1,
        max_df=0.95
    )

    try:
        vectors = vectorizer.fit_transform(concept_texts)
        return vectors, concept_texts, vectorizer
    except ValueError:
        # Fallback to simple word vectors
        return None, concept_texts, None

def calculate_semantic_similarity_matrix(vectors):
    """
    Calculate semantic similarity matrix using cosine similarity

    Args:
        vectors: Dense concept vectors

    Returns:
        numpy.array: Similarity matrix
    """
    if vectors is None:
        return None

    # Convert to dense array for cosine similarity
    dense_vectors = vectors.toarray()
    similarity_matrix = cosine_similarity(dense_vectors)

    return similarity_matrix

def find_embedding_expansions(concept, concepts, similarity_matrix, concept_index, similarity_threshold=0.3, max_expansions=5):
    """
    Find expansion terms using embedding similarity

    Args:
        concept: Target concept
        concepts: All concepts
        similarity_matrix: Precomputed similarity matrix
        concept_index: Index of target concept
        similarity_threshold: Minimum similarity threshold
        max_expansions: Maximum expansion terms

    Returns:
        list: Embedding-based expansion terms
    """
    if similarity_matrix is None or concept_index >= len(similarity_matrix):
        return []

    original_keywords = set(kw.lower().strip() for kw in concept.get("keywords", []))
    expansion_candidates = []

    # Get similarity scores for this concept
    similarities = similarity_matrix[concept_index]

    # Find similar concepts
    for i, similarity_score in enumerate(similarities):
        if i != concept_index and similarity_score >= similarity_threshold:
            similar_concept = concepts[i]
            similar_keywords = similar_concept.get("keywords", [])

            # Extract new keywords from similar concept
            for keyword in similar_keywords:
                if keyword.lower().strip() not in original_keywords:
                    expansion_candidates.append({
                        "term": keyword,
                        "similarity_score": similarity_score,
                        "source_concept_id": similar_concept.get("concept_id"),
                        "source_concept_name": similar_concept.get("canonical_name", "")
                    })

    # Sort by similarity score and remove duplicates
    seen_terms = set()
    unique_candidates = []
    for candidate in sorted(expansion_candidates, key=lambda x: x["similarity_score"], reverse=True):
        if candidate["term"] not in seen_terms:
            seen_terms.add(candidate["term"])
            unique_candidates.append(candidate)

    return unique_candidates[:max_expansions]

def expand_concept_with_embeddings(concept, concepts, similarity_matrix, concept_index, max_expansions=5):
    """
    Expand a concept using contextual embeddings

    Args:
        concept: Target concept to expand
        concepts: All concepts
        similarity_matrix: Similarity matrix
        concept_index: Index of concept
        max_expansions: Maximum expansion terms

    Returns:
        dict: Expanded concept with embedding-based terms
    """
    # Get embedding expansions
    embedding_expansions = find_embedding_expansions(
        concept, concepts, similarity_matrix, concept_index, max_expansions=max_expansions
    )

    # Add expansion terms to keywords
    original_keywords = concept.get("keywords", [])
    expanded_keywords = original_keywords.copy()

    for expansion in embedding_expansions:
        expanded_keywords.append(expansion["term"])

    # Calculate embedding quality metrics
    avg_similarity = sum(exp["similarity_score"] for exp in embedding_expansions) / max(len(embedding_expansions), 1)

    # Create expanded concept
    expanded_concept = concept.copy()
    expanded_concept["keywords"] = expanded_keywords
    expanded_concept["expansion_metadata"] = {
        "strategy": "contextual_embedding",
        "original_keyword_count": len(original_keywords),
        "expanded_keyword_count": len(expanded_keywords),
        "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
        "embedding_expansions": embedding_expansions,
        "average_similarity": avg_similarity,
        "concept_index": concept_index
    }

    return expanded_concept

def process_contextual_embedding_expansion(core_concepts):
    """
    Process contextual embedding expansion for all concepts

    Args:
        core_concepts: List of core concepts from A2.4

    Returns:
        dict: Contextual embedding expansion results
    """
    # Create embedding vectors
    print("  Computing contextual embeddings...")
    vectors, concept_texts, vectorizer = create_embedding_vectors(core_concepts)

    # Calculate similarity matrix
    similarity_matrix = calculate_semantic_similarity_matrix(vectors)

    # Expand concepts
    expanded_concepts = []
    expansion_stats = {
        "total_concepts": len(core_concepts),
        "concepts_expanded": 0,
        "total_original_keywords": 0,
        "total_expanded_keywords": 0,
        "expansion_ratios": [],
        "similarity_scores": []
    }

    for i, concept in enumerate(core_concepts):
        # Expand the concept
        expanded_concept = expand_concept_with_embeddings(
            concept, core_concepts, similarity_matrix, i
        )
        expanded_concepts.append(expanded_concept)

        # Update statistics
        metadata = expanded_concept["expansion_metadata"]
        expansion_stats["total_original_keywords"] += metadata["original_keyword_count"]
        expansion_stats["total_expanded_keywords"] += metadata["expanded_keyword_count"]
        expansion_stats["expansion_ratios"].append(metadata["expansion_ratio"])
        expansion_stats["similarity_scores"].append(metadata["average_similarity"])

        if metadata["expansion_ratio"] > 1.0:
            expansion_stats["concepts_expanded"] += 1

    # Calculate overall statistics
    expansion_stats["average_expansion_ratio"] = sum(expansion_stats["expansion_ratios"]) / len(expansion_stats["expansion_ratios"]) if expansion_stats["expansion_ratios"] else 0
    expansion_stats["expansion_coverage"] = expansion_stats["concepts_expanded"] / expansion_stats["total_concepts"]
    expansion_stats["average_similarity"] = sum(expansion_stats["similarity_scores"]) / len(expansion_stats["similarity_scores"]) if expansion_stats["similarity_scores"] else 0

    return {
        "strategy": "contextual_embedding",
        "expansions": expanded_concepts,
        "statistics": expansion_stats,
        "embedding_metadata": {
            "vectorization_successful": vectors is not None,
            "vector_dimensions": vectors.shape[1] if vectors is not None else 0,
            "similarity_matrix_computed": similarity_matrix is not None
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
    print("A2.5.5: Contextual Embedding Expansion Strategy")
    print("="*60)

    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()

        # Extract core concepts from document structure
        core_concepts = []
        for doc in input_data.get("documents", []):
            core_concepts.extend(doc.get("core_concepts", []))

        print(f"Processing contextual embedding expansion for {len(core_concepts)} concepts...")

        # Process contextual embedding expansion
        expansion_results = process_contextual_embedding_expansion(core_concepts)

        # Display results
        stats = expansion_results["statistics"]
        embedding_meta = expansion_results["embedding_metadata"]

        print(f"\nContextual Embedding Expansion Results:")
        print(f"  Concepts Processed: {stats['total_concepts']}")
        print(f"  Concepts Expanded: {stats['concepts_expanded']}")
        print(f"  Expansion Coverage: {stats['expansion_coverage']:.1%}")
        print(f"  Original Keywords: {stats['total_original_keywords']}")
        print(f"  Expanded Keywords: {stats['total_expanded_keywords']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")
        print(f"  Average Similarity Score: {stats['average_similarity']:.3f}")

        print(f"\nEmbedding Information:")
        print(f"  Vectorization Successful: {embedding_meta['vectorization_successful']}")
        print(f"  Vector Dimensions: {embedding_meta['vector_dimensions']}")
        print(f"  Similarity Matrix Computed: {embedding_meta['similarity_matrix_computed']}")

        # Show sample expansions
        print(f"\nSample Embedding Expansions:")
        for i, concept in enumerate(expansion_results["expansions"][:3], 1):
            metadata = concept["expansion_metadata"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Expansion: {metadata['original_keyword_count']} -> {metadata['expanded_keyword_count']} keywords")
            print(f"     Ratio: {metadata['expansion_ratio']:.2f}")
            print(f"     Avg Similarity: {metadata['average_similarity']:.3f}")

        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "contextual_embedding",
            "results": expansion_results,
            "processing_timestamp": datetime.now().isoformat()
        }

        output_path = Path(__file__).parent.parent / "outputs/A2.5.5_contextual_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.5 Contextual Embedding Expansion completed successfully!")

    except Exception as e:
        print(f"Error in A2.5.5: {str(e)}")
        raise

if __name__ == "__main__":
    main()