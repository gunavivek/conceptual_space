#!/usr/bin/env python3
"""
A2.5.1: Advanced Semantic Similarity Expansion
Uses vector embeddings + cosine similarity for semantic neighbor discovery
Implements sophisticated embedding-based concept expansion
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add expansion_modules to path
sys.path.append(str(Path(__file__).parent.parent / "expansion_modules"))

try:
    from embedding_manager import EmbeddingManager
except ImportError:
    print("[ERROR] Could not import EmbeddingManager. Check expansion_modules installation.")
    sys.exit(1)

def expand_concept_with_semantic_similarity(concept, all_concepts, embedding_manager, max_expansions=5):
    """
    Expand a concept using advanced semantic similarity

    Args:
        concept: Target concept to expand
        all_concepts: All available concepts
        embedding_manager: EmbeddingManager instance
        max_expansions: Maximum expansion terms

    Returns:
        dict: Expanded concept with semantic neighbor terms
    """
    # Find semantic neighbors
    semantic_neighbors = embedding_manager.find_semantic_neighbors(
        concept, all_concepts, similarity_threshold=0.6, max_neighbors=10
    )

    # Extract expansion terms
    expansion_terms = embedding_manager.extract_expansion_terms(
        concept, semantic_neighbors, max_terms=max_expansions
    )

    # Add expansion terms to keywords
    original_keywords = concept.get("keywords", [])
    expanded_keywords = original_keywords.copy()

    for expansion in expansion_terms:
        expanded_keywords.append(expansion["term"])

    # Create expanded concept
    expanded_concept = concept.copy()
    expanded_concept["keywords"] = expanded_keywords
    expanded_concept["expansion_metadata"] = {
        "strategy": "semantic_similarity",
        "original_keyword_count": len(original_keywords),
        "expanded_keyword_count": len(expanded_keywords),
        "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
        "semantic_neighbors_found": len(semantic_neighbors),
        "expansion_terms": expansion_terms,
        "embedding_model": embedding_manager.get_model_info()
    }

    return expanded_concept

def process_semantic_similarity_expansion(core_concepts):
    """
    Process advanced semantic similarity expansion for all concepts

    Args:
        core_concepts: List of core concepts from A2.4

    Returns:
        dict: Semantic similarity expansion results
    """
    # Initialize embedding manager
    print("  Initializing embedding manager...")
    embedding_manager = EmbeddingManager(model_type='sentence_transformer')

    expanded_concepts = []
    expansion_stats = {
        "total_concepts": len(core_concepts),
        "concepts_expanded": 0,
        "total_original_keywords": 0,
        "total_expanded_keywords": 0,
        "expansion_ratios": [],
        "semantic_neighbors_total": 0
    }

    print("  Computing semantic similarities...")
    for concept in core_concepts:
        # Expand the concept
        expanded_concept = expand_concept_with_semantic_similarity(
            concept, core_concepts, embedding_manager
        )
        expanded_concepts.append(expanded_concept)

        # Update statistics
        metadata = expanded_concept["expansion_metadata"]
        expansion_stats["total_original_keywords"] += metadata["original_keyword_count"]
        expansion_stats["total_expanded_keywords"] += metadata["expanded_keyword_count"]
        expansion_stats["expansion_ratios"].append(metadata["expansion_ratio"])
        expansion_stats["semantic_neighbors_total"] += metadata["semantic_neighbors_found"]

        if metadata["expansion_ratio"] > 1.0:
            expansion_stats["concepts_expanded"] += 1

    # Calculate overall statistics
    expansion_stats["average_expansion_ratio"] = sum(expansion_stats["expansion_ratios"]) / len(expansion_stats["expansion_ratios"]) if expansion_stats["expansion_ratios"] else 0
    expansion_stats["expansion_coverage"] = expansion_stats["concepts_expanded"] / expansion_stats["total_concepts"]
    expansion_stats["avg_semantic_neighbors"] = expansion_stats["semantic_neighbors_total"] / len(core_concepts)

    return {
        "strategy": "semantic_similarity",
        "expansions": expanded_concepts,
        "statistics": expansion_stats,
        "model_info": embedding_manager.get_model_info()
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
    print("A2.5.1: Advanced Semantic Similarity Expansion")
    print("="*60)

    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()

        # Extract core concepts from document structure
        core_concepts = []
        for doc in input_data.get("documents", []):
            core_concepts.extend(doc.get("core_concepts", []))

        print(f"Processing semantic similarity expansion for {len(core_concepts)} concepts...")

        # Process semantic similarity expansion
        expansion_results = process_semantic_similarity_expansion(core_concepts)

        # Display results
        stats = expansion_results["statistics"]
        model_info = expansion_results["model_info"]

        print(f"\nAdvanced Semantic Similarity Results:")
        print(f"  Concepts Processed: {stats['total_concepts']}")
        print(f"  Concepts Expanded: {stats['concepts_expanded']}")
        print(f"  Expansion Coverage: {stats['expansion_coverage']:.1%}")
        print(f"  Original Keywords: {stats['total_original_keywords']}")
        print(f"  Expanded Keywords: {stats['total_expanded_keywords']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")
        print(f"  Avg Semantic Neighbors: {stats['avg_semantic_neighbors']:.1f}")

        print(f"\nEmbedding Model Info:")
        print(f"  Model Type: {model_info['model_type']}")
        print(f"  Model Name: {model_info['model_name']}")
        print(f"  Cache Size: {model_info['cache_size']}")

        # Show sample expansions
        print(f"\nSample Semantic Expansions:")
        for i, concept in enumerate(expansion_results["expansions"][:3], 1):
            metadata = concept["expansion_metadata"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Expansion: {metadata['original_keyword_count']} -> {metadata['expanded_keyword_count']} keywords")
            print(f"     Ratio: {metadata['expansion_ratio']:.2f}")
            print(f"     Semantic Neighbors: {metadata['semantic_neighbors_found']}")

        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "semantic_similarity",
            "results": expansion_results,
            "processing_timestamp": datetime.now().isoformat()
        }

        output_path = Path(__file__).parent.parent / "outputs/A2.5.1_semantic_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.1 Advanced Semantic Similarity Expansion completed successfully!")

    except Exception as e:
        print(f"Error in A2.5.1: {str(e)}")
        raise

if __name__ == "__main__":
    main()