#!/usr/bin/env python3
"""
A2.5.1: Semantic Similarity Expansion Strategy
Expands concepts by adding keywords from semantically similar concepts using Jaccard similarity
"""

import json
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

def calculate_jaccard_similarity(concept1, concept2):
    """
    Calculate Jaccard similarity between two concepts based on their keywords

    Args:
        concept1: First concept with keywords
        concept2: Second concept with keywords

    Returns:
        float: Jaccard similarity score (0-1)
    """
    keywords1 = set(concept1.get("keywords", []))
    keywords2 = set(concept2.get("keywords", []))

    if not keywords1 or not keywords2:
        return 0.0

    intersection = keywords1 & keywords2
    union = keywords1 | keywords2

    return len(intersection) / len(union) if union else 0.0

def calculate_domain_bonus(concept1, concept2):
    """
    Calculate domain similarity bonus

    Args:
        concept1: First concept
        concept2: Second concept

    Returns:
        float: Domain bonus (0.0-0.1)
    """
    domain1 = concept1.get("business_category", concept1.get("domain", "general"))
    domain2 = concept2.get("business_category", concept2.get("domain", "general"))

    if domain1 == domain2 and domain1 != "general":
        return 0.1
    return 0.0

def calculate_theme_similarity(concept1, concept2):
    """
    Calculate theme similarity bonus

    Args:
        concept1: First concept
        concept2: Second concept

    Returns:
        float: Theme similarity bonus (0.0-0.1)
    """
    theme1 = concept1.get("concept_type", "")
    theme2 = concept2.get("concept_type", "")

    if theme1 and theme2 and theme1 == theme2:
        return 0.1
    return 0.0

def find_similar_concepts(target_concept, all_concepts, similarity_threshold=0.4):
    """
    Find concepts similar to the target concept

    Args:
        target_concept: Concept to find similarities for
        all_concepts: All available concepts
        similarity_threshold: Minimum similarity score

    Returns:
        list: Similar concepts with their similarity scores
    """
    similar_concepts = []
    target_id = target_concept.get("concept_id", "")

    for concept in all_concepts:
        if concept.get("concept_id") == target_id:
            continue  # Skip self

        # Calculate base Jaccard similarity
        jaccard_sim = calculate_jaccard_similarity(target_concept, concept)

        # Add domain bonus
        domain_bonus = calculate_domain_bonus(target_concept, concept)

        # Add theme similarity bonus
        theme_bonus = calculate_theme_similarity(target_concept, concept)

        # Total similarity score
        total_similarity = jaccard_sim + domain_bonus + theme_bonus

        if total_similarity >= similarity_threshold:
            similar_concepts.append({
                "concept": concept,
                "similarity_score": total_similarity,
                "jaccard_similarity": jaccard_sim,
                "domain_bonus": domain_bonus,
                "theme_bonus": theme_bonus
            })

    # Sort by similarity score
    similar_concepts.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similar_concepts

def expand_concept_with_similarity(concept, all_concepts, max_expansions=10):
    """
    Expand a concept using semantic similarity

    Args:
        concept: Target concept to expand
        all_concepts: All available concepts
        max_expansions: Maximum number of expansion terms to add

    Returns:
        dict: Expanded concept with additional keywords
    """
    # Find similar concepts
    similar_concepts = find_similar_concepts(concept, all_concepts)

    # Extract expansion terms from similar concepts
    original_keywords = set(concept.get("keywords", []))
    expansion_candidates = []

    for sim_data in similar_concepts[:5]:  # Top 5 similar concepts
        sim_concept = sim_data["concept"]
        sim_keywords = set(sim_concept.get("keywords", []))

        # Find keywords not in original concept
        new_keywords = sim_keywords - original_keywords

        for keyword in new_keywords:
            expansion_candidates.append({
                "term": keyword,
                "source_concept_id": sim_concept.get("concept_id"),
                "source_similarity": sim_data["similarity_score"],
                "jaccard_contribution": sim_data["jaccard_similarity"]
            })

    # Rank expansion candidates by source similarity
    expansion_candidates.sort(key=lambda x: x["source_similarity"], reverse=True)

    # Select top expansion terms
    selected_expansions = expansion_candidates[:max_expansions]
    expanded_keywords = list(original_keywords) + [exp["term"] for exp in selected_expansions]

    # Create expanded concept
    expanded_concept = concept.copy()
    expanded_concept["keywords"] = expanded_keywords
    expanded_concept["expansion_metadata"] = {
        "strategy": "semantic_similarity",
        "original_keyword_count": len(original_keywords),
        "expanded_keyword_count": len(expanded_keywords),
        "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
        "similar_concepts_found": len(similar_concepts),
        "expansion_sources": selected_expansions
    }

    return expanded_concept

def process_semantic_similarity_expansion(core_concepts):
    """
    Process semantic similarity expansion for all concepts

    Args:
        core_concepts: List of core concepts from A2.4

    Returns:
        dict: Semantic similarity expansion results
    """
    expanded_concepts = []
    expansion_stats = {
        "total_concepts": len(core_concepts),
        "concepts_expanded": 0,
        "total_original_keywords": 0,
        "total_expanded_keywords": 0,
        "expansion_ratios": []
    }

    for concept in core_concepts:
        # Expand the concept
        expanded_concept = expand_concept_with_similarity(concept, core_concepts)
        expanded_concepts.append(expanded_concept)

        # Update statistics
        metadata = expanded_concept["expansion_metadata"]
        expansion_stats["total_original_keywords"] += metadata["original_keyword_count"]
        expansion_stats["total_expanded_keywords"] += metadata["expanded_keyword_count"]
        expansion_stats["expansion_ratios"].append(metadata["expansion_ratio"])

        if metadata["expansion_ratio"] > 1.0:
            expansion_stats["concepts_expanded"] += 1

    # Calculate overall statistics
    expansion_stats["average_expansion_ratio"] = sum(expansion_stats["expansion_ratios"]) / len(expansion_stats["expansion_ratios"]) if expansion_stats["expansion_ratios"] else 0
    expansion_stats["expansion_coverage"] = expansion_stats["concepts_expanded"] / expansion_stats["total_concepts"]

    return {
        "strategy": "semantic_similarity",
        "expansions": expanded_concepts,
        "statistics": expansion_stats
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
    print("A2.5.1: Semantic Similarity Expansion Strategy")
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
        print(f"\nSemantic Similarity Expansion Results:")
        print(f"  Concepts Processed: {stats['total_concepts']}")
        print(f"  Concepts Expanded: {stats['concepts_expanded']}")
        print(f"  Expansion Coverage: {stats['expansion_coverage']:.1%}")
        print(f"  Original Keywords: {stats['total_original_keywords']}")
        print(f"  Expanded Keywords: {stats['total_expanded_keywords']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")

        # Show sample expansions
        print(f"\nSample Semantic Expansions:")
        for i, concept in enumerate(expansion_results["expansions"][:3], 1):
            metadata = concept["expansion_metadata"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Expansion: {metadata['original_keyword_count']} -> {metadata['expanded_keyword_count']} keywords")
            print(f"     Ratio: {metadata['expansion_ratio']:.2f}")
            print(f"     Similar concepts found: {metadata['similar_concepts_found']}")

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
        print("\nA2.5.1 Semantic Similarity Expansion completed successfully!")

    except Exception as e:
        print(f"Error in A2.5.1: {str(e)}")
        raise

if __name__ == "__main__":
    main()