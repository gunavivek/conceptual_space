#!/usr/bin/env python3
"""
A2.5.4: Frequency-Based Expansion Strategy
Expands concepts using co-occurrence analysis across corpus with minimum frequency threshold: 2 occurrences
"""

import json
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

def calculate_co_occurrence_matrix(concepts, min_frequency=2):
    """
    Calculate co-occurrence matrix for all terms across concepts

    Args:
        concepts: List of concepts
        min_frequency: Minimum frequency threshold

    Returns:
        dict: Co-occurrence data
    """
    # Collect all term pairs that co-occur in concepts
    cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
    term_frequency = Counter()

    for concept in concepts:
        keywords = [kw.lower().strip() for kw in concept.get("keywords", [])]

        # Count individual term frequencies
        for keyword in keywords:
            term_frequency[keyword] += 1

        # Count co-occurrences (term pairs in same concept)
        for i, term1 in enumerate(keywords):
            for j, term2 in enumerate(keywords):
                if i != j:  # Don't count self-co-occurrence
                    cooccurrence_matrix[term1][term2] += 1

    # Filter by minimum frequency
    filtered_cooccurrence = {}
    for term1, related_terms in cooccurrence_matrix.items():
        if term_frequency[term1] >= min_frequency:
            filtered_related = {
                term2: count for term2, count in related_terms.items()
                if count >= min_frequency and term_frequency[term2] >= min_frequency
            }
            if filtered_related:
                filtered_cooccurrence[term1] = filtered_related

    return {
        "cooccurrence_matrix": filtered_cooccurrence,
        "term_frequency": dict(term_frequency),
        "total_concepts": len(concepts)
    }

def find_frequency_expansions(concept, cooccurrence_data, max_expansions=5):
    """
    Find expansion terms based on frequency and co-occurrence

    Args:
        concept: Target concept
        cooccurrence_data: Co-occurrence analysis results
        max_expansions: Maximum expansion terms

    Returns:
        list: Frequency-based expansion terms
    """
    original_keywords = [kw.lower().strip() for kw in concept.get("keywords", [])]
    cooccurrence_matrix = cooccurrence_data["cooccurrence_matrix"]
    term_frequency = cooccurrence_data["term_frequency"]

    expansion_candidates = []

    # Find terms that frequently co-occur with concept's keywords
    for keyword in original_keywords:
        if keyword in cooccurrence_matrix:
            related_terms = cooccurrence_matrix[keyword]

            for related_term, cooccur_count in related_terms.items():
                # Don't add terms already in concept
                if related_term not in original_keywords:
                    # Calculate co-occurrence strength
                    term_freq = term_frequency.get(related_term, 0)
                    cooccurrence_strength = cooccur_count / max(term_freq, 1)

                    expansion_candidates.append({
                        "term": related_term,
                        "cooccurrence_count": cooccur_count,
                        "cooccurrence_strength": cooccurrence_strength,
                        "term_frequency": term_freq,
                        "source_keyword": keyword
                    })

    # Remove duplicates and sort by co-occurrence strength
    seen_terms = set()
    unique_candidates = []
    for candidate in expansion_candidates:
        if candidate["term"] not in seen_terms:
            seen_terms.add(candidate["term"])
            unique_candidates.append(candidate)

    # Sort by co-occurrence strength
    unique_candidates.sort(key=lambda x: (x["cooccurrence_strength"], x["cooccurrence_count"]), reverse=True)

    return unique_candidates[:max_expansions]

def expand_concept_with_frequency(concept, cooccurrence_data, max_expansions=5):
    """
    Expand a concept using frequency-based analysis

    Args:
        concept: Target concept to expand
        cooccurrence_data: Co-occurrence analysis results
        max_expansions: Maximum expansion terms

    Returns:
        dict: Expanded concept with frequency-based terms
    """
    # Get frequency expansions
    frequency_expansions = find_frequency_expansions(concept, cooccurrence_data, max_expansions)

    # Add expansion terms to keywords
    original_keywords = concept.get("keywords", [])
    expanded_keywords = original_keywords.copy()

    for expansion in frequency_expansions:
        expanded_keywords.append(expansion["term"])

    # Create expanded concept
    expanded_concept = concept.copy()
    expanded_concept["keywords"] = expanded_keywords
    expanded_concept["expansion_metadata"] = {
        "strategy": "frequency_based",
        "original_keyword_count": len(original_keywords),
        "expanded_keyword_count": len(expanded_keywords),
        "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
        "frequency_expansions": frequency_expansions
    }

    return expanded_concept

def process_frequency_based_expansion(core_concepts):
    """
    Process frequency-based expansion for all concepts

    Args:
        core_concepts: List of core concepts from A2.4

    Returns:
        dict: Frequency-based expansion results
    """
    # Calculate co-occurrence matrix
    print("  Computing co-occurrence matrix...")
    cooccurrence_data = calculate_co_occurrence_matrix(core_concepts)

    # Expand concepts
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
        expanded_concept = expand_concept_with_frequency(concept, cooccurrence_data)
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
        "strategy": "frequency_based",
        "expansions": expanded_concepts,
        "statistics": expansion_stats,
        "cooccurrence_metadata": {
            "total_unique_terms": len(cooccurrence_data["term_frequency"]),
            "terms_with_cooccurrence": len(cooccurrence_data["cooccurrence_matrix"])
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
    print("A2.5.4: Frequency-Based Expansion Strategy")
    print("="*60)

    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()

        # Extract core concepts from document structure
        core_concepts = []
        for doc in input_data.get("documents", []):
            core_concepts.extend(doc.get("core_concepts", []))

        print(f"Processing frequency-based expansion for {len(core_concepts)} concepts...")

        # Process frequency-based expansion
        expansion_results = process_frequency_based_expansion(core_concepts)

        # Display results
        stats = expansion_results["statistics"]
        cooccur_meta = expansion_results["cooccurrence_metadata"]

        print(f"\nFrequency-Based Expansion Results:")
        print(f"  Concepts Processed: {stats['total_concepts']}")
        print(f"  Concepts Expanded: {stats['concepts_expanded']}")
        print(f"  Expansion Coverage: {stats['expansion_coverage']:.1%}")
        print(f"  Original Keywords: {stats['total_original_keywords']}")
        print(f"  Expanded Keywords: {stats['total_expanded_keywords']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")

        print(f"\nCo-occurrence Analysis:")
        print(f"  Total Unique Terms: {cooccur_meta['total_unique_terms']}")
        print(f"  Terms with Co-occurrence: {cooccur_meta['terms_with_cooccurrence']}")

        # Show sample expansions
        print(f"\nSample Frequency Expansions:")
        for i, concept in enumerate(expansion_results["expansions"][:3], 1):
            metadata = concept["expansion_metadata"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Expansion: {metadata['original_keyword_count']} -> {metadata['expanded_keyword_count']} keywords")
            print(f"     Ratio: {metadata['expansion_ratio']:.2f}")

        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "frequency_based",
            "results": expansion_results,
            "processing_timestamp": datetime.now().isoformat()
        }

        output_path = Path(__file__).parent.parent / "outputs/A2.5.4_frequency_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.4 Frequency-Based Expansion completed successfully!")

    except Exception as e:
        print(f"Error in A2.5.4: {str(e)}")
        raise

if __name__ == "__main__":
    main()