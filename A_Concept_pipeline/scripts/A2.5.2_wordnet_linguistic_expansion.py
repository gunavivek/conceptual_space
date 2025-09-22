#!/usr/bin/env python3
"""
A2.5.2: Advanced WordNet Linguistic Expansion
Uses NLTK WordNet for sophisticated linguistic expansion via semantic relations
Implements synonyms, hypernyms, hyponyms, and coordinate terms expansion
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add expansion_modules to path
sys.path.append(str(Path(__file__).parent.parent / "expansion_modules"))

try:
    from wordnet_processor import WordNetProcessor
except ImportError:
    print("[ERROR] Could not import WordNetProcessor. Check expansion_modules installation.")
    sys.exit(1)

def expand_concept_with_wordnet(concept, wordnet_processor, max_expansions_per_type=3):
    """
    Expand a concept using advanced WordNet linguistic relationships

    Args:
        concept: Target concept to expand
        wordnet_processor: WordNetProcessor instance
        max_expansions_per_type: Maximum expansions per relation type

    Returns:
        dict: Expanded concept with linguistic terms
    """
    # Get comprehensive linguistic expansions
    linguistic_expansions = wordnet_processor.expand_concept_linguistically(
        concept, max_expansions_per_type
    )

    # Extract expansion terms from all relation types
    all_expansion_terms = []
    relation_stats = {}

    for relation_type, expansions in linguistic_expansions.items():
        relation_stats[relation_type] = len(expansions)
        for expansion in expansions:
            all_expansion_terms.append({
                "term": expansion["term"],
                "relation_type": expansion["relation_type"],
                "source_keyword": expansion["source_keyword"],
                "confidence": expansion["confidence"],
                "synset": expansion.get("synset", ""),
                "definition": expansion.get("definition", ""),
                "pos": expansion.get("pos", ""),
                "depth": expansion.get("depth", 0)
            })

    # Add expansion terms to keywords
    original_keywords = concept.get("keywords", [])
    expanded_keywords = original_keywords.copy()

    for expansion in all_expansion_terms:
        expanded_keywords.append(expansion["term"])

    # Create expanded concept
    expanded_concept = concept.copy()
    expanded_concept["keywords"] = expanded_keywords
    expanded_concept["expansion_metadata"] = {
        "strategy": "wordnet_linguistic",
        "original_keyword_count": len(original_keywords),
        "expanded_keyword_count": len(expanded_keywords),
        "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
        "relation_statistics": relation_stats,
        "total_linguistic_relations": len(all_expansion_terms),
        "expansion_terms": all_expansion_terms,
        "wordnet_processor_info": wordnet_processor.get_processor_info()
    }

    return expanded_concept

def process_wordnet_linguistic_expansion(core_concepts):
    """
    Process advanced WordNet linguistic expansion for all concepts

    Args:
        core_concepts: List of core concepts from A2.4

    Returns:
        dict: WordNet linguistic expansion results
    """
    # Initialize WordNet processor
    print("  Initializing WordNet processor...")
    wordnet_processor = WordNetProcessor(max_depth=2, include_definitions=True)

    expanded_concepts = []
    expansion_stats = {
        "total_concepts": len(core_concepts),
        "concepts_expanded": 0,
        "total_original_keywords": 0,
        "total_expanded_keywords": 0,
        "expansion_ratios": [],
        "relation_type_totals": {
            "synonyms": 0,
            "hypernyms": 0,
            "hyponyms": 0,
            "coordinates": 0
        },
        "total_linguistic_relations": 0
    }

    print("  Computing WordNet linguistic expansions...")
    for concept in core_concepts:
        # Expand the concept
        expanded_concept = expand_concept_with_wordnet(
            concept, wordnet_processor
        )
        expanded_concepts.append(expanded_concept)

        # Update statistics
        metadata = expanded_concept["expansion_metadata"]
        expansion_stats["total_original_keywords"] += metadata["original_keyword_count"]
        expansion_stats["total_expanded_keywords"] += metadata["expanded_keyword_count"]
        expansion_stats["expansion_ratios"].append(metadata["expansion_ratio"])
        expansion_stats["total_linguistic_relations"] += metadata["total_linguistic_relations"]

        # Update relation type statistics
        for relation_type, count in metadata["relation_statistics"].items():
            if relation_type in expansion_stats["relation_type_totals"]:
                expansion_stats["relation_type_totals"][relation_type] += count

        if metadata["expansion_ratio"] > 1.0:
            expansion_stats["concepts_expanded"] += 1

    # Calculate overall statistics
    expansion_stats["average_expansion_ratio"] = sum(expansion_stats["expansion_ratios"]) / len(expansion_stats["expansion_ratios"]) if expansion_stats["expansion_ratios"] else 0
    expansion_stats["expansion_coverage"] = expansion_stats["concepts_expanded"] / expansion_stats["total_concepts"]
    expansion_stats["avg_linguistic_relations"] = expansion_stats["total_linguistic_relations"] / len(core_concepts)

    return {
        "strategy": "wordnet_linguistic",
        "expansions": expanded_concepts,
        "statistics": expansion_stats,
        "processor_info": wordnet_processor.get_processor_info()
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
    print("A2.5.2: Advanced WordNet Linguistic Expansion")
    print("="*60)

    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()

        # Extract core concepts from document structure
        core_concepts = []
        for doc in input_data.get("documents", []):
            core_concepts.extend(doc.get("core_concepts", []))

        print(f"Processing WordNet linguistic expansion for {len(core_concepts)} concepts...")

        # Process WordNet linguistic expansion
        expansion_results = process_wordnet_linguistic_expansion(core_concepts)

        # Display results
        stats = expansion_results["statistics"]
        processor_info = expansion_results["processor_info"]

        print(f"\nAdvanced WordNet Linguistic Results:")
        print(f"  Concepts Processed: {stats['total_concepts']}")
        print(f"  Concepts Expanded: {stats['concepts_expanded']}")
        print(f"  Expansion Coverage: {stats['expansion_coverage']:.1%}")
        print(f"  Original Keywords: {stats['total_original_keywords']}")
        print(f"  Expanded Keywords: {stats['total_expanded_keywords']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")
        print(f"  Avg Linguistic Relations: {stats['avg_linguistic_relations']:.1f}")

        print(f"\nLinguistic Relation Statistics:")
        for relation_type, count in stats["relation_type_totals"].items():
            print(f"  {relation_type.capitalize()}: {count}")

        print(f"\nWordNet Processor Info:")
        print(f"  WordNet Available: {processor_info['wordnet_available']}")
        print(f"  Max Depth: {processor_info['max_depth']}")
        print(f"  Include Definitions: {processor_info['include_definitions']}")
        print(f"  Cache Size: {processor_info['cache_size']}")

        # Show sample expansions
        print(f"\nSample Linguistic Expansions:")
        for i, concept in enumerate(expansion_results["expansions"][:3], 1):
            metadata = concept["expansion_metadata"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Expansion: {metadata['original_keyword_count']} -> {metadata['expanded_keyword_count']} keywords")
            print(f"     Ratio: {metadata['expansion_ratio']:.2f}")
            print(f"     Relations: {metadata['relation_statistics']}")

        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "wordnet_linguistic",
            "results": expansion_results,
            "processing_timestamp": datetime.now().isoformat()
        }

        output_path = Path(__file__).parent.parent / "outputs/A2.5.2_wordnet_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.2 Advanced WordNet Linguistic Expansion completed successfully!")

    except Exception as e:
        print(f"Error in A2.5.2: {str(e)}")
        raise

if __name__ == "__main__":
    main()