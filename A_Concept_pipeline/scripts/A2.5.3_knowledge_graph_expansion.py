#!/usr/bin/env python3
"""
A2.5.3: Advanced Knowledge Graph Relationship Expansion
Uses semantic relationships for sophisticated concept expansion
Implements is-a, part-of, causes, requires relationship-based expansion
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add expansion_modules to path
sys.path.append(str(Path(__file__).parent.parent / "expansion_modules"))

try:
    from knowledge_graph import KnowledgeGraphProcessor
except ImportError:
    print("[ERROR] Could not import KnowledgeGraphProcessor. Check expansion_modules installation.")
    sys.exit(1)

def expand_concept_with_knowledge_graph(concept, all_concepts, kg_processor, max_expansions=5):
    """
    Expand a concept using knowledge graph relationships

    Args:
        concept: Target concept to expand
        all_concepts: All available concepts for relationship discovery
        kg_processor: KnowledgeGraphProcessor instance
        max_expansions: Maximum expansion terms

    Returns:
        dict: Expanded concept with relationship-based terms
    """
    # Learn relationships from all concepts first
    learned_relationships = kg_processor.learn_relationships_from_concepts(all_concepts)

    # Get relationship-based expansions
    relationship_expansions = kg_processor.expand_with_relationships(concept, max_expansions)

    # Discover implicit relationships
    implicit_relationships = kg_processor.discover_implicit_relationships(
        concept, all_concepts, max_discoveries=max_expansions
    )

    # Combine all relationship expansions
    all_expansion_terms = relationship_expansions + implicit_relationships

    # Add expansion terms to keywords
    original_keywords = concept.get("keywords", [])
    expanded_keywords = original_keywords.copy()

    for expansion in all_expansion_terms:
        expanded_keywords.append(expansion["term"])

    # Create expanded concept
    expanded_concept = concept.copy()
    expanded_concept["keywords"] = expanded_keywords
    expanded_concept["expansion_metadata"] = {
        "strategy": "knowledge_graph",
        "original_keyword_count": len(original_keywords),
        "expanded_keyword_count": len(expanded_keywords),
        "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
        "relationship_expansions": len(relationship_expansions),
        "implicit_relationships": len(implicit_relationships),
        "total_kg_expansions": len(all_expansion_terms),
        "learned_relationships": learned_relationships,
        "expansion_terms": all_expansion_terms,
        "kg_processor_info": kg_processor.get_processor_info()
    }

    return expanded_concept

def process_knowledge_graph_expansion(core_concepts):
    """
    Process advanced knowledge graph expansion for all concepts

    Args:
        core_concepts: List of core concepts from A2.4

    Returns:
        dict: Knowledge graph expansion results
    """
    # Initialize knowledge graph processor
    print("  Initializing knowledge graph processor...")
    kg_processor = KnowledgeGraphProcessor()

    # Pre-populate with common relationships
    kg_processor.populate_common_relationships()

    expanded_concepts = []
    expansion_stats = {
        "total_concepts": len(core_concepts),
        "concepts_expanded": 0,
        "total_original_keywords": 0,
        "total_expanded_keywords": 0,
        "expansion_ratios": [],
        "relationship_type_totals": {
            "is_a": 0,
            "part_of": 0,
            "causes": 0,
            "requires": 0,
            "implicit": 0
        },
        "total_kg_expansions": 0
    }

    print("  Computing knowledge graph expansions...")
    for concept in core_concepts:
        # Expand the concept
        expanded_concept = expand_concept_with_knowledge_graph(
            concept, core_concepts, kg_processor
        )
        expanded_concepts.append(expanded_concept)

        # Update statistics
        metadata = expanded_concept["expansion_metadata"]
        expansion_stats["total_original_keywords"] += metadata["original_keyword_count"]
        expansion_stats["total_expanded_keywords"] += metadata["expanded_keyword_count"]
        expansion_stats["expansion_ratios"].append(metadata["expansion_ratio"])
        expansion_stats["total_kg_expansions"] += metadata["total_kg_expansions"]

        # Update relationship type statistics
        for expansion in metadata["expansion_terms"]:
            rel_type = expansion.get("relation_type", "unknown")
            if rel_type in expansion_stats["relationship_type_totals"]:
                expansion_stats["relationship_type_totals"][rel_type] += 1
            elif "implicit" in rel_type:
                expansion_stats["relationship_type_totals"]["implicit"] += 1

        if metadata["expansion_ratio"] > 1.0:
            expansion_stats["concepts_expanded"] += 1

    # Calculate overall statistics
    expansion_stats["average_expansion_ratio"] = sum(expansion_stats["expansion_ratios"]) / len(expansion_stats["expansion_ratios"]) if expansion_stats["expansion_ratios"] else 0
    expansion_stats["expansion_coverage"] = expansion_stats["concepts_expanded"] / expansion_stats["total_concepts"]
    expansion_stats["avg_kg_expansions"] = expansion_stats["total_kg_expansions"] / len(core_concepts)

    return {
        "strategy": "knowledge_graph",
        "expansions": expanded_concepts,
        "statistics": expansion_stats,
        "processor_info": kg_processor.get_processor_info()
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
    print("A2.5.3: Advanced Knowledge Graph Expansion")
    print("="*60)

    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()

        # Extract core concepts from document structure
        core_concepts = []
        for doc in input_data.get("documents", []):
            core_concepts.extend(doc.get("core_concepts", []))

        print(f"Processing knowledge graph expansion for {len(core_concepts)} concepts...")

        # Process knowledge graph expansion
        expansion_results = process_knowledge_graph_expansion(core_concepts)

        # Display results
        stats = expansion_results["statistics"]
        processor_info = expansion_results["processor_info"]

        print(f"\nAdvanced Knowledge Graph Results:")
        print(f"  Concepts Processed: {stats['total_concepts']}")
        print(f"  Concepts Expanded: {stats['concepts_expanded']}")
        print(f"  Expansion Coverage: {stats['expansion_coverage']:.1%}")
        print(f"  Original Keywords: {stats['total_original_keywords']}")
        print(f"  Expanded Keywords: {stats['total_expanded_keywords']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")
        print(f"  Avg KG Expansions: {stats['avg_kg_expansions']:.1f}")

        print(f"\nRelationship Type Statistics:")
        for rel_type, count in stats["relationship_type_totals"].items():
            print(f"  {rel_type.replace('_', '-').title()}: {count}")

        print(f"\nKnowledge Graph Processor Info:")
        print(f"  Total Relationships: {processor_info['total_relationships']}")
        print(f"  Relationship Types: {', '.join(processor_info['relationship_types'])}")
        print(f"  Learning Enabled: {processor_info['learning_enabled']}")

        # Show sample expansions
        print(f"\nSample Knowledge Graph Expansions:")
        for i, concept in enumerate(expansion_results["expansions"][:3], 1):
            metadata = concept["expansion_metadata"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Expansion: {metadata['original_keyword_count']} -> {metadata['expanded_keyword_count']} keywords")
            print(f"     Ratio: {metadata['expansion_ratio']:.2f}")
            print(f"     KG Expansions: {metadata['total_kg_expansions']}")
            print(f"     Relationships: {metadata['relationship_expansions']} explicit, {metadata['implicit_relationships']} implicit")

        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "knowledge_graph",
            "results": expansion_results,
            "processing_timestamp": datetime.now().isoformat()
        }

        output_path = Path(__file__).parent.parent / "outputs/A2.5.3_knowledge_graph_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.3 Advanced Knowledge Graph Expansion completed successfully!")

    except Exception as e:
        print(f"Error in A2.5.3: {str(e)}")
        raise

if __name__ == "__main__":
    main()