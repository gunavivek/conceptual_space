#!/usr/bin/env python3
"""
A2.5.4: Advanced Domain Ontology Expansion
Uses domain-specific ontologies and automatic vocabulary learning
Implements industry standard ontologies and adaptive domain vocabulary expansion
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add expansion_modules to path
sys.path.append(str(Path(__file__).parent.parent / "expansion_modules"))

try:
    from domain_ontology import DomainOntologyManager
except ImportError:
    print("[ERROR] Could not import DomainOntologyManager. Check expansion_modules installation.")
    sys.exit(1)

def expand_concept_with_domain_ontology(concept, domain_manager, max_expansions=5):
    """
    Expand a concept using domain ontology and learned vocabulary

    Args:
        concept: Target concept to expand
        domain_manager: DomainOntologyManager instance
        max_expansions: Maximum expansion terms

    Returns:
        dict: Expanded concept with domain-specific terms
    """
    # Get domain-specific expansions
    domain_expansions = domain_manager.expand_concept_with_domain_ontology(
        concept, max_expansions
    )

    # Add expansion terms to keywords
    original_keywords = concept.get("keywords", [])
    expanded_keywords = original_keywords.copy()

    expansion_sources = {
        "predefined_ontology": 0,
        "learned_vocabulary": 0,
        "learned_cooccurrence": 0
    }

    for expansion in domain_expansions:
        expanded_keywords.append(expansion["term"])
        source = expansion.get("source", "unknown")
        if source in expansion_sources:
            expansion_sources[source] += 1

    # Create expanded concept
    expanded_concept = concept.copy()
    expanded_concept["keywords"] = expanded_keywords
    expanded_concept["expansion_metadata"] = {
        "strategy": "domain_ontology",
        "original_keyword_count": len(original_keywords),
        "expanded_keyword_count": len(expanded_keywords),
        "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
        "domain_expansions": len(domain_expansions),
        "expansion_sources": expansion_sources,
        "expansion_terms": domain_expansions,
        "domain_manager_info": domain_manager.get_ontology_info()
    }

    return expanded_concept

def process_domain_ontology_expansion(core_concepts, domain="general"):
    """
    Process advanced domain ontology expansion for all concepts

    Args:
        core_concepts: List of core concepts from A2.4
        domain: Target domain for ontology

    Returns:
        dict: Domain ontology expansion results
    """
    # Initialize domain ontology manager
    print(f"  Initializing domain ontology manager (domain: {domain})...")
    domain_manager = DomainOntologyManager(domain=domain, auto_learn=True, min_frequency=2)

    # Learn domain vocabulary from concepts
    print("  Learning domain vocabulary from concepts...")
    learning_stats = domain_manager.learn_domain_vocabulary(core_concepts)

    expanded_concepts = []
    expansion_stats = {
        "total_concepts": len(core_concepts),
        "concepts_expanded": 0,
        "total_original_keywords": 0,
        "total_expanded_keywords": 0,
        "expansion_ratios": [],
        "expansion_source_totals": {
            "predefined_ontology": 0,
            "learned_vocabulary": 0,
            "learned_cooccurrence": 0
        },
        "total_domain_expansions": 0,
        "learning_statistics": learning_stats
    }

    print("  Computing domain ontology expansions...")
    for concept in core_concepts:
        # Expand the concept
        expanded_concept = expand_concept_with_domain_ontology(
            concept, domain_manager
        )
        expanded_concepts.append(expanded_concept)

        # Update statistics
        metadata = expanded_concept["expansion_metadata"]
        expansion_stats["total_original_keywords"] += metadata["original_keyword_count"]
        expansion_stats["total_expanded_keywords"] += metadata["expanded_keyword_count"]
        expansion_stats["expansion_ratios"].append(metadata["expansion_ratio"])
        expansion_stats["total_domain_expansions"] += metadata["domain_expansions"]

        # Update expansion source statistics
        for source, count in metadata["expansion_sources"].items():
            if source in expansion_stats["expansion_source_totals"]:
                expansion_stats["expansion_source_totals"][source] += count

        if metadata["expansion_ratio"] > 1.0:
            expansion_stats["concepts_expanded"] += 1

    # Calculate overall statistics
    expansion_stats["average_expansion_ratio"] = sum(expansion_stats["expansion_ratios"]) / len(expansion_stats["expansion_ratios"]) if expansion_stats["expansion_ratios"] else 0
    expansion_stats["expansion_coverage"] = expansion_stats["concepts_expanded"] / expansion_stats["total_concepts"]
    expansion_stats["avg_domain_expansions"] = expansion_stats["total_domain_expansions"] / len(core_concepts)

    return {
        "strategy": "domain_ontology",
        "domain": domain,
        "expansions": expanded_concepts,
        "statistics": expansion_stats,
        "manager_info": domain_manager.get_ontology_info()
    }

def detect_likely_domain(core_concepts):
    """
    Detect the most likely domain from concept analysis

    Args:
        core_concepts: List of core concepts

    Returns:
        str: Most likely domain
    """
    domain_indicators = {
        "technical": ["algorithm", "system", "data", "computer", "software", "code", "programming"],
        "business": ["strategy", "management", "revenue", "profit", "analysis", "market", "customer"],
        "academic": ["research", "theory", "method", "study", "analysis", "framework", "approach"],
        "medical": ["patient", "treatment", "diagnosis", "medical", "health", "clinical", "therapy"],
        "legal": ["law", "legal", "court", "contract", "regulation", "compliance", "policy"]
    }

    domain_scores = {domain: 0 for domain in domain_indicators}

    for concept in core_concepts:
        keywords = concept.get("keywords", [])
        canonical_name = concept.get("canonical_name", "")
        all_terms = keywords + [canonical_name] if canonical_name else keywords

        for term in all_terms:
            term_lower = term.lower()
            for domain, indicators in domain_indicators.items():
                for indicator in indicators:
                    if indicator in term_lower:
                        domain_scores[domain] += 1

    # Return domain with highest score, fallback to "general"
    best_domain = max(domain_scores, key=domain_scores.get)
    return best_domain if domain_scores[best_domain] > 0 else "general"

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
    print("A2.5.4: Advanced Domain Ontology Expansion")
    print("="*60)

    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()

        # Extract core concepts from document structure
        core_concepts = []
        for doc in input_data.get("documents", []):
            core_concepts.extend(doc.get("core_concepts", []))

        print(f"Processing domain ontology expansion for {len(core_concepts)} concepts...")

        # Detect likely domain
        detected_domain = detect_likely_domain(core_concepts)
        print(f"  Detected likely domain: {detected_domain}")

        # Process domain ontology expansion
        expansion_results = process_domain_ontology_expansion(core_concepts, detected_domain)

        # Display results
        stats = expansion_results["statistics"]
        manager_info = expansion_results["manager_info"]
        learning_stats = stats["learning_statistics"]

        print(f"\nAdvanced Domain Ontology Results:")
        print(f"  Domain: {expansion_results['domain']}")
        print(f"  Concepts Processed: {stats['total_concepts']}")
        print(f"  Concepts Expanded: {stats['concepts_expanded']}")
        print(f"  Expansion Coverage: {stats['expansion_coverage']:.1%}")
        print(f"  Original Keywords: {stats['total_original_keywords']}")
        print(f"  Expanded Keywords: {stats['total_expanded_keywords']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")
        print(f"  Avg Domain Expansions: {stats['avg_domain_expansions']:.1f}")

        print(f"\nExpansion Source Statistics:")
        for source, count in stats["expansion_source_totals"].items():
            print(f"  {source.replace('_', ' ').title()}: {count}")

        print(f"\nDomain Learning Statistics:")
        print(f"  Learned Terms: {learning_stats['learned_terms_count']}")
        print(f"  Total Frequency: {learning_stats['total_term_frequency']}")
        print(f"  Likely Domain: {learning_stats['likely_domain']}")
        print(f"  Domain Indicators: {learning_stats['domain_indicators']}")

        print(f"\nDomain Manager Info:")
        print(f"  Auto Learning: {manager_info['auto_learn']}")
        print(f"  Min Frequency: {manager_info['min_frequency']}")
        print(f"  Predefined Ontology Size: {manager_info['predefined_ontology_size']}")
        print(f"  Learned Vocabulary Size: {manager_info['learned_vocabulary_size']}")
        print(f"  Cache Sizes: {manager_info['term_frequency_cache']} terms, {manager_info['cooccurrence_cache']} cooccurrences")

        # Show sample expansions
        print(f"\nSample Domain Ontology Expansions:")
        for i, concept in enumerate(expansion_results["expansions"][:3], 1):
            metadata = concept["expansion_metadata"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Expansion: {metadata['original_keyword_count']} -> {metadata['expanded_keyword_count']} keywords")
            print(f"     Ratio: {metadata['expansion_ratio']:.2f}")
            print(f"     Domain Expansions: {metadata['domain_expansions']}")
            print(f"     Sources: {metadata['expansion_sources']}")

        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "domain_ontology",
            "results": expansion_results,
            "processing_timestamp": datetime.now().isoformat()
        }

        output_path = Path(__file__).parent.parent / "outputs/A2.5.4_domain_ontology_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.4 Advanced Domain Ontology Expansion completed successfully!")

    except Exception as e:
        print(f"Error in A2.5.4: {str(e)}")
        raise

if __name__ == "__main__":
    main()