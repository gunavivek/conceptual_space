#!/usr/bin/env python3
"""
A2.5.2: Domain Knowledge Concept Generation Strategy
Generates NEW concept entities using domain-specific ontologies and knowledge patterns
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Domain-specific concept relationships
DOMAIN_ONTOLOGIES = {
    "finance": {
        "revenue": ["income", "sales", "earnings", "turnover", "receipts"],
        "cost": ["expense", "expenditure", "outflow", "spending", "charges"],
        "profit": ["earnings", "income", "margin", "return", "gain"],
        "assets": ["holdings", "resources", "capital", "property", "investments"],
        "liability": ["debt", "obligation", "payable", "owing", "commitments"],
        "equity": ["ownership", "capital", "shares", "stock", "net_worth"],
        "cash": ["money", "liquidity", "funds", "currency", "cash_flow"],
        "investment": ["portfolio", "securities", "bonds", "stocks", "mutual_funds"],
        "balance": ["equilibrium", "statement", "account", "ledger", "books"],
        "contract": ["agreement", "deal", "terms", "obligations", "provisions"]
    },
    "operations": {
        "process": ["workflow", "procedure", "method", "operation", "system"],
        "management": ["administration", "control", "oversight", "leadership", "governance"],
        "efficiency": ["productivity", "optimization", "performance", "effectiveness", "throughput"],
        "quality": ["standards", "excellence", "control", "assurance", "improvement"],
        "inventory": ["stock", "supplies", "materials", "goods", "products"],
        "discontinued": ["terminated", "ended", "ceased", "stopped", "abandoned"],
        "valuation": ["assessment", "appraisal", "evaluation", "pricing", "worth"]
    },
    "accounting": {
        "depreciation": ["amortization", "write_down", "allocation", "expense", "reduction"],
        "receivable": ["outstanding", "due", "collectible", "accounts", "invoiced"],
        "deferred": ["postponed", "delayed", "accrued", "unearned", "prepaid"],
        "tax": ["levy", "duty", "assessment", "obligation", "liability"],
        "audit": ["examination", "review", "verification", "inspection", "compliance"],
        "journal": ["record", "entry", "transaction", "posting", "documentation"]
    },
    "general": {
        "change": ["modification", "alteration", "adjustment", "variation", "shift"],
        "increase": ["growth", "rise", "expansion", "improvement", "enhancement"],
        "decrease": ["reduction", "decline", "fall", "drop", "contraction"],
        "analysis": ["study", "examination", "review", "assessment", "evaluation"],
        "report": ["document", "summary", "statement", "account", "disclosure"]
    }
}

def generate_domain_specific_concepts(seed_concept, expansion_id_base):
    """
    Generate domain-specific concept entities from seed concept
    
    Args:
        seed_concept: Seed concept to expand from
        expansion_id_base: Base ID for new concepts
        
    Returns:
        list: List of newly generated domain-specific concepts
    """
    new_concepts = []
    seed_domain = seed_concept.get("domain", "general")
    seed_keywords = set(seed_concept.get("primary_keywords", []))
    
    # Strategy 1: Generate specialized domain subconcepts
    domain_ontology = DOMAIN_ONTOLOGIES.get(seed_domain, DOMAIN_ONTOLOGIES["general"])
    
    for ontology_key, ontology_terms in domain_ontology.items():
        # Check if seed keywords relate to this ontology concept
        keyword_overlap = any(keyword.lower() in ontology_key or ontology_key in keyword.lower() 
                            for keyword in seed_keywords)
        term_overlap = any(term in " ".join(seed_keywords).lower() for term in ontology_terms)
        
        if keyword_overlap or term_overlap:
            # Create specialized domain concept
            specialized_concept = {
                "concept_id": f"{expansion_id_base}_domain_{ontology_key}",
                "canonical_name": f"{seed_concept.get('canonical_name', 'Unknown')}_{ontology_key.title()}",
                "primary_keywords": ontology_terms,
                "domain": f"{seed_domain}_specialized",
                "related_documents": seed_concept.get("related_documents", []),
                "generation_method": "domain_specialization",
                "seed_concept_id": seed_concept.get("concept_id"),
                "ontology_source": ontology_key,
                "domain_depth": "specialized"
            }
            new_concepts.append(specialized_concept)
    
    # Strategy 2: Generate cross-domain bridge concepts
    for other_domain, other_ontology in DOMAIN_ONTOLOGIES.items():
        if other_domain == seed_domain:
            continue
            
        # Find conceptual bridges between domains
        bridge_terms = []
        bridge_connections = []
        
        for seed_keyword in seed_keywords:
            for ontology_key, ontology_terms in other_ontology.items():
                # Check for conceptual similarity
                if (seed_keyword.lower() in ontology_key or 
                    any(term in seed_keyword.lower() for term in ontology_terms)):
                    bridge_terms.extend(ontology_terms[:3])  # Top 3 terms
                    bridge_connections.append(f"{seed_keyword}→{ontology_key}")
        
        if len(bridge_terms) >= 3:  # Sufficient bridge content
            bridge_concept = {
                "concept_id": f"{expansion_id_base}_bridge_{other_domain}",
                "canonical_name": f"{seed_concept.get('canonical_name', 'Unknown')}_Bridge_{other_domain.title()}",
                "primary_keywords": list(set(bridge_terms)),
                "domain": "cross_domain",
                "related_documents": seed_concept.get("related_documents", []),
                "generation_method": "cross_domain_bridge",
                "seed_concept_id": seed_concept.get("concept_id"),
                "source_domain": seed_domain,
                "target_domain": other_domain,
                "bridge_connections": bridge_connections
            }
            new_concepts.append(bridge_concept)
    
    # Strategy 3: Generate hierarchical subconcepts
    # Create more granular concepts from seed terms
    if len(seed_keywords) >= 3:
        for i, keyword in enumerate(list(seed_keywords)[:3]):
            # Find domain expansions for this specific keyword
            related_terms = []
            for ontology_key, ontology_terms in domain_ontology.items():
                if keyword.lower() in ontology_key or ontology_key in keyword.lower():
                    related_terms.extend(ontology_terms)
            
            if len(related_terms) >= 2:
                hierarchical_concept = {
                    "concept_id": f"{expansion_id_base}_hier_{i+1}",
                    "canonical_name": f"{keyword.title()}_Specialized",
                    "primary_keywords": list(set(related_terms)),
                    "domain": seed_domain,
                    "related_documents": seed_concept.get("related_documents", []),
                    "generation_method": "hierarchical_specialization",
                    "seed_concept_id": seed_concept.get("concept_id"),
                    "parent_keyword": keyword,
                    "hierarchy_level": "subconcept"
                }
                new_concepts.append(hierarchical_concept)
    
    return new_concepts

def process_domain_concept_generation(core_concepts):
    """
    Generate new concept entities using domain knowledge from all seed concepts
    
    Args:
        core_concepts: List of A2.4 seed concepts
        
    Returns:
        dict: Domain concept generation results
    """
    all_new_concepts = []
    generation_log = []
    
    # Generate new concepts from each seed concept
    for i, seed_concept in enumerate(core_concepts):
        seed_id = seed_concept.get("concept_id", f"seed_{i}")
        expansion_id_base = f"a252_{seed_id}"
        
        new_concepts = generate_domain_specific_concepts(seed_concept, expansion_id_base)
        all_new_concepts.extend(new_concepts)
        
        generation_log.append({
            "seed_concept_id": seed_id,
            "seed_canonical_name": seed_concept.get("canonical_name", "Unknown"),
            "seed_domain": seed_concept.get("domain", "general"),
            "concepts_generated": len(new_concepts),
            "generation_methods": list(set(c["generation_method"] for c in new_concepts))
        })
    
    # Calculate strategy statistics
    total_generated = len(all_new_concepts)
    avg_per_seed = total_generated / len(core_concepts) if core_concepts else 0
    
    # Analyze generation methods and domains
    method_counts = {}
    domain_counts = {}
    for concept in all_new_concepts:
        method = concept["generation_method"]
        method_counts[method] = method_counts.get(method, 0) + 1
        
        domain = concept.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    return {
        "strategy": "domain_knowledge_generation",
        "generated_concepts": all_new_concepts,
        "generation_log": generation_log,
        "statistics": {
            "seed_concepts_processed": len(core_concepts),
            "total_concepts_generated": total_generated,
            "average_concepts_per_seed": avg_per_seed,
            "generation_methods": method_counts,
            "domain_distribution": domain_counts,
            "cross_domain_bridges": len([c for c in all_new_concepts if c["generation_method"] == "cross_domain_bridge"])
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
    print("A2.5.2: Domain Knowledge Concept Generation Strategy")
    print("="*60)
    
    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()
        core_concepts = input_data.get("core_concepts", [])
        
        # Generate new concepts using domain knowledge
        print(f"Generating domain-specific concepts from {len(core_concepts)} seed concepts...")
        generation_results = process_domain_concept_generation(core_concepts)
        
        # Display results
        stats = generation_results["statistics"]
        print(f"\nDomain Knowledge Concept Generation Results:")
        print(f"  Seed Concepts: {stats['seed_concepts_processed']}")
        print(f"  New Concepts Generated: {stats['total_concepts_generated']}")
        print(f"  Average per Seed: {stats['average_concepts_per_seed']:.1f}")
        print(f"  Cross-Domain Bridges: {stats['cross_domain_bridges']}")
        
        print(f"\nGeneration Methods:")
        for method, count in stats["generation_methods"].items():
            print(f"  {method}: {count} concepts")
        
        print(f"\nDomain Distribution:")
        for domain, count in stats["domain_distribution"].items():
            print(f"  {domain}: {count} concepts")
        
        # Show sample generated concepts
        print(f"\nSample Generated Concepts:")
        for i, concept in enumerate(generation_results["generated_concepts"][:5], 1):
            print(f"  {i}. {concept['canonical_name']} ({concept['concept_id']})")
            print(f"     Method: {concept['generation_method']}")
            print(f"     Keywords: {len(concept['primary_keywords'])}")
            print(f"     Domain: {concept.get('domain', 'unknown')}")
        
        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "domain_knowledge_generation",
            "results": generation_results,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        output_path = Path(__file__).parent.parent / "outputs/A2.5.2_domain_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.2 Domain Knowledge Concept Generation completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.5.2: {str(e)}")
        raise

if __name__ == "__main__":
    main()