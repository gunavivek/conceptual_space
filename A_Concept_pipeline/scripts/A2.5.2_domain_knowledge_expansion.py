#!/usr/bin/env python3
"""
A2.5.2: Domain Knowledge Expansion Strategy
Expands concepts using domain-specific knowledge and ontologies
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Domain-specific concept relationships
DOMAIN_ONTOLOGIES = {
    "finance": {
        "revenue": ["income", "sales", "earnings", "turnover"],
        "cost": ["expense", "expenditure", "outflow", "spending"],
        "profit": ["earnings", "income", "margin", "return"],
        "assets": ["holdings", "resources", "capital", "property"],
        "liability": ["debt", "obligation", "payable", "owing"],
        "equity": ["ownership", "capital", "shares", "stock"],
        "cash": ["money", "liquidity", "funds", "currency"],
        "investment": ["portfolio", "securities", "bonds", "stocks"]
    },
    "healthcare": {
        "patient": ["individual", "person", "subject", "case"],
        "treatment": ["therapy", "intervention", "care", "procedure"],
        "diagnosis": ["condition", "disease", "disorder", "illness"],
        "medication": ["drug", "pharmaceutical", "prescription", "medicine"],
        "symptom": ["sign", "indication", "manifestation", "feature"],
        "outcome": ["result", "effect", "consequence", "prognosis"]
    },
    "technology": {
        "system": ["platform", "framework", "infrastructure", "architecture"],
        "data": ["information", "dataset", "records", "content"],
        "process": ["workflow", "procedure", "method", "operation"],
        "interface": ["ui", "api", "connection", "interaction"],
        "security": ["protection", "safety", "privacy", "encryption"],
        "performance": ["speed", "efficiency", "optimization", "throughput"]
    },
    "general": {
        "change": ["modification", "alteration", "adjustment", "variation"],
        "increase": ["growth", "rise", "expansion", "improvement"],
        "decrease": ["reduction", "decline", "fall", "drop"],
        "analysis": ["study", "examination", "review", "assessment"],
        "report": ["document", "summary", "statement", "account"]
    }
}

def get_domain_expansions(keyword, domain):
    """
    Get domain-specific expansions for a keyword
    
    Args:
        keyword: Keyword to expand
        domain: Domain context
        
    Returns:
        list: Related terms from domain ontology
    """
    domain_dict = DOMAIN_ONTOLOGIES.get(domain.lower(), {})
    
    # Direct lookup
    if keyword.lower() in domain_dict:
        return domain_dict[keyword.lower()]
    
    # Fuzzy matching - check if keyword is in any value list
    expansions = []
    for key, values in domain_dict.items():
        if keyword.lower() in values or any(keyword.lower() in v for v in values):
            expansions.extend(values)
            expansions.append(key)
    
    # Fallback to general domain if specific domain has no matches
    if not expansions and domain != "general":
        return get_domain_expansions(keyword, "general")
    
    return list(set(expansions))

def expand_concept_domain_knowledge(concept):
    """
    Expand a concept using domain knowledge
    
    Args:
        concept: Concept to expand
        
    Returns:
        dict: Domain-expanded concept
    """
    domain = concept.get("domain", "general")
    primary_keywords = concept.get("primary_keywords", [])
    
    domain_expansions = {}
    all_expanded_terms = set(primary_keywords)
    
    for keyword in primary_keywords:
        expansions = get_domain_expansions(keyword, domain)
        if expansions:
            domain_expansions[keyword] = expansions
            all_expanded_terms.update(expansions)
    
    # Calculate expansion metrics
    original_count = len(primary_keywords)
    expanded_count = len(all_expanded_terms)
    expansion_ratio = expanded_count / max(original_count, 1)
    
    # Determine expansion quality
    domain_coverage = len([k for k in primary_keywords if k.lower() in DOMAIN_ONTOLOGIES.get(domain, {})])
    coverage_ratio = domain_coverage / max(len(primary_keywords), 1)
    
    return {
        "original_concept": concept,
        "domain_expansions": domain_expansions,
        "expanded_terms": list(all_expanded_terms),
        "expansion_metrics": {
            "original_terms": original_count,
            "expanded_terms": expanded_count,
            "expansion_ratio": expansion_ratio,
            "domain_coverage": coverage_ratio,
            "expansion_strength": len(domain_expansions)
        }
    }

def identify_cross_domain_connections(concepts):
    """
    Identify connections between concepts across domains
    
    Args:
        concepts: List of concepts
        
    Returns:
        dict: Cross-domain connections
    """
    domain_groups = defaultdict(list)
    
    # Group concepts by domain
    for concept in concepts:
        domain = concept.get("domain", "general")
        domain_groups[domain].append(concept)
    
    connections = []
    
    # Find cross-domain keyword overlaps
    for domain1, concepts1 in domain_groups.items():
        for domain2, concepts2 in domain_groups.items():
            if domain1 >= domain2:  # Avoid duplicates
                continue
                
            for c1 in concepts1:
                for c2 in concepts2:
                    kw1 = set(c1.get("primary_keywords", []))
                    kw2 = set(c2.get("primary_keywords", []))
                    overlap = kw1 & kw2
                    
                    if len(overlap) > 0:
                        connections.append({
                            "concept1": c1["concept_id"],
                            "concept2": c2["concept_id"],
                            "domain1": domain1,
                            "domain2": domain2,
                            "shared_keywords": list(overlap),
                            "connection_strength": len(overlap)
                        })
    
    return connections

def process_domain_knowledge_expansion(core_concepts):
    """
    Process domain knowledge expansion for all concepts
    
    Args:
        core_concepts: List of core concepts
        
    Returns:
        dict: Domain expansion results
    """
    expansions = []
    
    for concept in core_concepts:
        expansion = expand_concept_domain_knowledge(concept)
        expansions.append(expansion)
    
    # Find cross-domain connections
    cross_connections = identify_cross_domain_connections(core_concepts)
    
    # Calculate strategy statistics
    total_expanded_terms = sum(exp["expansion_metrics"]["expanded_terms"] for exp in expansions)
    total_original_terms = sum(exp["expansion_metrics"]["original_terms"] for exp in expansions)
    
    return {
        "strategy": "domain_knowledge",
        "expansions": expansions,
        "cross_domain_connections": cross_connections,
        "statistics": {
            "concepts_processed": len(expansions),
            "total_original_terms": total_original_terms,
            "total_expanded_terms": total_expanded_terms,
            "average_expansion_ratio": total_expanded_terms / max(total_original_terms, 1),
            "cross_connections_found": len(cross_connections),
            "high_expansion_concepts": len([e for e in expansions if e["expansion_metrics"]["expansion_ratio"] > 2.0])
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
    print("A2.5.2: Domain Knowledge Expansion Strategy")
    print("="*60)
    
    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()
        core_concepts = input_data.get("core_concepts", [])
        
        # Process domain expansion
        print(f"Processing domain knowledge expansion for {len(core_concepts)} concepts...")
        expansion_results = process_domain_knowledge_expansion(core_concepts)
        
        # Display results
        stats = expansion_results["statistics"]
        print(f"\nDomain Knowledge Expansion Results:")
        print(f"  Concepts Processed: {stats['concepts_processed']}")
        print(f"  Original Terms: {stats['total_original_terms']}")
        print(f"  Expanded Terms: {stats['total_expanded_terms']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")
        print(f"  Cross-Domain Connections: {stats['cross_connections_found']}")
        print(f"  High Expansion Concepts: {stats['high_expansion_concepts']}")
        
        # Show sample expansions
        print(f"\nSample Domain Expansions:")
        for i, exp in enumerate(expansion_results["expansions"][:3], 1):
            concept = exp["original_concept"]
            metrics = exp["expansion_metrics"]
            print(f"  {i}. {concept['theme_name']} ({concept['domain']})")
            print(f"     Expansion: {metrics['original_terms']} -> {metrics['expanded_terms']} terms")
            print(f"     Domain Coverage: {metrics['domain_coverage']:.2f}")
        
        # Show cross-domain connections
        if expansion_results["cross_domain_connections"]:
            print(f"\nCross-Domain Connections:")
            for i, conn in enumerate(expansion_results["cross_domain_connections"][:3], 1):
                print(f"  {i}. {conn['domain1']} ↔ {conn['domain2']}")
                print(f"     Shared: {', '.join(conn['shared_keywords'])}")
        
        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "domain_knowledge",
            "results": expansion_results,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        output_path = Path(__file__).parent.parent / "outputs/A2.5.2_domain_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.2 Domain Knowledge Expansion completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.5.2: {str(e)}")
        raise

if __name__ == "__main__":
    main()