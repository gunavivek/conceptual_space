#!/usr/bin/env python3
"""
A2.5.2: Domain Knowledge Expansion Strategy
Expands concepts using domain-specific term dictionaries for Financial, Operational, Tax, and Accounting domains
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# Domain-specific term dictionaries as specified in architecture
DOMAIN_DICTIONARIES = {
    "Financial Concepts": {
        "revenue": ["sales", "income", "turnover", "earnings", "receipts"],
        "cost": ["expense", "expenditure", "outflow", "spending", "charges"],
        "profit": ["earnings", "income", "margin", "return", "gain"],
        "assets": ["holdings", "resources", "capital", "property", "investments"],
        "liability": ["debt", "obligation", "payable", "owing", "commitments"],
        "equity": ["ownership", "capital", "shares", "stock", "net_worth"],
        "cash": ["money", "liquidity", "funds", "currency", "cash_flow"],
        "investment": ["portfolio", "securities", "bonds", "stocks", "mutual_funds"],
        "balance": ["equilibrium", "statement", "account", "ledger", "books"],
        "contract": ["agreement", "deal", "terms", "obligations", "provisions"],
        "receivable": ["journal", "audit", "accrual", "ledger", "reconciliation"],
        "income": ["expense", "asset", "loss", "profit", "revenue"]
    },
    "Operational": {
        "process": ["workflow", "procedure", "method", "operation", "system"],
        "management": ["administration", "control", "oversight", "leadership", "governance"],
        "efficiency": ["productivity", "optimization", "performance", "effectiveness", "throughput"],
        "quality": ["standards", "excellence", "control", "assurance", "improvement"],
        "operations": ["activities", "functions", "processes", "procedures", "workflow"],
        "service": ["delivery", "support", "assistance", "help", "maintenance"],
        "customer": ["client", "consumer", "buyer", "user", "patron"],
        "product": ["offering", "goods", "merchandise", "item", "commodity"]
    },
    "Tax": {
        "tax": ["levy", "duty", "assessment", "charge", "impost"],
        "deduction": ["allowance", "reduction", "discount", "exemption", "credit"],
        "depreciation": ["amortization", "depletion", "write_off", "decline", "reduction"],
        "liability": ["obligation", "debt", "responsibility", "burden", "duty"],
        "credit": ["allowance", "benefit", "reduction", "offset", "deduction"],
        "rate": ["percentage", "ratio", "proportion", "level", "amount"],
        "income": ["earnings", "revenue", "profit", "gain", "proceeds"]
    },
    "Accounting": {
        "journal": ["record", "entry", "log", "register", "book"],
        "ledger": ["account", "book", "register", "record", "statement"],
        "audit": ["examination", "review", "inspection", "verification", "check"],
        "accrual": ["accumulation", "buildup", "growth", "increase", "addition"],
        "reconciliation": ["matching", "balancing", "adjustment", "verification", "alignment"],
        "expense": ["cost", "expenditure", "outlay", "charge", "payment"],
        "asset": ["resource", "property", "holding", "investment", "capital"],
        "statement": ["report", "summary", "account", "record", "document"]
    }
}

def identify_concept_domain(concept):
    """
    Identify the primary domain of a concept

    Args:
        concept: Concept to analyze

    Returns:
        str: Primary domain name
    """
    # Check business_category field first
    business_category = concept.get("business_category", "")
    if business_category in DOMAIN_DICTIONARIES:
        return business_category

    # Check concept_type field
    concept_type = concept.get("concept_type", "")
    if concept_type in DOMAIN_DICTIONARIES:
        return concept_type

    # Analyze keywords to infer domain
    keywords = [kw.lower() for kw in concept.get("keywords", [])]
    domain_scores = defaultdict(int)

    for domain, domain_terms in DOMAIN_DICTIONARIES.items():
        for base_term, related_terms in domain_terms.items():
            # Check if base term or related terms appear in keywords
            all_terms = [base_term] + related_terms
            for term in all_terms:
                if any(term.lower() in keyword for keyword in keywords):
                    domain_scores[domain] += 1

    # Return domain with highest score, default to Financial Concepts
    if domain_scores:
        return max(domain_scores, key=domain_scores.get)
    return "Financial Concepts"  # Default domain

def find_domain_expansions(concept, max_expansions=5):
    """
    Find domain-specific expansion terms for a concept

    Args:
        concept: Target concept
        max_expansions: Maximum number of terms to add

    Returns:
        list: Domain expansion terms with metadata
    """
    domain = identify_concept_domain(concept)
    keywords = [kw.lower() for kw in concept.get("keywords", [])]
    expansion_candidates = []

    domain_dict = DOMAIN_DICTIONARIES.get(domain, {})

    # Find relevant domain terms
    for base_term, related_terms in domain_dict.items():
        # Check if this base term is relevant to the concept
        relevance_score = 0

        # Direct keyword match
        if any(base_term.lower() in keyword for keyword in keywords):
            relevance_score += 3

        # Related term match
        for related_term in related_terms:
            if any(related_term.lower() in keyword for keyword in keywords):
                relevance_score += 1

        # If relevant, add related terms as expansion candidates
        if relevance_score > 0:
            for related_term in related_terms:
                # Don't add terms already in keywords
                if not any(related_term.lower() in keyword for keyword in keywords):
                    expansion_candidates.append({
                        "term": related_term,
                        "base_term": base_term,
                        "domain": domain,
                        "relevance_score": relevance_score
                    })

    # Sort by relevance and select top terms
    expansion_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    return expansion_candidates[:max_expansions]

def expand_concept_with_domain_knowledge(concept, max_expansions=5):
    """
    Expand a concept using domain knowledge

    Args:
        concept: Target concept to expand
        max_expansions: Maximum number of expansion terms

    Returns:
        dict: Expanded concept with domain-specific terms
    """
    # Get domain expansions
    domain_expansions = find_domain_expansions(concept, max_expansions)

    # Add expansion terms to keywords
    original_keywords = concept.get("keywords", [])
    expanded_keywords = original_keywords.copy()

    for expansion in domain_expansions:
        expanded_keywords.append(expansion["term"])

    # Create expanded concept
    expanded_concept = concept.copy()
    expanded_concept["keywords"] = expanded_keywords
    expanded_concept["expansion_metadata"] = {
        "strategy": "domain_knowledge",
        "original_keyword_count": len(original_keywords),
        "expanded_keyword_count": len(expanded_keywords),
        "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
        "identified_domain": identify_concept_domain(concept),
        "domain_expansions": domain_expansions
    }

    return expanded_concept

def process_domain_knowledge_expansion(core_concepts):
    """
    Process domain knowledge expansion for all concepts

    Args:
        core_concepts: List of core concepts from A2.4

    Returns:
        dict: Domain knowledge expansion results
    """
    expanded_concepts = []
    expansion_stats = {
        "total_concepts": len(core_concepts),
        "concepts_expanded": 0,
        "total_original_keywords": 0,
        "total_expanded_keywords": 0,
        "expansion_ratios": [],
        "domain_distribution": Counter()
    }

    for concept in core_concepts:
        # Expand the concept
        expanded_concept = expand_concept_with_domain_knowledge(concept)
        expanded_concepts.append(expanded_concept)

        # Update statistics
        metadata = expanded_concept["expansion_metadata"]
        expansion_stats["total_original_keywords"] += metadata["original_keyword_count"]
        expansion_stats["total_expanded_keywords"] += metadata["expanded_keyword_count"]
        expansion_stats["expansion_ratios"].append(metadata["expansion_ratio"])
        expansion_stats["domain_distribution"][metadata["identified_domain"]] += 1

        if metadata["expansion_ratio"] > 1.0:
            expansion_stats["concepts_expanded"] += 1

    # Calculate overall statistics
    expansion_stats["average_expansion_ratio"] = sum(expansion_stats["expansion_ratios"]) / len(expansion_stats["expansion_ratios"]) if expansion_stats["expansion_ratios"] else 0
    expansion_stats["expansion_coverage"] = expansion_stats["concepts_expanded"] / expansion_stats["total_concepts"]

    return {
        "strategy": "domain_knowledge",
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
    print("A2.5.2: Domain Knowledge Expansion Strategy")
    print("="*60)

    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()

        # Extract core concepts from document structure
        core_concepts = []
        for doc in input_data.get("documents", []):
            core_concepts.extend(doc.get("core_concepts", []))

        print(f"Processing domain knowledge expansion for {len(core_concepts)} concepts...")

        # Process domain knowledge expansion
        expansion_results = process_domain_knowledge_expansion(core_concepts)

        # Display results
        stats = expansion_results["statistics"]
        print(f"\nDomain Knowledge Expansion Results:")
        print(f"  Concepts Processed: {stats['total_concepts']}")
        print(f"  Concepts Expanded: {stats['concepts_expanded']}")
        print(f"  Expansion Coverage: {stats['expansion_coverage']:.1%}")
        print(f"  Original Keywords: {stats['total_original_keywords']}")
        print(f"  Expanded Keywords: {stats['total_expanded_keywords']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")

        print(f"\nDomain Distribution:")
        for domain, count in stats["domain_distribution"].items():
            print(f"  {domain}: {count} concepts")

        # Show sample expansions
        print(f"\nSample Domain Expansions:")
        for i, concept in enumerate(expansion_results["expansions"][:3], 1):
            metadata = concept["expansion_metadata"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Domain: {metadata['identified_domain']}")
            print(f"     Expansion: {metadata['original_keyword_count']} -> {metadata['expanded_keyword_count']} keywords")
            print(f"     Ratio: {metadata['expansion_ratio']:.2f}")

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