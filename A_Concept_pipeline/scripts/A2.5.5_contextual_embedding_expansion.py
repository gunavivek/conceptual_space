#!/usr/bin/env python3
"""
A2.5.5: Contextual Embedding Expansion Strategy
Expands concepts using contextual embeddings and vector similarity
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

# Simple contextual patterns for concept expansion
CONTEXTUAL_PATTERNS = {
    "financial": {
        "patterns": [
            r"(\w+)\s+(?:increased|decreased|changed)\s+by",
            r"(?:the|a)\s+(\w+)\s+of\s+\$",
            r"(\w+)\s+(?:revenue|income|cost|expense)",
            r"(?:total|net|gross)\s+(\w+)",
            r"(\w+)\s+(?:margin|ratio|percentage)"
        ],
        "expansion_terms": {
            "revenue": ["sales", "income", "turnover", "earnings"],
            "cost": ["expense", "expenditure", "outlay"],
            "profit": ["earnings", "margin", "return", "gain"],
            "loss": ["deficit", "shortfall", "decline"],
            "growth": ["increase", "expansion", "rise", "improvement"]
        }
    },
    "healthcare": {
        "patterns": [
            r"(\w+)\s+(?:treatment|therapy|procedure)",
            r"(\w+)\s+(?:patient|individual|case)",
            r"(?:diagnosis|condition)\s+of\s+(\w+)",
            r"(\w+)\s+(?:symptom|sign|indication)"
        ],
        "expansion_terms": {
            "patient": ["individual", "subject", "case", "person"],
            "treatment": ["therapy", "intervention", "care", "procedure"],
            "diagnosis": ["condition", "disease", "disorder", "illness"],
            "outcome": ["result", "effect", "prognosis", "consequence"]
        }
    },
    "general": {
        "patterns": [
            r"(\w+)\s+(?:analysis|study|review)",
            r"(\w+)\s+(?:data|information|record)",
            r"(?:the|this)\s+(\w+)\s+(?:shows|indicates|reveals)"
        ],
        "expansion_terms": {
            "analysis": ["study", "examination", "review", "assessment"],
            "data": ["information", "records", "content", "dataset"],
            "report": ["document", "summary", "statement", "account"],
            "change": ["modification", "alteration", "adjustment", "variation"]
        }
    }
}

def extract_contextual_features(text, domain="general"):
    """
    Extract contextual features from text
    
    Args:
        text: Text to analyze
        domain: Domain context
        
    Returns:
        dict: Contextual features
    """
    domain_patterns = CONTEXTUAL_PATTERNS.get(domain.lower(), CONTEXTUAL_PATTERNS["general"])
    
    extracted_terms = []
    context_matches = []
    
    # Apply patterns to extract contextual terms
    for pattern in domain_patterns["patterns"]:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            if isinstance(match, tuple):
                extracted_terms.extend(match)
            else:
                extracted_terms.append(match)
            context_matches.append({
                "pattern": pattern,
                "match": match,
                "context": "pattern_based"
            })
    
    # Filter out common stop words and short terms
    stop_words = {"the", "a", "an", "is", "was", "are", "were", "of", "in", "to", "for", "with"}
    filtered_terms = [term for term in extracted_terms if len(term) > 2 and term not in stop_words]
    
    return {
        "extracted_terms": list(set(filtered_terms)),
        "context_matches": context_matches,
        "term_count": len(filtered_terms)
    }

def simulate_embedding_similarity(term1, term2, domain="general"):
    """
    Simulate embedding-based similarity (placeholder for actual embeddings)
    
    Args:
        term1: First term
        term2: Second term
        domain: Domain context
        
    Returns:
        float: Simulated similarity score
    """
    # Simple similarity based on string patterns and domain knowledge
    term1_lower = term1.lower()
    term2_lower = term2.lower()
    
    # Exact match
    if term1_lower == term2_lower:
        return 1.0
    
    # Check domain expansion terms
    domain_expansions = CONTEXTUAL_PATTERNS.get(domain.lower(), {}).get("expansion_terms", {})
    
    for base_term, expansions in domain_expansions.items():
        if term1_lower == base_term and term2_lower in [e.lower() for e in expansions]:
            return 0.8
        if term2_lower == base_term and term1_lower in [e.lower() for e in expansions]:
            return 0.8
        if term1_lower in [e.lower() for e in expansions] and term2_lower in [e.lower() for e in expansions]:
            return 0.7
    
    # String similarity (simple)
    if term1_lower in term2_lower or term2_lower in term1_lower:
        return 0.6
    
    # Common prefixes/suffixes
    if len(term1_lower) > 3 and len(term2_lower) > 3:
        if term1_lower[:3] == term2_lower[:3] or term1_lower[-3:] == term2_lower[-3:]:
            return 0.4
    
    return 0.1  # Base similarity

def find_contextually_similar_terms(target_terms, all_available_terms, domain="general", threshold=0.5):
    """
    Find contextually similar terms using simulated embeddings
    
    Args:
        target_terms: Terms to find similarities for
        all_available_terms: Pool of available terms
        domain: Domain context
        threshold: Similarity threshold
        
    Returns:
        list: Similar terms with scores
    """
    similar_terms = []
    
    for target_term in target_terms:
        for available_term in all_available_terms:
            if available_term.lower() == target_term.lower():
                continue
                
            similarity = simulate_embedding_similarity(target_term, available_term, domain)
            if similarity >= threshold:
                similar_terms.append({
                    "original_term": target_term,
                    "similar_term": available_term,
                    "similarity_score": similarity,
                    "expansion_type": "contextual_embedding"
                })
    
    # Sort by similarity score
    similar_terms.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return similar_terms

def expand_concept_contextually(concept, all_concepts):
    """
    Expand concept using contextual embedding approach
    
    Args:
        concept: Target concept
        all_concepts: All available concepts
        
    Returns:
        dict: Contextual expansion
    """
    domain = concept.get("domain", "general")
    primary_keywords = concept.get("primary_keywords", [])
    
    # Collect all terms from other concepts
    all_available_terms = set()
    for other_concept in all_concepts:
        if other_concept.get("concept_id") != concept.get("concept_id"):
            all_available_terms.update(other_concept.get("primary_keywords", []))
    
    # Find contextually similar terms
    similar_terms = find_contextually_similar_terms(
        primary_keywords, 
        list(all_available_terms), 
        domain
    )
    
    # Group by original term
    expansion_by_term = defaultdict(list)
    for similar in similar_terms:
        expansion_by_term[similar["original_term"]].append(similar)
    
    # Create expanded term set
    expanded_terms = set(primary_keywords)
    context_expansions = []
    
    for original_term, expansions in expansion_by_term.items():
        # Take top 3 most similar terms per original term
        top_expansions = expansions[:3]
        for exp in top_expansions:
            expanded_terms.add(exp["similar_term"])
            context_expansions.append(exp)
    
    # Calculate expansion metrics
    original_count = len(primary_keywords)
    expanded_count = len(expanded_terms)
    expansion_ratio = expanded_count / max(original_count, 1)
    
    # Calculate average similarity
    avg_similarity = sum(exp["similarity_score"] for exp in context_expansions) / max(len(context_expansions), 1)
    
    # Determine context richness
    context_diversity = len(set(exp["original_term"] for exp in context_expansions))
    context_richness = context_diversity / max(len(primary_keywords), 1)
    
    return {
        "original_concept": concept,
        "contextual_expansions": context_expansions,
        "expanded_terms": list(expanded_terms),
        "expansion_by_term": dict(expansion_by_term),
        "expansion_metrics": {
            "original_terms": original_count,
            "expanded_terms": expanded_count,
            "expansion_ratio": expansion_ratio,
            "context_expansions": len(context_expansions),
            "average_similarity": avg_similarity,
            "context_richness": context_richness
        }
    }

def process_contextual_embedding_expansion(core_concepts):
    """
    Process contextual embedding expansion for all concepts
    
    Args:
        core_concepts: List of core concepts
        
    Returns:
        dict: Contextual expansion results
    """
    expansions = []
    
    for concept in core_concepts:
        expansion = expand_concept_contextually(concept, core_concepts)
        expansions.append(expansion)
    
    # Calculate global statistics
    total_expansions = sum(exp["expansion_metrics"]["context_expansions"] for exp in expansions)
    avg_similarity = sum(exp["expansion_metrics"]["average_similarity"] for exp in expansions) / max(len(expansions), 1)
    
    return {
        "strategy": "contextual_embedding",
        "expansions": expansions,
        "statistics": {
            "concepts_processed": len(expansions),
            "total_context_expansions": total_expansions,
            "average_expansion_ratio": sum(exp["expansion_metrics"]["expansion_ratio"] for exp in expansions) / max(len(expansions), 1),
            "average_similarity_score": avg_similarity,
            "high_richness_concepts": len([e for e in expansions if e["expansion_metrics"]["context_richness"] > 0.7]),
            "high_expansion_concepts": len([e for e in expansions if e["expansion_metrics"]["expansion_ratio"] > 1.3])
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
        core_concepts = input_data.get("core_concepts", [])
        
        # Process contextual embedding expansion
        print(f"Processing contextual embedding expansion for {len(core_concepts)} concepts...")
        expansion_results = process_contextual_embedding_expansion(core_concepts)
        
        # Display results
        stats = expansion_results["statistics"]
        print(f"\nContextual Embedding Expansion Results:")
        print(f"  Concepts Processed: {stats['concepts_processed']}")
        print(f"  Total Context Expansions: {stats['total_context_expansions']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")
        print(f"  Average Similarity: {stats['average_similarity_score']:.3f}")
        print(f"  High Richness Concepts: {stats['high_richness_concepts']}")
        print(f"  High Expansion Concepts: {stats['high_expansion_concepts']}")
        
        # Show sample expansions
        print(f"\nSample Contextual Expansions:")
        for i, exp in enumerate(expansion_results["expansions"][:3], 1):
            concept = exp["original_concept"]
            metrics = exp["expansion_metrics"]
            print(f"  {i}. {concept['theme_name']} ({concept['domain']})")
            print(f"     Expansion: {metrics['original_terms']} -> {metrics['expanded_terms']} terms")
            print(f"     Context richness: {metrics['context_richness']:.3f}")
            if exp["contextual_expansions"]:
                sample_exp = exp["contextual_expansions"][0]
                print(f"     Example: '{sample_exp['original_term']}' -> '{sample_exp['similar_term']}' ({sample_exp['similarity_score']:.3f})")
        
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