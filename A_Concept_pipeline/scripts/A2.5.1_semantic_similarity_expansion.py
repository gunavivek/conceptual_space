#!/usr/bin/env python3
"""
A2.5.1: Semantic Similarity Expansion Strategy
Expands concepts using semantic similarity relationships
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import math

def calculate_semantic_similarity(concept1, concept2):
    """
    Calculate semantic similarity between concepts
    
    Args:
        concept1: First concept data
        concept2: Second concept data
        
    Returns:
        float: Similarity score (0-1)
    """
    # Extract keywords for comparison
    kw1 = set(concept1.get("primary_keywords", []))
    kw2 = set(concept2.get("primary_keywords", []))
    
    # Jaccard similarity as base
    if not kw1 or not kw2:
        return 0.0
    
    intersection = len(kw1 & kw2)
    union = len(kw1 | kw2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Domain bonus for same domain
    domain1 = concept1.get("domain", "general")
    domain2 = concept2.get("domain", "general")
    domain_bonus = 0.2 if domain1 == domain2 else 0.0
    
    # Theme similarity (if available)
    theme1 = concept1.get("theme_name", "").lower()
    theme2 = concept2.get("theme_name", "").lower()
    theme_similarity = 0.1 if any(word in theme2 for word in theme1.split()) else 0.0
    
    return min(1.0, jaccard + domain_bonus + theme_similarity)

def find_similar_concepts(target_concept, all_concepts, threshold=0.4, max_similar=5):
    """
    Find concepts similar to target concept
    
    Args:
        target_concept: Target concept to expand
        all_concepts: All available concepts
        threshold: Minimum similarity threshold
        max_similar: Maximum similar concepts to return
        
    Returns:
        list: Similar concepts with scores
    """
    similarities = []
    target_id = target_concept.get("concept_id")
    
    for concept in all_concepts:
        if concept.get("concept_id") == target_id:
            continue
            
        similarity = calculate_semantic_similarity(target_concept, concept)
        if similarity >= threshold:
            similarities.append({
                "concept": concept,
                "similarity_score": similarity,
                "expansion_type": "semantic_similar"
            })
    
    # Sort by similarity and return top matches
    similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similarities[:max_similar]

def expand_concept_semantically(concept, all_concepts):
    """
    Expand a single concept using semantic similarity
    
    Args:
        concept: Concept to expand
        all_concepts: All available concepts
        
    Returns:
        dict: Expanded concept with similar concepts
    """
    similar_concepts = find_similar_concepts(concept, all_concepts)
    
    # Aggregate keywords from similar concepts
    expanded_keywords = set(concept.get("primary_keywords", []))
    related_domains = set([concept.get("domain", "general")])
    total_documents = set(concept.get("related_documents", []))
    
    for similar in similar_concepts:
        sim_concept = similar["concept"]
        weight = similar["similarity_score"]
        
        # Add weighted keywords
        sim_keywords = sim_concept.get("primary_keywords", [])
        if weight > 0.6:  # High similarity - include all keywords
            expanded_keywords.update(sim_keywords)
        else:  # Medium similarity - include top keywords only
            expanded_keywords.update(sim_keywords[:3])
        
        related_domains.add(sim_concept.get("domain", "general"))
        total_documents.update(sim_concept.get("related_documents", []))
    
    expansion_data = {
        "original_concept": concept,
        "similar_concepts": similar_concepts,
        "expanded_keywords": list(expanded_keywords),
        "related_domains": list(related_domains),
        "total_document_coverage": len(total_documents),
        "expansion_strength": len(similar_concepts),
        "semantic_coherence": sum(s["similarity_score"] for s in similar_concepts) / max(len(similar_concepts), 1)
    }
    
    return expansion_data

def process_semantic_expansion(core_concepts):
    """
    Process semantic similarity expansion for all concepts
    
    Args:
        core_concepts: List of core concepts
        
    Returns:
        dict: Semantic expansion results
    """
    expansions = []
    
    for concept in core_concepts:
        expansion = expand_concept_semantically(concept, core_concepts)
        expansions.append(expansion)
    
    # Calculate strategy statistics
    total_similarities = sum(len(exp["similar_concepts"]) for exp in expansions)
    avg_expansion = total_similarities / len(expansions) if expansions else 0
    
    return {
        "strategy": "semantic_similarity",
        "expansions": expansions,
        "statistics": {
            "concepts_processed": len(expansions),
            "total_similarities_found": total_similarities,
            "average_expansion_per_concept": avg_expansion,
            "high_coherence_concepts": len([e for e in expansions if e["semantic_coherence"] > 0.7])
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
    print("A2.5.1: Semantic Similarity Expansion Strategy")
    print("="*60)
    
    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()
        core_concepts = input_data.get("core_concepts", [])
        
        # Process semantic expansion
        print(f"Processing semantic expansion for {len(core_concepts)} concepts...")
        expansion_results = process_semantic_expansion(core_concepts)
        
        # Display results
        stats = expansion_results["statistics"]
        print(f"\nSemantic Expansion Results:")
        print(f"  Concepts Processed: {stats['concepts_processed']}")
        print(f"  Total Similarities: {stats['total_similarities_found']}")
        print(f"  Average Expansion: {stats['average_expansion_per_concept']:.1f}")
        print(f"  High Coherence: {stats['high_coherence_concepts']}")
        
        # Show sample expansions
        print(f"\nSample Expansions:")
        for i, exp in enumerate(expansion_results["expansions"][:3], 1):
            concept = exp["original_concept"]
            print(f"  {i}. {concept['theme_name']}")
            print(f"     Similar concepts: {exp['expansion_strength']}")
            print(f"     Coherence: {exp['semantic_coherence']:.3f}")
        
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