#!/usr/bin/env python3
"""
A2.5.1: Semantic Similarity Concept Generation Strategy
Generates NEW concept entities from semantic neighborhoods of A2.4 core concepts
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
    
    # Theme similarity (if available) - using canonical_name from A2.4
    theme1 = concept1.get("canonical_name", "").lower()
    theme2 = concept2.get("canonical_name", "").lower()
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

def generate_semantic_neighbor_concepts(seed_concept, all_concepts, expansion_id_base):
    """
    Generate new concept entities from semantic neighborhoods of seed concept
    
    Args:
        seed_concept: Seed concept to expand from
        all_concepts: All available concepts for similarity calculation
        expansion_id_base: Base ID for new concepts
        
    Returns:
        list: List of newly generated concept entities
    """
    similar_concepts = find_similar_concepts(seed_concept, all_concepts, threshold=0.3, max_similar=8)
    
    new_concepts = []
    seed_keywords = set(seed_concept.get("primary_keywords", []))
    
    # Strategy 1: Generate concepts from high-similarity clusters
    high_sim_concepts = [s for s in similar_concepts if s["similarity_score"] > 0.6]
    if len(high_sim_concepts) >= 2:
        # Cluster high-similarity concepts into new concept entity
        clustered_keywords = set()
        clustered_docs = set(seed_concept.get("related_documents", []))
        
        for sim_data in high_sim_concepts[:3]:  # Top 3 high-similarity
            sim_concept = sim_data["concept"]
            clustered_keywords.update(sim_concept.get("primary_keywords", []))
            clustered_docs.update(sim_concept.get("related_documents", []))
        
        # Create new concept from high-similarity cluster
        new_concept = {
            "concept_id": f"{expansion_id_base}_sem_cluster",
            "canonical_name": f"{seed_concept.get('canonical_name', 'Unknown')}_Semantic_Cluster",
            "primary_keywords": list(clustered_keywords - seed_keywords),  # Exclude seed keywords
            "domain": seed_concept.get("domain", "general"),
            "related_documents": list(clustered_docs),
            "generation_method": "semantic_similarity_cluster",
            "seed_concept_id": seed_concept.get("concept_id"),
            "semantic_coherence": sum(s["similarity_score"] for s in high_sim_concepts) / len(high_sim_concepts)
        }
        new_concepts.append(new_concept)
    
    # Strategy 2: Generate concepts from individual semantic neighbors
    for i, sim_data in enumerate(similar_concepts[:4]):  # Top 4 individual neighbors
        sim_concept = sim_data["concept"]
        similarity_score = sim_data["similarity_score"]
        
        # Create intersection-based new concept
        neighbor_keywords = set(sim_concept.get("primary_keywords", []))
        intersection_keywords = seed_keywords & neighbor_keywords
        unique_neighbor_keywords = neighbor_keywords - seed_keywords
        
        if len(unique_neighbor_keywords) >= 2:  # Must have unique contribution
            new_concept = {
                "concept_id": f"{expansion_id_base}_sem_neighbor_{i+1}",
                "canonical_name": f"{seed_concept.get('canonical_name', 'Unknown')}_Neighbor_{i+1}",
                "primary_keywords": list(unique_neighbor_keywords),
                "domain": sim_concept.get("domain", seed_concept.get("domain", "general")),
                "related_documents": sim_concept.get("related_documents", []),
                "generation_method": "semantic_neighbor_extraction",
                "seed_concept_id": seed_concept.get("concept_id"),
                "neighbor_concept_id": sim_concept.get("concept_id"),
                "semantic_distance": 1.0 - similarity_score,
                "shared_keywords": list(intersection_keywords)
            }
            new_concepts.append(new_concept)
    
    # Strategy 3: Generate cross-domain bridge concepts
    cross_domain_concepts = [s for s in similar_concepts 
                           if s["concept"].get("domain") != seed_concept.get("domain") 
                           and s["similarity_score"] > 0.4]
    
    if cross_domain_concepts:
        bridge_keywords = set()
        bridge_domains = set([seed_concept.get("domain", "general")])
        bridge_docs = set()
        
        for cross_sim in cross_domain_concepts[:2]:  # Top 2 cross-domain
            cross_concept = cross_sim["concept"]
            bridge_keywords.update(cross_concept.get("primary_keywords", []))
            bridge_domains.add(cross_concept.get("domain", "general"))
            bridge_docs.update(cross_concept.get("related_documents", []))
        
        if len(bridge_keywords - seed_keywords) >= 3:  # Sufficient unique terms
            bridge_concept = {
                "concept_id": f"{expansion_id_base}_sem_bridge",
                "canonical_name": f"{seed_concept.get('canonical_name', 'Unknown')}_Cross_Domain",
                "primary_keywords": list(bridge_keywords - seed_keywords),
                "domain": "interdisciplinary",
                "related_documents": list(bridge_docs),
                "generation_method": "cross_domain_bridge",
                "seed_concept_id": seed_concept.get("concept_id"),
                "bridge_domains": list(bridge_domains)
            }
            new_concepts.append(bridge_concept)
    
    return new_concepts

def process_semantic_concept_generation(core_concepts):
    """
    Generate new concept entities using semantic similarity from all seed concepts
    
    Args:
        core_concepts: List of A2.4 seed concepts
        
    Returns:
        dict: Semantic concept generation results
    """
    all_new_concepts = []
    generation_log = []
    
    # Generate new concepts from each seed concept
    for i, seed_concept in enumerate(core_concepts):
        seed_id = seed_concept.get("concept_id", f"seed_{i}")
        expansion_id_base = f"a251_{seed_id}"
        
        new_concepts = generate_semantic_neighbor_concepts(seed_concept, core_concepts, expansion_id_base)
        all_new_concepts.extend(new_concepts)
        
        generation_log.append({
            "seed_concept_id": seed_id,
            "seed_canonical_name": seed_concept.get("canonical_name", "Unknown"),
            "concepts_generated": len(new_concepts),
            "generation_methods": list(set(c["generation_method"] for c in new_concepts))
        })
    
    # Calculate strategy statistics
    total_generated = len(all_new_concepts)
    avg_per_seed = total_generated / len(core_concepts) if core_concepts else 0
    
    # Analyze generation methods
    method_counts = {}
    for concept in all_new_concepts:
        method = concept["generation_method"]
        method_counts[method] = method_counts.get(method, 0) + 1
    
    return {
        "strategy": "semantic_similarity_generation",
        "generated_concepts": all_new_concepts,
        "generation_log": generation_log,
        "statistics": {
            "seed_concepts_processed": len(core_concepts),
            "total_concepts_generated": total_generated,
            "average_concepts_per_seed": avg_per_seed,
            "generation_methods": method_counts,
            "unique_domains": len(set(c.get("domain", "unknown") for c in all_new_concepts))
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
        
        # Generate new concepts using semantic similarity
        print(f"Generating new concepts from {len(core_concepts)} seed concepts...")
        generation_results = process_semantic_concept_generation(core_concepts)
        
        # Display results
        stats = generation_results["statistics"]
        print(f"\nSemantic Concept Generation Results:")
        print(f"  Seed Concepts: {stats['seed_concepts_processed']}")
        print(f"  New Concepts Generated: {stats['total_concepts_generated']}")
        print(f"  Average per Seed: {stats['average_concepts_per_seed']:.1f}")
        print(f"  Unique Domains: {stats['unique_domains']}")
        
        print(f"\nGeneration Methods:")
        for method, count in stats["generation_methods"].items():
            print(f"  {method}: {count} concepts")
        
        # Show sample generated concepts
        print(f"\nSample Generated Concepts:")
        for i, concept in enumerate(generation_results["generated_concepts"][:5], 1):
            print(f"  {i}. {concept['canonical_name']} ({concept['concept_id']})")
            print(f"     Method: {concept['generation_method']}")
            print(f"     Keywords: {len(concept['primary_keywords'])}")
            print(f"     Domain: {concept.get('domain', 'unknown')}")
        
        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "semantic_similarity_generation",
            "results": generation_results,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        output_path = Path(__file__).parent.parent / "outputs/A2.5.1_semantic_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.1 Semantic Concept Generation completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.5.1: {str(e)}")
        raise

if __name__ == "__main__":
    main()