#!/usr/bin/env python3
"""
A2.5.3: Hierarchical Clustering Concept Generation Strategy
Generates NEW concept entities using hierarchical clustering of term relationships
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import math

def generate_hierarchical_concepts(seed_concept, all_concepts, expansion_id_base):
    """
    Generate hierarchical concept entities from clustering patterns
    
    Args:
        seed_concept: Seed concept to expand from
        all_concepts: All seed concepts for clustering analysis
        expansion_id_base: Base ID for new concepts
        
    Returns:
        list: List of newly generated hierarchical concepts
    """
    new_concepts = []
    seed_keywords = set(seed_concept.get("primary_keywords", []))
    
    # Strategy 1: Create parent-level concepts by clustering with similar concepts
    similar_concepts = []
    for other_concept in all_concepts:
        if other_concept.get("concept_id") == seed_concept.get("concept_id"):
            continue
        
        other_keywords = set(other_concept.get("primary_keywords", []))
        intersection = seed_keywords & other_keywords
        union = seed_keywords | other_keywords
        
        if len(intersection) > 0 and len(union) > 0:
            jaccard_sim = len(intersection) / len(union)
            if jaccard_sim > 0.2:  # Minimum similarity
                similar_concepts.append({
                    "concept": other_concept,
                    "similarity": jaccard_sim,
                    "shared_terms": list(intersection)
                })
    
    # Create parent concept from high-level clusters
    if len(similar_concepts) >= 2:
        # Find most common keywords across similar concepts
        all_keywords = list(seed_keywords)
        for sim_data in similar_concepts[:3]:  # Top 3 similar
            all_keywords.extend(sim_data["concept"].get("primary_keywords", []))
        
        keyword_freq = Counter(all_keywords)
        common_keywords = [kw for kw, freq in keyword_freq.items() if freq >= 2]
        
        if len(common_keywords) >= 3:
            parent_concept = {
                "concept_id": f"{expansion_id_base}_parent_cluster",
                "canonical_name": f"{seed_concept.get('canonical_name', 'Unknown')}_Parent_Cluster",
                "primary_keywords": common_keywords,
                "domain": seed_concept.get("domain", "general"),
                "related_documents": seed_concept.get("related_documents", []),
                "generation_method": "hierarchical_parent",
                "seed_concept_id": seed_concept.get("concept_id"),
                "cluster_size": len(similar_concepts) + 1,
                "hierarchy_level": "parent"
            }
            new_concepts.append(parent_concept)
    
    # Strategy 2: Create child concepts from keyword subclusters
    if len(seed_keywords) >= 4:
        # Group keywords by first letter/semantic similarity (simplified)
        keyword_groups = defaultdict(list)
        for keyword in seed_keywords:
            if keyword:
                # Simple grouping by first character and length
                group_key = f"{keyword[0].lower()}_{len(keyword)//3}"
                keyword_groups[group_key].append(keyword)
        
        # Create child concepts from groups with multiple keywords
        child_index = 1
        for group_key, group_keywords in keyword_groups.items():
            if len(group_keywords) >= 2:
                child_concept = {
                    "concept_id": f"{expansion_id_base}_child_{child_index}",
                    "canonical_name": f"{seed_concept.get('canonical_name', 'Unknown')}_Child_{child_index}",
                    "primary_keywords": group_keywords,
                    "domain": seed_concept.get("domain", "general"),
                    "related_documents": seed_concept.get("related_documents", []),
                    "generation_method": "hierarchical_child",
                    "seed_concept_id": seed_concept.get("concept_id"),
                    "parent_concept": seed_concept.get("canonical_name", "Unknown"),
                    "hierarchy_level": "child"
                }
                new_concepts.append(child_concept)
                child_index += 1
    
    # Strategy 3: Create sibling concepts from related domains
    seed_domain = seed_concept.get("domain", "general")
    related_concepts = [c for c in all_concepts 
                       if c.get("domain") == seed_domain 
                       and c.get("concept_id") != seed_concept.get("concept_id")]
    
    if len(related_concepts) >= 1:
        # Create sibling by combining characteristics
        sibling_keywords = []
        for related in related_concepts[:2]:  # Max 2 related concepts
            related_kw = related.get("primary_keywords", [])
            # Take keywords not in seed concept
            unique_kw = [kw for kw in related_kw if kw not in seed_keywords]
            sibling_keywords.extend(unique_kw[:3])  # Max 3 from each
        
        if len(sibling_keywords) >= 2:
            sibling_concept = {
                "concept_id": f"{expansion_id_base}_sibling",
                "canonical_name": f"{seed_concept.get('canonical_name', 'Unknown')}_Sibling",
                "primary_keywords": list(set(sibling_keywords)),
                "domain": seed_domain,
                "related_documents": seed_concept.get("related_documents", []),
                "generation_method": "hierarchical_sibling",
                "seed_concept_id": seed_concept.get("concept_id"),
                "sibling_source": [c.get("concept_id") for c in related_concepts[:2]],
                "hierarchy_level": "sibling"
            }
            new_concepts.append(sibling_concept)
    
    return new_concepts

def process_hierarchical_concept_generation(core_concepts):
    """
    Generate new concept entities using hierarchical clustering from all seed concepts
    
    Args:
        core_concepts: List of A2.4 seed concepts
        
    Returns:
        dict: Hierarchical concept generation results
    """
    all_new_concepts = []
    generation_log = []
    
    # Generate new concepts from each seed concept
    for i, seed_concept in enumerate(core_concepts):
        seed_id = seed_concept.get("concept_id", f"seed_{i}")
        expansion_id_base = f"a253_{seed_id}"
        
        new_concepts = generate_hierarchical_concepts(seed_concept, core_concepts, expansion_id_base)
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
    
    # Analyze hierarchy levels and methods
    method_counts = {}
    hierarchy_counts = {}
    for concept in all_new_concepts:
        method = concept["generation_method"]
        method_counts[method] = method_counts.get(method, 0) + 1
        
        hierarchy = concept.get("hierarchy_level", "unknown")
        hierarchy_counts[hierarchy] = hierarchy_counts.get(hierarchy, 0) + 1
    
    return {
        "strategy": "hierarchical_clustering_generation",
        "generated_concepts": all_new_concepts,
        "generation_log": generation_log,
        "statistics": {
            "seed_concepts_processed": len(core_concepts),
            "total_concepts_generated": total_generated,
            "average_concepts_per_seed": avg_per_seed,
            "generation_methods": method_counts,
            "hierarchy_levels": hierarchy_counts
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
    print("A2.5.3: Hierarchical Clustering Concept Generation Strategy")
    print("="*60)
    
    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()
        core_concepts = input_data.get("core_concepts", [])
        
        # Generate new concepts using hierarchical clustering
        print(f"Generating hierarchical concepts from {len(core_concepts)} seed concepts...")
        generation_results = process_hierarchical_concept_generation(core_concepts)
        
        # Display results
        stats = generation_results["statistics"]
        print(f"\nHierarchical Clustering Concept Generation Results:")
        print(f"  Seed Concepts: {stats['seed_concepts_processed']}")
        print(f"  New Concepts Generated: {stats['total_concepts_generated']}")
        print(f"  Average per Seed: {stats['average_concepts_per_seed']:.1f}")
        
        print(f"\nGeneration Methods:")
        for method, count in stats["generation_methods"].items():
            print(f"  {method}: {count} concepts")
        
        print(f"\nHierarchy Levels:")
        for level, count in stats["hierarchy_levels"].items():
            print(f"  {level}: {count} concepts")
        
        # Show sample generated concepts
        print(f"\nSample Generated Concepts:")
        for i, concept in enumerate(generation_results["generated_concepts"][:5], 1):
            print(f"  {i}. {concept['canonical_name']} ({concept['concept_id']})")
            print(f"     Method: {concept['generation_method']}")
            print(f"     Keywords: {len(concept['primary_keywords'])}")
            print(f"     Level: {concept.get('hierarchy_level', 'unknown')}")
        
        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "hierarchical_clustering_generation",
            "results": generation_results,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        output_path = Path(__file__).parent.parent / "outputs/A2.5.3_hierarchical_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.3 Hierarchical Clustering Concept Generation completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.5.3: {str(e)}")
        raise

if __name__ == "__main__":
    main()