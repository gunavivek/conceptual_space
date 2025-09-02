#!/usr/bin/env python3
"""
A2.4: Synthesize Core Concepts
Identifies and synthesizes the most important core concepts from thematic groups
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import math

def calculate_concept_importance(group, total_docs):
    """
    Calculate importance score for a concept group
    
    Args:
        group: Thematic group
        total_docs: Total number of documents
        
    Returns:
        float: Importance score
    """
    # Factors for importance:
    # 1. Number of documents in group (coverage)
    # 2. Keyword frequency/strength
    # 3. Domain relevance
    
    doc_coverage = len(group["documents"]) / total_docs
    
    # Calculate keyword strength (sum of frequencies)
    keyword_strength = sum(group["common_keywords"].values()) / len(group["documents"])
    
    # Domain bonus for non-general domains
    domain_bonus = 1.2 if group.get("dominant_domain", "general") != "general" else 1.0
    
    importance = (doc_coverage * 0.4 + 
                 min(keyword_strength / 10, 1.0) * 0.4 + 
                 len(group["representative_keywords"]) / 20 * 0.2) * domain_bonus
    
    return min(1.0, importance)

def identify_core_concepts(groups, total_docs, top_k=10):
    """
    Identify core concepts from thematic groups
    
    Args:
        groups: Thematic groups
        total_docs: Total number of documents
        top_k: Number of core concepts to identify
        
    Returns:
        list: Core concepts with metadata
    """
    core_concepts = []
    
    for group in groups:
        importance = calculate_concept_importance(group, total_docs)
        
        core_concept = {
            "concept_id": f"core_{len(core_concepts) + 1}",
            "theme_name": group["theme_name"],
            "importance_score": importance,
            "document_count": len(group["documents"]),
            "coverage_ratio": len(group["documents"]) / total_docs,
            "primary_keywords": group["representative_keywords"][:5],
            "domain": group.get("dominant_domain", "general"),
            "related_documents": group["documents"],
            "keyword_frequencies": dict(sorted(group["common_keywords"].items(), key=lambda x: x[1], reverse=True)[:10])
        }
        
        core_concepts.append(core_concept)
    
    # Sort by importance and take top k
    core_concepts.sort(key=lambda x: x["importance_score"], reverse=True)
    return core_concepts[:top_k]

def create_concept_hierarchy(core_concepts):
    """
    Create hierarchical structure of concepts
    
    Args:
        core_concepts: List of core concepts
        
    Returns:
        dict: Hierarchical concept structure
    """
    # Group concepts by domain
    domain_hierarchy = {}
    
    for concept in core_concepts:
        domain = concept["domain"]
        if domain not in domain_hierarchy:
            domain_hierarchy[domain] = {
                "domain": domain,
                "concepts": [],
                "total_importance": 0,
                "document_coverage": set()
            }
        
        domain_hierarchy[domain]["concepts"].append(concept)
        domain_hierarchy[domain]["total_importance"] += concept["importance_score"]
        domain_hierarchy[domain]["document_coverage"].update(concept["related_documents"])
    
    # Calculate domain-level statistics
    for domain_data in domain_hierarchy.values():
        domain_data["concept_count"] = len(domain_data["concepts"])
        domain_data["avg_importance"] = domain_data["total_importance"] / domain_data["concept_count"]
        domain_data["document_coverage"] = len(domain_data["document_coverage"])
    
    return domain_hierarchy

def generate_concept_mappings(core_concepts):
    """
    Generate mappings between concepts and documents
    
    Args:
        core_concepts: List of core concepts
        
    Returns:
        dict: Concept-document mappings
    """
    concept_to_docs = {}
    doc_to_concepts = {}
    
    for concept in core_concepts:
        concept_id = concept["concept_id"]
        concept_to_docs[concept_id] = concept["related_documents"]
        
        for doc_id in concept["related_documents"]:
            if doc_id not in doc_to_concepts:
                doc_to_concepts[doc_id] = []
            doc_to_concepts[doc_id].append(concept_id)
    
    return {
        "concept_to_documents": concept_to_docs,
        "document_to_concepts": doc_to_concepts
    }

def load_input(input_path="outputs/A2.3_concept_grouping_thematic.json"):
    """Load concept grouping from A2.3"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_core_concepts(data):
    """
    Process core concept synthesis
    
    Args:
        data: Concept grouping data
        
    Returns:
        dict: Core concepts with metadata
    """
    groups = data.get("thematic_groups", [])
    total_docs = data.get("statistics", {}).get("total_documents", 0)
    
    # Identify core concepts
    core_concepts = identify_core_concepts(groups, total_docs)
    
    # Create hierarchy
    hierarchy = create_concept_hierarchy(core_concepts)
    
    # Generate mappings
    mappings = generate_concept_mappings(core_concepts)
    
    # Calculate overall statistics
    total_coverage = len(set().union(*[c["related_documents"] for c in core_concepts]))
    
    return {
        "core_concepts": core_concepts,
        "concept_hierarchy": hierarchy,
        "mappings": mappings,
        "statistics": {
            "total_core_concepts": len(core_concepts),
            "total_documents": total_docs,
            "document_coverage": total_coverage,
            "coverage_percentage": (total_coverage / total_docs * 100) if total_docs > 0 else 0,
            "avg_importance": sum(c["importance_score"] for c in core_concepts) / len(core_concepts) if core_concepts else 0,
            "domains_covered": len(hierarchy)
        },
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/A2.4_core_concepts.json"):
    """Save core concepts"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved core concepts to {full_path}")
    
    # Save domain mappings separately
    domain_path = full_path.with_name("A2.4_domain_mappings.json")
    with open(domain_path, 'w') as f:
        json.dump(data["mappings"], f, indent=2)
    
    # Save statistics
    stats_path = full_path.with_name("A2.4_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(data["statistics"], f, indent=2)

def main():
    """Main execution"""
    print("="*60)
    print("A2.4: Synthesize Core Concepts")
    print("="*60)
    
    try:
        # Load concept grouping
        print("Loading concept grouping...")
        input_data = load_input()
        
        # Process core concepts
        print("Synthesizing core concepts...")
        output_data = process_core_concepts(input_data)
        
        # Display results
        stats = output_data["statistics"]
        print(f"\nCore Concept Statistics:")
        print(f"  Total Core Concepts: {stats['total_core_concepts']}")
        print(f"  Document Coverage: {stats['document_coverage']}/{stats['total_documents']} ({stats['coverage_percentage']:.1f}%)")
        print(f"  Average Importance: {stats['avg_importance']:.3f}")
        print(f"  Domains Covered: {stats['domains_covered']}")
        
        print(f"\nTop 5 Core Concepts:")
        for i, concept in enumerate(output_data["core_concepts"][:5], 1):
            print(f"  {i}. {concept['theme_name']}")
            print(f"     Importance: {concept['importance_score']:.3f}")
            print(f"     Domain: {concept['domain']}")
            print(f"     Documents: {concept['document_count']}")
            print(f"     Keywords: {', '.join(concept['primary_keywords'])}")
        
        # Save output
        save_output(output_data)
        
        print("\nA2.4 Core Concept Synthesis completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.4 Core Concept Synthesis: {str(e)}")
        raise

if __name__ == "__main__":
    main()