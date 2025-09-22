#!/usr/bin/env python3
"""
A2.4: Synthesize Core Concepts (INDEPENDENT DOCUMENTS)
Identifies and synthesizes the most important core concepts within each document independently
WITHOUT cross-document aggregation
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import math
import re

def calculate_concept_importance_within_doc(cluster, total_clusters_in_doc):
    """
    Calculate importance score for a concept cluster within a single document

    Args:
        cluster: Keyword cluster from A2.3
        total_clusters_in_doc: Total clusters in this document

    Returns:
        float: Importance score
    """
    # Factors for concept importance within a document:
    # 1. Number of keywords in cluster (cluster size)
    # 2. Average keyword scores (concept strength)
    # 3. Business term relevance
    # 4. Semantic coherence

    cluster_keywords = cluster.get("keywords", [])
    num_keywords = len(cluster_keywords)

    if num_keywords == 0:
        return 0.0

    # Calculate average keyword strength
    total_score = sum(kw.get("score", 0) for kw in cluster_keywords)
    avg_keyword_strength = total_score / num_keywords

    # Size factor (larger clusters are more important, but with diminishing returns)
    size_factor = min(num_keywords / 8, 1.0)  # Normalize to 8 keywords max

    # Business relevance bonus
    theme_name = cluster.get("theme_name", "").lower()
    business_bonus = 1.3 if any(business_term in theme_name
                               for business_term in ['revenue', 'income', 'contract', 'balance', 'inventory',
                                                    'customer', 'operation', 'process', 'management', 'tax']) else 1.0

    # Relative importance within document
    cluster_prominence = min(num_keywords / max(total_clusters_in_doc * 2, 1), 1.0)

    importance = (
        size_factor * 0.3 +                    # Cluster size
        min(avg_keyword_strength, 1.0) * 0.4 + # Keyword strength
        cluster_prominence * 0.3               # Relative prominence in doc
    ) * business_bonus

    return min(1.0, importance)

def extract_core_concepts_per_document(doc, min_quality_threshold=0.05):
    """
    Extract ALL core concepts from a single document independently

    Args:
        doc: Document with keyword_clusters from A2.3
        min_quality_threshold: Minimum importance score to preserve concept (default: 0.05)

    Returns:
        list: ALL core concepts for this document above quality threshold
    """
    doc_id = doc.get("doc_id", "unknown")
    clusters = doc.get("keyword_clusters", [])

    if not clusters:
        return []

    doc_concepts = []
    total_clusters = len(clusters)

    for cluster in clusters:
        theme_name = cluster.get("theme_name", "Unknown Theme")
        keywords = cluster.get("keywords", [])

        if not keywords:
            continue

        # Calculate importance within this document only
        importance = calculate_concept_importance_within_doc(cluster, total_clusters)

        # Skip concepts below quality threshold
        if importance < min_quality_threshold:
            continue

        # Create concept definition
        concept_definition = generate_concept_definition(theme_name, keywords)

        # Categorize business concept
        business_category = categorize_business_concept(theme_name)

        concept = {
            "concept_id": f"{doc_id}_{theme_name.lower().replace(' ', '_')}",
            "canonical_name": theme_name.lower(),
            "original_theme_name": theme_name,
            "importance_score": importance,
            "document_id": doc_id,
            "keywords": [kw.get("term", "") for kw in keywords],
            "keyword_scores": [kw.get("score", 0) for kw in keywords],
            "keyword_count": len(keywords),
            "avg_keyword_score": sum(kw.get("score", 0) for kw in keywords) / len(keywords),
            "business_category": business_category,
            "concept_definition": concept_definition,
            "concept_type": determine_concept_type(theme_name, keywords)
        }

        doc_concepts.append(concept)

    # Sort by importance and return ALL concepts above quality threshold
    doc_concepts.sort(key=lambda x: x["importance_score"], reverse=True)
    return doc_concepts  # PRESERVE ALL CONCEPTS - No top_k limitation!

def generate_concept_definition(theme_name, keywords):
    """Generate a definition for the concept based on theme and keywords"""
    keyword_terms = [kw.get("term", "") for kw in keywords[:5]]  # Use top 5 keywords

    if "revenue" in theme_name.lower():
        return f"Financial concept related to {theme_name.lower()}, encompassing terms like {', '.join(keyword_terms[:3])}."
    elif "tax" in theme_name.lower():
        return f"Tax and regulatory concept involving {theme_name.lower()}, including {', '.join(keyword_terms[:3])}."
    elif "customer" in theme_name.lower():
        return f"Customer relationship concept focusing on {theme_name.lower()}, involving {', '.join(keyword_terms[:3])}."
    else:
        return f"Business concept related to {theme_name.lower()}, encompassing {', '.join(keyword_terms[:3])}."

def categorize_business_concept(theme_name):
    """Categorize business concept into predefined categories"""
    name_lower = theme_name.lower()

    if any(term in name_lower for term in ['revenue', 'income', 'expense', 'cost', 'tax', 'balance']):
        return "Financial Concepts"
    elif any(term in name_lower for term in ['customer', 'contract', 'service']):
        return "Customer & Contract Concepts"
    elif any(term in name_lower for term in ['operation', 'process', 'management', 'business']):
        return "Operational Concepts"
    else:
        return "General Business Concepts"

def determine_concept_type(theme_name, keywords):
    """Determine the type of business concept"""
    name_lower = theme_name.lower()
    keyword_text = ' '.join([kw.get("term", "") for kw in keywords]).lower()

    if any(term in name_lower or term in keyword_text for term in ['revenue', 'income', 'tax']):
        return "Financial Performance"
    elif any(term in name_lower or term in keyword_text for term in ['customer', 'client']):
        return "Business Relationship"
    elif any(term in name_lower or term in keyword_text for term in ['contract', 'agreement']):
        return "Legal Instrument"
    else:
        return "General Business Concept"

def process_documents_independently():
    """
    Process all documents independently for core concept extraction
    """
    print("============================================================")
    print("A2.4: Synthesize Core Concepts (INDEPENDENT DOCUMENTS)")
    print("============================================================")

    # Load input from A2.3
    base_path = Path(__file__).parent.parent
    input_file = base_path / "outputs" / "A2.3_concept_grouping_thematic.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    documents = input_data.get("documents", [])

    print(f"Processing {len(documents)} documents independently for core concept extraction...")

    # Process each document independently
    all_results = []
    total_concepts = 0

    for doc in documents:
        doc_id = doc.get("doc_id", "unknown")

        # Extract ALL core concepts for this document (no top_k limitation)
        doc_concepts = extract_core_concepts_per_document(doc, min_quality_threshold=0.05)

        doc_result = {
            "doc_id": doc_id,
            "core_concepts": doc_concepts,
            "concept_count": len(doc_concepts),
            "processing_metadata": {
                "extraction_method": "independent_document_ALL_CONCEPTS",
                "quality_threshold": 0.05,
                "clusters_processed": len(doc.get("keyword_clusters", [])),
                "concepts_preserved": len(doc_concepts),
                "concept_extraction_timestamp": datetime.now().isoformat()
            }
        }

        all_results.append(doc_result)
        total_concepts += len(doc_concepts)

        # Print summary for this document
        print(f"\\nDocument {doc_id}:")
        print(f"  Clusters processed: {len(doc.get('keyword_clusters', []))}")
        print(f"  Core concepts extracted: {len(doc_concepts)}")

        if doc_concepts:
            print(f"  Top concept: {doc_concepts[0]['canonical_name']} (importance: {doc_concepts[0]['importance_score']:.3f})")
            print(f"  Categories: {set(c['business_category'] for c in doc_concepts)}")

    # Create output summary
    output_data = {
        "documents": all_results,
        "processing_summary": {
            "total_documents": len(documents),
            "total_core_concepts": total_concepts,
            "avg_concepts_per_doc": total_concepts / len(documents) if documents else 0,
            "processing_method": "independent_document_synthesis",
            "cross_document_analysis": False,
            "timestamp": datetime.now().isoformat()
        },
        "methodology": {
            "approach": "Independent document processing - ALL CONCEPTS PRESERVED",
            "description": "Each document processed separately without cross-document aggregation. ALL concepts above quality threshold preserved.",
            "concept_ranking": "Within-document importance scoring",
            "quality_threshold": 0.05,
            "concept_preservation": "ALL concepts above threshold (no top_k limitation)"
        }
    }

    # Save results
    output_file = base_path / "outputs" / "A2.4_core_concepts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\\n[OK] Saved independent core concepts to {output_file}")

    # Print overall statistics
    print(f"\\nIndependent Processing Summary:")
    print(f"  Documents processed: {len(documents)}")
    print(f"  Total core concepts: {total_concepts}")
    print(f"  Average concepts per document: {total_concepts/len(documents):.1f}")
    print(f"  Processing approach: Independent (no cross-document analysis)")

    print("\\nA2.4 Independent Core Concept Synthesis completed successfully!")

if __name__ == "__main__":
    process_documents_independently()