#!/usr/bin/env python3
"""
Analyze A2.4 Core Concepts and Document Relationships
"""

import json
import pandas as pd
from pathlib import Path
from tabulate import tabulate
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_concepts_and_documents():
    # Load the core concepts
    concepts_path = Path("A_Concept_pipeline/outputs/A2.4_core_concepts.json")
    
    with open(concepts_path, 'r') as f:
        data = json.load(f)
    
    # Extract core concepts
    concepts = data.get("core_concepts", [])
    
    # Create document-concept mapping
    doc_concept_map = {}
    concept_details = {}
    
    for concept in concepts:
        concept_id = concept["concept_id"]
        canonical_name = concept["canonical_name"]
        importance = concept["importance_score"]
        keyword_count = concept["unique_keyword_count"]
        
        concept_details[concept_id] = {
            "name": canonical_name,
            "importance": round(importance, 3),
            "keywords": keyword_count,
            "doc_count": concept["document_count"]
        }
        
        # Map documents to concepts
        for doc_id in concept["related_documents"]:
            if doc_id not in doc_concept_map:
                doc_concept_map[doc_id] = []
            doc_concept_map[doc_id].append(concept_id)
    
    # Get all unique document IDs
    all_docs = sorted(set(doc_concept_map.keys()))
    
    print("\n" + "="*80)
    print("A2.4 CONCEPT ANALYSIS: CONCEPTS BY DOCUMENT")
    print("="*80)
    
    # Display document-concept relationships
    print("\n📊 DOCUMENT-TO-CONCEPT MAPPING:")
    print("-" * 80)
    
    for doc_id in all_docs:
        concepts_in_doc = doc_concept_map.get(doc_id, [])
        print(f"\n📄 Document: {doc_id}")
        print(f"   Total Concepts: {len(concepts_in_doc)}")
        
        for concept_id in concepts_in_doc:
            details = concept_details[concept_id]
            print(f"   └─ {concept_id}: {details['name']}")
            print(f"      • Importance: {details['importance']}")
            print(f"      • Keywords: {details['keywords']}")
            print(f"      • Appears in {details['doc_count']} docs")
    
    # Create summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n📈 Overview:")
    print(f"   • Total Documents Analyzed: {len(all_docs)}")
    print(f"   • Total Core Concepts Identified: {len(concepts)}")
    print(f"   • Average Concepts per Document: {sum(len(c) for c in doc_concept_map.values()) / len(doc_concept_map):.1f}")
    
    # Create concept importance ranking
    print("\n📊 TOP CONCEPTS BY IMPORTANCE:")
    print("-" * 80)
    
    sorted_concepts = sorted(concepts, key=lambda x: x["importance_score"], reverse=True)[:10]
    
    table_data = []
    for i, concept in enumerate(sorted_concepts, 1):
        table_data.append([
            i,
            concept["concept_id"],
            concept["canonical_name"][:40],
            f"{concept['importance_score']:.3f}",
            concept["document_count"],
            concept["unique_keyword_count"]
        ])
    
    headers = ["Rank", "Concept ID", "Concept Name", "Importance", "Doc Count", "Keywords"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Cross-document concepts
    print("\n🔗 CROSS-DOCUMENT CONCEPTS (appearing in multiple documents):")
    print("-" * 80)
    
    cross_doc_concepts = [c for c in concepts if c["document_count"] > 1]
    
    if cross_doc_concepts:
        for concept in sorted(cross_doc_concepts, key=lambda x: x["document_count"], reverse=True):
            print(f"\n   {concept['concept_id']}: {concept['canonical_name']}")
            print(f"   Documents ({concept['document_count']}): {', '.join(concept['related_documents'])}")
            print(f"   Primary Keywords: {', '.join(concept['primary_keywords'][:5])}")
    else:
        print("   No concepts found across multiple documents")
    
    # Document similarity based on shared concepts
    print("\n🔄 DOCUMENT SIMILARITY (based on shared concepts):")
    print("-" * 80)
    
    doc_pairs = []
    for i, doc1 in enumerate(all_docs):
        for doc2 in all_docs[i+1:]:
            shared = set(doc_concept_map[doc1]) & set(doc_concept_map[doc2])
            if shared:
                doc_pairs.append((doc1, doc2, len(shared), list(shared)))
    
    if doc_pairs:
        doc_pairs.sort(key=lambda x: x[2], reverse=True)
        for doc1, doc2, count, shared_concepts in doc_pairs[:5]:
            print(f"\n   {doc1} ↔ {doc2}")
            print(f"   Shared Concepts ({count}): {', '.join(shared_concepts)}")
            for concept_id in shared_concepts[:3]:
                print(f"      • {concept_id}: {concept_details[concept_id]['name']}")
    else:
        print("   No documents share concepts")
    
    return doc_concept_map, concept_details

if __name__ == "__main__":
    try:
        doc_concept_map, concept_details = analyze_concepts_and_documents()
        print("\n✅ Analysis complete!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()