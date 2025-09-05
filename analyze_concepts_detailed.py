#!/usr/bin/env python3
"""
Detailed Analysis of A2.4 Core Concepts with Document-Concept Comparison
"""

import json
from pathlib import Path
from tabulate import tabulate
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_detailed_concept_document_matrix():
    # Load the core concepts
    concepts_path = Path("A_Concept_pipeline/outputs/A2.4_core_concepts.json")
    
    with open(concepts_path, 'r') as f:
        data = json.load(f)
    
    concepts = data.get("core_concepts", [])
    
    # Build comprehensive matrix
    doc_concept_details = {}
    all_docs = set()
    
    for concept in concepts:
        concept_id = concept["concept_id"]
        
        for instance in concept.get("document_instances", []):
            doc_id = instance["doc_id"]
            all_docs.add(doc_id)
            
            if doc_id not in doc_concept_details:
                doc_concept_details[doc_id] = []
            
            doc_concept_details[doc_id].append({
                "concept_id": concept_id,
                "canonical_name": concept["canonical_name"],
                "importance": concept["importance_score"],
                "cluster_id": instance.get("cluster_id"),
                "theme_name": instance.get("original_theme_name"),
                "keywords_count": len(instance.get("keywords", [])),
                "top_keywords": instance.get("keywords", [])[:5],
                "keyword_ids_sample": instance.get("keyword_ids", [])[:5]
            })
    
    # Sort documents for consistent display
    all_docs = sorted(all_docs)
    
    print("\n" + "="*120)
    print("DETAILED A2.4 CONCEPT ANALYSIS: DOCUMENT-CONCEPT COMPARISON")
    print("="*120)
    
    # Display detailed document-concept matrix
    for doc_id in all_docs:
        print(f"\n{'='*80}")
        print(f"DOCUMENT: {doc_id}")
        print(f"{'='*80}")
        
        concepts_in_doc = doc_concept_details.get(doc_id, [])
        print(f"Total Concepts: {len(concepts_in_doc)}\n")
        
        for i, concept_detail in enumerate(concepts_in_doc, 1):
            print(f"  [{i}] CONCEPT: {concept_detail['concept_id']}")
            print(f"      Name: {concept_detail['canonical_name']}")
            print(f"      Importance Score: {concept_detail['importance']:.3f}")
            print(f"      Source Cluster ID: {concept_detail['cluster_id']}")
            print(f"      Original Theme: {concept_detail['theme_name']}")
            print(f"      Keywords Count: {concept_detail['keywords_count']}")
            print(f"      Top Keywords: {', '.join(concept_detail['top_keywords'])}")
            print(f"      Sample Keyword IDs: {', '.join(concept_detail['keyword_ids_sample'])}")
            print()
    
    # Create comparison matrix table
    print("\n" + "="*120)
    print("CONCEPT DISTRIBUTION MATRIX")
    print("="*120)
    
    # Build matrix data
    concept_list = sorted(set(c["concept_id"] for doc_concepts in doc_concept_details.values() 
                             for c in doc_concepts))
    
    matrix_data = []
    for concept_id in concept_list:
        # Find concept details
        concept_info = next((c for c in concepts if c["concept_id"] == concept_id), None)
        if not concept_info:
            continue
            
        row = [concept_id, concept_info["canonical_name"][:30]]
        
        for doc_id in all_docs:
            doc_concepts = doc_concept_details.get(doc_id, [])
            has_concept = any(c["concept_id"] == concept_id for c in doc_concepts)
            
            if has_concept:
                # Find the importance score for this document
                concept_detail = next((c for c in doc_concepts if c["concept_id"] == concept_id), None)
                if concept_detail:
                    row.append(f"✓ ({concept_detail['importance']:.2f})")
                else:
                    row.append("✓")
            else:
                row.append("-")
        
        matrix_data.append(row)
    
    headers = ["Concept ID", "Concept Name"] + [doc.split("_")[-1] for doc in all_docs]
    print(tabulate(matrix_data, headers=headers, tablefmt="grid"))
    
    # Summary statistics per document
    print("\n" + "="*120)
    print("DOCUMENT STATISTICS")
    print("="*120)
    
    doc_stats = []
    for doc_id in all_docs:
        concepts_in_doc = doc_concept_details.get(doc_id, [])
        
        if concepts_in_doc:
            avg_importance = sum(c["importance"] for c in concepts_in_doc) / len(concepts_in_doc)
            total_keywords = sum(c["keywords_count"] for c in concepts_in_doc)
            
            doc_stats.append([
                doc_id.split("_")[-1],
                doc_id,
                len(concepts_in_doc),
                f"{avg_importance:.3f}",
                total_keywords,
                ", ".join([c["concept_id"] for c in concepts_in_doc][:3])
            ])
    
    headers = ["Doc#", "Full Doc ID", "Concepts", "Avg Import", "Total KW", "Top Concept IDs"]
    print(tabulate(doc_stats, headers=headers, tablefmt="grid"))
    
    # Concept type analysis
    print("\n" + "="*120)
    print("CONCEPT TYPE ANALYSIS")
    print("="*120)
    
    # Categorize concepts by business domain
    business_categories = {
        "Financial": ["income", "revenue", "balance", "receivable", "contract", "deferred"],
        "Operational": ["operation", "inventory", "valuation"],
        "Tax": ["tax", "twdv"],
        "Accounting": ["nbv", "net", "book", "unearned"]
    }
    
    categorized_concepts = {}
    for concept in concepts:
        name_lower = concept["canonical_name"].lower()
        categorized = False
        
        for category, keywords in business_categories.items():
            if any(kw in name_lower for kw in keywords):
                if category not in categorized_concepts:
                    categorized_concepts[category] = []
                categorized_concepts[category].append({
                    "id": concept["concept_id"],
                    "name": concept["canonical_name"],
                    "docs": len(concept["related_documents"])
                })
                categorized = True
                break
        
        if not categorized:
            if "Other" not in categorized_concepts:
                categorized_concepts["Other"] = []
            categorized_concepts["Other"].append({
                "id": concept["concept_id"],
                "name": concept["canonical_name"],
                "docs": len(concept["related_documents"])
            })
    
    for category, concepts_list in sorted(categorized_concepts.items()):
        print(f"\n{category} Concepts ({len(concepts_list)}):")
        for c in concepts_list:
            print(f"  • {c['id']}: {c['name']} (appears in {c['docs']} doc(s))")

if __name__ == "__main__":
    try:
        create_detailed_concept_document_matrix()
        print("\n✓ Detailed analysis complete!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()