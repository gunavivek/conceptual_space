#!/usr/bin/env python3
"""
Detailed analysis of finqa_test_96 concepts
"""

import json
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_finqa_96_concepts():
    """Analyze concepts from document finqa_test_96"""
    
    with open("A_Concept_pipeline/outputs/A2.4_core_concepts.json", 'r') as f:
        data = json.load(f)
    
    all_concepts = data.get("core_concepts", [])
    
    # Filter for finqa_test_96
    concepts_96 = [c for c in all_concepts if 'finqa_test_96' in c.get('related_documents', [])]
    
    print("="*80)
    print("FINQA_TEST_96 CONCEPT ANALYSIS")
    print("="*80)
    
    print(f"\nDocument: finqa_test_96")
    print(f"Total Concepts: {len(concepts_96)}")
    print(f"Domain: Financial/Revenue Recognition")
    
    print("\n" + "-"*80)
    print("CONVEX BALL PROPERTIES")
    print("-"*80)
    
    for i, concept in enumerate(concepts_96, 1):
        print(f"\n[{i}] CONCEPT: {concept['canonical_name'].upper()}")
        print(f"    ID: {concept['concept_id']}")
        print(f"    Importance Score: {concept['importance_score']:.3f}")
        print(f"    Document Coverage: {concept['coverage_ratio']:.1%}")
        print(f"    Unique Keywords: {concept['unique_keyword_count']}")
        
        # Get specific instance for finqa_test_96
        for instance in concept.get('document_instances', []):
            if instance.get('doc_id') == 'finqa_test_96':
                print(f"    Original Theme: {instance.get('original_theme_name', 'N/A')}")
                print(f"    Cluster ID: {instance.get('cluster_id', 'N/A')}")
                print(f"    Keywords in Doc: {len(instance.get('keywords', []))}")
                
                # Show top keywords
                keywords = instance.get('keywords', [])[:10]
                print(f"    Top Keywords: {', '.join(keywords)}")
                
                # Show keyword IDs for traceability
                keyword_ids = instance.get('keyword_ids', [])[:5]
                print(f"    Sample Keyword IDs: {', '.join(keyword_ids)}")
                break
    
    print("\n" + "-"*80)
    print("CONVEX BALL RELATIONSHIPS")
    print("-"*80)
    
    # Check for concept relationships
    print("\nSemantic Relationships:")
    concept_names = [c['canonical_name'] for c in concepts_96]
    
    # Simple semantic analysis
    financial_terms = {
        'contract balances': ['contract', 'balances', 'agreements'],
        'revenue unearned': ['revenue', 'unearned', 'deferred', 'recognition'],
        'receivable balance': ['receivable', 'balance', 'accounts', 'outstanding']
    }
    
    for name in concept_names:
        related_concepts = []
        for other_name in concept_names:
            if name != other_name:
                # Check for shared semantic space
                name_terms = set(name.lower().split())
                other_terms = set(other_name.lower().split())
                if name_terms & other_terms:  # Intersection
                    related_concepts.append(other_name)
        
        if related_concepts:
            print(f"• {name} overlaps with: {', '.join(related_concepts)}")
        else:
            print(f"• {name} is semantically distinct")
    
    print("\n" + "-"*80)
    print("CONVEX BALL SPATIAL PROPERTIES")
    print("-"*80)
    
    # Calculate approximate ball properties
    total_importance = sum(c['importance_score'] for c in concepts_96)
    avg_importance = total_importance / len(concepts_96)
    
    print(f"Conceptual Space Properties:")
    print(f"• Number of Convex Balls: {len(concepts_96)}")
    print(f"• Total Semantic Volume: {total_importance:.3f}")
    print(f"• Average Ball Density: {avg_importance:.3f}")
    print(f"• Domain Cohesion: High (all financial concepts)")
    
    # Ball sizes (importance * keyword_count)
    print(f"\nBall Size Rankings:")
    sorted_concepts = sorted(concepts_96, key=lambda x: x['importance_score'], reverse=True)
    for i, concept in enumerate(sorted_concepts, 1):
        ball_volume = concept['importance_score'] * concept['unique_keyword_count']
        print(f"{i}. {concept['canonical_name']}: Volume = {ball_volume:.2f} "
              f"(importance: {concept['importance_score']:.3f}, keywords: {concept['unique_keyword_count']})")
    
    print(f"\n" + "="*80)
    print("VISUALIZATION NOTES")
    print("="*80)
    print("In the 3D visualization (finqa_test_96_concepts_3d.html):")
    print("• Each sphere represents a concept's convex ball in semantic space")
    print("• Sphere size = importance score + keyword count")
    print("• Position = semantic similarity (closer = more related)")
    print("• Transparency shows overlap regions")
    print("• Labels show concept NAMES (not IDs)")
    print("• Hover for detailed information")

if __name__ == "__main__":
    analyze_finqa_96_concepts()