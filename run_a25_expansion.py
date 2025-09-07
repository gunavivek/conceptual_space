#!/usr/bin/env python3
"""
Run A2.5 Concept Expansion with proper error handling
"""

import json
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

def load_core_concepts():
    """Load A2.4 core concepts"""
    with open("A_Concept_pipeline/outputs/A2.4_core_concepts.json", 'r') as f:
        data = json.load(f)
    return data['core_concepts']

def semantic_expansion(concepts):
    """Simple semantic similarity expansion"""
    expanded = []
    
    for concept in concepts:
        expansion = {
            "original_concept": concept,
            "expanded_keywords": [],
            "similarity_scores": {}
        }
        
        # Find similar concepts based on keyword overlap
        concept_keywords = set(concept.get('primary_keywords', []))
        
        for other in concepts:
            if other['concept_id'] == concept['concept_id']:
                continue
                
            other_keywords = set(other.get('primary_keywords', []))
            overlap = concept_keywords & other_keywords
            
            if len(overlap) > 0:
                similarity = len(overlap) / len(concept_keywords | other_keywords)
                if similarity > 0.2:  # Threshold
                    # Add some keywords from similar concept
                    new_keywords = list(other_keywords - concept_keywords)[:3]
                    expansion["expanded_keywords"].extend(new_keywords)
                    expansion["similarity_scores"][other['concept_id']] = similarity
        
        # Remove duplicates
        expansion["expanded_keywords"] = list(set(expansion["expanded_keywords"]))
        expanded.append(expansion)
    
    return expanded

def domain_knowledge_expansion(concepts):
    """Expand based on domain knowledge"""
    # Domain-specific term mappings
    domain_terms = {
        'Financial': ['revenue', 'income', 'expense', 'profit', 'loss', 'asset', 'liability'],
        'Operational': ['process', 'efficiency', 'workflow', 'capacity', 'utilization', 'performance'],
        'Tax': ['deduction', 'credit', 'liability', 'assessment', 'compliance', 'filing'],
        'Accounting': ['balance', 'ledger', 'journal', 'reconciliation', 'audit', 'accrual']
    }
    
    expanded = []
    
    for concept in concepts:
        expansion = {
            "original_concept": concept,
            "expanded_terms": [],
            "expansion_source": "domain_knowledge"
        }
        
        # Determine domain from keywords
        keywords = concept.get('primary_keywords', [])
        detected_domain = None
        
        for domain, terms in domain_terms.items():
            if any(term in ' '.join(keywords).lower() for term in terms):
                detected_domain = domain
                break
        
        if detected_domain:
            # Add relevant domain terms not already in keywords
            for term in domain_terms[detected_domain]:
                if term not in ' '.join(keywords).lower():
                    expansion["expanded_terms"].append(term)
        
        expansion["expanded_terms"] = expansion["expanded_terms"][:5]  # Limit expansion
        expanded.append(expansion)
    
    return expanded

def frequency_based_expansion(concepts):
    """Expand based on keyword frequency patterns"""
    # Count all keywords across concepts
    keyword_counter = Counter()
    
    for concept in concepts:
        keywords = concept.get('primary_keywords', [])
        keyword_counter.update(keywords)
    
    # Find frequently co-occurring terms
    expanded = []
    
    for concept in concepts:
        expansion = {
            "original_concept": concept,
            "all_expanded_terms": [],
            "frequency_scores": {}
        }
        
        keywords = set(concept.get('primary_keywords', []))
        
        # Find terms that frequently appear with this concept's keywords
        related_terms = []
        for term, count in keyword_counter.most_common():
            if term not in keywords and count >= 2:  # Appears in at least 2 concepts
                related_terms.append(term)
        
        expansion["all_expanded_terms"] = related_terms[:5]  # Top 5 related terms
        expanded.append(expansion)
    
    return expanded

def combine_expansions(semantic, domain, frequency):
    """Combine all expansion strategies"""
    combined = {}
    
    # Process each concept
    all_concept_ids = set()
    for exp_list in [semantic, domain, frequency]:
        for exp in exp_list:
            all_concept_ids.add(exp['original_concept']['concept_id'])
    
    for concept_id in all_concept_ids:
        combined[concept_id] = {
            "concept_id": concept_id,
            "original_concept": None,
            "all_expanded_terms": set(),
            "expansion_sources": {},
            "expansion_summary": {}
        }
        
        # Get semantic expansion
        for exp in semantic:
            if exp['original_concept']['concept_id'] == concept_id:
                combined[concept_id]['original_concept'] = exp['original_concept']
                terms = exp.get('expanded_keywords', [])
                combined[concept_id]['all_expanded_terms'].update(terms)
                combined[concept_id]['expansion_sources']['semantic'] = terms
                break
        
        # Get domain expansion
        for exp in domain:
            if exp['original_concept']['concept_id'] == concept_id:
                terms = exp.get('expanded_terms', [])
                combined[concept_id]['all_expanded_terms'].update(terms)
                combined[concept_id]['expansion_sources']['domain'] = terms
                break
        
        # Get frequency expansion
        for exp in frequency:
            if exp['original_concept']['concept_id'] == concept_id:
                terms = exp.get('all_expanded_terms', [])
                combined[concept_id]['all_expanded_terms'].update(terms)
                combined[concept_id]['expansion_sources']['frequency'] = terms
                break
        
        # Convert set to list
        combined[concept_id]['all_expanded_terms'] = list(combined[concept_id]['all_expanded_terms'])
        
        # Calculate expansion metrics
        original_count = len(combined[concept_id]['original_concept'].get('primary_keywords', []))
        expanded_count = len(combined[concept_id]['all_expanded_terms'])
        
        combined[concept_id]['expansion_summary'] = {
            'original_keyword_count': original_count,
            'expanded_term_count': expanded_count,
            'expansion_ratio': expanded_count / max(original_count, 1),
            'expansion_strategies_used': len(combined[concept_id]['expansion_sources'])
        }
    
    return combined

def main():
    print("="*60)
    print("A2.5: Concept Expansion Analysis")
    print("="*60)
    
    # Load core concepts
    concepts = load_core_concepts()
    print(f"\nLoaded {len(concepts)} core concepts from A2.4")
    
    # Run expansion strategies
    print("\nRunning expansion strategies...")
    
    print("  1. Semantic similarity expansion...")
    semantic = semantic_expansion(concepts)
    
    print("  2. Domain knowledge expansion...")
    domain = domain_knowledge_expansion(concepts)
    
    print("  3. Frequency-based expansion...")
    frequency = frequency_based_expansion(concepts)
    
    # Combine results
    print("\nCombining expansion results...")
    combined = combine_expansions(semantic, domain, frequency)
    
    # Generate summary
    print("\n" + "="*60)
    print("EXPANSION SUMMARY")
    print("="*60)
    
    total_original_terms = sum(c['expansion_summary']['original_keyword_count'] 
                               for c in combined.values())
    total_expanded_terms = sum(c['expansion_summary']['expanded_term_count'] 
                               for c in combined.values())
    
    print(f"\nOverall Statistics:")
    print(f"  Total Concepts: {len(combined)}")
    print(f"  Original Keywords: {total_original_terms}")
    print(f"  Expanded Terms: {total_expanded_terms}")
    print(f"  Overall Expansion Ratio: {total_expanded_terms/max(total_original_terms,1):.2f}")
    
    print(f"\nPer-Concept Expansion:")
    for concept_id, data in combined.items():
        concept = data['original_concept']
        summary = data['expansion_summary']
        print(f"\n  {concept_id}: {concept['canonical_name']}")
        print(f"    Original Keywords: {summary['original_keyword_count']}")
        print(f"    Expanded Terms: {summary['expanded_term_count']}")
        print(f"    Expansion Ratio: {summary['expansion_ratio']:.2f}")
        print(f"    Strategies Used: {summary['expansion_strategies_used']}")
        
        if data['all_expanded_terms']:
            print(f"    Sample Expansions: {', '.join(data['all_expanded_terms'][:5])}")
    
    # Save expanded concepts
    output_file = Path("A_Concept_pipeline/outputs/A2.5_expanded_concepts.json")
    output_data = {
        "expanded_concepts": list(combined.values()),
        "summary": {
            "total_concepts": len(combined),
            "original_keywords": total_original_terms,
            "expanded_terms": total_expanded_terms,
            "expansion_ratio": total_expanded_terms/max(total_original_terms,1),
            "strategies_used": ["semantic_similarity", "domain_knowledge", "frequency_based"]
        },
        "metadata": {
            "source": "A2.4_core_concepts.json",
            "expansion_strategies": 3,
            "version": "1.0"
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Expanded concepts saved to: {output_file}")
    
    return combined

if __name__ == "__main__":
    main()