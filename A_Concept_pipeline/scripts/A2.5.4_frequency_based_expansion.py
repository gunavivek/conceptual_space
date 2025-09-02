#!/usr/bin/env python3
"""
A2.5.4: Frequency-Based Expansion Strategy
Expands concepts using term frequency analysis and statistical co-occurrence patterns
"""

import json
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

def calculate_term_frequency_stats(concepts):
    """
    Calculate term frequency statistics across all concepts
    
    Args:
        concepts: List of concepts
        
    Returns:
        dict: Term frequency statistics
    """
    # Collect all terms and their document frequencies
    term_doc_freq = defaultdict(int)  # Number of documents containing term
    term_total_freq = defaultdict(int)  # Total frequency across all documents
    doc_terms = defaultdict(set)  # Terms in each document
    
    total_documents = 0
    
    for concept in concepts:
        concept_id = concept.get("concept_id", "")
        keywords = concept.get("primary_keywords", [])
        
        # Count unique documents
        related_docs = concept.get("related_documents", [])
        total_documents = max(total_documents, len(related_docs))
        
        # Process keywords
        for keyword in keywords:
            term_doc_freq[keyword] += 1
            term_total_freq[keyword] += 1
            doc_terms[concept_id].add(keyword)
    
    # Calculate TF-IDF-like scores
    term_stats = {}
    for term in term_doc_freq:
        df = term_doc_freq[term]  # Document frequency
        tf = term_total_freq[term]  # Term frequency
        
        # IDF calculation
        idf = math.log(len(concepts) / max(df, 1))
        
        # TF-IDF score
        tf_idf = tf * idf
        
        term_stats[term] = {
            "document_frequency": df,
            "term_frequency": tf,
            "idf": idf,
            "tf_idf": tf_idf,
            "prevalence": df / len(concepts)  # What fraction of concepts contain this term
        }
    
    return term_stats, doc_terms

def find_high_frequency_expansions(concept, term_stats, threshold_percentile=75):
    """
    Find expansion terms based on high frequency patterns
    
    Args:
        concept: Target concept
        term_stats: Term frequency statistics
        threshold_percentile: Percentile threshold for frequency
        
    Returns:
        dict: Frequency-based expansions
    """
    concept_keywords = set(concept.get("primary_keywords", []))
    
    # Calculate frequency thresholds
    all_tf_idf = [stats["tf_idf"] for stats in term_stats.values()]
    all_prevalence = [stats["prevalence"] for stats in term_stats.values()]
    
    tf_idf_threshold = sorted(all_tf_idf)[int(len(all_tf_idf) * threshold_percentile / 100)] if all_tf_idf else 0
    prevalence_threshold = sorted(all_prevalence)[int(len(all_prevalence) * threshold_percentile / 100)] if all_prevalence else 0
    
    # Find expansion candidates
    expansion_candidates = []
    
    for term, stats in term_stats.items():
        if term in concept_keywords:
            continue  # Skip terms already in concept
        
        # Check if term meets frequency criteria
        is_high_tf_idf = stats["tf_idf"] >= tf_idf_threshold
        is_moderate_prevalence = 0.1 <= stats["prevalence"] <= 0.8  # Not too rare, not too common
        
        if is_high_tf_idf and is_moderate_prevalence:
            expansion_candidates.append({
                "term": term,
                "reason": "high_tf_idf",
                "tf_idf_score": stats["tf_idf"],
                "prevalence": stats["prevalence"],
                "document_frequency": stats["document_frequency"]
            })
    
    # Sort by TF-IDF score
    expansion_candidates.sort(key=lambda x: x["tf_idf_score"], reverse=True)
    
    return expansion_candidates[:10]  # Top 10 expansion terms

def find_co_occurrence_patterns(concept, doc_terms, min_cooccurrence=2):
    """
    Find terms that co-occur with concept terms
    
    Args:
        concept: Target concept
        doc_terms: Terms in each document
        min_cooccurrence: Minimum co-occurrence count
        
    Returns:
        list: Co-occurring terms
    """
    concept_keywords = set(concept.get("primary_keywords", []))
    cooccurrence_counts = defaultdict(int)
    
    # Find documents that contain concept keywords
    relevant_docs = []
    for doc_id, terms in doc_terms.items():
        if concept_keywords & terms:  # If document contains any concept keywords
            relevant_docs.append(doc_id)
    
    # Count co-occurrences
    for doc_id in relevant_docs:
        doc_terms_set = doc_terms[doc_id]
        for term in doc_terms_set:
            if term not in concept_keywords:
                cooccurrence_counts[term] += 1
    
    # Filter by minimum co-occurrence
    cooccurring_terms = []
    for term, count in cooccurrence_counts.items():
        if count >= min_cooccurrence:
            cooccurrence_strength = count / len(relevant_docs) if relevant_docs else 0
            cooccurring_terms.append({
                "term": term,
                "cooccurrence_count": count,
                "cooccurrence_strength": cooccurrence_strength,
                "documents_with_cooccurrence": count
            })
    
    # Sort by co-occurrence strength
    cooccurring_terms.sort(key=lambda x: x["cooccurrence_strength"], reverse=True)
    
    return cooccurring_terms[:8]  # Top 8 co-occurring terms

def expand_concept_frequency_based(concept, term_stats, doc_terms):
    """
    Expand a concept using frequency-based analysis
    
    Args:
        concept: Target concept
        term_stats: Term frequency statistics
        doc_terms: Terms in documents
        
    Returns:
        dict: Frequency-based expansion
    """
    # Get high-frequency expansions
    high_freq_expansions = find_high_frequency_expansions(concept, term_stats)
    
    # Get co-occurrence expansions
    cooccurrence_expansions = find_co_occurrence_patterns(concept, doc_terms)
    
    # Combine all expansion terms
    all_expansion_terms = set(concept.get("primary_keywords", []))
    
    # Add high frequency terms
    for exp in high_freq_expansions:
        all_expansion_terms.add(exp["term"])
    
    # Add co-occurrence terms
    for exp in cooccurrence_expansions:
        all_expansion_terms.add(exp["term"])
    
    # Calculate expansion metrics
    original_count = len(concept.get("primary_keywords", []))
    expanded_count = len(all_expansion_terms)
    expansion_ratio = expanded_count / max(original_count, 1)
    
    # Calculate quality scores
    avg_tf_idf = sum(exp["tf_idf_score"] for exp in high_freq_expansions) / max(len(high_freq_expansions), 1)
    avg_cooccurrence = sum(exp["cooccurrence_strength"] for exp in cooccurrence_expansions) / max(len(cooccurrence_expansions), 1)
    
    return {
        "original_concept": concept,
        "high_frequency_expansions": high_freq_expansions,
        "cooccurrence_expansions": cooccurrence_expansions,
        "all_expanded_terms": list(all_expansion_terms),
        "expansion_metrics": {
            "original_terms": original_count,
            "expanded_terms": expanded_count,
            "expansion_ratio": expansion_ratio,
            "high_freq_terms": len(high_freq_expansions),
            "cooccurrence_terms": len(cooccurrence_expansions),
            "avg_tf_idf_score": avg_tf_idf,
            "avg_cooccurrence_strength": avg_cooccurrence
        }
    }

def process_frequency_based_expansion(core_concepts):
    """
    Process frequency-based expansion for all concepts
    
    Args:
        core_concepts: List of core concepts
        
    Returns:
        dict: Frequency expansion results
    """
    # Calculate term frequency statistics
    term_stats, doc_terms = calculate_term_frequency_stats(core_concepts)
    
    # Expand each concept
    expansions = []
    for concept in core_concepts:
        expansion = expand_concept_frequency_based(concept, term_stats, doc_terms)
        expansions.append(expansion)
    
    # Calculate global statistics
    total_unique_terms = len(term_stats)
    avg_term_prevalence = sum(stats["prevalence"] for stats in term_stats.values()) / max(len(term_stats), 1)
    
    high_value_terms = [term for term, stats in term_stats.items() if stats["tf_idf"] > avg_term_prevalence]
    
    return {
        "strategy": "frequency_based",
        "term_statistics": {
            "total_unique_terms": total_unique_terms,
            "average_prevalence": avg_term_prevalence,
            "high_value_terms_count": len(high_value_terms)
        },
        "expansions": expansions,
        "statistics": {
            "concepts_processed": len(expansions),
            "avg_expansion_ratio": sum(exp["expansion_metrics"]["expansion_ratio"] for exp in expansions) / max(len(expansions), 1),
            "avg_high_freq_terms": sum(exp["expansion_metrics"]["high_freq_terms"] for exp in expansions) / max(len(expansions), 1),
            "avg_cooccurrence_terms": sum(exp["expansion_metrics"]["cooccurrence_terms"] for exp in expansions) / max(len(expansions), 1),
            "high_expansion_concepts": len([e for e in expansions if e["expansion_metrics"]["expansion_ratio"] > 1.5])
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
    print("A2.5.4: Frequency-Based Expansion Strategy")
    print("="*60)
    
    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()
        core_concepts = input_data.get("core_concepts", [])
        
        # Process frequency-based expansion
        print(f"Processing frequency-based expansion for {len(core_concepts)} concepts...")
        expansion_results = process_frequency_based_expansion(core_concepts)
        
        # Display results
        stats = expansion_results["statistics"]
        term_stats = expansion_results["term_statistics"]
        
        print(f"\nFrequency-Based Expansion Results:")
        print(f"  Concepts Processed: {stats['concepts_processed']}")
        print(f"  Average Expansion Ratio: {stats['avg_expansion_ratio']:.2f}")
        print(f"  Avg High-Freq Terms: {stats['avg_high_freq_terms']:.1f}")
        print(f"  Avg Co-occurrence Terms: {stats['avg_cooccurrence_terms']:.1f}")
        print(f"  High Expansion Concepts: {stats['high_expansion_concepts']}")
        
        print(f"\nTerm Statistics:")
        print(f"  Total Unique Terms: {term_stats['total_unique_terms']}")
        print(f"  Average Prevalence: {term_stats['average_prevalence']:.3f}")
        print(f"  High-Value Terms: {term_stats['high_value_terms_count']}")
        
        # Show sample expansions
        print(f"\nSample Frequency Expansions:")
        for i, exp in enumerate(expansion_results["expansions"][:3], 1):
            concept = exp["original_concept"]
            metrics = exp["expansion_metrics"]
            print(f"  {i}. {concept['theme_name']}")
            print(f"     Expansion: {metrics['original_terms']} -> {metrics['expanded_terms']} terms")
            print(f"     High-freq: {metrics['high_freq_terms']}, Co-occurrence: {metrics['cooccurrence_terms']}")
        
        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "frequency_based",
            "results": expansion_results,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        output_path = Path(__file__).parent.parent / "outputs/A2.5.4_frequency_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.4 Frequency-Based Expansion completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.5.4: {str(e)}")
        raise

if __name__ == "__main__":
    main()