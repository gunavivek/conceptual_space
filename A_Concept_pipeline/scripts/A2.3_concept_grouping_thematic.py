#!/usr/bin/env python3
"""
A2.3: Concept Grouping Thematic - Intra-Document Keyword Clustering
Groups related keywords within each document into thematic clusters
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import math

def calculate_keyword_similarity(kw1_data, kw2_data, doc_text):
    """
    Calculate semantic similarity between two keywords within a document context
    
    Args:
        kw1_data: First keyword data {term, score}
        kw2_data: Second keyword data {term, score}
        doc_text: Document text for context analysis
        
    Returns:
        float: Similarity score (0-1)
    """
    kw1 = kw1_data['term'].lower()
    kw2 = kw2_data['term'].lower()
    
    if kw1 == kw2:
        return 1.0
    
    # Factor 1: Semantic relationship (basic word similarity)
    semantic_score = 0.0
    
    # Check for shared word parts or roots
    if len(kw1) > 3 and len(kw2) > 3:
        if kw1 in kw2 or kw2 in kw1:
            semantic_score += 0.3
        elif kw1[:4] == kw2[:4]:  # Same prefix
            semantic_score += 0.2
    
    # Factor 2: Contextual proximity in document
    proximity_score = 0.0
    kw1_positions = [m.start() for m in re.finditer(re.escape(kw1), doc_text.lower())]
    kw2_positions = [m.start() for m in re.finditer(re.escape(kw2), doc_text.lower())]
    
    if kw1_positions and kw2_positions:
        min_distance = min(abs(p1 - p2) for p1 in kw1_positions for p2 in kw2_positions)
        # Closer keywords are more related (inverse relationship)
        proximity_score = max(0, 1 - (min_distance / 500))  # Normalize by 500 chars
    
    # Factor 3: Financial domain relationships
    financial_clusters = {
        'monetary': ['million', 'billion', 'thousand', '$', 'dollar', 'amount'],
        'time_periods': ['2019', '2018', '2020', '2021', 'year', 'period', 'quarter'],
        'accounting': ['deferred', 'revenue', 'income', 'expense', 'assets', 'liability'],
        'operations': ['contract', 'operations', 'business', 'company', 'service'],
        'financial_reporting': ['balance', 'statement', 'report', 'analysis', 'total']
    }
    
    domain_score = 0.0
    for cluster_name, terms in financial_clusters.items():
        kw1_in = any(term in kw1 for term in terms)
        kw2_in = any(term in kw2 for term in terms)
        if kw1_in and kw2_in:
            domain_score = 0.4
            break
    
    # Factor 4: TF-IDF score similarity (keywords with similar importance)
    score_similarity = 0.0
    if abs(kw1_data['score'] - kw2_data['score']) < 0.05:  # Similar TF-IDF scores
        score_similarity = 0.2
    
    # Weighted combination
    total_score = (semantic_score * 0.3 + 
                  proximity_score * 0.3 + 
                  domain_score * 0.3 + 
                  score_similarity * 0.1)
    
    return min(total_score, 1.0)

def cluster_keywords_within_document(doc_data, similarity_threshold=0.3):
    """
    Cluster keywords within a single document
    
    Args:
        doc_data: Document data with keywords
        similarity_threshold: Threshold for grouping keywords
        
    Returns:
        dict: Document with keyword clusters
    """
    doc_id = doc_data['doc_id']
    keywords = doc_data.get('keywords', [])
    doc_text = doc_data.get('text', '')
    
    if not keywords:
        return {
            'doc_id': doc_id,
            'keyword_clusters': [],
            'cluster_count': 0,
            'keywords_clustered': 0,
            'keywords_total': 0
        }
    
    print(f"\\nClustering {len(keywords)} keywords in document {doc_id}")
    
    # Calculate similarity matrix
    n_keywords = len(keywords)
    similarity_matrix = {}
    
    for i in range(n_keywords):
        for j in range(i + 1, n_keywords):
            similarity = calculate_keyword_similarity(keywords[i], keywords[j], doc_text)
            similarity_matrix[(i, j)] = similarity
            if similarity > 0.1:  # Only show meaningful similarities
                print(f"  '{keywords[i]['term']}' vs '{keywords[j]['term']}': {similarity:.3f}")
    
    # Perform agglomerative clustering
    clusters = []
    used_keywords = set()
    
    for i, kw_data in enumerate(keywords):
        if i in used_keywords:
            continue
            
        # Start new cluster
        cluster = {
            'cluster_id': len(clusters) + 1,
            'theme_name': '',
            'keywords': [kw_data],
            'keyword_indices': [i],
            'avg_tfidf_score': kw_data['score'],
            'cluster_coherence': 1.0
        }
        used_keywords.add(i)
        
        # Find similar keywords to add to cluster
        for j in range(i + 1, n_keywords):
            if j in used_keywords:
                continue
                
            similarity = similarity_matrix.get((i, j), 0.0)
            if similarity > similarity_threshold:
                cluster['keywords'].append(keywords[j])
                cluster['keyword_indices'].append(j)
                used_keywords.add(j)
                print(f"    Clustered '{kw_data['term']}' with '{keywords[j]['term']}' (sim: {similarity:.3f})")
        
        # Calculate cluster statistics
        if len(cluster['keywords']) > 1:
            # Average TF-IDF score
            cluster['avg_tfidf_score'] = sum(kw['score'] for kw in cluster['keywords']) / len(cluster['keywords'])
            
            # Cluster coherence (average similarity within cluster)
            coherence_scores = []
            for idx1 in cluster['keyword_indices']:
                for idx2 in cluster['keyword_indices']:
                    if idx1 < idx2:
                        coherence_scores.append(similarity_matrix.get((idx1, idx2), 0.0))
            cluster['cluster_coherence'] = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
            
            # Generate theme name from top 2-3 keywords
            top_keywords = sorted(cluster['keywords'], key=lambda x: x['score'], reverse=True)[:3]
            cluster['theme_name'] = ' & '.join(kw['term'].title() for kw in top_keywords[:2])
        else:
            cluster['theme_name'] = cluster['keywords'][0]['term'].title()
        
        clusters.append(cluster)
        print(f"  Created cluster '{cluster['theme_name']}' with {len(cluster['keywords'])} keywords")
    
    return {
        'doc_id': doc_id,
        'keyword_clusters': clusters,
        'cluster_count': len(clusters),
        'keywords_clustered': sum(len(c['keywords']) for c in clusters),
        'keywords_total': len(keywords),
        'clustering_effectiveness': len(clusters) / len(keywords) if keywords else 0
    }

def load_input(input_path="outputs/A2.2_keyword_extractions.json"):
    """Load keyword extractions from A2.2"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data["documents"] if "documents" in data else []

def process_all_documents(documents):
    """
    Process all documents for intra-document keyword clustering
    
    Args:
        documents: List of documents with keywords
        
    Returns:
        dict: Complete clustering results
    """
    results = {
        'documents': [],
        'statistics': {
            'total_documents': len(documents),
            'total_clusters': 0,
            'total_keywords': 0,
            'keywords_clustered': 0,
            'avg_clusters_per_doc': 0,
            'clustering_effectiveness': 0
        },
        'processing_timestamp': datetime.now().isoformat()
    }
    
    for doc in documents:
        doc_result = cluster_keywords_within_document(doc)
        results['documents'].append(doc_result)
        
        # Update statistics
        results['statistics']['total_clusters'] += doc_result['cluster_count']
        results['statistics']['total_keywords'] += doc_result['keywords_total']
        results['statistics']['keywords_clustered'] += doc_result['keywords_clustered']
    
    # Calculate averages
    if results['statistics']['total_documents'] > 0:
        results['statistics']['avg_clusters_per_doc'] = results['statistics']['total_clusters'] / results['statistics']['total_documents']
    
    if results['statistics']['total_keywords'] > 0:
        results['statistics']['clustering_effectiveness'] = results['statistics']['total_clusters'] / results['statistics']['total_keywords']
    
    return results

def save_output(data, output_path="outputs/A2.3_concept_grouping_thematic.json"):
    """Save intra-document clustering results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved intra-document clustering to {full_path}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("A2.3: Intra-Document Keyword Clustering")
    print("=" * 60)
    print("Loading keyword extractions...")
    
    # Load documents
    documents = load_input()
    print(f"Processing {len(documents)} documents for intra-document clustering...")
    
    # Process all documents
    results = process_all_documents(documents)
    
    # Save results
    save_output(results)
    
    # Print summary
    stats = results['statistics']
    print(f"\\nClustering Statistics:")
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Total Keyword Clusters: {stats['total_clusters']}")
    print(f"  Average Clusters per Document: {stats['avg_clusters_per_doc']:.1f}")
    print(f"  Keywords Processed: {stats['total_keywords']}")
    print(f"  Clustering Effectiveness: {stats['clustering_effectiveness']:.2f}")
    
    # Show sample clusters
    print(f"\\nSample Keyword Clusters:")
    for doc_result in results['documents'][:3]:  # Show first 3 documents
        print(f"\\n  Document: {doc_result['doc_id']}")
        for cluster in doc_result['keyword_clusters'][:3]:  # Show first 3 clusters per doc
            keywords = [kw['term'] for kw in cluster['keywords']]
            print(f"    Cluster: {cluster['theme_name']} - {keywords}")
    
    print(f"\\nA2.3 Intra-Document Keyword Clustering completed successfully!")

if __name__ == "__main__":
    main()