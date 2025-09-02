#!/usr/bin/env python3
"""
A2.2: Keyword and Phrase Extraction
Extract important terms and phrases from preprocessed documents
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import re
import math

def calculate_tf(text):
    """
    Calculate term frequency for words in text
    
    Args:
        text: Input text
        
    Returns:
        dict: Term frequency scores
    """
    words = text.lower().split()
    word_count = len(words)
    
    if word_count == 0:
        return {}
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Calculate TF scores
    tf_scores = {}
    for word, freq in word_freq.items():
        tf_scores[word] = freq / word_count
    
    return tf_scores

def calculate_idf(documents):
    """
    Calculate inverse document frequency across all documents
    
    Args:
        documents: List of document texts
        
    Returns:
        dict: IDF scores
    """
    num_docs = len(documents)
    if num_docs == 0:
        return {}
    
    # Count document frequency for each word
    word_doc_freq = {}
    
    for doc in documents:
        words_in_doc = set(doc.lower().split())
        for word in words_in_doc:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
    
    # Calculate IDF scores
    idf_scores = {}
    for word, doc_freq in word_doc_freq.items():
        idf_scores[word] = math.log(num_docs / doc_freq)
    
    return idf_scores

def calculate_tfidf(tf_scores, idf_scores):
    """
    Calculate TF-IDF scores
    
    Args:
        tf_scores: Term frequency scores
        idf_scores: Inverse document frequency scores
        
    Returns:
        dict: TF-IDF scores
    """
    tfidf_scores = {}
    
    for word, tf in tf_scores.items():
        if word in idf_scores:
            tfidf_scores[word] = tf * idf_scores[word]
    
    return tfidf_scores

def extract_phrases(text, max_words=3):
    """
    Extract multi-word phrases from text
    
    Args:
        text: Input text
        max_words: Maximum words in a phrase
        
    Returns:
        list: Extracted phrases
    """
    # Clean text for phrase extraction
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    phrases = []
    
    # Extract n-grams
    for n in range(2, min(max_words + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            
            # Filter out phrases with stop words at beginning/end
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            first_word = words[i]
            last_word = words[i+n-1]
            
            if first_word not in stop_words and last_word not in stop_words:
                phrases.append(phrase)
    
    return phrases

def filter_keywords(tfidf_scores, min_length=2, top_k=50):
    """
    Filter and select top keywords
    
    Args:
        tfidf_scores: TF-IDF scores for words
        min_length: Minimum word length
        top_k: Number of top keywords to select
        
    Returns:
        list: Top keywords with scores
    """
    # Filter by length and common words
    stop_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
        'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
        'do', 'at', 'this', 'but', 'his', 'by', 'from', 'is', 'was'
    }
    
    filtered_scores = {
        word: score for word, score in tfidf_scores.items()
        if len(word) >= min_length and word not in stop_words
    }
    
    # Sort by score and get top k
    sorted_keywords = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_keywords[:top_k]

def extract_concept_keywords(concept_definitions):
    """
    Extract seed keywords from BIZBOK concept definitions for domain-aware weighting
    
    Args:
        concept_definitions: Dict of concept_id -> definition text
        
    Returns:
        set: Set of domain-relevant seed keywords
    """
    concept_keywords = set()
    
    for concept_id, definition in concept_definitions.items():
        if not definition:
            continue
            
        # Extract meaningful words from definition (excluding stop words)
        words = re.findall(r'\b[a-z]{3,}\b', definition.lower())
        
        # Filter common stop words and focus on business terms
        stop_words = {'the', 'and', 'for', 'are', 'that', 'with', 'such', 'this', 'can', 'may'}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        concept_keywords.update(meaningful_words[:5])  # Top 5 words per concept
        
        # Also extract the base concept name (e.g., "financial_account" -> ["financial", "account"])
        base_concept = concept_id.split('.')[-1]
        concept_keywords.update(base_concept.replace('_', ' ').split())
    
    return concept_keywords

def get_domain_specific_terms(domain):
    """
    Get domain-specific terminology for targeted boosting
    
    Args:
        domain: BIZBOK domain classification (e.g., 'finance', 'insurance', 'manufacturing')
        
    Returns:
        set: Domain-specific terms that should receive relevance boosts
    """
    domain_terms = {
        'finance': {'income', 'revenue', 'payment', 'transaction', 'account', 'asset', 'liability', 
                   'deferred', 'current', 'million', 'thousand', 'balance', 'financial', 'monetary'},
        
        'insurance': {'claim', 'policy', 'premium', 'coverage', 'benefit', 'deductible', 'risk', 
                     'underwriting', 'actuarial', 'insured', 'insurer', 'policyholder'},
        
        'manufacturing': {'production', 'assembly', 'quality', 'process', 'material', 'inventory', 
                         'supply', 'manufacturing', 'operation', 'equipment', 'facility'},
        
        'government': {'regulation', 'compliance', 'policy', 'public', 'citizen', 'service', 
                      'administration', 'governance', 'authority', 'jurisdiction'},
        
        'transportation': {'logistics', 'shipping', 'delivery', 'route', 'vehicle', 'freight', 
                          'carrier', 'transit', 'cargo', 'transportation'},
        
        'telecom': {'network', 'service', 'infrastructure', 'communication', 'bandwidth', 
                   'connectivity', 'subscriber', 'telecommunication'},
        
        'international development org': {'development', 'program', 'project', 'community', 
                                        'beneficiary', 'intervention', 'capacity', 'sustainability'}
    }
    
    return domain_terms.get(domain.lower(), set())

def calculate_domain_relevance_multiplier(keyword, concept_keywords, matched_keywords, document_domain=None):
    """
    Calculate relevance multiplier for domain-specific terms based on document's actual domain
    
    Args:
        keyword: Term to evaluate
        concept_keywords: Set of domain concept keywords
        matched_keywords: Dict of concept -> matched keywords from R4S
        document_domain: BIZBOK domain classification for this specific document
        
    Returns:
        float: Relevance multiplier (1.0 = no boost, >1.0 = boosted)
    """
    multiplier = 1.0
    
    # Boost if keyword appears in concept vocabulary
    if keyword.lower() in concept_keywords:
        multiplier *= 2.0
    
    # Additional boost if keyword was matched by R4S semantic analysis
    for concept_matches in matched_keywords.values():
        if keyword.lower() in [kw.lower() for kw in concept_matches]:
            multiplier *= 1.5
            break
    
    # Domain-specific boosts based on document's actual domain
    if document_domain:
        domain_terms = get_domain_specific_terms(document_domain)
        if keyword.lower() in domain_terms:
            multiplier *= 1.3
    
    return multiplier

def load_input(preprocessed_path="outputs/A2.1_preprocessed_documents.json", 
               domain_path="outputs/A1.2_domain_detection_output.json"):
    """
    Load both preprocessed documents and domain detection data for enhanced keyword extraction
    
    Args:
        preprocessed_path: Path to A2.1 preprocessed documents
        domain_path: Path to A1.2 domain detection output with BIZBOK concepts
        
    Returns:
        dict: Combined data with preprocessed documents and domain intelligence
    """
    script_dir = Path(__file__).parent.parent
    
    # Load A2.1 preprocessed documents (primary input for cleaned text)
    preprocessed_file = script_dir / preprocessed_path
    if not preprocessed_file.exists():
        raise FileNotFoundError(f"Preprocessed documents not found: {preprocessed_file}")
    
    with open(preprocessed_file, 'r', encoding='utf-8') as f:
        preprocessed_data = json.load(f)
    
    # Load A1.2 domain detection (secondary input for BIZBOK concept intelligence)
    domain_file = script_dir / domain_path
    if not domain_file.exists():
        print(f"Warning: Domain detection file not found: {domain_file}")
        print("Proceeding with basic keyword extraction without domain enhancement...")
        return {
            "documents": preprocessed_data["documents"],
            "domain_enhanced": False,
            "preprocessing_stats": {k: v for k, v in preprocessed_data.items() if k != "documents"}
        }
    
    with open(domain_file, 'r', encoding='utf-8') as f:
        domain_data = json.load(f)
    
    # Build concept definitions lookup for domain-aware keyword extraction
    concept_definitions = {}
    matched_keywords_lookup = {}
    
    for doc in domain_data["documents"]:
        doc_id = doc["doc_id"]
        if "concept_definitions" in doc:
            concept_definitions[doc_id] = doc["concept_definitions"]
        if "matched_keywords" in doc:
            matched_keywords_lookup[doc_id] = doc["matched_keywords"]
    
    return {
        "documents": preprocessed_data["documents"],
        "domain_enhanced": True,
        "concept_definitions": concept_definitions,
        "matched_keywords": matched_keywords_lookup,
        "preprocessing_stats": {k: v for k, v in preprocessed_data.items() if k != "documents"}
    }

def process_documents(data):
    """
    Extract keywords and phrases from documents with BIZBOK domain intelligence
    
    Args:
        data: Combined data with preprocessed documents and domain concepts
        
    Returns:
        dict: Documents with domain-enhanced keyword extractions
    """
    documents = data.get("documents", [])
    domain_enhanced = data.get("domain_enhanced", False)
    
    # Collect all document texts for IDF calculation  
    all_texts = [doc.get("cleaned_text", "") for doc in documents]
    idf_scores = calculate_idf(all_texts)
    
    all_keywords = []
    all_phrases = []
    concept_keyword_stats = {"total_concepts": 0, "boosted_terms": 0}
    
    for doc in documents:
        doc_id = doc.get("doc_id", "")
        text = doc.get("cleaned_text", "")
        
        # Calculate base TF-IDF scores
        tf_scores = calculate_tf(text)
        tfidf_scores = calculate_tfidf(tf_scores, idf_scores)
        
        # Apply domain-aware enhancement if available
        if domain_enhanced and doc_id in data.get("concept_definitions", {}):
            concept_defs = data["concept_definitions"][doc_id]
            matched_kws = data.get("matched_keywords", {}).get(doc_id, {})
            document_domain = doc.get("bizbok_domain") or doc.get("domain")  # Get document's actual domain
            
            # Extract concept keywords for this document's domain
            concept_keywords = extract_concept_keywords(concept_defs)
            concept_keyword_stats["total_concepts"] += len(concept_defs)
            
            # Apply domain relevance multipliers to TF-IDF scores with document's specific domain
            enhanced_scores = {}
            for term, score in tfidf_scores.items():
                multiplier = calculate_domain_relevance_multiplier(term, concept_keywords, matched_kws, document_domain)
                enhanced_scores[term] = score * multiplier
                if multiplier > 1.0:
                    concept_keyword_stats["boosted_terms"] += 1
                    
            tfidf_scores = enhanced_scores
            doc["domain_enhanced"] = True
            doc["document_domain"] = document_domain  # Track which domain was used for boosting
            doc["concept_boost_applied"] = len([k for k, v in enhanced_scores.items() 
                                              if calculate_domain_relevance_multiplier(k, concept_keywords, matched_kws, document_domain) > 1.0])
        else:
            doc["domain_enhanced"] = False
            doc["concept_boost_applied"] = 0
        
        # Extract top keywords with domain-aware scoring
        keywords = filter_keywords(tfidf_scores, top_k=30)
        doc["keywords"] = [{"term": word, "score": score} for word, score in keywords]
        all_keywords.extend([word for word, _ in keywords])
        
        # Extract phrases
        phrases = extract_phrases(text)
        phrase_freq = Counter(phrases)
        top_phrases = phrase_freq.most_common(20)
        doc["phrases"] = [{"phrase": phrase, "count": count} for phrase, count in top_phrases]
        all_phrases.extend([phrase for phrase, _ in top_phrases])
    
    # Calculate corpus-level statistics
    keyword_freq = Counter(all_keywords)
    phrase_freq = Counter(all_phrases)
    
    return {
        "documents": documents,
        "count": len(documents),
        "total_unique_keywords": len(set(all_keywords)),
        "total_unique_phrases": len(set(all_phrases)),
        "top_keywords": keyword_freq.most_common(20),
        "top_phrases": phrase_freq.most_common(20),
        "domain_enhancement": {
            "enabled": domain_enhanced,
            "total_concepts_processed": concept_keyword_stats["total_concepts"],
            "terms_boosted": concept_keyword_stats["boosted_terms"],
            "documents_enhanced": sum(1 for doc in documents if doc.get("domain_enhanced", False))
        },
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/A2.2_keyword_extractions.json"):
    """Save keyword extraction results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved keyword extractions to {full_path}")
    
    # Save metadata
    meta_path = full_path.with_suffix('.meta.json')
    metadata = {
        "script": "A2.2_keyword_phrase_extraction.py",
        "timestamp": data["processing_timestamp"],
        "document_count": data["count"],
        "unique_keywords": data["total_unique_keywords"],
        "unique_phrases": data["total_unique_phrases"],
        "output_file": str(full_path)
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    """Main execution - BIZBOK Domain-Enhanced Keyword Extraction"""
    print("="*70)
    print("A2.2: Keyword and Phrase Extraction (BIZBOK DOMAIN-ENHANCED)")
    print("="*70)
    
    try:
        # Load both preprocessed documents and domain intelligence
        print("Loading preprocessed documents and BIZBOK domain intelligence...")
        input_data = load_input()
        
        # Extract keywords with domain enhancement
        print(f"Extracting keywords from {len(input_data['documents'])} documents...")
        if input_data.get("domain_enhanced"):
            print("[OK] BIZBOK domain enhancement: ENABLED")
        else:
            print("[WARNING] BIZBOK domain enhancement: DISABLED (A1.2 not found)")
            
        output_data = process_documents(input_data)
        
        # Display domain enhancement results
        domain_stats = output_data.get("domain_enhancement", {})
        if domain_stats.get("enabled"):
            print(f"\nBIZBOK Domain Enhancement Results:")
            print(f"  Concepts processed: {domain_stats.get('total_concepts_processed', 0)}")
            print(f"  Terms boosted: {domain_stats.get('terms_boosted', 0)}")
            print(f"  Documents enhanced: {domain_stats.get('documents_enhanced', 0)}/{output_data['count']}")
        
        # Display extraction results  
        print(f"\nKeyword Extraction Statistics:")
        print(f"  Unique Keywords: {output_data['total_unique_keywords']}")
        print(f"  Unique Phrases: {output_data['total_unique_phrases']}")
        
        print(f"\nTop 10 Domain-Enhanced Keywords:")
        for keyword, count in output_data["top_keywords"][:10]:
            print(f"  - {keyword}: {count}")
        
        print(f"\nTop 10 Extracted Phrases:")
        for phrase, count in output_data["top_phrases"][:10]:
            print(f"  - {phrase}: {count}")
        
        # Save output
        save_output(output_data)
        
        print("\nA2.2 Keyword Extraction completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.2 Keyword Extraction: {str(e)}")
        raise

if __name__ == "__main__":
    main()