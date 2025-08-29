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

def load_input(input_path="outputs/A2.1_preprocessed_documents.json"):
    """Load preprocessed documents from A2.1"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_documents(data):
    """
    Extract keywords and phrases from documents
    
    Args:
        data: Preprocessed document data
        
    Returns:
        dict: Documents with extracted keywords
    """
    documents = data.get("documents", [])
    
    # Collect all document texts for IDF calculation
    all_texts = [doc.get("cleaned_text", "") for doc in documents]
    idf_scores = calculate_idf(all_texts)
    
    all_keywords = []
    all_phrases = []
    
    for doc in documents:
        text = doc.get("cleaned_text", "")
        
        # Calculate TF-IDF for this document
        tf_scores = calculate_tf(text)
        tfidf_scores = calculate_tfidf(tf_scores, idf_scores)
        
        # Extract top keywords
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
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/A2.2_keyword_extractions.json"):
    """Save keyword extraction results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved keyword extractions to {full_path}")
    
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
    """Main execution"""
    print("="*60)
    print("A2.2: Keyword and Phrase Extraction")
    print("="*60)
    
    try:
        # Load preprocessed documents
        print("Loading preprocessed documents...")
        input_data = load_input()
        
        # Extract keywords and phrases
        print(f"Extracting keywords from {input_data['count']} documents...")
        output_data = process_documents(input_data)
        
        # Display results
        print(f"\nExtraction Statistics:")
        print(f"  Unique Keywords: {output_data['total_unique_keywords']}")
        print(f"  Unique Phrases: {output_data['total_unique_phrases']}")
        
        print(f"\nTop 10 Keywords:")
        for keyword, count in output_data["top_keywords"][:10]:
            print(f"  - {keyword}: {count}")
        
        print(f"\nTop 10 Phrases:")
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