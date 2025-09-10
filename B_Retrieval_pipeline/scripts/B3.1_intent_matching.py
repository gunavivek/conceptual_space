#!/usr/bin/env python3
"""
B3.1: Intent Matching Strategy - FIXED VERSION
Matches questions to chunks based on semantic similarity and intent analysis
"""

import json
import math
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

def preprocess_text(text):
    """Clean and preprocess text for similarity calculation"""
    if not text:
        return ""
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def calculate_tf_idf(text, all_texts):
    """
    Calculate TF-IDF scores for text
    
    Args:
        text: Target text
        all_texts: All texts in corpus for IDF calculation
    
    Returns:
        dict: TF-IDF scores for each word
    """
    if not text or not all_texts:
        return {}
    
    # Calculate term frequency (TF)
    words = preprocess_text(text).split()
    if not words:
        return {}
    
    word_count = Counter(words)
    total_words = len(words)
    tf_scores = {word: count / total_words for word, count in word_count.items()}
    
    # Calculate inverse document frequency (IDF)
    total_docs = len(all_texts)
    idf_scores = {}
    
    for word in tf_scores:
        docs_containing_word = sum(1 for doc in all_texts if word in preprocess_text(doc))
        if docs_containing_word > 0:
            idf_scores[word] = math.log(total_docs / docs_containing_word)
        else:
            idf_scores[word] = 0
    
    # Calculate TF-IDF
    tfidf_scores = {word: tf_scores[word] * idf_scores[word] for word in tf_scores}
    return tfidf_scores

def cosine_similarity(text1, text2, all_texts):
    """
    Calculate cosine similarity between two texts using TF-IDF
    
    Args:
        text1: First text
        text2: Second text  
        all_texts: All texts for TF-IDF calculation
        
    Returns:
        float: Cosine similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # Get TF-IDF vectors
    tfidf1 = calculate_tf_idf(text1, all_texts)
    tfidf2 = calculate_tf_idf(text2, all_texts)
    
    if not tfidf1 or not tfidf2:
        return 0.0
    
    # Get all unique words
    all_words = set(tfidf1.keys()) | set(tfidf2.keys())
    
    if not all_words:
        return 0.0
    
    # Create vectors
    vector1 = [tfidf1.get(word, 0) for word in all_words]
    vector2 = [tfidf2.get(word, 0) for word in all_words]
    
    # Calculate dot product
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(v * v for v in vector1))
    magnitude2 = math.sqrt(sum(v * v for v in vector2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    return max(0.0, min(1.0, similarity))

def simple_keyword_similarity(question, chunk_content):
    """
    Simple keyword-based similarity as fallback
    
    Args:
        question: Question text
        chunk_content: Chunk content
        
    Returns:
        float: Keyword similarity score (0-1)
    """
    if not question or not chunk_content:
        return 0.0
    
    question_words = set(preprocess_text(question).split())
    chunk_words = set(preprocess_text(chunk_content).split())
    
    if not question_words or not chunk_words:
        return 0.0
    
    intersection = len(question_words & chunk_words)
    union = len(question_words | chunk_words)
    
    return intersection / union if union > 0 else 0.0

def calculate_intent_boost(question_intent, chunk_content):
    """
    Calculate intent-based boost for similarity score
    
    Args:
        question_intent: Intent analysis from B2.1
        chunk_content: Chunk content
        
    Returns:
        float: Intent boost factor (0.8-1.2)
    """
    if not question_intent or not chunk_content:
        return 1.0
    
    intent_type = question_intent.get("primary_intent", "").lower()
    content_lower = chunk_content.lower()
    
    boost = 1.0
    
    # Intent-specific keyword boosting
    if intent_type == "comparison":
        comparison_keywords = ["higher", "lower", "increase", "decrease", "change", "compared", "versus", "than"]
        if any(keyword in content_lower for keyword in comparison_keywords):
            boost += 0.2
    
    elif intent_type == "calculation":
        calc_keywords = ["total", "sum", "average", "percentage", "calculate", "compute"]
        if any(keyword in content_lower for keyword in calc_keywords):
            boost += 0.15
    
    elif intent_type == "identification":
        id_keywords = ["is", "was", "are", "were", "identify", "name", "which"]
        if any(keyword in content_lower for keyword in id_keywords):
            boost += 0.1
    
    elif intent_type == "factual":
        # Factual questions benefit from concrete numbers and facts
        if re.search(r'\d+', content_lower):
            boost += 0.1
    
    elif intent_type == "temporal":
        temporal_keywords = ["year", "month", "when", "time", "period", "2018", "2019", "2020"]
        if any(keyword in content_lower for keyword in temporal_keywords):
            boost += 0.2
    
    return max(0.8, min(1.2, boost))

def calculate_semantic_similarity(question, chunk_content, all_chunk_texts, intent_analysis=None):
    """
    Calculate comprehensive semantic similarity between question and chunk
    
    Args:
        question: Question text
        chunk_content: Chunk content
        all_chunk_texts: All chunk texts for TF-IDF corpus
        intent_analysis: Intent analysis from B2.1
        
    Returns:
        float: Semantic similarity score (0-1)
    """
    if not question or not chunk_content:
        return 0.0
    
    # Primary similarity using TF-IDF cosine similarity
    tfidf_similarity = cosine_similarity(question, chunk_content, all_chunk_texts + [question])
    
    # Fallback to keyword similarity if TF-IDF fails
    keyword_similarity_score = simple_keyword_similarity(question, chunk_content)
    
    # Use the higher of the two scores
    base_similarity = max(tfidf_similarity, keyword_similarity_score)
    
    # Apply intent boost
    intent_boost = calculate_intent_boost(intent_analysis, chunk_content)
    final_similarity = base_similarity * intent_boost
    
    return max(0.0, min(1.0, final_similarity))

def match_chunks_by_intent(question_text, chunks, intent_modeling, temporal_analysis=None):
    """
    Interface function for orchestrator to match chunks by intent using semantic similarity
    
    Args:
        question_text: Question string
        chunks: List of chunks from A-Pipeline
        intent_modeling: Intent analysis from B2.1
        temporal_analysis: Temporal analysis from B2.4 (optional)
        
    Returns:
        dict: Intent matching results with ranked chunks
    """
    if not question_text or not chunks:
        return {"ranked_chunks": [], "matching_stats": {"total_chunks": 0}}
    
    # Extract all chunk texts for TF-IDF corpus
    all_chunk_texts = [chunk.get("content", "") for chunk in chunks if chunk.get("content")]
    
    if not all_chunk_texts:
        return {"ranked_chunks": [], "matching_stats": {"total_chunks": len(chunks)}}
    
    # Calculate similarity for each chunk
    ranked_chunks = []
    
    for chunk in chunks:
        chunk_content = chunk.get("content", "")
        if not chunk_content:
            continue
        
        # Calculate semantic similarity
        similarity_score = calculate_semantic_similarity(
            question_text, 
            chunk_content, 
            all_chunk_texts,
            intent_modeling.get("intent_analysis", {}) if intent_modeling else None
        )
        
        # Only include chunks with meaningful similarity
        if similarity_score > 0.01:  # Very low threshold to include most matches
            ranked_chunks.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "content": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
                "full_content": chunk_content,  # Keep full content for further processing
                "similarity_score": round(similarity_score, 4),
                "match_strategy": "semantic_similarity",
                "match_details": {
                    "content_length": len(chunk_content),
                    "intent_type": intent_modeling.get("intent_analysis", {}).get("primary_intent", "unknown") if intent_modeling else "unknown",
                    "has_temporal": temporal_analysis is not None and temporal_analysis.get("temporal_confidence", 0) > 0.5
                }
            })
    
    # Sort by similarity score (highest first)
    ranked_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Calculate matching statistics
    matching_stats = {
        "total_chunks": len(chunks),
        "matched_chunks": len(ranked_chunks),
        "avg_similarity": round(sum(chunk["similarity_score"] for chunk in ranked_chunks) / len(ranked_chunks), 4) if ranked_chunks else 0.0,
        "max_similarity": round(ranked_chunks[0]["similarity_score"], 4) if ranked_chunks else 0.0,
        "min_similarity": round(ranked_chunks[-1]["similarity_score"], 4) if ranked_chunks else 0.0,
        "similarity_method": "tf_idf_cosine_with_intent_boost"
    }
    
    return {
        "ranked_chunks": ranked_chunks[:10],  # Return top 10 matches
        "matching_stats": matching_stats,
        "question_processed": question_text[:100] + "..." if len(question_text) > 100 else question_text,
        "processing_timestamp": datetime.now().isoformat()
    }

# Backward compatibility functions (in case other parts of pipeline call them)
def calculate_intent_similarity(question_intent, concept_keywords, concept_domain):
    """Legacy function - now redirects to semantic similarity"""
    return 0.5  # Default moderate similarity

def match_intent_to_concepts(question_data, expanded_concepts):
    """Legacy function - maintained for compatibility"""
    return []

def analyze_intent_matching_quality(matches, question_data):
    """Analyze the quality of intent matching"""
    if not matches:
        return {"quality_score": 0.0, "analysis": "No matches found"}
    
    scores = [match["similarity_score"] for match in matches]
    quality_score = sum(scores) / len(scores)
    
    return {
        "quality_score": round(quality_score, 4),
        "score_distribution": {
            "high_quality": len([s for s in scores if s > 0.7]),
            "medium_quality": len([s for s in scores if 0.3 <= s <= 0.7]),  
            "low_quality": len([s for s in scores if s < 0.3])
        },
        "analysis": f"Average similarity: {quality_score:.4f}, Best match: {max(scores):.4f}"
    }

if __name__ == "__main__":
    # Test the semantic similarity function
    test_question = "What was the revenue in 2019?"
    test_chunks = [
        {"chunk_id": "test_1", "content": "Revenue for 2019 was $100 million, an increase from 2018."},
        {"chunk_id": "test_2", "content": "The company expenses increased significantly during the year."},
        {"chunk_id": "test_3", "content": "Total revenue grew by 15% in fiscal year 2019."}
    ]
    
    test_intent = {
        "intent_analysis": {
            "primary_intent": "factual",
            "keywords": ["revenue", "2019"]
        }
    }
    
    result = match_chunks_by_intent(test_question, test_chunks, test_intent)
    print("Test Results:")
    print(f"Matched chunks: {result['matching_stats']['matched_chunks']}")
    for chunk in result['ranked_chunks']:
        print(f"  {chunk['chunk_id']}: {chunk['similarity_score']:.4f}")