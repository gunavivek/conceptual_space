#!/usr/bin/env python3
"""
B3.3: Answer-Backward Matching Strategy - FIXED VERSION
Matches chunks based on expected answer types using semantic similarity
"""

import json
import re
import math
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

def extract_answer_relevant_features(chunk_content, expected_answer_type):
    """
    Extract features from chunk that are relevant to the expected answer type
    
    Args:
        chunk_content: Chunk text content
        expected_answer_type: Type of answer expected (numeric, text, date, etc.)
        
    Returns:
        dict: Extracted features with relevance scores
    """
    features = {
        "has_numbers": False,
        "numeric_values": [],
        "has_dates": False,
        "date_values": [],
        "has_entities": False,
        "entity_count": 0,
        "has_comparisons": False,
        "content_length": len(chunk_content),
        "relevance_score": 0.0
    }
    
    if not chunk_content:
        return features
    
    content_lower = chunk_content.lower()
    
    # Extract numbers
    numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', chunk_content)
    if numbers:
        features["has_numbers"] = True
        features["numeric_values"] = numbers[:5]  # Keep first 5 numbers
    
    # Extract years/dates
    years = re.findall(r'\b20[0-2]\d\b|\b19[0-9]\d\b', chunk_content)
    if years:
        features["has_dates"] = True
        features["date_values"] = list(set(years))
    
    # Check for comparison words
    comparison_words = ["increase", "decrease", "higher", "lower", "more", "less", "change", "growth", "decline"]
    if any(word in content_lower for word in comparison_words):
        features["has_comparisons"] = True
    
    # Count potential entities (capitalized words)
    entities = re.findall(r'\b[A-Z][a-z]+\b', chunk_content)
    features["entity_count"] = len(entities)
    features["has_entities"] = len(entities) > 0
    
    # Calculate relevance based on expected answer type
    if expected_answer_type == "numeric":
        if features["has_numbers"]:
            features["relevance_score"] += 0.6
        if features["has_comparisons"]:
            features["relevance_score"] += 0.2
        if "percentage" in content_lower or "%" in chunk_content:
            features["relevance_score"] += 0.2
            
    elif expected_answer_type == "date":
        if features["has_dates"]:
            features["relevance_score"] += 0.7
        if any(word in content_lower for word in ["year", "month", "quarter", "period"]):
            features["relevance_score"] += 0.3
            
    elif expected_answer_type == "text":
        # Text answers benefit from entities and context
        if features["has_entities"]:
            features["relevance_score"] += 0.3
        if features["content_length"] > 50:
            features["relevance_score"] += 0.2
        # Default base relevance for text
        features["relevance_score"] += 0.3
    
    # Normalize relevance score
    features["relevance_score"] = min(1.0, features["relevance_score"])
    
    return features

def calculate_answer_similarity(chunk_content, answer_expectation):
    """
    Calculate semantic similarity between chunk and answer expectation
    
    Args:
        chunk_content: Chunk text content
        answer_expectation: Answer expectations from B2.3
        
    Returns:
        float: Similarity score (0-1)
    """
    if not chunk_content or not answer_expectation:
        return 0.0
    
    # Get expected answer type
    answer_prediction = answer_expectation.get("answer_prediction", {})
    expected_type = answer_prediction.get("primary_type", "text")
    confidence_scores = answer_expectation.get("confidence_scores", {})
    
    # Extract answer-relevant features from chunk
    features = extract_answer_relevant_features(chunk_content, expected_type)
    
    # Base score from feature relevance
    base_score = features["relevance_score"]
    
    # Adjust based on validation criteria
    validation_criteria = answer_expectation.get("validation_checks", {})
    
    # Check for required elements
    if validation_criteria.get("needs_context") and features["content_length"] > 100:
        base_score += 0.1
    
    if validation_criteria.get("needs_evidence") and features["has_numbers"]:
        base_score += 0.15
    
    if validation_criteria.get("temporal_requirement") and features["has_dates"]:
        base_score += 0.15
    
    # Keyword matching bonus
    chunk_words = set(preprocess_text(chunk_content).split())
    
    # Check for answer type specific keywords
    type_keywords = {
        "numeric": ["value", "amount", "total", "sum", "cost", "revenue", "price", "percentage"],
        "date": ["year", "month", "date", "when", "time", "period", "quarter"],
        "text": ["is", "are", "was", "were", "define", "describe", "include", "contain"]
    }
    
    relevant_keywords = type_keywords.get(expected_type, [])
    keyword_matches = sum(1 for kw in relevant_keywords if kw in chunk_words)
    keyword_bonus = min(0.2, keyword_matches * 0.05)
    
    # Confidence adjustment
    type_confidence = confidence_scores.get(expected_type, 0.5)
    confidence_factor = 0.8 + (type_confidence * 0.4)  # Range: 0.8 to 1.2
    
    # Calculate final score
    final_score = (base_score + keyword_bonus) * confidence_factor
    
    return max(0.0, min(1.0, final_score))

def calculate_validation_alignment(chunk_content, validation_criteria):
    """
    Calculate how well chunk aligns with answer validation criteria
    
    Args:
        chunk_content: Chunk text content
        validation_criteria: Validation criteria from B2.3
        
    Returns:
        float: Alignment score (0-1)
    """
    if not chunk_content or not validation_criteria:
        return 0.3  # Low default score instead of 0.5
    
    alignment_score = 0.0
    criteria_count = 0
    
    content_lower = chunk_content.lower()
    
    # Check format requirements
    format_spec = validation_criteria.get("format", {})
    if format_spec:
        criteria_count += 1
        if format_spec.get("requires_number") and re.search(r'\d+', chunk_content):
            alignment_score += 1.0
        elif format_spec.get("requires_text") and len(chunk_content) > 20:
            alignment_score += 1.0
    
    # Check content requirements
    must_contain = validation_criteria.get("must_contain", [])
    if must_contain:
        criteria_count += len(must_contain)
        for requirement in must_contain:
            if requirement.lower() in content_lower:
                alignment_score += 1.0
    
    # Check completeness
    if validation_criteria.get("requires_explanation") and len(chunk_content) > 100:
        criteria_count += 1
        alignment_score += 1.0
    
    if validation_criteria.get("requires_comparison"):
        criteria_count += 1
        comparison_words = ["increase", "decrease", "higher", "lower", "change", "versus", "compared"]
        if any(word in content_lower for word in comparison_words):
            alignment_score += 1.0
    
    # Calculate normalized alignment
    if criteria_count > 0:
        return alignment_score / criteria_count
    else:
        # If no specific criteria, check general quality
        quality_score = 0.3  # Base score
        if len(chunk_content) > 50:
            quality_score += 0.2
        if re.search(r'\d+', chunk_content):
            quality_score += 0.1
        if re.search(r'\b[A-Z][a-z]+\b', chunk_content):  # Has entities
            quality_score += 0.1
        return min(1.0, quality_score)

def match_by_answer_expectations(chunks, answer_expectation):
    """
    Interface function for orchestrator to match chunks by answer expectations
    
    Args:
        chunks: List of chunks from A-Pipeline
        answer_expectation: Answer expectations from B2.3
        
    Returns:
        dict: Answer-backward matching results with ranked chunks
    """
    if not chunks:
        return {
            "strategy": "answer_backward_matching",
            "ranked_chunks": [],
            "total_matches": 0,
            "processing_timestamp": datetime.now().isoformat()
        }
    
    # Process each chunk
    ranked_chunks = []
    
    for chunk in chunks:
        chunk_content = chunk.get("content", "")
        if not chunk_content:
            continue
        
        # Calculate answer similarity
        answer_similarity = calculate_answer_similarity(chunk_content, answer_expectation)
        
        # Calculate validation alignment
        validation_criteria = answer_expectation.get("validation_criteria", {})
        validation_alignment = calculate_validation_alignment(chunk_content, validation_criteria)
        
        # Combined score with weighted components
        combined_score = (answer_similarity * 0.6) + (validation_alignment * 0.4)
        
        # Only include chunks with meaningful scores
        if combined_score > 0.1:
            # Extract answer type for details
            answer_prediction = answer_expectation.get("answer_prediction", {})
            expected_type = answer_prediction.get("primary_type", "text")
            
            ranked_chunks.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "content": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
                "similarity_score": round(combined_score, 4),
                "match_strategy": "answer_backward",
                "match_details": {
                    "answer_similarity": round(answer_similarity, 4),
                    "validation_alignment": round(validation_alignment, 4),
                    "expected_answer_type": expected_type,
                    "can_provide_answer": answer_similarity > 0.5,
                    "meets_validation": validation_alignment > 0.4,
                    "content_features": {
                        "has_numbers": bool(re.search(r'\d+', chunk_content)),
                        "has_dates": bool(re.search(r'\b20[0-2]\d\b|\b19[0-9]\d\b', chunk_content)),
                        "content_length": len(chunk_content)
                    }
                }
            })
    
    # Sort by similarity score
    ranked_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Return top matches (limit to 8 as this is a supporting strategy)
    return {
        "strategy": "answer_backward_matching",
        "ranked_chunks": ranked_chunks[:8],
        "total_matches": len(ranked_chunks),
        "matching_stats": {
            "total_chunks_processed": len(chunks),
            "chunks_with_scores": len(ranked_chunks),
            "avg_score": round(sum(c["similarity_score"] for c in ranked_chunks) / len(ranked_chunks), 4) if ranked_chunks else 0.0,
            "max_score": round(ranked_chunks[0]["similarity_score"], 4) if ranked_chunks else 0.0
        },
        "processing_timestamp": datetime.now().isoformat()
    }

# Legacy functions for compatibility
def determine_concept_answer_capability(concept, answer_expectation):
    """Legacy function - redirects to new implementation"""
    content = concept.get("content", "")
    return calculate_answer_similarity(content, answer_expectation)

def calculate_answer_alignment_score(concept_keywords, answer_validation_criteria):
    """Legacy function - redirects to new implementation"""
    content = " ".join(concept_keywords) if concept_keywords else ""
    return calculate_validation_alignment(content, answer_validation_criteria)

def match_answer_backward_to_concepts(question_data, expanded_concepts):
    """Legacy function - maintained for compatibility"""
    return []

def analyze_answer_matching_quality(matches, question_data):
    """Analyze the quality of answer-backward matching"""
    if not matches:
        return {"quality_score": 0.0, "analysis": "No matches found"}
    
    scores = [match["similarity_score"] for match in matches]
    quality_score = sum(scores) / len(scores)
    
    return {
        "quality_score": round(quality_score, 4),
        "score_distribution": {
            "high_quality": len([s for s in scores if s > 0.7]),
            "medium_quality": len([s for s in scores if 0.4 <= s <= 0.7]),
            "low_quality": len([s for s in scores if s < 0.4])
        },
        "analysis": f"Average similarity: {quality_score:.4f}, Best match: {max(scores):.4f}"
    }

def load_inputs():
    """Load B2.3 answer expectation data and A-pipeline chunks"""
    script_dir = Path(__file__).parent.parent
    
    # Load B2.3 answer expectations
    b2_3_path = script_dir / "outputs" / "B2.3_answer_expectation_output.json"
    if not b2_3_path.exists():
        raise FileNotFoundError(f"B2.3 output not found: {b2_3_path}")
    
    with open(b2_3_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # Load A-pipeline chunks from A3_multi_strategy_chunks.json (same as B3.1)
    a_pipeline_path = script_dir.parent / "A_Concept_pipeline" / "outputs" / "A3_multi_strategy_chunks.json"
    chunks = []
    
    if a_pipeline_path.exists():
        with open(a_pipeline_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
            # Extract chunks from A3 output
            if isinstance(chunk_data, dict) and "chunks" in chunk_data:
                for chunk in chunk_data["chunks"]:
                    chunks.append({
                        "chunk_id": chunk.get("chunk_id", ""),
                        "content": chunk.get("content", ""),
                        "doc_id": chunk.get("doc_id", ""),
                        "chunk_type": chunk.get("chunk_type", "")
                    })
    
    if not chunks:
        raise FileNotFoundError(f"No real chunks found in A-pipeline output: {a_pipeline_path}")
    
    return questions_data, chunks

def save_output(data, output_path="outputs/B3.3_answer_backward_matching_output.json"):
    """Save answer backward matching results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    # Create output directory if needed
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved answer backward matching results to {full_path}")

def main():
    """Main execution for processing all 20 questions"""
    print("="*60)
    print("B3.3: Answer Backward Matching Strategy")
    print("="*60)
    
    try:
        # Load inputs
        print("Loading questions and chunks...")
        questions_data, chunks = load_inputs()
        
        if not isinstance(questions_data, list):
            questions_data = [questions_data]
        
        all_results = []
        
        print(f"Processing {len(questions_data)} questions with {len(chunks)} chunks...\n")
        
        # Process each question
        for i, question_data in enumerate(questions_data):
            question_text = question_data.get("question", "")
            print(f"[{i+1:2d}/20] Backward matching: {question_text[:60]}...")
            
            # Perform answer backward matching
            matching_result = match_by_answer_expectations(chunks, question_data)
            
            # Add question metadata
            result = {
                "question_id": question_data.get("question_id", f"q_{i}"),
                "question": question_text,
                "strategy": "answer_backward_matching",
                "ranked_chunks": matching_result.get("ranked_chunks", []),
                "total_matches": matching_result.get("total_matches", 0),
                "quality_analysis": analyze_answer_matching_quality(
                    matching_result.get("ranked_chunks", []), 
                    question_data
                ),
                "processing_timestamp": datetime.now().isoformat()
            }
            
            all_results.append(result)
            
            # Brief results display
            matches = result["total_matches"]
            quality = result["quality_analysis"]
            print(f"       Matches: {matches}/{len(chunks)} chunks (quality: {quality.get('quality_score', 0):.3f})")
        
        # Save all results
        save_output(all_results)
        
        # Summary statistics
        print(f"\n{'='*60}")
        print("ANSWER BACKWARD MATCHING SUMMARY")
        print(f"{'='*60}")
        print(f"Total questions processed: {len(all_results)}")
        
        # Quality statistics
        qualities = [result["quality_analysis"]["quality_score"] for result in all_results]
        avg_quality = sum(qualities) / len(qualities) if qualities else 0
        
        print(f"Average matching quality: {avg_quality:.3f}")
        
        # Match distribution
        high_quality = sum(1 for q in qualities if q > 0.7)
        medium_quality = sum(1 for q in qualities if 0.4 <= q <= 0.7)
        low_quality = sum(1 for q in qualities if q < 0.4)
        
        print(f"Quality distribution:")
        print(f"  High quality (>0.7): {high_quality} questions")
        print(f"  Medium quality (0.4-0.7): {medium_quality} questions") 
        print(f"  Low quality (<0.4): {low_quality} questions")
        
        print("\nB3.3 Answer Backward Matching completed successfully!")
        
    except Exception as e:
        print(f"Error in B3.3 Answer Backward Matching: {str(e)}")
        raise

if __name__ == "__main__":
    main()