#!/usr/bin/env python3
"""
B3.3: Answer Capability Assessment - ENHANCED VERSION
Assesses chunks' capability to provide required answer types using semantic analysis and concept integration
"""

import json
import re
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

def load_concept_data():
    """
    Load concept data from A-pipeline for enhancement
    
    Returns:
        dict: Concept data with keywords, importance scores, and metadata
    """
    script_dir = Path(__file__).parent.parent
    concept_path = script_dir.parent / "A_Concept_pipeline" / "outputs" / "A2.4_core_concepts.json"
    
    if not concept_path.exists():
        print(f"Warning: Concept data not found at {concept_path}")
        return {}
    
    try:
        with open(concept_path, 'r', encoding='utf-8') as f:
            concept_data = json.load(f)
        
        # Convert to lookup dict by concept_id
        concepts_lookup = {}
        for concept in concept_data.get("core_concepts", []):
            concept_id = concept.get("concept_id")
            if concept_id:
                concepts_lookup[concept_id] = {
                    "canonical_name": concept.get("canonical_name", ""),
                    "importance_score": concept.get("importance_score", 0.5),
                    "primary_keywords": concept.get("primary_keywords", []),
                    "keyword_frequencies": concept.get("keyword_frequencies", {}),
                    "document_count": concept.get("document_count", 0),
                    "coverage_ratio": concept.get("coverage_ratio", 0.0)
                }
        
        return concepts_lookup
    except Exception as e:
        print(f"Warning: Error loading concept data: {e}")
        return {}

def calculate_concept_enhancement(chunk, answer_expectation, concepts_lookup):
    """
    Calculate concept-based enhancement for answer capability assessment
    
    Args:
        chunk: Chunk with concept memberships
        answer_expectation: Answer expectations from B2.3
        concepts_lookup: Concept data lookup
        
    Returns:
        dict: Concept enhancement scores and details
    """
    concept_memberships = chunk.get("concept_memberships", [])
    membership_scores = chunk.get("membership_scores", {})
    
    if not concept_memberships or not concepts_lookup:
        return {
            "concept_boost": 0.0,
            "importance_multiplier": 1.0,
            "keyword_enhancement": 0.0,
            "concept_details": []
        }
    
    # Get expected answer type for concept matching
    answer_prediction = answer_expectation.get("answer_prediction", {})
    expected_type = answer_prediction.get("primary_type", "text")
    
    concept_boost = 0.0
    total_importance = 0.0
    enhanced_keywords = set()
    concept_details = []
    
    for concept_id in concept_memberships:
        if concept_id in concepts_lookup:
            concept_info = concepts_lookup[concept_id]
            membership_score = membership_scores.get(concept_id, 0.0)
            importance_score = concept_info.get("importance_score", 0.5)
            concept_keywords = concept_info.get("primary_keywords", [])
            
            # Calculate concept relevance to answer type
            concept_relevance = calculate_concept_answer_relevance(
                concept_keywords, 
                concept_info.get("canonical_name", ""),
                expected_type
            )
            
            # Concept boost based on membership * importance * relevance
            individual_boost = membership_score * importance_score * concept_relevance * 0.15
            concept_boost += individual_boost
            
            # Accumulate importance for multiplier
            total_importance += importance_score * membership_score
            
            # Add concept keywords to enhanced set
            enhanced_keywords.update([kw.lower() for kw in concept_keywords])
            
            concept_details.append({
                "concept_id": concept_id,
                "canonical_name": concept_info.get("canonical_name", ""),
                "membership_score": membership_score,
                "importance_score": importance_score,
                "concept_relevance": concept_relevance,
                "individual_boost": individual_boost
            })
    
    # Calculate importance multiplier (0.9 to 1.2)
    avg_importance = total_importance / len(concept_memberships) if concept_memberships else 0.5
    importance_multiplier = 0.9 + (avg_importance * 0.3)
    
    # Calculate keyword enhancement bonus
    chunk_content = chunk.get("content", "").lower()
    enhanced_keyword_matches = sum(1 for kw in enhanced_keywords if kw in chunk_content)
    keyword_enhancement = min(0.1, enhanced_keyword_matches * 0.02)
    
    return {
        "concept_boost": min(0.3, concept_boost),  # Cap at 0.3
        "importance_multiplier": min(1.2, importance_multiplier),  # Cap at 1.2
        "keyword_enhancement": keyword_enhancement,
        "concept_details": concept_details,
        "enhanced_keywords_count": len(enhanced_keywords),
        "enhanced_keyword_matches": enhanced_keyword_matches
    }

def calculate_concept_answer_relevance(concept_keywords, concept_name, expected_answer_type):
    """
    Calculate how relevant a concept is to the expected answer type
    
    Args:
        concept_keywords: List of concept keywords
        concept_name: Concept canonical name
        expected_answer_type: Expected answer type (numeric, date, text)
        
    Returns:
        float: Relevance score (0-1)
    """
    if not concept_keywords:
        return 0.3  # Default relevance
    
    concept_text = f"{concept_name} {' '.join(concept_keywords)}".lower()
    
    # Type-specific relevance patterns
    type_relevance_patterns = {
        "numeric": [
            "revenue", "income", "cost", "expense", "profit", "sales", "total", "amount",
            "price", "value", "percentage", "rate", "change", "increase", "decrease",
            "million", "billion", "thousand", "dollar", "financial"
        ],
        "date": [
            "year", "quarter", "period", "time", "when", "date", "fiscal", "annual",
            "monthly", "reporting", "2018", "2019", "2020", "2021", "timeline"
        ],
        "text": [
            "policy", "method", "approach", "description", "definition", "explanation",
            "standard", "principle", "guideline", "procedure", "practice"
        ]
    }
    
    relevant_patterns = type_relevance_patterns.get(expected_answer_type, [])
    
    # Count pattern matches
    pattern_matches = sum(1 for pattern in relevant_patterns if pattern in concept_text)
    
    # Calculate relevance score
    if pattern_matches > 0:
        relevance = min(1.0, 0.5 + (pattern_matches * 0.1))
    else:
        relevance = 0.3  # Base relevance for all concepts
    
    return relevance

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

def assess_answer_capability(chunks, answer_expectation):
    """
    Interface function for orchestrator to assess chunks' capability to provide required answers
    
    Args:
        chunks: List of chunks from A-Pipeline with concept memberships
        answer_expectation: Answer expectations from B2.3
        
    Returns:
        dict: Answer capability assessment results with ranked chunks
    """
    if not chunks:
        return {
            "strategy": "answer_capability_assessment",
            "ranked_chunks": [],
            "total_matches": 0,
            "processing_timestamp": datetime.now().isoformat()
        }
    
    # Load concept data for enhancement
    concepts_lookup = load_concept_data()
    concept_enhancement_enabled = len(concepts_lookup) > 0
    
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
        
        # Calculate concept enhancement (if available)
        if concept_enhancement_enabled:
            concept_enhancement = calculate_concept_enhancement(chunk, answer_expectation, concepts_lookup)
            concept_boost = concept_enhancement["concept_boost"]
            importance_multiplier = concept_enhancement["importance_multiplier"]
            keyword_enhancement = concept_enhancement["keyword_enhancement"]
        else:
            concept_enhancement = {
                "concept_boost": 0.0,
                "importance_multiplier": 1.0,
                "keyword_enhancement": 0.0,
                "concept_details": []
            }
            concept_boost = 0.0
            importance_multiplier = 1.0
            keyword_enhancement = 0.0
        
        # Enhanced combined score with concept integration
        base_score = (answer_similarity * 0.6) + (validation_alignment * 0.4)
        enhanced_score = base_score + concept_boost + keyword_enhancement
        combined_score = enhanced_score * importance_multiplier
        
        # Only include chunks with meaningful scores
        if combined_score > 0.1:
            # Extract answer type for details
            answer_prediction = answer_expectation.get("answer_prediction", {})
            expected_type = answer_prediction.get("primary_type", "text")
            
            ranked_chunks.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "content": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
                "similarity_score": round(combined_score, 4),
                "match_strategy": "capability_assessment",
                "assessment_details": {
                    "answer_similarity": round(answer_similarity, 4),
                    "validation_alignment": round(validation_alignment, 4),
                    "expected_answer_type": expected_type,
                    "can_provide_answer": answer_similarity > 0.5,
                    "meets_validation": validation_alignment > 0.4,
                    "content_features": {
                        "has_numbers": bool(re.search(r'\d+', chunk_content)),
                        "has_dates": bool(re.search(r'\b20[0-2]\d\b|\b19[0-9]\d\b', chunk_content)),
                        "content_length": len(chunk_content)
                    },
                    "concept_enhancement": {
                        "enabled": concept_enhancement_enabled,
                        "concept_memberships": chunk.get("concept_memberships", []),
                        "membership_scores": chunk.get("membership_scores", {}),
                        "concept_boost": round(concept_boost, 4),
                        "importance_multiplier": round(importance_multiplier, 4),
                        "keyword_enhancement": round(keyword_enhancement, 4),
                        "base_score": round(base_score, 4),
                        "enhanced_score": round(enhanced_score, 4),
                        "concept_details": concept_enhancement.get("concept_details", [])
                    }
                }
            })
    
    # Sort by similarity score
    ranked_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Return top matches (limit to 8 as this is a supporting strategy)
    return {
        "strategy": "answer_capability_assessment",
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

# Backward compatibility alias
match_by_answer_expectations = assess_answer_capability

def analyze_capability_assessment_quality(matches, question_data):
    """Analyze the quality of answer capability assessment"""
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

def assess_answer_capability_with_b25(chunks, answer_expectation, question_concepts, concepts_lookup):
    """
    B2.5-Enhanced answer capability assessment with question-concept mapping integration

    Args:
        chunks: List of chunks from A-Pipeline with concept memberships
        answer_expectation: Answer expectations from B2.3
        question_concepts: B2.5 question-concept mappings
        concepts_lookup: A2.4 concept data

    Returns:
        dict: Enhanced answer capability assessment results with B2.5 integration
    """
    if not chunks:
        return {
            "strategy": "answer_capability_assessment_b25_enhanced",
            "ranked_chunks": [],
            "total_matches": 0,
            "processing_timestamp": datetime.now().isoformat()
        }

    concept_enhancement_enabled = len(concepts_lookup) > 0
    b25_enhancement_enabled = bool(question_concepts)

    # Process each chunk
    ranked_chunks = []

    for chunk in chunks:
        chunk_content = chunk.get("content", "")
        if not chunk_content:
            continue

        # Calculate answer similarity (original)
        answer_similarity = calculate_answer_similarity(chunk_content, answer_expectation)

        # Calculate validation alignment (original)
        validation_criteria = answer_expectation.get("validation_criteria", {})
        validation_alignment = calculate_validation_alignment(chunk_content, validation_criteria)

        # Calculate A2.4 concept enhancement (original)
        if concept_enhancement_enabled:
            concept_enhancement = calculate_concept_enhancement(chunk, answer_expectation, concepts_lookup)
            a24_boost = concept_enhancement["concept_boost"]
            importance_multiplier = concept_enhancement["importance_multiplier"]
            keyword_enhancement = concept_enhancement["keyword_enhancement"]
        else:
            concept_enhancement = {
                "concept_boost": 0.0,
                "importance_multiplier": 1.0,
                "keyword_enhancement": 0.0,
                "concept_details": []
            }
            a24_boost = 0.0
            importance_multiplier = 1.0
            keyword_enhancement = 0.0

        # Calculate B2.5 enhancement (new)
        if b25_enhancement_enabled:
            b25_enhancement = calculate_b25_enhancement_score(chunk, question_concepts, concepts_lookup)
            b25_multiplier = b25_enhancement["enhancement_multiplier"]
            b25_details = b25_enhancement["details"]
        else:
            b25_enhancement = {
                "enhancement_multiplier": 1.0,
                "primary_concept_boost": 0.0,
                "membership_boost": 0.0,
                "details": []
            }
            b25_multiplier = 1.0
            b25_details = []

        # Enhanced combined score with dual concept integration
        base_score = (answer_similarity * 0.6) + (validation_alignment * 0.4)
        a24_enhanced_score = base_score + a24_boost + keyword_enhancement
        final_score = a24_enhanced_score * importance_multiplier * b25_multiplier

        # Only include chunks with meaningful scores
        if final_score > 0.1:
            # Extract answer type for details
            answer_prediction = answer_expectation.get("answer_prediction", {})
            expected_type = answer_prediction.get("primary_type", "text")

            ranked_chunks.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "content": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
                "similarity_score": round(final_score, 4),
                "match_strategy": "capability_assessment_b25_enhanced",
                "assessment_details": {
                    "answer_similarity": round(answer_similarity, 4),
                    "validation_alignment": round(validation_alignment, 4),
                    "expected_answer_type": expected_type,
                    "can_provide_answer": answer_similarity > 0.5,
                    "meets_validation": validation_alignment > 0.4,
                    "content_features": {
                        "has_numbers": bool(re.search(r'\d+', chunk_content)),
                        "has_dates": bool(re.search(r'\b20[0-2]\d\b|\b19[0-9]\d\b', chunk_content)),
                        "content_length": len(chunk_content),
                        "readability": "high" if len(chunk_content) > 100 else "medium"
                    }
                },
                "concept_enhancement": {
                    "a24_enabled": concept_enhancement_enabled,
                    "a24_boost": round(a24_boost, 4),
                    "a24_importance_multiplier": round(importance_multiplier, 4),
                    "a24_keyword_enhancement": round(keyword_enhancement, 4),
                    "b25_enabled": b25_enhancement_enabled,
                    "b25_enhancement_multiplier": round(b25_multiplier, 4),
                    "b25_details": b25_details,
                    "final_enhancement": round((importance_multiplier * b25_multiplier), 4)
                },
                "doc_id": chunk.get("doc_id", ""),
                "chunk_type": chunk.get("chunk_type", ""),
                "confidence": round(min(1.0, final_score * 0.8), 4)
            })

    # Sort by similarity score
    ranked_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Limit to top 8 matches
    ranked_chunks = ranked_chunks[:8]

    return {
        "strategy": "answer_capability_assessment_b25_enhanced",
        "ranked_chunks": ranked_chunks,
        "total_matches": len(ranked_chunks),
        "processing_timestamp": datetime.now().isoformat(),
        "enhancement_summary": {
            "a24_concept_enhancement": concept_enhancement_enabled,
            "b25_question_mapping": b25_enhancement_enabled,
            "dual_enhancement": concept_enhancement_enabled and b25_enhancement_enabled,
            "chunks_processed": len(chunks),
            "chunks_with_scores": len([c for c in chunks if calculate_answer_similarity(c.get("content", ""), answer_expectation) > 0.1])
        }
    }

def load_b25_concept_mappings():
    """
    Load B2.5 question-concept mapping data for enhanced retrieval

    Returns:
        dict: B2.5 concept mappings by question_id
    """
    script_dir = Path(__file__).parent.parent
    b25_path = script_dir / "outputs" / "B2.5_question_concept_mapping_output.json"

    if not b25_path.exists():
        print(f"Warning: B2.5 concept mappings not found at {b25_path}")
        return {}

    try:
        with open(b25_path, 'r', encoding='utf-8') as f:
            b25_data = json.load(f)

        # Convert to lookup by question_id
        concept_mappings = {}
        for result in b25_data.get("results", []):
            question_id = result.get("question_id")
            if question_id:
                concept_mappings[question_id] = {
                    "fuzzy_memberships": result.get("concept_mappings", {}).get("fuzzy_memberships", {}),
                    "primary_concepts": result.get("concept_mappings", {}).get("primary_concepts", []),
                    "mapping_confidence": result.get("quality_indicators", {}).get("mapping_confidence", 0.5)
                }

        print(f"Loaded B2.5 concept mappings for {len(concept_mappings)} questions")
        return concept_mappings

    except Exception as e:
        print(f"Warning: Error loading B2.5 concept mappings: {e}")
        return {}

def load_inputs():
    """Load B2.3 answer expectation data, B2.5 concept mappings, and A-pipeline chunks"""
    script_dir = Path(__file__).parent.parent

    # Load B2.3 answer expectations
    b2_3_path = script_dir / "outputs" / "B2.3_answer_expectation_output.json"
    if not b2_3_path.exists():
        raise FileNotFoundError(f"B2.3 output not found: {b2_3_path}")

    with open(b2_3_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    # Load B2.5 concept mappings
    b25_mappings = load_b25_concept_mappings()

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
                        "chunk_type": chunk.get("chunk_type", ""),
                        "concept_memberships": chunk.get("concept_memberships", []),
                        "membership_scores": chunk.get("membership_scores", {}),
                        "metadata": chunk.get("metadata", {})
                    })

    if not chunks:
        raise FileNotFoundError(f"No real chunks found in A-pipeline output: {a_pipeline_path}")

    return questions_data, chunks, b25_mappings

def apply_b25_concept_filtering(chunks, question_id, b25_mappings, membership_threshold=0.3):
    """
    Filter chunks based on B2.5 concept mappings for enhanced retrieval

    Args:
        chunks: List of chunks with concept memberships
        question_id: Question identifier
        b25_mappings: B2.5 concept mappings
        membership_threshold: Minimum membership score for inclusion

    Returns:
        tuple: (filtered_chunks, question_concepts)
    """
    if question_id not in b25_mappings:
        return chunks, {}

    question_concepts = b25_mappings[question_id]
    fuzzy_memberships = question_concepts["fuzzy_memberships"]
    primary_concepts = question_concepts["primary_concepts"]

    # Get relevant concept IDs based on membership threshold
    relevant_concepts = set()
    for concept_id, membership_data in fuzzy_memberships.items():
        if membership_data["membership_score"] >= membership_threshold:
            relevant_concepts.add(concept_id)

    # Add primary concepts regardless of threshold
    relevant_concepts.update(primary_concepts)

    if not relevant_concepts:
        # If no concepts meet threshold, return all chunks
        return chunks, question_concepts

    # Filter chunks that have membership in relevant concepts
    filtered_chunks = []
    for chunk in chunks:
        chunk_concepts = set(chunk.get("concept_memberships", []))
        if chunk_concepts.intersection(relevant_concepts):
            filtered_chunks.append(chunk)

    # If filtering removes all chunks, return original set
    if not filtered_chunks:
        return chunks, question_concepts

    return filtered_chunks, question_concepts

def calculate_b25_enhancement_score(chunk, question_concepts, concepts_lookup):
    """
    Calculate enhancement score based on B2.5 concept mappings

    Args:
        chunk: Chunk with concept memberships
        question_concepts: B2.5 concept mappings for question
        concepts_lookup: A2.4 concept data

    Returns:
        dict: Enhancement scores and details
    """
    fuzzy_memberships = question_concepts.get("fuzzy_memberships", {})
    primary_concepts = question_concepts.get("primary_concepts", [])
    chunk_concepts = chunk.get("concept_memberships", [])

    if not fuzzy_memberships or not chunk_concepts:
        return {
            "enhancement_multiplier": 1.0,
            "primary_concept_boost": 0.0,
            "membership_boost": 0.0,
            "details": {"no_concept_overlap": True}
        }

    enhancement_multiplier = 1.0
    primary_concept_boost = 0.0
    membership_boost = 0.0
    concept_details = []

    for concept_id in chunk_concepts:
        # Check if concept has fuzzy membership for this question
        if concept_id in fuzzy_memberships:
            membership_score = fuzzy_memberships[concept_id]["membership_score"]
            confidence = fuzzy_memberships[concept_id]["confidence"]

            # Apply membership boost
            membership_contribution = membership_score * confidence * 0.3
            membership_boost += membership_contribution

            concept_details.append({
                "concept_id": concept_id,
                "membership_score": membership_score,
                "confidence": confidence,
                "contribution": membership_contribution
            })

        # Check if concept is a primary concept
        if concept_id in primary_concepts:
            primary_concept_boost += 0.4  # 40% boost per primary concept
            concept_details.append({
                "concept_id": concept_id,
                "primary_concept": True,
                "boost": 0.4
            })

    # Calculate total enhancement multiplier
    enhancement_multiplier += membership_boost + primary_concept_boost
    enhancement_multiplier = min(enhancement_multiplier, 2.5)  # Cap at 2.5x

    return {
        "enhancement_multiplier": enhancement_multiplier,
        "primary_concept_boost": primary_concept_boost,
        "membership_boost": membership_boost,
        "details": concept_details
    }

def save_output(data, output_path="outputs/B3.3_answer_capability_assessment_output.json"):
    """Save answer capability assessment results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    # Create output directory if needed
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved answer capability assessment results to {full_path}")

def main():
    """Main execution for processing all 20 questions"""
    print("="*60)
    print("B3.3: Answer Capability Assessment Strategy - CONCEPT ENHANCED")
    print("="*60)
    
    try:
        # Load inputs
        print("Loading questions, chunks, and B2.5 concept mappings...")
        questions_data, chunks, b25_mappings = load_inputs()

        # Load concept data for enhancement
        concepts_lookup = load_concept_data()
        if concepts_lookup:
            print(f"A2.4 Concept enhancement: ENABLED ({len(concepts_lookup)} concepts loaded)")
        else:
            print("A2.4 Concept enhancement: DISABLED (no concept data found)")

        if b25_mappings:
            print(f"B2.5 Question-Concept mapping: ENABLED ({len(b25_mappings)} questions mapped)")
        else:
            print("B2.5 Question-Concept mapping: DISABLED (no mapping data found)")
        
        if not isinstance(questions_data, list):
            questions_data = [questions_data]
        
        all_results = []
        
        print(f"Processing {len(questions_data)} questions with {len(chunks)} chunks...\n")
        
        # Process each question
        for i, question_data in enumerate(questions_data):
            question_text = question_data.get("question", "")
            question_id = question_data.get("question_id", f"q_{i}")
            print(f"[{i+1:2d}/20] Capability assessment: {question_text[:60]}...")

            # Apply B2.5 concept-guided filtering
            if b25_mappings and question_id in b25_mappings:
                filtered_chunks, question_concepts = apply_b25_concept_filtering(
                    chunks, question_id, b25_mappings
                )
                print(f"    B2.5 filtering: {len(chunks)} -> {len(filtered_chunks)} chunks")
            else:
                filtered_chunks = chunks
                question_concepts = {}

            # Perform answer capability assessment on filtered chunks
            matching_result = assess_answer_capability_with_b25(
                filtered_chunks, question_data, question_concepts, concepts_lookup
            )

            # Add question metadata
            result = {
                "question_id": question_id,
                "question": question_text,
                "strategy": "answer_capability_assessment_b25_enhanced",
                "ranked_chunks": matching_result.get("ranked_chunks", []),
                "total_matches": matching_result.get("total_matches", 0),
                "b25_integration": {
                    "concept_filtering_enabled": bool(b25_mappings and question_id in b25_mappings),
                    "chunks_before_filtering": len(chunks),
                    "chunks_after_filtering": len(filtered_chunks),
                    "primary_concepts": question_concepts.get("primary_concepts", []),
                    "mapping_confidence": question_concepts.get("mapping_confidence", 0.0)
                },
                "quality_analysis": analyze_capability_assessment_quality(
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
        print("ANSWER CAPABILITY ASSESSMENT SUMMARY")
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
        
        print("\nB3.3 Answer Capability Assessment completed successfully!")
        
    except Exception as e:
        print(f"Error in B3.3 Answer Capability Assessment: {str(e)}")
        raise

if __name__ == "__main__":
    main()