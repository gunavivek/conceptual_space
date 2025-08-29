#!/usr/bin/env python3
"""
B3.1: Intent Matching Strategy
Matches questions to concepts based on intent analysis
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re
import math

def calculate_intent_similarity(question_intent, concept_keywords, concept_domain):
    """
    Calculate similarity between question intent and concept
    
    Args:
        question_intent: Intent analysis from B2.1
        concept_keywords: Concept keywords
        concept_domain: Concept domain
        
    Returns:
        float: Intent similarity score (0-1)
    """
    # Extract intent features
    intent_type = question_intent.get("primary_intent", "")
    intent_keywords = question_intent.get("keywords", [])
    question_domain = question_intent.get("domain", "general")
    
    # Keyword overlap scoring
    intent_kw_set = set([kw.lower() for kw in intent_keywords])
    concept_kw_set = set([kw.lower() for kw in concept_keywords])
    
    if not intent_kw_set or not concept_kw_set:
        keyword_score = 0.0
    else:
        intersection = len(intent_kw_set & concept_kw_set)
        union = len(intent_kw_set | concept_kw_set)
        keyword_score = intersection / union
    
    # Domain alignment scoring
    domain_score = 1.0 if question_domain.lower() == concept_domain.lower() else 0.5
    if question_domain == "general" or concept_domain == "general":
        domain_score = 0.7  # Neutral for general domains
    
    # Intent type scoring
    intent_score = 1.0  # Base score
    
    # Adjust based on intent type
    if intent_type == "factual":
        # Factual questions prefer concepts with high importance
        intent_score = 1.0
    elif intent_type == "analytical":
        # Analytical questions prefer concepts with relationships
        intent_score = 1.1
    elif intent_type == "comparative":
        # Comparative questions need multiple related concepts
        intent_score = 0.9
    
    # Combine scores
    final_score = (keyword_score * 0.5 + domain_score * 0.3 + (intent_score - 1.0) * 0.2)
    return min(1.0, max(0.0, final_score))

def match_intent_to_concepts(question_data, expanded_concepts):
    """
    Match question intent to expanded concepts
    
    Args:
        question_data: Question with intent analysis
        expanded_concepts: Expanded concept data
        
    Returns:
        list: Intent-based matches
    """
    question_intent = question_data.get("intent_analysis", {})
    matches = []
    
    # Handle both formats of expanded concepts
    if isinstance(expanded_concepts, dict):
        if "expanded_concepts" in expanded_concepts:
            concepts_to_match = expanded_concepts["expanded_concepts"]
        else:
            concepts_to_match = expanded_concepts
    else:
        concepts_to_match = expanded_concepts
    
    for concept_id, concept_data in concepts_to_match.items():
        if isinstance(concept_data, dict) and "original_concept" in concept_data:
            # Format from A2.5 orchestrator
            original_concept = concept_data["original_concept"]
            expanded_terms = concept_data.get("all_expanded_terms", [])
            
            # Use expanded terms for matching
            all_keywords = list(set(original_concept.get("primary_keywords", []) + expanded_terms))
        else:
            # Direct concept format
            original_concept = concept_data
            all_keywords = original_concept.get("primary_keywords", [])
        
        concept_domain = original_concept.get("domain", "general")
        
        # Calculate intent similarity
        similarity = calculate_intent_similarity(question_intent, all_keywords, concept_domain)
        
        if similarity > 0.1:  # Minimum threshold
            matches.append({
                "concept_id": concept_id,
                "concept": original_concept,
                "similarity_score": similarity,
                "matching_strategy": "intent_based",
                "match_details": {
                    "keyword_overlap": len(set([kw.lower() for kw in question_intent.get("keywords", [])]) & 
                                          set([kw.lower() for kw in all_keywords])),
                    "domain_match": concept_domain == question_intent.get("domain", "general"),
                    "intent_type": question_intent.get("primary_intent", "unknown")
                }
            })
    
    # Sort by similarity score
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return matches[:10]  # Top 10 matches

def analyze_intent_matching_quality(matches, question_data):
    """
    Analyze the quality of intent matching
    
    Args:
        matches: Intent-based matches
        question_data: Original question data
        
    Returns:
        dict: Quality analysis
    """
    if not matches:
        return {
            "match_quality": "poor",
            "confidence": 0.0,
            "analysis": "No intent matches found"
        }
    
    top_score = matches[0]["similarity_score"]
    avg_score = sum(match["similarity_score"] for match in matches) / len(matches)
    score_variance = sum((match["similarity_score"] - avg_score) ** 2 for match in matches) / len(matches)
    
    # Determine quality
    if top_score >= 0.8:
        quality = "excellent"
    elif top_score >= 0.6:
        quality = "good"
    elif top_score >= 0.4:
        quality = "fair"
    else:
        quality = "poor"
    
    # Calculate confidence
    confidence = min(1.0, top_score + (0.1 if score_variance < 0.1 else 0))
    
    return {
        "match_quality": quality,
        "confidence": confidence,
        "top_score": top_score,
        "average_score": avg_score,
        "score_variance": score_variance,
        "total_matches": len(matches)
    }

def load_inputs():
    """Load question data and expanded concepts"""
    script_dir = Path(__file__).parent.parent
    
    # Load question with intent analysis from B2.1
    b2_1_path = script_dir / "outputs/B2_1_intent_layer_output.json"
    if b2_1_path.exists():
        with open(b2_1_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)
    else:
        # Fallback to B1
        b1_path = script_dir / "outputs/B1_current_question.json"
        if b1_path.exists():
            with open(b1_path, 'r', encoding='utf-8') as f:
                question_data = json.load(f)
        else:
            # Mock data for testing
            question_data = {
                "question": "What was the change in Current deferred income?",
                "intent_analysis": {
                    "primary_intent": "factual",
                    "keywords": ["change", "current", "deferred", "income"],
                    "domain": "finance"
                }
            }
    
    # Load expanded concepts from A2.5
    a2_5_path = Path(__file__).parent.parent.parent / "A_concept_pipeline/outputs/A2.5_expanded_concepts.json"
    if a2_5_path.exists():
        with open(a2_5_path, 'r', encoding='utf-8') as f:
            expanded_concepts = json.load(f)
    else:
        # Fallback to core concepts
        a2_4_path = Path(__file__).parent.parent.parent / "A_concept_pipeline/outputs/A2.4_core_concepts.json"
        if a2_4_path.exists():
            with open(a2_4_path, 'r', encoding='utf-8') as f:
                core_data = json.load(f)
                expanded_concepts = {
                    c["concept_id"]: c for c in core_data.get("core_concepts", [])
                }
        else:
            # Mock concepts for testing
            expanded_concepts = {
                "core_1": {
                    "concept_id": "core_1",
                    "theme_name": "Financial Metrics",
                    "primary_keywords": ["revenue", "income", "financial", "current"],
                    "domain": "finance",
                    "importance_score": 0.8
                }
            }
    
    return question_data, expanded_concepts

def process_intent_matching(question_data, expanded_concepts):
    """
    Process intent-based matching
    
    Args:
        question_data: Question with intent analysis
        expanded_concepts: Expanded concept data
        
    Returns:
        dict: Intent matching results
    """
    # Perform intent matching
    matches = match_intent_to_concepts(question_data, expanded_concepts)
    
    # Analyze matching quality
    quality_analysis = analyze_intent_matching_quality(matches, question_data)
    
    return {
        "question": question_data.get("question", ""),
        "intent_analysis": question_data.get("intent_analysis", {}),
        "matches": matches,
        "quality_analysis": quality_analysis,
        "strategy_name": "intent_matching",
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/B3_1_intent_matching_output.json"):
    """Save intent matching results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved intent matching results to {full_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B3.1: Intent Matching Strategy")
    print("="*60)
    
    try:
        # Load inputs
        print("Loading question data and expanded concepts...")
        question_data, expanded_concepts = load_inputs()
        
        # Process intent matching
        question = question_data.get("question", "")
        print(f"Processing intent matching for: {question}")
        output_data = process_intent_matching(question_data, expanded_concepts)
        
        # Display results
        quality = output_data["quality_analysis"]
        print(f"\nIntent Matching Results:")
        print(f"  Total Matches: {len(output_data['matches'])}")
        print(f"  Match Quality: {quality['match_quality']}")
        print(f"  Confidence: {quality['confidence']:.3f}")
        print(f"  Top Score: {quality['top_score']:.3f}")
        
        if output_data["matches"]:
            print(f"\nTop 3 Intent Matches:")
            for i, match in enumerate(output_data["matches"][:3], 1):
                concept = match["concept"]
                print(f"  {i}. {concept['theme_name']}")
                print(f"     Score: {match['similarity_score']:.3f}")
                print(f"     Domain: {concept.get('domain', 'N/A')}")
                print(f"     Keywords overlap: {match['match_details']['keyword_overlap']}")
        
        # Save output
        save_output(output_data)
        
        print("\nB3.1 Intent Matching completed successfully!")
        
    except Exception as e:
        print(f"Error in B3.1 Intent Matching: {str(e)}")
        raise

if __name__ == "__main__":
    main()