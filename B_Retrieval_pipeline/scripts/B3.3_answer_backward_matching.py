#!/usr/bin/env python3
"""
B3.3: Answer-Backward Matching Strategy
Matches concepts based on expected answer types and formats working backward from answer expectations
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

def determine_concept_answer_capability(concept, answer_expectation):
    """
    Determine if a concept can provide the expected answer type
    
    Args:
        concept: Concept data
        answer_expectation: Expected answer type and format from B2.3
        
    Returns:
        float: Answer capability score (0-1)
    """
    expected_type = answer_expectation.get("primary_type", "text")
    expected_format = answer_expectation.get("format_specification", {})
    concept_keywords = concept.get("primary_keywords", [])
    concept_domain = concept.get("domain", "general")
    
    capability_score = 0.0
    
    # Type-specific capability scoring
    if expected_type == "numeric":
        # Check if concept contains numeric-relevant keywords
        numeric_indicators = ["amount", "value", "number", "total", "sum", "count", "quantity", 
                            "revenue", "cost", "price", "percentage", "ratio", "rate"]
        keyword_matches = sum(1 for kw in concept_keywords if any(ni in kw.lower() for ni in numeric_indicators))
        capability_score += min(0.6, keyword_matches * 0.2)
        
        # Domain bonus for financial numeric questions
        if concept_domain == "finance" and expected_format.get("units"):
            if "dollar" in expected_format.get("units", "").lower():
                capability_score += 0.3
            elif "percent" in expected_format.get("units", "").lower():
                capability_score += 0.2
    
    elif expected_type == "date":
        # Check for time-related keywords
        date_indicators = ["year", "date", "time", "period", "quarter", "month", "when"]
        keyword_matches = sum(1 for kw in concept_keywords if any(di in kw.lower() for di in date_indicators))
        capability_score += min(0.5, keyword_matches * 0.25)
    
    elif expected_type == "boolean":
        # Boolean questions are generally answerable by most concepts
        capability_score += 0.4
    
    elif expected_type == "list":
        # List questions need concepts with multiple related items
        if len(concept_keywords) >= 3:
            capability_score += 0.5
        else:
            capability_score += 0.2
    
    else:  # text type
        # Text answers are generally answerable
        capability_score += 0.6
    
    # Format specification bonus
    format_spec = answer_expectation.get("format_specification", {})
    if format_spec.get("context_required", False):
        # Concepts with higher importance can better provide context
        importance = concept.get("importance_score", 0.5)
        capability_score += importance * 0.2
    
    # Complexity analysis bonus
    complexity = answer_expectation.get("complexity_analysis", {})
    if complexity.get("requires_calculation", False):
        # Financial domain concepts are better for calculations
        if concept_domain == "finance":
            capability_score += 0.2
    
    if complexity.get("requires_comparison", False):
        # Concepts with many documents can better support comparisons
        doc_count = len(concept.get("related_documents", []))
        if doc_count > 5:
            capability_score += 0.1
    
    return min(1.0, capability_score)

def calculate_answer_alignment_score(concept_keywords, answer_validation_criteria):
    """
    Calculate how well concept aligns with answer validation criteria
    
    Args:
        concept_keywords: Keywords from concept
        answer_validation_criteria: Validation criteria from B2.3
        
    Returns:
        float: Alignment score (0-1)
    """
    criteria = answer_validation_criteria or {}
    must_contain = criteria.get("must_contain", [])
    
    if not must_contain:
        return 0.5  # Neutral score if no specific requirements
    
    alignment_score = 0.0
    
    for requirement in must_contain:
        if requirement.lower() == "number":
            # Check for numeric keywords
            numeric_keywords = ["amount", "value", "total", "revenue", "cost", "income"]
            if any(kw in [ckw.lower() for ckw in concept_keywords] for kw in numeric_keywords):
                alignment_score += 0.4
        
        elif "dollar" in requirement.lower():
            # Check for financial keywords
            financial_keywords = ["revenue", "cost", "price", "income", "expense", "profit"]
            if any(kw in [ckw.lower() for ckw in concept_keywords] for kw in financial_keywords):
                alignment_score += 0.3
        
        elif "percent" in requirement.lower():
            # Check for ratio/percentage keywords
            ratio_keywords = ["rate", "ratio", "percentage", "margin", "growth"]
            if any(kw in [ckw.lower() for ckw in concept_keywords] for kw in ratio_keywords):
                alignment_score += 0.3
        
        else:
            # Direct keyword match
            if any(requirement.lower() in kw.lower() for kw in concept_keywords):
                alignment_score += 0.2
    
    return min(1.0, alignment_score)

def match_answer_backward_to_concepts(question_data, expanded_concepts):
    """
    Match concepts using answer-backward strategy
    
    Args:
        question_data: Question with answer expectations
        expanded_concepts: Expanded concept data
        
    Returns:
        list: Answer-backward matches
    """
    answer_prediction = question_data.get("answer_prediction", {})
    validation_criteria = question_data.get("validation_criteria", {})
    
    if not answer_prediction:
        return []
    
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
        
        # Calculate answer capability
        capability_score = determine_concept_answer_capability(original_concept, answer_prediction)
        
        # Calculate answer alignment
        alignment_score = calculate_answer_alignment_score(all_keywords, validation_criteria)
        
        # Combined score
        combined_score = capability_score * 0.7 + alignment_score * 0.3
        
        if combined_score > 0.1:  # Minimum threshold
            matches.append({
                "concept_id": concept_id,
                "concept": original_concept,
                "similarity_score": combined_score,
                "matching_strategy": "answer_backward",
                "match_details": {
                    "answer_capability": capability_score,
                    "validation_alignment": alignment_score,
                    "expected_answer_type": answer_prediction.get("primary_type", "text"),
                    "can_provide_format": capability_score > 0.5,
                    "meets_validation": alignment_score > 0.3
                }
            })
    
    # Sort by similarity score
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return matches[:8]  # Top 8 matches (fewer since this is usually a supporting strategy)

def analyze_answer_backward_quality(matches, question_data):
    """
    Analyze the quality of answer-backward matching
    
    Args:
        matches: Answer-backward matches
        question_data: Original question data
        
    Returns:
        dict: Quality analysis
    """
    if not matches:
        return {
            "match_quality": "poor",
            "confidence": 0.0,
            "analysis": "No answer-backward matches found"
        }
    
    top_score = matches[0]["similarity_score"]
    avg_capability = sum(match["match_details"]["answer_capability"] for match in matches) / len(matches)
    avg_alignment = sum(match["match_details"]["validation_alignment"] for match in matches) / len(matches)
    
    # Count concepts that can provide expected format
    can_provide_format = sum(1 for match in matches if match["match_details"]["can_provide_format"])
    format_coverage = can_provide_format / len(matches)
    
    # Determine quality
    answer_confidence = question_data.get("confidence", 0.5)
    
    if top_score >= 0.7 and format_coverage >= 0.6:
        quality = "excellent"
    elif top_score >= 0.5 and avg_capability >= 0.4:
        quality = "good"
    elif top_score >= 0.3:
        quality = "fair"
    else:
        quality = "poor"
    
    # Calculate confidence
    confidence = min(1.0, (top_score * 0.4 + avg_capability * 0.3 + answer_confidence * 0.3))
    
    return {
        "match_quality": quality,
        "confidence": confidence,
        "top_score": top_score,
        "average_capability": avg_capability,
        "average_alignment": avg_alignment,
        "format_coverage": format_coverage,
        "total_matches": len(matches)
    }

def load_inputs():
    """Load question data and expanded concepts"""
    script_dir = Path(__file__).parent.parent
    
    # Load question with answer expectations from B2.3
    b2_3_path = script_dir / "outputs/B2_3_answer_expectation_output.json"
    if b2_3_path.exists():
        with open(b2_3_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)
    else:
        # Fallback chain
        b2_2_path = script_dir / "outputs/B2_2_declarative_transformation_output.json"
        if b2_2_path.exists():
            with open(b2_2_path, 'r', encoding='utf-8') as f:
                question_data = json.load(f)
        else:
            # Mock data for testing
            question_data = {
                "question": "What was the change in Current deferred income?",
                "answer_prediction": {
                    "primary_type": "numeric",
                    "confidence": 0.8
                },
                "format_specification": {
                    "units": "million dollars",
                    "precision": 2,
                    "context_required": True
                },
                "validation_criteria": {
                    "must_contain": ["number", "million dollars"]
                },
                "confidence": 0.8
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
                    "theme_name": "Financial Values",
                    "primary_keywords": ["income", "deferred", "amount", "revenue", "financial"],
                    "domain": "finance",
                    "importance_score": 0.8,
                    "related_documents": ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
                }
            }
    
    return question_data, expanded_concepts

def process_answer_backward_matching(question_data, expanded_concepts):
    """
    Process answer-backward matching
    
    Args:
        question_data: Question with answer expectations
        expanded_concepts: Expanded concept data
        
    Returns:
        dict: Answer-backward matching results
    """
    # Perform answer-backward matching
    matches = match_answer_backward_to_concepts(question_data, expanded_concepts)
    
    # Analyze matching quality
    quality_analysis = analyze_answer_backward_quality(matches, question_data)
    
    return {
        "question": question_data.get("question", ""),
        "answer_prediction": question_data.get("answer_prediction", {}),
        "validation_criteria": question_data.get("validation_criteria", {}),
        "matches": matches,
        "quality_analysis": quality_analysis,
        "strategy_name": "answer_backward",
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/B3_3_answer_backward_output.json"):
    """Save answer-backward matching results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved answer-backward matching results to {full_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B3.3: Answer-Backward Matching Strategy")
    print("="*60)
    
    try:
        # Load inputs
        print("Loading question data and expanded concepts...")
        question_data, expanded_concepts = load_inputs()
        
        # Process answer-backward matching
        question = question_data.get("question", "")
        print(f"Processing answer-backward matching for: {question}")
        output_data = process_answer_backward_matching(question_data, expanded_concepts)
        
        # Display results
        quality = output_data["quality_analysis"]
        answer_pred = output_data["answer_prediction"]
        
        print(f"\nAnswer-Backward Matching Results:")
        print(f"  Expected Answer Type: {answer_pred.get('primary_type', 'N/A')}")
        print(f"  Total Matches: {len(output_data['matches'])}")
        print(f"  Match Quality: {quality['match_quality']}")
        print(f"  Confidence: {quality['confidence']:.3f}")
        print(f"  Format Coverage: {quality['format_coverage']:.1%}")
        print(f"  Average Capability: {quality['average_capability']:.3f}")
        
        if output_data["matches"]:
            print(f"\nTop 3 Answer-Backward Matches:")
            for i, match in enumerate(output_data["matches"][:3], 1):
                concept = match["concept"]
                details = match["match_details"]
                print(f"  {i}. {concept['theme_name']}")
                print(f"     Score: {match['similarity_score']:.3f}")
                print(f"     Answer Capability: {details['answer_capability']:.3f}")
                print(f"     Can Provide Format: {details['can_provide_format']}")
                print(f"     Meets Validation: {details['meets_validation']}")
        
        # Save output
        save_output(output_data)
        
        print("\nB3.3 Answer-Backward Matching completed successfully!")
        
    except Exception as e:
        print(f"Error in B3.3 Answer-Backward Matching: {str(e)}")
        raise

if __name__ == "__main__":
    main()