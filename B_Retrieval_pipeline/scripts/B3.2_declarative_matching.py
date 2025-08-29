#!/usr/bin/env python3
"""
B3.2: Declarative Matching Strategy
Matches declarative forms of questions to concept patterns
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

def calculate_declarative_similarity(declarative_forms, concept_keywords, concept_domain):
    """
    Calculate similarity between declarative forms and concept
    
    Args:
        declarative_forms: List of declarative forms from B2.2
        concept_keywords: Concept keywords
        concept_domain: Concept domain
        
    Returns:
        float: Declarative similarity score (0-1)
    """
    if not declarative_forms:
        return 0.0
    
    best_score = 0.0
    
    for decl_form in declarative_forms:
        declarative_text = decl_form.get("declarative", "").lower()
        quality_score = decl_form.get("quality_score", 0.0)
        
        if not declarative_text:
            continue
        
        # Extract words from declarative
        decl_words = set(re.findall(r'\b\w{3,}\b', declarative_text))
        concept_words = set([kw.lower() for kw in concept_keywords])
        
        # Calculate word overlap
        if not decl_words or not concept_words:
            word_overlap = 0.0
        else:
            intersection = len(decl_words & concept_words)
            union = len(decl_words | concept_words)
            word_overlap = intersection / union
        
        # Pattern-based scoring for specific declarative patterns
        pattern_score = 0.0
        
        # Financial patterns
        if concept_domain == "finance":
            financial_patterns = [
                r'\b(revenue|income|sales)\s+(?:is|was|are)\b',
                r'\b\w+\s+changed\s+by\b',
                r'\b(?:cost|expense)\s+(?:is|was)\b',
                r'\b\w+\s+increased\s+(?:by|to)\b',
                r'\b\w+\s+decreased\s+(?:by|to)\b'
            ]
            
            for pattern in financial_patterns:
                if re.search(pattern, declarative_text):
                    pattern_score = 0.3
                    break
        
        # General patterns
        if pattern_score == 0.0:
            general_patterns = [
                r'\b\w+\s+(?:is|was|are|were)\b',
                r'\b\w+\s+occurred\s+(?:in|at)\b',
                r'\bthe\s+\w+\s+(?:is|was)\b'
            ]
            
            for pattern in general_patterns:
                if re.search(pattern, declarative_text):
                    pattern_score = 0.2
                    break
        
        # Calculate combined score
        form_score = (word_overlap * 0.6 + pattern_score + quality_score * 0.2)
        best_score = max(best_score, form_score)
    
    return min(1.0, best_score)

def extract_declarative_components(declarative_forms):
    """
    Extract key components from declarative forms
    
    Args:
        declarative_forms: List of declarative forms
        
    Returns:
        dict: Extracted components
    """
    subjects = set()
    predicates = set()
    objects = set()
    
    for decl_form in declarative_forms:
        declarative_text = decl_form.get("declarative", "").lower()
        
        # Simple extraction - look for common patterns
        # Subject is usually the first noun/noun phrase
        words = declarative_text.split()
        
        if len(words) > 0:
            # First meaningful word as potential subject
            first_word = words[0] if len(words[0]) > 2 else (words[1] if len(words) > 1 and len(words[1]) > 2 else "")
            if first_word:
                subjects.add(first_word)
        
        # Look for predicates (verbs)
        predicates_found = re.findall(r'\b(?:is|was|are|were|changed|increased|decreased|occurred|happened)\b', declarative_text)
        predicates.update(predicates_found)
        
        # Extract potential objects (after predicates)
        for predicate in predicates_found:
            pattern = rf'{predicate}\s+(\w+(?:\s+\w+)*?)(?:\s|$|,|\.)'
            matches = re.findall(pattern, declarative_text)
            objects.update(matches)
    
    return {
        "subjects": list(subjects),
        "predicates": list(predicates),
        "objects": list(objects)
    }

def match_declarative_to_concepts(question_data, expanded_concepts):
    """
    Match declarative forms to expanded concepts
    
    Args:
        question_data: Question with declarative transformations
        expanded_concepts: Expanded concept data
        
    Returns:
        list: Declarative-based matches
    """
    declarative_forms = question_data.get("declarative_forms", [])
    if not declarative_forms:
        return []
    
    matches = []
    
    # Extract declarative components
    decl_components = extract_declarative_components(declarative_forms)
    
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
        
        # Calculate declarative similarity
        similarity = calculate_declarative_similarity(declarative_forms, all_keywords, concept_domain)
        
        if similarity > 0.1:  # Minimum threshold
            # Check component alignment
            concept_words = set([kw.lower() for kw in all_keywords])
            component_alignment = {
                "subject_match": bool(set(decl_components["subjects"]) & concept_words),
                "predicate_relevance": len(decl_components["predicates"]) > 0,
                "object_match": bool(set(decl_components["objects"]) & concept_words)
            }
            
            matches.append({
                "concept_id": concept_id,
                "concept": original_concept,
                "similarity_score": similarity,
                "matching_strategy": "declarative_based",
                "match_details": {
                    "best_declarative": max(declarative_forms, key=lambda x: x.get("quality_score", 0))["declarative"],
                    "component_alignment": component_alignment,
                    "keyword_overlap": len(concept_words & set(sum([decl["declarative"].lower().split() for decl in declarative_forms], []))),
                    "declarative_quality": max(decl_form.get("quality_score", 0) for decl_form in declarative_forms)
                }
            })
    
    # Sort by similarity score
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return matches[:10]  # Top 10 matches

def analyze_declarative_matching_quality(matches, question_data):
    """
    Analyze the quality of declarative matching
    
    Args:
        matches: Declarative-based matches
        question_data: Original question data
        
    Returns:
        dict: Quality analysis
    """
    if not matches:
        return {
            "match_quality": "poor",
            "confidence": 0.0,
            "analysis": "No declarative matches found"
        }
    
    top_score = matches[0]["similarity_score"]
    avg_score = sum(match["similarity_score"] for match in matches) / len(matches)
    
    # Check declarative transformation quality
    declarative_forms = question_data.get("declarative_forms", [])
    transformation_confidence = question_data.get("transformation_confidence", 0.0)
    
    # Determine quality based on multiple factors
    if top_score >= 0.7 and transformation_confidence >= 0.7:
        quality = "excellent"
    elif top_score >= 0.5 and transformation_confidence >= 0.5:
        quality = "good"
    elif top_score >= 0.3:
        quality = "fair"
    else:
        quality = "poor"
    
    # Calculate confidence based on transformation quality and match scores
    confidence = min(1.0, (top_score * 0.6 + transformation_confidence * 0.4))
    
    return {
        "match_quality": quality,
        "confidence": confidence,
        "top_score": top_score,
        "average_score": avg_score,
        "transformation_confidence": transformation_confidence,
        "total_matches": len(matches)
    }

def load_inputs():
    """Load question data and expanded concepts"""
    script_dir = Path(__file__).parent.parent
    
    # Load question with declarative transformations from B2.2
    b2_2_path = script_dir / "outputs/B2_2_declarative_transformation_output.json"
    if b2_2_path.exists():
        with open(b2_2_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)
    else:
        # Fallback to B2.1 or B1
        b2_1_path = script_dir / "outputs/B2_1_intent_layer_output.json"
        if b2_1_path.exists():
            with open(b2_1_path, 'r', encoding='utf-8') as f:
                question_data = json.load(f)
        else:
            # Mock data for testing
            question_data = {
                "question": "What was the change in Current deferred income?",
                "declarative_forms": [
                    {
                        "declarative": "Current deferred income changed by",
                        "quality_score": 0.8
                    },
                    {
                        "declarative": "the information about current deferred income change",
                        "quality_score": 0.6
                    }
                ],
                "transformation_confidence": 0.8
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
                    "theme_name": "Financial Changes",
                    "primary_keywords": ["income", "deferred", "current", "change"],
                    "domain": "finance",
                    "importance_score": 0.9
                }
            }
    
    return question_data, expanded_concepts

def process_declarative_matching(question_data, expanded_concepts):
    """
    Process declarative-based matching
    
    Args:
        question_data: Question with declarative transformations
        expanded_concepts: Expanded concept data
        
    Returns:
        dict: Declarative matching results
    """
    # Perform declarative matching
    matches = match_declarative_to_concepts(question_data, expanded_concepts)
    
    # Analyze matching quality
    quality_analysis = analyze_declarative_matching_quality(matches, question_data)
    
    return {
        "question": question_data.get("question", ""),
        "declarative_forms": question_data.get("declarative_forms", []),
        "matches": matches,
        "quality_analysis": quality_analysis,
        "strategy_name": "declarative_matching",
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/B3_2_declarative_matching_output.json"):
    """Save declarative matching results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved declarative matching results to {full_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B3.2: Declarative Matching Strategy")
    print("="*60)
    
    try:
        # Load inputs
        print("Loading question data and expanded concepts...")
        question_data, expanded_concepts = load_inputs()
        
        # Process declarative matching
        question = question_data.get("question", "")
        print(f"Processing declarative matching for: {question}")
        output_data = process_declarative_matching(question_data, expanded_concepts)
        
        # Display results
        quality = output_data["quality_analysis"]
        print(f"\nDeclarative Matching Results:")
        print(f"  Total Matches: {len(output_data['matches'])}")
        print(f"  Match Quality: {quality['match_quality']}")
        print(f"  Confidence: {quality['confidence']:.3f}")
        print(f"  Top Score: {quality['top_score']:.3f}")
        print(f"  Transformation Confidence: {quality['transformation_confidence']:.3f}")
        
        if output_data["declarative_forms"]:
            print(f"\nDeclarative Forms:")
            for i, decl_form in enumerate(output_data["declarative_forms"][:3], 1):
                print(f"  {i}. \"{decl_form['declarative']}\"")
                print(f"     Quality: {decl_form.get('quality_score', 0):.3f}")
        
        if output_data["matches"]:
            print(f"\nTop 3 Declarative Matches:")
            for i, match in enumerate(output_data["matches"][:3], 1):
                concept = match["concept"]
                details = match["match_details"]
                print(f"  {i}. {concept['theme_name']}")
                print(f"     Score: {match['similarity_score']:.3f}")
                print(f"     Domain: {concept.get('domain', 'N/A')}")
                print(f"     Best declarative: \"{details['best_declarative'][:50]}...\"")
        
        # Save output
        save_output(output_data)
        
        print("\nB3.2 Declarative Matching completed successfully!")
        
    except Exception as e:
        print(f"Error in B3.2 Declarative Matching: {str(e)}")
        raise

if __name__ == "__main__":
    main()