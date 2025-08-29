#!/usr/bin/env python3
"""
B2.2: Declarative Transformation
Transforms questions into declarative forms for better concept matching
"""

import json
from pathlib import Path
from datetime import datetime
import re

def transform_to_declarative(question):
    """
    Transform a question into declarative form
    
    Args:
        question: Original question
        
    Returns:
        list: Possible declarative transformations
    """
    question_lower = question.lower().strip()
    declaratives = []
    
    # Remove question marks and normalize
    clean_question = re.sub(r'[?!.]', '', question_lower).strip()
    
    # Pattern-based transformations
    patterns = [
        # "What was X?" -> "X was [value]"
        (r'^what\s+was\s+(.+)', r'\1 was'),
        (r'^what\s+is\s+(.+)', r'\1 is'),
        
        # "How much is X?" -> "X is [amount]"
        (r'^how\s+much\s+(?:is|was)\s+(.+)', r'\1 is'),
        (r'^how\s+many\s+(.+)', r'the number of \1 is'),
        
        # "When did X?" -> "X happened in [time]"
        (r'^when\s+(?:did|was)\s+(.+)', r'\1 occurred in'),
        
        # "Where is X?" -> "X is located at"
        (r'^where\s+(?:is|was)\s+(.+)', r'\1 is located'),
        
        # "Why did X?" -> "X happened because"
        (r'^why\s+(?:did|was)\s+(.+)', r'\1 occurred because'),
        
        # "Which X?" -> "The X is"
        (r'^which\s+(.+)', r'the \1 is'),
        
        # "Who is X?" -> "X is [person]"
        (r'^who\s+(?:is|was)\s+(.+)', r'\1 is'),
    ]
    
    # Apply pattern transformations
    for pattern, replacement in patterns:
        match = re.match(pattern, clean_question)
        if match:
            declarative = re.sub(pattern, replacement, clean_question)
            declaratives.append(declarative.strip())
    
    # Additional transformations for financial questions
    if any(word in question_lower for word in ['change', 'increase', 'decrease', 'difference']):
        # "What was the change in X?" -> "X changed by [amount]"
        change_pattern = r'what\s+was\s+the\s+change\s+in\s+(.+)'
        match = re.match(change_pattern, clean_question)
        if match:
            declarative = f"{match.group(1)} changed by"
            declaratives.append(declarative)
            
            # Also add current value form
            declarative2 = f"{match.group(1)} is currently"
            declaratives.append(declarative2)
    
    # If no patterns matched, create a generic declarative
    if not declaratives:
        # Remove question words and create statement
        generic = re.sub(r'^(what|how|when|where|why|which|who)\s+', '', clean_question)
        generic = re.sub(r'\s+(is|are|was|were|do|does|did)\s+', ' ', generic)
        declaratives.append(f"the information about {generic}")
    
    return declaratives

def extract_key_components(question, declaratives):
    """
    Extract key components from question and declaratives
    
    Args:
        question: Original question
        declaratives: Declarative transformations
        
    Returns:
        dict: Key components
    """
    # Extract subject, predicate, object
    components = {
        "subjects": [],
        "predicates": [],
        "objects": [],
        "modifiers": []
    }
    
    # Simple extraction based on patterns
    for declarative in declaratives:
        words = declarative.split()
        
        # Identify potential subjects (nouns at beginning)
        for i, word in enumerate(words):
            if i < 3 and len(word) > 3:  # First few words, reasonable length
                components["subjects"].append(word)
                
        # Identify predicates (verbs)
        predicates = ['is', 'was', 'are', 'were', 'changed', 'increased', 'decreased', 'occurred']
        for word in words:
            if word in predicates:
                components["predicates"].append(word)
        
        # Extract time/date modifiers
        time_words = re.findall(r'\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b|\b(q1|q2|q3|q4)\b', declarative.lower())
        components["modifiers"].extend([match[0] or match[1] or match[2] for match in time_words])
    
    # Remove duplicates and empty strings
    for key in components:
        components[key] = list(filter(None, set(components[key])))
    
    return components

def score_declarative_quality(declarative, original_question):
    """
    Score the quality of a declarative transformation
    
    Args:
        declarative: Transformed declarative
        original_question: Original question
        
    Returns:
        float: Quality score (0-1)
    """
    # Factors for quality:
    # 1. Length preservation
    # 2. Key word preservation
    # 3. Grammatical structure
    
    orig_words = set(original_question.lower().split())
    decl_words = set(declarative.lower().split())
    
    # Remove question words from original for fair comparison
    question_words = {'what', 'how', 'when', 'where', 'why', 'which', 'who', 'is', 'was', 'did', 'do', 'does'}
    orig_words -= question_words
    
    # Word preservation score
    preservation = len(orig_words & decl_words) / max(len(orig_words), 1)
    
    # Length similarity score
    length_ratio = min(len(declarative), len(original_question)) / max(len(declarative), len(original_question))
    
    # Structure score (simple heuristic)
    structure_score = 0.8 if any(verb in declarative for verb in ['is', 'was', 'are', 'were']) else 0.5
    
    return (preservation * 0.5 + length_ratio * 0.3 + structure_score * 0.2)

def load_input(input_path="outputs/B2_1_intent_layer_output.json"):
    """Load intent analysis from B2.1"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        # Try B1 output as fallback
        alt_path = script_dir / "outputs/B1_current_question.json"
        if alt_path.exists():
            full_path = alt_path
        else:
            raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_transformation(data):
    """
    Process declarative transformation
    
    Args:
        data: Question data with intent analysis
        
    Returns:
        dict: Declarative transformations
    """
    question = data.get("question", "")
    
    # Transform to declaratives
    declaratives = transform_to_declarative(question)
    
    # Extract key components
    components = extract_key_components(question, declaratives)
    
    # Score declaratives
    scored_declaratives = []
    for decl in declaratives:
        score = score_declarative_quality(decl, question)
        scored_declaratives.append({
            "declarative": decl,
            "quality_score": score
        })
    
    # Sort by quality score
    scored_declaratives.sort(key=lambda x: x["quality_score"], reverse=True)
    
    return {
        "question_id": data.get("question_id"),
        "question": question,
        "intent_analysis": data.get("intent_analysis", {}),
        "declarative_forms": scored_declaratives,
        "key_components": components,
        "best_declarative": scored_declaratives[0]["declarative"] if scored_declaratives else "",
        "transformation_confidence": scored_declaratives[0]["quality_score"] if scored_declaratives else 0,
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/B2_2_declarative_transformation_output.json"):
    """Save declarative transformation results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved declarative transformation to {full_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B2.2: Declarative Transformation")
    print("="*60)
    
    try:
        # Load question data
        print("Loading question data...")
        input_data = load_input()
        
        # Process transformation
        question = input_data.get("question", "")
        print(f"Transforming question: {question}")
        output_data = process_transformation(input_data)
        
        # Display results
        print(f"\nDeclarative Transformations:")
        for i, decl_data in enumerate(output_data["declarative_forms"], 1):
            print(f"  {i}. {decl_data['declarative']}")
            print(f"     Quality Score: {decl_data['quality_score']:.3f}")
        
        print(f"\nBest Declarative: {output_data['best_declarative']}")
        print(f"Confidence: {output_data['transformation_confidence']:.3f}")
        
        if output_data["key_components"]["subjects"]:
            print(f"\nKey Subjects: {', '.join(output_data['key_components']['subjects'])}")
        
        # Save output
        save_output(output_data)
        
        print("\nB2.2 Declarative Transformation completed successfully!")
        
    except Exception as e:
        print(f"Error in B2.2 Declarative Transformation: {str(e)}")
        raise

if __name__ == "__main__":
    main()