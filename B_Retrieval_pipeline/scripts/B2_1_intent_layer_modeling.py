#!/usr/bin/env python3
"""
B2.1: Intent Layer Modeling
Analyze the intent behind the user's question
"""

import json
from pathlib import Path
from datetime import datetime
import re

def analyze_intent(question):
    """
    Analyze the intent of a question
    
    Args:
        question: Question text
        
    Returns:
        dict: Intent analysis
    """
    question_lower = question.lower()
    
    # Define intent patterns
    intent_patterns = {
        "comparison": [
            "compare", "difference", "versus", "vs", "between",
            "change", "increase", "decrease", "growth", "decline"
        ],
        "calculation": [
            "calculate", "compute", "how much", "how many", "total",
            "sum", "average", "percentage", "ratio", "rate"
        ],
        "definition": [
            "what is", "define", "meaning", "explain", "describe"
        ],
        "identification": [
            "which", "who", "what", "identify", "name", "list"
        ],
        "temporal": [
            "when", "time", "date", "year", "month", "period",
            "quarter", "fiscal", "annual"
        ],
        "causal": [
            "why", "because", "reason", "cause", "due to", "result"
        ],
        "procedural": [
            "how", "process", "method", "procedure", "steps"
        ],
        "factual": [
            "is", "are", "was", "were", "does", "do", "did"
        ]
    }
    
    # Detect primary intent
    detected_intents = []
    intent_scores = {}
    
    for intent, keywords in intent_patterns.items():
        score = sum(1 for keyword in keywords if keyword in question_lower)
        if score > 0:
            detected_intents.append(intent)
            intent_scores[intent] = score
    
    # Determine primary intent
    if intent_scores:
        primary_intent = max(intent_scores, key=intent_scores.get)
    else:
        primary_intent = "general"
    
    # Detect if question expects numeric answer
    expects_numeric = any(word in question_lower for word in [
        "how many", "how much", "number", "count", "total",
        "percentage", "percent", "%", "amount", "value"
    ])
    
    # Detect if question is about change/comparison
    is_comparative = any(word in question_lower for word in [
        "change", "difference", "increase", "decrease", "compare",
        "growth", "decline", "versus", "between"
    ])
    
    return {
        "primary_intent": primary_intent,
        "all_intents": detected_intents,
        "intent_scores": intent_scores,
        "expects_numeric": expects_numeric,
        "is_comparative": is_comparative,
        "confidence": min(1.0, len(detected_intents) * 0.3)
    }

def extract_key_entities(question):
    """
    Extract key entities from the question
    
    Args:
        question: Question text
        
    Returns:
        list: Key entities
    """
    # Simple entity extraction based on capitalization and patterns
    entities = []
    
    # Extract years (4-digit numbers)
    years = re.findall(r'\b(19|20)\d{2}\b', question)
    entities.extend([{"type": "year", "value": year} for year in years])
    
    # Extract percentages
    percentages = re.findall(r'\d+\.?\d*\s*%', question)
    entities.extend([{"type": "percentage", "value": pct} for pct in percentages])
    
    # Extract monetary values
    money = re.findall(r'\$\s*\d+\.?\d*\s*(?:million|billion|thousand)?', question)
    entities.extend([{"type": "money", "value": m} for m in money])
    
    # Extract capitalized terms (potential entity names)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
    for term in capitalized:
        if term.lower() not in ['what', 'how', 'when', 'where', 'why', 'which']:
            entities.append({"type": "entity", "value": term})
    
    return entities

def load_input(input_path="outputs/B1_current_question.json"):
    """Load question from B1"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_question(data):
    """
    Process question for intent analysis
    
    Args:
        data: Question data from B1
        
    Returns:
        dict: Question with intent analysis
    """
    question = data.get("question", "")
    
    # Analyze intent
    intent_analysis = analyze_intent(question)
    
    # Extract entities
    entities = extract_key_entities(question)
    
    # Identify focus terms (important words for retrieval)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'was', 'is', 'are'}
    words = question.lower().split()
    focus_terms = [w for w in words if w not in stop_words and len(w) > 2]
    
    return {
        "question_id": data.get("question_id"),
        "question": question,
        "intent_analysis": intent_analysis,
        "entities": entities,
        "focus_terms": focus_terms,
        "document_id": data.get("document_id"),
        "ground_truth": data.get("answer"),
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/B2_1_intent_layer_output.json"):
    """Save intent analysis results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved intent analysis to {full_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B2.1: Intent Layer Modeling")
    print("="*60)
    
    try:
        # Load question
        print("Loading question from B1...")
        input_data = load_input()
        
        # Process question
        print(f"Analyzing question: {input_data['question']}")
        output_data = process_question(input_data)
        
        # Display results
        intent = output_data["intent_analysis"]
        print(f"\nIntent Analysis:")
        print(f"  Primary Intent: {intent['primary_intent']}")
        print(f"  All Intents: {intent['all_intents']}")
        print(f"  Expects Numeric: {intent['expects_numeric']}")
        print(f"  Is Comparative: {intent['is_comparative']}")
        print(f"  Confidence: {intent['confidence']:.2f}")
        
        if output_data["entities"]:
            print(f"\nExtracted Entities:")
            for entity in output_data["entities"]:
                print(f"  - {entity['type']}: {entity['value']}")
        
        if output_data["focus_terms"]:
            print(f"\nFocus Terms: {', '.join(output_data['focus_terms'])}")
        
        # Save output
        save_output(output_data)
        
        print("\nB2.1 Intent Layer Modeling completed successfully!")
        
    except Exception as e:
        print(f"Error in B2.1 Intent Layer Modeling: {str(e)}")
        raise

if __name__ == "__main__":
    main()