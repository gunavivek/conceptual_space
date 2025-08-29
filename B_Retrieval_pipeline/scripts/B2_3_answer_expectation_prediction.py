#!/usr/bin/env python3
"""
B2.3: Answer Expectation Prediction
Predicts what type of answer is expected based on question analysis
"""

import json
from pathlib import Path
from datetime import datetime
import re

def predict_answer_type(question, intent_analysis):
    """
    Predict the expected answer type
    
    Args:
        question: Original question
        intent_analysis: Intent analysis from B2.1
        
    Returns:
        dict: Answer type predictions
    """
    question_lower = question.lower()
    predictions = {
        "primary_type": "text",
        "confidence": 0.5,
        "possible_types": [],
        "format_hints": []
    }
    
    # Numeric answer patterns
    numeric_patterns = [
        r'how\s+much',
        r'how\s+many',
        r'what\s+(?:is|was)\s+the\s+(?:amount|value|number|total|sum)',
        r'change\s+in',
        r'increase\s+(?:in|of)',
        r'decrease\s+(?:in|of)',
        r'percentage',
        r'ratio',
        r'rate'
    ]
    
    for pattern in numeric_patterns:
        if re.search(pattern, question_lower):
            predictions["primary_type"] = "numeric"
            predictions["confidence"] = 0.8
            predictions["possible_types"].append("numeric")
            break
    
    # Date/Time answer patterns
    date_patterns = [
        r'when\s+(?:did|was|will)',
        r'what\s+(?:date|time|year|month)',
        r'in\s+what\s+(?:year|period|quarter)'
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, question_lower):
            if predictions["primary_type"] == "text":
                predictions["primary_type"] = "date"
                predictions["confidence"] = 0.7
            predictions["possible_types"].append("date")
    
    # Boolean answer patterns
    boolean_patterns = [
        r'^(?:is|are|was|were|do|does|did)',
        r'^(?:can|could|will|would|should)',
        r'true\s+or\s+false',
        r'yes\s+or\s+no'
    ]
    
    for pattern in boolean_patterns:
        if re.search(pattern, question_lower):
            if predictions["primary_type"] == "text":
                predictions["primary_type"] = "boolean"
                predictions["confidence"] = 0.6
            predictions["possible_types"].append("boolean")
    
    # List/Multiple answer patterns
    list_patterns = [
        r'what\s+(?:are\s+the|were\s+the)',
        r'list\s+(?:all|the)',
        r'name\s+(?:all|the)',
        r'which\s+(?:of\s+the|are)'
    ]
    
    for pattern in list_patterns:
        if re.search(pattern, question_lower):
            if predictions["primary_type"] == "text":
                predictions["primary_type"] = "list"
                predictions["confidence"] = 0.7
            predictions["possible_types"].append("list")
    
    return predictions

def predict_answer_format(question, answer_type):
    """
    Predict specific format details for the answer
    
    Args:
        question: Original question
        answer_type: Predicted answer type
        
    Returns:
        dict: Format specifications
    """
    format_spec = {
        "units": None,
        "precision": None,
        "structure": "simple",
        "context_required": False
    }
    
    question_lower = question.lower()
    
    if answer_type == "numeric":
        # Detect units
        if any(unit in question_lower for unit in ['million', 'billion', 'thousand']):
            if 'million' in question_lower:
                format_spec["units"] = "million"
            elif 'billion' in question_lower:
                format_spec["units"] = "billion"
            elif 'thousand' in question_lower:
                format_spec["units"] = "thousand"
        
        if any(word in question_lower for word in ['dollar', '$', 'cost', 'price', 'revenue', 'income']):
            format_spec["units"] = format_spec.get("units", "") + " dollars"
            format_spec["precision"] = 2
        
        if any(word in question_lower for word in ['percentage', 'percent', '%', 'rate']):
            format_spec["units"] = "percent"
            format_spec["precision"] = 1
        
        if any(word in question_lower for word in ['change', 'difference', 'increase', 'decrease']):
            format_spec["context_required"] = True
            format_spec["structure"] = "comparative"
    
    elif answer_type == "date":
        if any(word in question_lower for word in ['year']):
            format_spec["precision"] = "year"
        elif any(word in question_lower for word in ['quarter', 'q1', 'q2', 'q3', 'q4']):
            format_spec["precision"] = "quarter"
        else:
            format_spec["precision"] = "full_date"
    
    return format_spec

def analyze_answer_complexity(question, declaratives):
    """
    Analyze expected answer complexity
    
    Args:
        question: Original question
        declaratives: Declarative forms
        
    Returns:
        dict: Complexity analysis
    """
    complexity = {
        "level": "simple",  # simple, moderate, complex
        "requires_calculation": False,
        "requires_comparison": False,
        "requires_context": False,
        "multi_part": False
    }
    
    question_lower = question.lower()
    
    # Check for calculation requirements
    calc_words = ['total', 'sum', 'average', 'difference', 'change', 'ratio', 'percentage']
    if any(word in question_lower for word in calc_words):
        complexity["requires_calculation"] = True
        complexity["level"] = "moderate"
    
    # Check for comparison requirements
    comp_words = ['compare', 'versus', 'between', 'higher', 'lower', 'more', 'less']
    if any(word in question_lower for word in comp_words):
        complexity["requires_comparison"] = True
        complexity["level"] = "moderate"
    
    # Check for context requirements
    if any(word in question_lower for word in ['why', 'because', 'reason', 'explain']):
        complexity["requires_context"] = True
        complexity["level"] = "complex"
    
    # Check for multi-part questions
    if len([d for d in declaratives if d]) > 2:
        complexity["multi_part"] = True
        if complexity["level"] == "simple":
            complexity["level"] = "moderate"
    
    return complexity

def load_input(input_path="outputs/B2_2_declarative_transformation_output.json"):
    """Load declarative transformation from B2.2"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        # Try B2.1 output as fallback
        alt_path = script_dir / "outputs/B2_1_intent_layer_output.json"
        if alt_path.exists():
            full_path = alt_path
        else:
            raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_answer_expectation(data):
    """
    Process answer expectation prediction
    
    Args:
        data: Question data with transformations
        
    Returns:
        dict: Answer expectation analysis
    """
    question = data.get("question", "")
    intent_analysis = data.get("intent_analysis", {})
    declaratives = [d.get("declarative", "") for d in data.get("declarative_forms", [])]
    
    # Predict answer type
    answer_prediction = predict_answer_type(question, intent_analysis)
    
    # Predict format
    format_spec = predict_answer_format(question, answer_prediction["primary_type"])
    
    # Analyze complexity
    complexity = analyze_answer_complexity(question, declaratives)
    
    # Create validation criteria
    validation_criteria = {
        "type_check": answer_prediction["primary_type"],
        "format_check": format_spec,
        "length_expectation": "short" if complexity["level"] == "simple" else "medium",
        "context_required": complexity["requires_context"],
        "must_contain": []
    }
    
    # Add specific validation requirements
    if answer_prediction["primary_type"] == "numeric":
        validation_criteria["must_contain"].append("number")
    if format_spec.get("units"):
        validation_criteria["must_contain"].append(format_spec["units"])
    
    return {
        "question_id": data.get("question_id"),
        "question": question,
        "answer_prediction": answer_prediction,
        "format_specification": format_spec,
        "complexity_analysis": complexity,
        "validation_criteria": validation_criteria,
        "confidence": answer_prediction["confidence"],
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/B2_3_answer_expectation_output.json"):
    """Save answer expectation results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved answer expectation to {full_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B2.3: Answer Expectation Prediction")
    print("="*60)
    
    try:
        # Load transformation data
        print("Loading question transformation data...")
        input_data = load_input()
        
        # Process expectation
        question = input_data.get("question", "")
        print(f"Analyzing expectations for: {question}")
        output_data = process_answer_expectation(input_data)
        
        # Display results
        prediction = output_data["answer_prediction"]
        format_spec = output_data["format_specification"]
        complexity = output_data["complexity_analysis"]
        
        print(f"\nAnswer Type Prediction:")
        print(f"  Primary Type: {prediction['primary_type']}")
        print(f"  Confidence: {prediction['confidence']:.3f}")
        if prediction["possible_types"]:
            print(f"  Possible Types: {', '.join(prediction['possible_types'])}")
        
        print(f"\nFormat Specification:")
        if format_spec["units"]:
            print(f"  Units: {format_spec['units']}")
        if format_spec["precision"]:
            print(f"  Precision: {format_spec['precision']}")
        print(f"  Structure: {format_spec['structure']}")
        
        print(f"\nComplexity Analysis:")
        print(f"  Level: {complexity['level']}")
        print(f"  Requires Calculation: {complexity['requires_calculation']}")
        print(f"  Requires Comparison: {complexity['requires_comparison']}")
        print(f"  Requires Context: {complexity['requires_context']}")
        
        # Save output
        save_output(output_data)
        
        print("\nB2.3 Answer Expectation Prediction completed successfully!")
        
    except Exception as e:
        print(f"Error in B2.3 Answer Expectation Prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()