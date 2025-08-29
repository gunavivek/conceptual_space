#!/usr/bin/env python3
"""
B6: Answer Comparison
Compare generated answer with ground truth
"""

import json
from pathlib import Path
from datetime import datetime
import re
from difflib import SequenceMatcher

def normalize_answer(answer):
    """
    Normalize answer for comparison
    
    Args:
        answer: Answer text
        
    Returns:
        str: Normalized answer
    """
    if not answer:
        return ""
    
    # Convert to lowercase
    normalized = answer.lower()
    
    # Remove punctuation except numbers and decimal points
    normalized = re.sub(r'[^\w\s\.\d]', ' ', normalized)
    
    # Normalize whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized

def extract_numeric_value(text):
    """
    Extract numeric value from answer
    
    Args:
        text: Answer text
        
    Returns:
        float or None: Extracted numeric value
    """
    # Look for patterns like "66.8 million", "42.5", "$100"
    patterns = [
        r'(\d+\.?\d*)\s*(?:million|billion|thousand)?',
        r'\$\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*%'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return float(match.group(1))
            except:
                pass
    
    return None

def calculate_similarity(answer1, answer2):
    """
    Calculate similarity between two answers
    
    Args:
        answer1: First answer
        answer2: Second answer
        
    Returns:
        float: Similarity score (0-1)
    """
    # Normalize answers
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)
    
    # Calculate text similarity
    text_similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Check numeric similarity if both contain numbers
    num1 = extract_numeric_value(answer1)
    num2 = extract_numeric_value(answer2)
    
    if num1 is not None and num2 is not None:
        # Calculate numeric similarity
        if num2 != 0:
            numeric_similarity = 1 - abs(num1 - num2) / abs(num2)
            numeric_similarity = max(0, numeric_similarity)
        else:
            numeric_similarity = 1.0 if num1 == 0 else 0.0
        
        # Weight numeric similarity higher for numeric answers
        return 0.3 * text_similarity + 0.7 * numeric_similarity
    
    return text_similarity

def evaluate_answer(generated, ground_truth):
    """
    Evaluate generated answer against ground truth
    
    Args:
        generated: Generated answer
        ground_truth: Ground truth answer
        
    Returns:
        dict: Evaluation metrics
    """
    similarity = calculate_similarity(generated, ground_truth)
    
    # Check for exact match
    exact_match = normalize_answer(generated) == normalize_answer(ground_truth)
    
    # Check if both contain the same numeric value
    gen_num = extract_numeric_value(generated)
    truth_num = extract_numeric_value(ground_truth)
    numeric_match = (gen_num is not None and 
                    truth_num is not None and 
                    abs(gen_num - truth_num) < 0.01)
    
    # Determine correctness based on thresholds
    is_correct = exact_match or numeric_match or similarity > 0.8
    is_partial = similarity > 0.5 and not is_correct
    
    return {
        "similarity_score": similarity,
        "exact_match": exact_match,
        "numeric_match": numeric_match,
        "is_correct": is_correct,
        "is_partial": is_partial,
        "extracted_numeric": {
            "generated": gen_num,
            "ground_truth": truth_num
        }
    }

def load_inputs():
    """Load generated answer and ground truth"""
    script_dir = Path(__file__).parent.parent
    
    # Load generated answer from B5
    b5_path = script_dir / "outputs/B5_generated_answer.json"
    if b5_path.exists():
        with open(b5_path, 'r', encoding='utf-8') as f:
            b5_data = json.load(f)
    else:
        # Mock data if B5 output doesn't exist
        b5_data = {
            "question": "What was the change in Current deferred income?",
            "generated_answer": "The deferred income for 2019 is 66.8 million",
            "model_used": "mock",
            "confidence": 0.7
        }
    
    # Load ground truth from B1
    b1_path = script_dir / "outputs/B1_current_question.json"
    if b1_path.exists():
        with open(b1_path, 'r', encoding='utf-8') as f:
            b1_data = json.load(f)
            ground_truth = b1_data.get("answer", "")
    else:
        ground_truth = "66.8"
    
    return b5_data, ground_truth

def process_comparison(generated_data, ground_truth):
    """
    Process answer comparison
    
    Args:
        generated_data: Generated answer data from B5
        ground_truth: Ground truth answer
        
    Returns:
        dict: Comparison results
    """
    generated_answer = generated_data.get("generated_answer", "")
    
    # Evaluate answer
    evaluation = evaluate_answer(generated_answer, ground_truth)
    
    return {
        "question": generated_data.get("question", ""),
        "generated_answer": generated_answer,
        "ground_truth": ground_truth,
        "evaluation": evaluation,
        "model_used": generated_data.get("model_used", "unknown"),
        "generation_confidence": generated_data.get("confidence", 0.0),
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/B6_answer_comparison.json"):
    """Save comparison results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved answer comparison to {full_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B6: Answer Comparison")
    print("="*60)
    
    try:
        # Load inputs
        print("Loading generated answer and ground truth...")
        generated_data, ground_truth = load_inputs()
        
        # Process comparison
        print(f"Comparing answers...")
        output_data = process_comparison(generated_data, ground_truth)
        
        # Display results
        print(f"\nQuestion: {output_data['question']}")
        print(f"\nGenerated Answer: {output_data['generated_answer']}")
        print(f"Ground Truth: {output_data['ground_truth']}")
        
        eval_results = output_data['evaluation']
        print(f"\nEvaluation:")
        print(f"  Similarity Score: {eval_results['similarity_score']:.3f}")
        print(f"  Exact Match: {eval_results['exact_match']}")
        print(f"  Numeric Match: {eval_results['numeric_match']}")
        print(f"  Is Correct: {eval_results['is_correct']}")
        print(f"  Is Partial: {eval_results['is_partial']}")
        
        if eval_results['extracted_numeric']['generated'] is not None:
            print(f"\nExtracted Numbers:")
            print(f"  Generated: {eval_results['extracted_numeric']['generated']}")
            print(f"  Ground Truth: {eval_results['extracted_numeric']['ground_truth']}")
        
        # Save output
        save_output(output_data)
        
        print("\nB6 Answer Comparison completed successfully!")
        
    except Exception as e:
        print(f"Error in B6 Answer Comparison: {str(e)}")
        raise

if __name__ == "__main__":
    main()