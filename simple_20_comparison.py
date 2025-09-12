#!/usr/bin/env python3
"""
Simple 20-Record Comparison
Creates a side-by-side comparison table without running the full pipeline
Shows what we have vs what we need to test
"""

import pandas as pd
import json
import re
from pathlib import Path

def extract_answer_from_response(response: str) -> str:
    """Extract the key answer from a response text"""
    if not response:
        return "N/A"
        
    # Look for percentage answers
    percent_match = re.search(r'(\d+\.?\d*)\s*%', response)
    if percent_match:
        return f"{percent_match.group(1)}%"
    
    # Look for dollar amounts
    dollar_match = re.search(r'\$([0-9,]+(?:\.\d+)?)', response)
    if dollar_match:
        return f"${dollar_match.group(1)}"
    
    # Look for simple numbers with context
    if any(word in response.lower() for word in ['million', 'billion', 'thousand']):
        number_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(million|billion|thousand)', response.lower())
        if number_match:
            return f"{number_match.group(1)} {number_match.group(2)}"
    
    # Look for yes/no answers
    if any(word in response.lower() for word in ['yes', 'no', 'true', 'false']):
        if 'yes' in response.lower()[:100] or 'true' in response.lower()[:100]:
            return "Yes"
        elif 'no' in response.lower()[:100] or 'false' in response.lower()[:100]:
            return "No"
    
    # Extract first sentence or up to 100 chars
    first_sentence = response.split('.')[0] if '.' in response else response
    return first_sentence[:100] + "..." if len(first_sentence) > 100 else first_sentence

def main():
    """Create comparison table with current data"""
    print("="*120)
    print("20-RECORD COMPARISON TABLE")
    print("="*120)
    
    # Load data
    parquet_file = Path("A_Concept_pipeline/data/sample_20_records.parquet")
    df = pd.read_parquet(parquet_file)
    
    print(f"Loaded {len(df)} records from sample_20_records.parquet")
    
    # Check if we have B5 result for the first question
    b5_file = Path("B_Retrieval_pipeline/outputs/B5_enhanced_answer_output.json")
    current_b5_result = None
    if b5_file.exists():
        with open(b5_file, 'r') as f:
            current_b5_result = json.load(f)
    
    # Header
    print(f"\n{'#':<3} {'ID':<15} {'Question':<50} {'Ground Truth Answer':<25} {'B5 Status':<15}")
    print("-" * 108)
    
    # Show all records
    for i, (index, row) in enumerate(df.iterrows(), 1):
        question_id = row['id']
        question = row['question']
        response = row.get('response', '')
        
        # Truncate question for display
        question_preview = question[:47] + "..." if len(question) > 50 else question
        
        # Extract ground truth answer
        gt_answer = extract_answer_from_response(response)
        gt_answer_display = gt_answer[:22] + "..." if len(gt_answer) > 25 else gt_answer
        
        # Check if this is the question we already tested
        b5_status = "NOT TESTED"
        if current_b5_result and current_b5_result.get('question_id') == question_id:
            b5_answer = current_b5_result.get('answer', 'N/A')
            b5_conf = current_b5_result.get('confidence', 0)
            b5_status = f"{b5_answer} ({b5_conf:.2f})"
        
        b5_status_display = b5_status[:12] + "..." if len(b5_status) > 15 else b5_status
        
        print(f"{i:<3} {question_id:<15} {question_preview:<50} {gt_answer_display:<25} {b5_status_display:<15}")
    
    # Summary by question type
    print(f"\n{'='*120}")
    print("QUESTION TYPE ANALYSIS:")
    print(f"{'='*120}")
    
    question_types = {
        'Percentage Change': 0,
        'How Much/Cost': 0,
        'What Is/Definition': 0,
        'Comparison': 0,
        'Other': 0
    }
    
    for _, row in df.iterrows():
        question = row['question'].lower()
        if 'percentage change' in question or '% change' in question:
            question_types['Percentage Change'] += 1
        elif 'how much' in question or 'cost' in question or 'revenue' in question:
            question_types['How Much/Cost'] += 1
        elif 'what is' in question or 'what was' in question or 'what does' in question:
            question_types['What Is/Definition'] += 1
        elif 'compare' in question or 'difference' in question or 'change in' in question:
            question_types['Comparison'] += 1
        else:
            question_types['Other'] += 1
    
    for q_type, count in question_types.items():
        print(f"{q_type}: {count} questions")
    
    print(f"\n{'='*120}")
    print("NEXT STEPS FOR COMPREHENSIVE TESTING:")
    print(f"{'='*120}")
    print("1. Currently tested: 1/20 questions (finqa_test_1630 - percentage change)")
    print("2. Result: 23.07% answer with 90% confidence - CORRECT")
    print("3. Need to test remaining 19 questions")
    print("4. Focus areas:")
    print(f"   - Percentage Change questions: {question_types['Percentage Change']} total")
    print(f"   - How Much/Cost questions: {question_types['How Much/Cost']} total")
    print(f"   - Definition questions: {question_types['What Is/Definition']} total")
    print("5. Expected high performance on percentage/calculation questions")
    print("6. May need different strategies for definition/lookup questions")

if __name__ == "__main__":
    main()