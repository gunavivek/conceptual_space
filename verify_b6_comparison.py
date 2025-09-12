#!/usr/bin/env python3
"""
Simple verification of B6 comparison between B5 enhanced output and ground truth
"""

import pandas as pd
import json
import re
from pathlib import Path

def verify_comparison():
    print("="*60)
    print("VERIFYING B6 COMPARISON")
    print("="*60)
    
    # Load ground truth from parquet
    parquet_file = Path("A_Concept_pipeline/data/sample_20_records.parquet")
    df = pd.read_parquet(parquet_file)
    
    # Find finqa_test_1630
    target_row = df[df['id'] == 'finqa_test_1630'].iloc[0]
    
    print(f"Question: {target_row['question']}")
    print(f"\nGround Truth Response (first 300 chars):")
    response = target_row.get('response', '')
    print(response[:300] + "..." if len(response) > 300 else response)
    
    # Extract any percentage from ground truth
    percent_matches = re.findall(r'(\d+\.?\d*)\s*%', response)
    print(f"\nPercentages found in ground truth: {percent_matches}")
    
    # Load B5 enhanced output
    b5_file = Path("B_Retrieval_pipeline/outputs/B5_enhanced_answer_output.json")
    with open(b5_file, 'r') as f:
        b5_data = json.load(f)
    
    print(f"\n" + "="*40)
    print("B5 ENHANCED OUTPUT:")
    print(f"Answer: {b5_data.get('answer', 'N/A')}")
    print(f"Confidence: {b5_data.get('confidence', 'N/A')}")
    print(f"Question Type: {b5_data.get('question_type', 'N/A')}")
    
    if 'calculation_details' in b5_data:
        calc = b5_data['calculation_details']
        print(f"Calculation: {calc}")
    
    # Manual comparison
    b5_answer = b5_data.get('answer', '')
    b5_numeric = None
    
    # Extract number from B5 answer
    b5_match = re.search(r'(\d+\.?\d*)', str(b5_answer))
    if b5_match:
        b5_numeric = float(b5_match.group(1))
    
    print(f"\n" + "="*40)
    print("COMPARISON RESULT:")
    print(f"B5 Generated: {b5_answer}")
    print(f"B5 Numeric: {b5_numeric}")
    print(f"Ground Truth Percentages: {percent_matches}")
    
    # Check if B5 answer matches any ground truth percentage
    match_found = False
    if b5_numeric and percent_matches:
        for gt_percent in percent_matches:
            gt_numeric = float(gt_percent)
            if abs(b5_numeric - gt_numeric) < 0.1:  # Within 0.1%
                match_found = True
                print(f"[SUCCESS] MATCH FOUND: {b5_numeric}% matches {gt_numeric}%")
                break
    
    if not match_found:
        print("[INFO] No exact numeric match found in ground truth response")
        print("[NOTE] Ground truth 'response' field contains full generated text, not just the answer")
    
    return {
        'b5_answer': b5_answer,
        'b5_numeric': b5_numeric,
        'ground_truth_percentages': percent_matches,
        'match_found': match_found
    }

if __name__ == "__main__":
    verify_comparison()