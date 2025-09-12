#!/usr/bin/env python3
"""
Load a specific question from sample_20_records.parquet for B-pipeline testing
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def load_question_from_sample(question_index=0):
    """Load a question from sample_20_records.parquet"""
    
    # Path to the sample file
    data_file = Path(__file__).parent.parent.parent / "A_Concept_pipeline" / "data" / "sample_20_records.parquet"
    
    if not data_file.exists():
        print(f"[ERROR] File not found: {data_file}")
        return None
    
    # Load the parquet file
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} records from sample_20_records.parquet")
    
    # Display all questions
    print("\nAvailable questions:")
    for i, row in df.iterrows():
        question = row['question']
        question_preview = question[:80] + "..." if len(question) > 80 else question
        print(f"  {i}: [{row['id']}] {question_preview}")
    
    # Load the specific question
    if question_index >= len(df):
        print(f"[ERROR] Question index {question_index} out of range (max: {len(df)-1})")
        return None
    
    row = df.iloc[question_index]
    
    # Analyze question type
    question_lower = row['question'].lower()
    if "percentage change" in question_lower or "% change" in question_lower:
        question_type = "percentage_change"
        expected_answer = "percentage"
    elif "how much" in question_lower or "how many" in question_lower:
        question_type = "how_much"
        expected_answer = "monetary"
    elif "what was" in question_lower or "what is" in question_lower:
        question_type = "what"
        expected_answer = "text"
    else:
        question_type = "general"
        expected_answer = "text"
    
    # Create B1 format question data
    question_data = {
        "question_id": row['id'],
        "question": row['question'],
        "metadata": {
            "source_file": str(data_file),
            "index": question_index,
            "loaded_at": datetime.now().isoformat()
        },
        "analysis": {
            "question_type": question_type,
            "expected_answer_type": expected_answer,
            "word_count": len(row['question'].split()),
            "contains_numbers": any(char.isdigit() for char in row['question'])
        }
    }
    
    # Save to B1 output
    output_file = Path(__file__).parent.parent / "outputs" / "B1_current_question.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(question_data, f, indent=2)
    
    print(f"\n[OK] Loaded question {question_index}: {row['question']}")
    print(f"[OK] Question type: {question_type}")
    print(f"[OK] Saved to: {output_file}")
    
    return question_data

if __name__ == "__main__":
    import sys
    
    # Get question index from command line or use default (0)
    if len(sys.argv) > 1:
        try:
            index = int(sys.argv[1])
        except ValueError:
            print(f"[ERROR] Invalid index: {sys.argv[1]}")
            index = 0
    else:
        index = 0
    
    load_question_from_sample(index)