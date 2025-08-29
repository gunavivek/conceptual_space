#!/usr/bin/env python3
"""
B1: Read Question
Loads and processes user questions for the QA pipeline
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def load_question_from_parquet(data_path="../../data/test_5_records.parquet", question_index=0):
    """
    Load a question from parquet file
    
    Args:
        data_path: Path to parquet file
        question_index: Index of question to load
        
    Returns:
        dict: Question data
    """
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / data_path
    
    if not full_path.exists():
        # Try alternative path
        full_path = Path(__file__).parent.parent.parent / "data/test_5_records.parquet"
        if not full_path.exists():
            raise FileNotFoundError(f"Data file not found: {full_path}")
    
    # Load parquet file
    df = pd.read_parquet(full_path)
    
    if question_index >= len(df):
        raise IndexError(f"Question index {question_index} out of range (max: {len(df)-1})")
    
    # Extract question data
    row = df.iloc[question_index]
    
    question_data = {
        "question_id": row.get("id", f"question_{question_index}"),
        "question": row.get("question", ""),
        "document_id": row.get("doc_id", ""),
        "answer": row.get("answer", ""),
        "metadata": {
            "source_file": str(full_path),
            "index": question_index,
            "loaded_at": datetime.now().isoformat()
        }
    }
    
    # Add any additional fields
    for col in df.columns:
        if col not in ["id", "question", "doc_id", "answer"]:
            question_data["metadata"][col] = row.get(col)
    
    return question_data

def analyze_question(question_text):
    """
    Basic question analysis
    
    Args:
        question_text: The question string
        
    Returns:
        dict: Question analysis
    """
    # Identify question type
    question_lower = question_text.lower()
    
    question_type = "unknown"
    if question_lower.startswith("what"):
        question_type = "what"
    elif question_lower.startswith("how"):
        question_type = "how"
    elif question_lower.startswith("why"):
        question_type = "why"
    elif question_lower.startswith("when"):
        question_type = "when"
    elif question_lower.startswith("where"):
        question_type = "where"
    elif question_lower.startswith("who"):
        question_type = "who"
    elif "?" in question_text:
        question_type = "yes/no"
    
    # Identify potential answer type
    answer_type = "text"
    if any(word in question_lower for word in ["how many", "how much", "number", "count", "total"]):
        answer_type = "numeric"
    elif any(word in question_lower for word in ["percentage", "percent", "%", "ratio"]):
        answer_type = "percentage"
    elif any(word in question_lower for word in ["when", "date", "year", "month"]):
        answer_type = "date"
    elif any(word in question_lower for word in ["yes", "no", "is", "are", "does", "do"]):
        answer_type = "boolean"
    
    return {
        "question_type": question_type,
        "expected_answer_type": answer_type,
        "word_count": len(question_text.split()),
        "contains_numbers": any(char.isdigit() for char in question_text)
    }

def save_output(data, output_path="outputs/B1_current_question.json"):
    """
    Save processed question to JSON
    
    Args:
        data: Question data to save
        output_path: Path for output file
    """
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    # Create output directory if it doesn't exist
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved question data to {full_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B1: Read Question")
    print("="*60)
    
    try:
        # Load question (default: first question)
        print("Loading question from dataset...")
        question_data = load_question_from_parquet(question_index=0)
        
        print(f"\nQuestion ID: {question_data['question_id']}")
        print(f"Question: {question_data['question']}")
        print(f"Document ID: {question_data['document_id']}")
        print(f"Ground Truth Answer: {question_data['answer']}")
        
        # Analyze question
        analysis = analyze_question(question_data['question'])
        question_data['analysis'] = analysis
        
        print(f"\nQuestion Analysis:")
        print(f"  Type: {analysis['question_type']}")
        print(f"  Expected Answer Type: {analysis['expected_answer_type']}")
        print(f"  Word Count: {analysis['word_count']}")
        
        # Save output
        save_output(question_data)
        
        print("\nB1 Read Question completed successfully!")
        
    except Exception as e:
        print(f"Error in B1 Read Question: {str(e)}")
        raise

if __name__ == "__main__":
    main()