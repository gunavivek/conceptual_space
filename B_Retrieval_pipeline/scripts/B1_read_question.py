#!/usr/bin/env python3
"""
B1: Read Question
Loads and processes user questions for the QA pipeline
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def load_question_from_parquet(data_path="../../A_Concept_pipeline/data/sample_20_records.parquet", question_index=0):
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
        full_path = Path(__file__).parent.parent.parent / "A_Concept_pipeline/data/sample_20_records.parquet"
        if not full_path.exists():
            raise FileNotFoundError(f"Data file not found: {full_path}")
    
    # Load parquet file - only read ID and Question columns
    df = pd.read_parquet(full_path, columns=['id', 'question'])
    
    if question_index >= len(df):
        raise IndexError(f"Question index {question_index} out of range (max: {len(df)-1})")
    
    # Extract question data
    row = df.iloc[question_index]
    
    question_data = {
        "question_id": row.get("id", f"question_{question_index}"),
        "question": row.get("question", ""),
        "metadata": {
            "source_file": str(full_path),
            "index": question_index,
            "loaded_at": datetime.now().isoformat()
        }
    }
    
    return question_data

def load_all_questions_from_parquet(data_path="../../A_Concept_pipeline/data/sample_20_records.parquet"):
    """
    Load all questions from parquet file
    
    Args:
        data_path: Path to parquet file
        
    Returns:
        list: List of question data dictionaries
    """
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / data_path
    
    if not full_path.exists():
        # Try alternative path
        full_path = Path(__file__).parent.parent.parent / "A_Concept_pipeline/data/sample_20_records.parquet"
        if not full_path.exists():
            raise FileNotFoundError(f"Data file not found: {full_path}")
    
    # Load parquet file - only read ID and Question columns
    df = pd.read_parquet(full_path, columns=['id', 'question'])
    
    all_questions = []
    
    for index, row in df.iterrows():
        question_data = {
            "question_id": row.get("id", f"question_{index}"),
            "question": row.get("question", ""),
            "metadata": {
                "source_file": str(full_path),
                "index": index,
                "loaded_at": datetime.now().isoformat()
            }
        }
        
        # Add question analysis
        analysis = analyze_question(question_data['question'])
        question_data['analysis'] = analysis
        
        all_questions.append(question_data)
    
    return all_questions

def load_question_by_id(record_id, data_path="data/sample_20_records.parquet"):
    """
    Load specific question by record ID
    
    Args:
        record_id: The specific record ID to load
        data_path: Path to the data file containing questions
        
    Returns:
        dict: Question data for the specific record
    """
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / data_path
    
    if not full_path.exists():
        # Try B-pipeline specific data directory
        b_data_path = script_dir / "data" / f"{record_id}.parquet"
        if b_data_path.exists():
            full_path = b_data_path
        else:
            # Try alternative path in A-pipeline
            full_path = Path(__file__).parent.parent.parent / "A_Concept_pipeline/data/sample_20_records.parquet"
            if not full_path.exists():
                raise FileNotFoundError(f"Data file not found for record {record_id}")
    
    # Load parquet file - only read ID and Question columns
    df = pd.read_parquet(full_path, columns=['id', 'question'])
    
    # Find the specific record
    matching_rows = df[df['id'] == record_id]
    if matching_rows.empty:
        raise ValueError(f"Record ID '{record_id}' not found in data file")
    
    # Extract question data
    row = matching_rows.iloc[0]
    
    question_data = {
        "question_id": row.get("id", record_id),
        "question": row.get("question", ""),
        "metadata": {
            "source_file": str(full_path),
            "record_id": record_id,
            "loaded_at": datetime.now().isoformat()
        }
    }
    
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
    
    print(f"[OK] Saved question data to {full_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B1: Read Question")
    print("="*60)
    
    try:
        # Check for command line argument for specific record ID or ALL flag
        import sys
        if len(sys.argv) > 1:
            if sys.argv[1].upper() == "ALL":
                print("Loading all questions from sample_20_records.parquet...")
                all_questions = load_all_questions_from_parquet()
                
                print(f"\nLoaded {len(all_questions)} questions:")
                for i, q in enumerate(all_questions):
                    print(f"  {i+1:2d}. [{q['question_id']}] {q['analysis']['question_type']} - {q['question'][:60]}...")
                
                # Save all questions as array
                save_output(all_questions)
                print(f"\nSaved all {len(all_questions)} questions to B1_current_question.json")
                
            else:
                record_id = sys.argv[1]
                print(f"Loading question for record: {record_id}")
                question_data = load_question_by_id(record_id)
                
                print(f"\nQuestion ID: {question_data['question_id']}")
                print(f"Question: {question_data['question']}")
                
                # Analyze question
                analysis = analyze_question(question_data['question'])
                question_data['analysis'] = analysis
                
                print(f"\nQuestion Analysis:")
                print(f"  Type: {analysis['question_type']}")
                print(f"  Expected Answer Type: {analysis['expected_answer_type']}")
                print(f"  Word Count: {analysis['word_count']}")
                
                # Save single question
                save_output(question_data)
                
        else:
            # Default: Load all questions
            print("Loading all questions from sample_20_records.parquet...")
            all_questions = load_all_questions_from_parquet()
            
            print(f"\nLoaded {len(all_questions)} questions:")
            for i, q in enumerate(all_questions):
                print(f"  {i+1:2d}. [{q['question_id']}] {q['analysis']['question_type']} - {q['question'][:60]}...")
            
            # Save all questions as array
            save_output(all_questions)
            print(f"\nSaved all {len(all_questions)} questions to B1_current_question.json")
        
        print("\nB1 Read Question completed successfully!")
        
    except Exception as e:
        print(f"Error in B1 Read Question: {str(e)}")
        raise

if __name__ == "__main__":
    main()