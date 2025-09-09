#!/usr/bin/env python3
"""
Run B-Pipeline for the 13 records that have chunks in A3
B1 -> (B2.1, B2.2, B2.3, B2.4) -> B5.2
"""

import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

def main():
    print("="*70)
    print("B-PIPELINE PROCESSOR - 13 RECORDS WITH CHUNKS")
    print("="*70)
    
    # The 13 records that have chunks in A3
    target_records = [
        'finqa_test_1212', 'finqa_test_1395', 'finqa_test_1431',
        'finqa_test_1485', 'finqa_test_1552', 'finqa_test_1630',
        'finqa_test_462', 'finqa_test_487', 'finqa_test_515',
        'finqa_test_607', 'finqa_test_723', 'finqa_test_734',
        'finqa_test_869'
    ]
    
    print(f"\nProcessing B-Pipeline for {len(target_records)} records:")
    for i, record_id in enumerate(target_records, 1):
        print(f"  {i:2d}. {record_id}")
    
    # Load the full sample data
    sample_file = Path("sample_20_records.parquet")
    df = pd.read_parquet(sample_file)
    
    # Filter to only our target records
    filtered_df = df[df['id'].isin(target_records)]
    print(f"\nLoaded {len(filtered_df)} matching records from sample data")
    
    # Extract questions for these records
    questions = []
    for _, row in filtered_df.iterrows():
        questions.append({
            'question_id': row['id'],
            'question': row['question'],
            'doc_id': row['id']  # Using same ID for document reference
        })
    
    # Save questions for B1.1
    b1_output = {
        'questions': questions,
        'count': len(questions),
        'metadata': {
            'source': 'Filtered from sample_20_records.parquet',
            'record_ids': target_records
        }
    }
    
    b1_path = Path("B_Question_pipeline/outputs/B1.1_raw_questions.json")
    b1_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Step 1] B1.1 - Saving {len(questions)} questions")
    with open(b1_path, 'w', encoding='utf-8') as f:
        json.dump(b1_output, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved to {b1_path}")
    
    # Run B2.1 - Question preprocessing
    print("\n[Step 2] B2.1 - Question Preprocessing")
    b21_script = Path("B_Question_pipeline/scripts/B2.1_preprocess_questions.py")
    if b21_script.exists():
        cmd = [sys.executable, str(b21_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              cwd="B_Question_pipeline")
        if result.returncode == 0:
            print("  [OK] B2.1 completed")
        else:
            print(f"  [ERROR] B2.1 failed: {result.stderr[:200]}")
    else:
        print(f"  [SKIP] Script not found: {b21_script}")
    
    # Run B2.2 - Extract question keywords
    print("\n[Step 3] B2.2 - Extract Question Keywords")
    b22_script = Path("B_Question_pipeline/scripts/B2.2_extract_question_keywords.py")
    if b22_script.exists():
        cmd = [sys.executable, str(b22_script)]
        result = subprocess.run(cmd, capture_output=True, text=True,
                              cwd="B_Question_pipeline")
        if result.returncode == 0:
            print("  [OK] B2.2 completed")
        else:
            print(f"  [ERROR] B2.2 failed: {result.stderr[:200]}")
    else:
        print(f"  [SKIP] Script not found: {b22_script}")
    
    # Run B2.3 - Identify question intent
    print("\n[Step 4] B2.3 - Identify Question Intent")
    b23_script = Path("B_Question_pipeline/scripts/B2.3_identify_question_intent.py")
    if b23_script.exists():
        cmd = [sys.executable, str(b23_script)]
        result = subprocess.run(cmd, capture_output=True, text=True,
                              cwd="B_Question_pipeline")
        if result.returncode == 0:
            print("  [OK] B2.3 completed")
        else:
            print(f"  [ERROR] B2.3 failed: {result.stderr[:200]}")
    else:
        print(f"  [SKIP] Script not found: {b23_script}")
    
    # Run B2.4 - Map to concepts
    print("\n[Step 5] B2.4 - Map Questions to Concepts")
    b24_script = Path("B_Question_pipeline/scripts/B2.4_map_to_concepts.py")
    if b24_script.exists():
        cmd = [sys.executable, str(b24_script)]
        result = subprocess.run(cmd, capture_output=True, text=True,
                              cwd="B_Question_pipeline")
        if result.returncode == 0:
            print("  [OK] B2.4 completed")
        else:
            print(f"  [ERROR] B2.4 failed: {result.stderr[:200]}")
    else:
        print(f"  [SKIP] Script not found: {b24_script}")
    
    # Run B5.2 - Retrieve context
    print("\n[Step 6] B5.2 - Retrieve Context from Chunks")
    b52_script = Path("B_Question_pipeline/scripts/B5.2_retrieve_context.py")
    if b52_script.exists():
        cmd = [sys.executable, str(b52_script)]
        result = subprocess.run(cmd, capture_output=True, text=True,
                              cwd="B_Question_pipeline")
        if result.returncode == 0:
            print("  [OK] B5.2 completed - Context retrieved")
        else:
            print(f"  [ERROR] B5.2 failed: {result.stderr[:200]}")
    else:
        print(f"  [SKIP] Script not found: {b52_script}")
    
    # Verify results
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Check B5.2 output
    b52_output = Path("B_Question_pipeline/outputs/B5.2_retrieved_contexts.json")
    if b52_output.exists():
        with open(b52_output, 'r') as f:
            contexts = json.load(f)
        
        retrieved = contexts.get('retrieved_contexts', [])
        print(f"Total contexts retrieved: {len(retrieved)}")
        
        # Check success rate
        successful = sum(1 for r in retrieved if r.get('chunks_found', 0) > 0)
        print(f"Successful retrievals: {successful}/{len(retrieved)}")
        
        # Show sample results
        print("\nSample results:")
        for i, ctx in enumerate(retrieved[:5], 1):
            qid = ctx.get('question_id', 'unknown')
            chunks = ctx.get('chunks_found', 0)
            status = "✓" if chunks > 0 else "✗"
            print(f"  {status} {qid}: {chunks} chunks found")
    else:
        print(f"[ERROR] B5.2 output not found: {b52_output}")
    
    print("\n" + "="*70)
    print("B-PIPELINE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()