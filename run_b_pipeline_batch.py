#!/usr/bin/env python3
"""
Run B-Pipeline in batch for the 13 records with chunks
Processes each question individually through B1 -> B2 -> B5.2
"""

import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

def main():
    print("="*70)
    print("B-PIPELINE BATCH PROCESSOR - 13 RECORDS")
    print("="*70)
    
    # The 13 records that have chunks in A3
    target_records = [
        'finqa_test_1212', 'finqa_test_1395', 'finqa_test_1431',
        'finqa_test_1485', 'finqa_test_1552', 'finqa_test_1630',
        'finqa_test_462', 'finqa_test_487', 'finqa_test_515',
        'finqa_test_607', 'finqa_test_723', 'finqa_test_734',
        'finqa_test_869'
    ]
    
    print(f"\nProcessing {len(target_records)} records through B-Pipeline")
    
    # Load the sample data
    sample_file = Path("sample_20_records.parquet")
    df = pd.read_parquet(sample_file)
    
    # Filter to our target records
    filtered_df = df[df['id'].isin(target_records)]
    print(f"Loaded {len(filtered_df)} records with questions\n")
    
    # Process each question through B-pipeline
    results = []
    successful = 0
    
    for idx, (_, row) in enumerate(filtered_df.iterrows(), 1):
        record_id = row['id']
        question = row['question']
        
        print(f"\n[{idx}/{len(filtered_df)}] Processing: {record_id}")
        print(f"  Question: {question[:80]}...")
        
        # Create B1 input for this question
        b1_data = {
            "question_id": record_id,
            "question": question,
            "metadata": {
                "source": "sample_20_records.parquet",
                "record_id": record_id,
                "loaded_at": datetime.now().isoformat()
            }
        }
        
        # Save as current question for B1
        b1_path = Path("B_Retrieval_pipeline/outputs/B1_current_question.json")
        b1_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(b1_path, 'w', encoding='utf-8') as f:
            json.dump(b1_data, f, indent=2)
        
        # Run B-pipeline orchestrator
        cmd = [sys.executable, "scripts/B_pipeline_orchestrator.py"]
        result = subprocess.run(cmd, capture_output=True, text=True,
                              cwd="B_Retrieval_pipeline")
        
        if result.returncode == 0:
            print("  [OK] B-pipeline completed")
            
            # Check B5.2 output
            b52_path = Path("B_Retrieval_pipeline/outputs/B5.2_direct_answer_output.json")
            if b52_path.exists():
                with open(b52_path, 'r') as f:
                    answer_data = json.load(f)
                
                chunks_found = len(answer_data.get('ranked_chunks', []))
                confidence = answer_data.get('confidence', 0)
                answer = answer_data.get('answer', 'No answer generated')
                
                if chunks_found > 0:
                    successful += 1
                    status = "[OK]"
                else:
                    status = "[NO CHUNKS]"
                
                print(f"  {status} Chunks found: {chunks_found}, Confidence: {confidence:.2f}")
                print(f"  Answer: {answer[:100]}...")
                
                results.append({
                    'record_id': record_id,
                    'question': question,
                    'chunks_found': chunks_found,
                    'confidence': confidence,
                    'answer': answer,
                    'status': 'success' if chunks_found > 0 else 'no_chunks'
                })
            else:
                print("  [ERROR] No B5.2 output found")
                results.append({
                    'record_id': record_id,
                    'question': question,
                    'status': 'error',
                    'error': 'No B5.2 output'
                })
        else:
            print(f"  [ERROR] B-pipeline failed: {result.stderr[:200]}")
            results.append({
                'record_id': record_id,
                'question': question,
                'status': 'error',
                'error': result.stderr[:200]
            })
    
    # Save batch results
    batch_results = {
        'timestamp': datetime.now().isoformat(),
        'total_processed': len(results),
        'successful': successful,
        'results': results
    }
    
    batch_output = Path("B_Retrieval_pipeline/outputs/B_pipeline_batch_results.json")
    with open(batch_output, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "="*70)
    print("B-PIPELINE BATCH SUMMARY")
    print("="*70)
    print(f"Total processed: {len(results)}")
    print(f"Successful (chunks found): {successful}/{len(results)}")
    print(f"Failed (no chunks): {len(results) - successful}")
    
    print("\nPer-record results:")
    for r in results:
        if r['status'] == 'success':
            print(f"  [OK] {r['record_id']}: {r['chunks_found']} chunks, confidence: {r['confidence']:.2f}")
        elif r['status'] == 'no_chunks':
            print(f"  [NO CHUNKS] {r['record_id']}: No chunks found")
        else:
            print(f"  [ERROR] {r['record_id']}: Error - {r.get('error', 'Unknown')}")
    
    print(f"\nResults saved to: {batch_output}")
    print("\n" + "="*70)
    print("B-PIPELINE BATCH PROCESSING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()