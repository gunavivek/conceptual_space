#!/usr/bin/env python3
"""
Test script to process just 3 records through the fixed A-Pipeline
This will help verify the batch aggregation fix before running all 20 records
"""

import pandas as pd
import subprocess
import json
import sys
from pathlib import Path

def main():
    print("="*70)
    print("TEST: Processing 3 Records with Fixed A-Pipeline")
    print("="*70)
    
    # Load sample records
    df = pd.read_parquet("sample_20_records.parquet")
    records = df.head(3).to_dict('records')  # Get first 3 records
    
    print(f"\nProcessing {len(records)} test records:")
    for r in records:
        print(f"  - {r['id']}")
    
    # Process each record through steps 1-7
    for i, record in enumerate(records, 1):
        record_id = record['id']
        print(f"\n[{i}/3] Processing: {record_id}")
        
        # Create individual record file
        record_df = pd.DataFrame([record])
        record_file = Path("A_Concept_pipeline/data") / f"{record_id}.parquet"
        record_df.to_parquet(record_file, index=False)
        print(f"  Created: {record_file.name}")
        
        # Run A1.1 with append mode (except for first record)
        a11_script = "scripts/A1.1_document_reader.py"
        data_file = f"data/{record_id}.parquet"
        
        if i == 1:
            cmd = [sys.executable, a11_script, data_file]
            print("  Running A1.1 in overwrite mode...")
        else:
            cmd = [sys.executable, a11_script, data_file, "--append"]
            print("  Running A1.1 in append mode...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              cwd="A_Concept_pipeline")
        
        if result.returncode != 0:
            print(f"  [ERROR] A1.1 failed: {result.stderr[:200]}")
            sys.exit(1)
        print(f"  [OK] A1.1 completed")
        
        # Run remaining steps (A1.2 through A2.5)
        steps = [
            ("A1.2", "A1.2_domain_concept_enrichment.py"),
            ("A2.1", "A2.1_preprocess_document_analysis.py"),
            ("A2.2", "A2.2_keyword_phrase_extraction.py"),
            ("A2.3", "A2.3_concept_grouping_thematic.py"),
            ("A2.4", "A2.4_synthesize_core_concepts.py"),
            ("A2.5", "A2.5_expanded_concepts_orchestrator.py")
        ]
        
        for step_name, script_name in steps:
            print(f"  Running {step_name}...")
            script = f"scripts/{script_name}"
            cmd = [sys.executable, script]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd="A_Concept_pipeline")
            
            if result.returncode != 0:
                print(f"  [ERROR] {step_name} failed")
                sys.exit(1)
            print(f"  [OK] {step_name} completed")
    
    # Run A3 once for all records
    print("\n" + "="*70)
    print("Running A3 - Multi-Strategy Chunking for ALL 3 Records")
    print("="*70)
    
    a3_script = "scripts/A3_concept_based_chunking.py"
    cmd = [sys.executable, a3_script]
    result = subprocess.run(cmd, capture_output=True, text=True,
                          cwd="A_Concept_pipeline")
    
    if result.returncode != 0:
        print(f"[ERROR] A3 failed: {result.stderr[:500]}")
        sys.exit(1)
    print("[OK] A3 completed")
    
    # Verify chunks
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    chunk_file = Path("A_Concept_pipeline/outputs/A3_multi_strategy_chunks.json")
    with open(chunk_file, 'r') as f:
        chunk_data = json.load(f)
    
    chunks = chunk_data.get('chunks', [])
    print(f"Total chunks: {len(chunks)}")
    
    # Check which records have chunks
    chunk_record_ids = set(c.get('doc_id') for c in chunks)
    print(f"Unique record IDs in chunks: {len(chunk_record_ids)}")
    
    for record_id in ['finqa_test_1630', 'finqa_test_1431', 'finqa_test_1212']:
        chunk_count = sum(1 for c in chunks if c.get('doc_id') == record_id)
        status = "✓" if chunk_count > 0 else "✗"
        print(f"  {status} {record_id}: {chunk_count} chunks")
    
    if len(chunk_record_ids) == 3:
        print("\n[SUCCESS] All 3 records have chunks in the output!")
        print("The batch aggregation fix is working correctly.")
    else:
        print(f"\n[FAILURE] Only {len(chunk_record_ids)} records have chunks.")
        print("The batch aggregation issue persists.")

if __name__ == "__main__":
    main()