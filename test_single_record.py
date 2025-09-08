#!/usr/bin/env python3
"""
Test Single Record Processing
"""

import pandas as pd
import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime

def test_single_record():
    """Test processing a single record through A and B pipelines"""
    
    # Load one record from sample data
    sample_file = Path("sample_20_records.parquet")
    df = pd.read_parquet(sample_file)
    record = df.iloc[0].to_dict()
    record_id = record['id']
    
    print(f"Testing record: {record_id}")
    
    # Create individual record file for A-pipeline
    a_pipeline_dir = Path("A_Concept_pipeline")
    a_record_file = a_pipeline_dir / "data" / f"{record_id}.parquet"
    record_df = pd.DataFrame([record])
    record_df.to_parquet(a_record_file, index=False)
    print(f"Created A-pipeline record file: {a_record_file}")
    
    # Test A1.1 
    base_dir = Path.cwd()
    a11_script = base_dir / "A_Concept_pipeline" / "scripts" / "A1.1_document_reader.py"
    data_file_path = f"data/{record_id}.parquet"
    cmd = [sys.executable, str(a11_script), data_file_path]
    print(f"Running A1.1: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(base_dir / "A_Concept_pipeline"))
    
    if result.returncode == 0:
        print("A1.1 SUCCESS!")
        print(f"A1.1 Output: {result.stdout}")
    else:
        print("A1.1 FAILED!")
        print(f"A1.1 Error: {result.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_single_record()
    if success:
        print("Single record test completed successfully!")
    else:
        print("Single record test failed!")