#!/usr/bin/env python3
"""
Test 3 Records Processing
"""

import pandas as pd
import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime

class Test3RecordsController:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.a_pipeline_dir = self.base_dir / "A_Concept_pipeline"
        self.b_pipeline_dir = self.base_dir / "B_Retrieval_pipeline"
        self.sample_file = self.base_dir / "sample_20_records.parquet"
        self.batch_results = []
        
    def load_sample_records(self, count=3):
        """Load first N sample records"""
        df = pd.read_parquet(self.sample_file)
        print(f"Loading {count} records from: {self.sample_file}")
        return df.head(count).to_dict('records')
    
    def create_individual_record_file(self, record_data, record_id):
        """Create individual record parquet file from batch data"""
        record_df = pd.DataFrame([record_data])
        record_file = self.a_pipeline_dir / "data" / f"{record_id}.parquet"
        record_df.to_parquet(record_file, index=False)
        print(f"Created individual record file: {record_file}")
        return record_file
    
    def run_a_pipeline(self, record_data, record_id):
        """Run A-pipeline for a specific record"""
        print(f"\n=== Running A-pipeline for record: {record_id} ===")
        
        # Create individual record file
        record_file = self.create_individual_record_file(record_data, record_id)
        
        # Run A1.1 only for now to test
        a11_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A1.1_document_reader.py"
        data_file_path = f"data/{record_id}.parquet"
        cmd = [sys.executable, str(a11_script), data_file_path]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"A1.1 failed for {record_id}: {result.stderr}")
            return False
            
        print(f"A1.1 completed successfully for {record_id}")
        return True
    
    def test_records(self):
        """Test processing 3 records"""
        print("=== TESTING 3 RECORDS PIPELINE ===")
        
        # Load sample records
        records = self.load_sample_records(3)
        
        # Process each record
        for i, record in enumerate(records, 1):
            record_id = record['id']
            print(f"\n{'='*60}")
            print(f"PROCESSING RECORD {i}/3: {record_id}")
            print(f"{'='*60}")
            
            # Run A-pipeline
            if not self.run_a_pipeline(record, record_id):
                print(f"A-pipeline failed for {record_id}")
                continue
                
            print(f"Successfully processed {record_id} ({i}/3)")
        
        print(f"\n=== TEST COMPLETED ===")
        return True

def main():
    controller = Test3RecordsController()
    success = controller.test_records()
    
    if success:
        print("3-record test completed!")
    else:
        print("3-record test failed!")

if __name__ == "__main__":
    main()