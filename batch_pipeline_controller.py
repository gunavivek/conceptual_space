#!/usr/bin/env python3
"""
Batch Pipeline Controller
Processes multiple records through complete A-pipeline and B-pipeline
Ensures proper record ID coupling and no data leakage
"""

import pandas as pd
import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime

class BatchPipelineController:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.a_pipeline_dir = self.base_dir / "A_Concept_pipeline"
        self.b_pipeline_dir = self.base_dir / "B_Retrieval_pipeline"
        self.sample_file = self.base_dir / "sample_20_records.parquet"
        self.b52_output_file = self.b_pipeline_dir / "outputs" / "B5.2_direct_answer_output.json"
        self.batch_results = []
        
    def load_sample_records(self):
        """Load the 20 sample records"""
        print(f"Loading sample records from: {self.sample_file}")
        df = pd.read_parquet(self.sample_file)
        print(f"Loaded {len(df)} records")
        return df.to_dict('records')
    
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
        
        # Run A1.1 - Document reader with absolute path
        a11_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A1.1_document_reader.py"
        data_file_path = f"data/{record_id}.parquet"
        cmd = [sys.executable, str(a11_script), data_file_path]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"A1.1 failed for {record_id}: {result.stderr}")
            return False
            
        # Run A2.2 - Enhanced keyword extraction
        a22_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A2.2_keyword_phrase_extraction.py"
        cmd = [sys.executable, str(a22_script)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"A2.2 failed for {record_id}: {result.stderr}")
            return False
            
        # Run A3 - Concept-based chunking
        a3_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A3_concept_based_chunking.py"
        cmd = [sys.executable, str(a3_script)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"A3 failed for {record_id}: {result.stderr}")
            return False
            
        print(f"A-pipeline completed successfully for {record_id}")
        return True
    
    def run_b_pipeline(self, record_data, record_id):
        """Run B-pipeline for a specific record"""
        print(f"\n=== Running B-pipeline for record: {record_id} ===")
        
        # Create individual record file in B-pipeline data directory if needed
        b_data_dir = self.b_pipeline_dir / "data"
        b_data_dir.mkdir(exist_ok=True)
        b_record_file = b_data_dir / f"{record_id}.parquet"
        if not b_record_file.exists():
            record_df = pd.DataFrame([record_data])
            record_df.to_parquet(b_record_file, index=False)
            print(f"Created B-pipeline record file: {b_record_file}")
        
        # Run B1 - Read question
        b1_script = self.base_dir / "B_Retrieval_pipeline" / "scripts" / "B1_read_question.py"
        cmd = [sys.executable, str(b1_script), record_id]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "B_Retrieval_pipeline"))
        
        if result.returncode != 0:
            print(f"B1 failed for {record_id}: {result.stderr}")
            return False
            
        # Run B2 orchestrator (includes conditional B2.4)
        b2_orch_script = self.base_dir / "B_Retrieval_pipeline" / "scripts" / "B_pipeline_orchestrator.py"
        cmd = [sys.executable, str(b2_orch_script)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "B_Retrieval_pipeline"))
        
        if result.returncode != 0:
            print(f"B2 orchestrator failed for {record_id}: {result.stderr}")
            return False
            
        # Run B5.2 - Answer generation with record ID filtering
        b52_script = self.base_dir / "B_Retrieval_pipeline" / "scripts" / "B5.2_generate_answer.py"
        cmd = [sys.executable, str(b52_script), record_id]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "B_Retrieval_pipeline"))
        
        if result.returncode != 0:
            print(f"B5.2 failed for {record_id}: {result.stderr}")
            return False
            
        print(f"B-pipeline completed successfully for {record_id}")
        return True
    
    def append_b52_result(self, record_id):
        """Read the latest B5.2 result and append to batch results"""
        try:
            if self.b52_output_file.exists():
                with open(self.b52_output_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    result['record_id'] = record_id
                    result['batch_processing_timestamp'] = datetime.now().isoformat()
                    self.batch_results.append(result)
                    print(f"Captured B5.2 result for {record_id}")
                    return True
            else:
                print(f"B5.2 output file not found for {record_id}")
                return False
        except Exception as e:
            print(f"Error reading B5.2 result for {record_id}: {e}")
            return False
    
    def save_batch_results(self):
        """Save all batch results to the B5.2 output file"""
        batch_output = {
            "batch_processing": {
                "total_records": len(self.batch_results),
                "processing_timestamp": datetime.now().isoformat(),
                "pipeline_version": "A-B_Full_Pipeline_v2.0"
            },
            "results": self.batch_results
        }
        
        with open(self.b52_output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_output, f, indent=2, ensure_ascii=False)
        
        print(f"\nBatch results saved to: {self.b52_output_file}")
        print(f"Total records processed: {len(self.batch_results)}")
    
    def process_all_records(self):
        """Main processing function - runs A and B pipelines for all records"""
        print("=== BATCH PIPELINE CONTROLLER STARTING ===")
        
        # Load sample records
        records = self.load_sample_records()
        
        # Initialize batch results file (clear previous results)
        self.batch_results = []
        
        # Process each record
        for i, record in enumerate(records, 1):
            record_id = record['id']
            print(f"\n{'='*60}")
            print(f"PROCESSING RECORD {i}/20: {record_id}")
            print(f"{'='*60}")
            
            # Run A-pipeline
            if not self.run_a_pipeline(record, record_id):
                print(f"Skipping B-pipeline for {record_id} due to A-pipeline failure")
                continue
                
            # Run B-pipeline  
            if not self.run_b_pipeline(record, record_id):
                print(f"B-pipeline failed for {record_id}")
                continue
                
            # Capture B5.2 result
            if not self.append_b52_result(record_id):
                print(f"Failed to capture B5.2 result for {record_id}")
                continue
                
            print(f"Successfully processed {record_id} ({i}/20)")
        
        # Save all results
        self.save_batch_results()
        
        print(f"\n=== BATCH PROCESSING COMPLETED ===")
        print(f"Successfully processed: {len(self.batch_results)}/20 records")
        return len(self.batch_results)

def main():
    controller = BatchPipelineController()
    success_count = controller.process_all_records()
    
    if success_count == 20:
        print("All records processed successfully!")
    else:
        print(f"Warning: Only {success_count}/20 records processed successfully")

if __name__ == "__main__":
    main()