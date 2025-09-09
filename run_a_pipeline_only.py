#!/usr/bin/env python3
"""
Run COMPLETE A-Pipeline for 20 Sample Records
Processes documents through proper sequence: A1.1 → A1.2 → A2.1 → A2.2 → A2.3 → A2.4 → A2.5 → A3
Ensures all intermediate steps are executed to generate proper chunks with correct record IDs
"""

import pandas as pd
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

class APipelineProcessor:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.a_pipeline_dir = self.base_dir / "A_Concept_pipeline"
        self.sample_file = self.base_dir / "sample_20_records.parquet"
        self.processed_records = []
        
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
        print(f"  Created record file: {record_file.name}")
        return record_file
    
    def run_a_pipeline_for_record(self, record_data, record_id, is_first_record=False):
        """Run A-pipeline steps A1.1 -> A1.2 -> A2.1 -> A2.2 -> A2.3 -> A2.4 -> A2.5 for a specific record
        Note: A3 will be run once at the end for all records together"""
        print(f"\n{'='*60}")
        print(f"Processing A-Pipeline Steps 1-7 for: {record_id}")
        print(f"{'='*60}")
        
        # Create individual record file
        record_file = self.create_individual_record_file(record_data, record_id)
        
        # Step 1: A1.1 - Document reader (with append mode for non-first records)
        print("\n[Step 1/7] A1.1 - Document Reader")
        a11_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A1.1_document_reader.py"
        data_file_path = f"data/{record_id}.parquet"
        # Use append mode for all records except the first one
        if is_first_record:
            cmd = [sys.executable, str(a11_script), data_file_path]
        else:
            cmd = [sys.executable, str(a11_script), data_file_path, "--append"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"  [ERROR] A1.1 failed: {result.stderr[:200]}")
            return False
        mode_str = " (overwrite mode)" if is_first_record else " (append mode)"
        print(f"  [OK] A1.1 completed - Document loaded and processed{mode_str}")
            
        # Step 2: A1.2 - Domain concept enrichment
        print("\n[Step 2/7] A1.2 - Domain Concept Enrichment")
        a12_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A1.2_domain_concept_enrichment.py"
        cmd = [sys.executable, str(a12_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"  [ERROR] A1.2 failed: {result.stderr[:200]}")
            return False
        print(f"  [OK] A1.2 completed - Domain concepts enriched")
            
        # Step 3: A2.1 - Document preprocessing and analysis
        print("\n[Step 3/7] A2.1 - Document Preprocessing & Analysis")
        a21_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A2.1_preprocess_document_analysis.py"
        cmd = [sys.executable, str(a21_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"  [ERROR] A2.1 failed: {result.stderr[:200]}")
            return False
        print(f"  [OK] A2.1 completed - Documents preprocessed and analyzed")
            
        # Step 4: A2.2 - Keyword & phrase extraction
        print("\n[Step 4/7] A2.2 - Keyword & Phrase Extraction")
        a22_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A2.2_keyword_phrase_extraction.py"
        cmd = [sys.executable, str(a22_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"  [ERROR] A2.2 failed: {result.stderr[:200]}")
            return False
        print(f"  [OK] A2.2 completed - Keywords and phrases extracted")
            
        # Step 5: A2.3 - Concept grouping thematic
        print("\n[Step 5/7] A2.3 - Concept Grouping Thematic")
        a23_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A2.3_concept_grouping_thematic.py"
        cmd = [sys.executable, str(a23_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"  [ERROR] A2.3 failed: {result.stderr[:200]}")
            return False
        print(f"  [OK] A2.3 completed - Concepts grouped thematically")
            
        # Step 6: A2.4 - Synthesize core concepts
        print("\n[Step 6/7] A2.4 - Synthesize Core Concepts")
        a24_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A2.4_synthesize_core_concepts.py"
        cmd = [sys.executable, str(a24_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"  [ERROR] A2.4 failed: {result.stderr[:200]}")
            return False
        print(f"  [OK] A2.4 completed - Core concepts synthesized")
            
        # Step 7: A2.5 - Expanded concepts orchestrator
        print("\n[Step 7/7] A2.5 - Expanded Concepts Orchestrator")
        a25_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A2.5_expanded_concepts_orchestrator.py"
        cmd = [sys.executable, str(a25_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"  [ERROR] A2.5 failed: {result.stderr[:200]}")
            return False
        print(f"  [OK] A2.5 completed - Concepts expanded")
            
        # Note: A3 will be run once at the end for all records
        print(f"\n[SUCCESS] A-Pipeline steps 1-7 completed for {record_id}")
        return True
    
    def run_a3_batch_chunking(self):
        """Run A3 chunking once for all processed records"""
        print("\n" + "="*70)
        print("Running A3 - Multi-Strategy Chunking for ALL Records")
        print("="*70)
        
        a3_script = self.base_dir / "A_Concept_pipeline" / "scripts" / "A3_concept_based_chunking.py"
        cmd = [sys.executable, str(a3_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir / "A_Concept_pipeline"))
        
        if result.returncode != 0:
            print(f"  [ERROR] A3 failed: {result.stderr[:500]}")
            return False
        print(f"  [OK] A3 completed - Chunks generated for all records")
        return True
    
    def process_all_records(self):
        """Process all 20 records through A-pipeline"""
        print("\n" + "="*70)
        print("A-PIPELINE BATCH PROCESSOR - 20 RECORDS")
        print("="*70)
        
        # Load sample records
        records = self.load_sample_records()
        
        success_count = 0
        failed_records = []
        
        # Process each record (steps 1-7)
        for i, record in enumerate(records, 1):
            record_id = record['id']
            print(f"\n[{i}/20] Processing: {record_id}")
            
            is_first = (i == 1)  # First record will overwrite, others will append
            if self.run_a_pipeline_for_record(record, record_id, is_first_record=is_first):
                success_count += 1
                self.processed_records.append(record_id)
            else:
                failed_records.append(record_id)
                print(f"[WARNING] Failed to process {record_id}")
        
        # Summary of record processing
        print("\n" + "="*70)
        print("RECORD PROCESSING COMPLETE")
        print("="*70)
        print(f"[SUCCESS] Successfully processed: {success_count}/20 records")
        
        if failed_records:
            print(f"[ERROR] Failed records: {', '.join(failed_records)}")
            print("[WARNING] A3 will only process successful records")
        
        # Run A3 once for all records
        if success_count > 0:
            print("\n" + "="*70)
            print("FINAL STEP: A3 BATCH CHUNKING")
            print("="*70)
            if self.run_a3_batch_chunking():
                print("[SUCCESS] A3 batch chunking completed successfully")
            else:
                print("[ERROR] A3 batch chunking failed")
                success_count = 0  # Mark as failure if A3 fails
        
        # Check final chunk output
        self.verify_chunk_output()
        
        return success_count
    
    def verify_chunk_output(self):
        """Verify the chunks generated have correct record IDs"""
        print("\n" + "-"*70)
        print("VERIFYING CHUNK OUTPUT")
        print("-"*70)
        
        chunk_file = self.a_pipeline_dir / "outputs" / "A3_multi_strategy_chunks.json"
        
        if not chunk_file.exists():
            print("[ERROR] Chunk file not found!")
            return
        
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
        
        if 'chunks' not in chunk_data:
            print("[ERROR] No chunks found in output!")
            return
            
        chunks = chunk_data['chunks']
        print(f"Total chunks generated: {len(chunks)}")
        
        # Check record IDs in chunks
        chunk_record_ids = set()
        for chunk in chunks:
            doc_id = chunk.get('doc_id', '')
            chunk_record_ids.add(doc_id)
        
        print(f"\nUnique record IDs in chunks: {len(chunk_record_ids)}")
        print("Sample chunk record IDs:")
        for rid in list(chunk_record_ids)[:5]:
            chunk_count = sum(1 for c in chunks if c.get('doc_id') == rid)
            print(f"  - {rid}: {chunk_count} chunks")
        
        # Check if processed records have chunks
        print("\nVerifying processed records have chunks:")
        for record_id in self.processed_records[:5]:
            has_chunks = record_id in chunk_record_ids
            status = "[OK]" if has_chunks else "[FAIL]"
            print(f"  {status} {record_id}")
        
        if not chunk_record_ids.intersection(self.processed_records):
            print("\n[WARNING] Chunks don't match processed records!")
            print("   The A3 script may be using cached data.")

def main():
    processor = APipelineProcessor()
    success_count = processor.process_all_records()
    
    print("\n" + "="*70)
    print("A-PIPELINE BATCH PROCESSING COMPLETED")
    print("="*70)
    print(f"Results: {success_count}/20 records processed successfully")
    print("\nNext steps:")
    print("1. Review the chunk output in A_Concept_pipeline/outputs/")
    print("2. Verify chunks have correct record IDs")
    print("3. Then run B-pipeline processing")
    
    return success_count == 20

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)