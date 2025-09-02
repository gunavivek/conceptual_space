#!/usr/bin/env python3
"""
A-Pipeline Orchestrator - Run all A-Pipeline scripts in sequence
"""

import subprocess
import sys
import time
from pathlib import Path

def run_script(script_name, description):
    """Run a single script and capture its output"""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Description: {description}")
    print('='*60)
    
    script_path = Path(__file__).parent / "scripts" / script_name
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"[ERROR] Script failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            return False
        
        print(f"[OK] {script_name} completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Script timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run script: {e}")
        return False

def main():
    print("="*80)
    print("A-PIPELINE ORCHESTRATOR")
    print("Document Processing and Concept Extraction Pipeline")
    print("="*80)
    
    # Define the pipeline scripts in order
    pipeline_scripts = [
        ("A1.1_document_reader.py", "Read raw documents from data source"),
        ("A1.2_document_domain_detector.py", "Detect document domains"),
        ("A2.1_preprocess_document_analysis.py", "Preprocess and analyze documents"),
        ("A2.2_keyword_phrase_extraction.py", "Extract keywords and phrases"),
        ("A2.3_concept_grouping_thematic.py", "Group concepts thematically"),
        ("A2.4_synthesize_core_concepts.py", "Synthesize core concepts"),
        ("A2.5_expanded_concepts_orchestrator.py", "Expand concepts using multiple methods"),
        ("A2.59_review_expanded_concepts.py", "Review expanded concepts"),
        ("A2.9_r4x_semantic_enhancement.py", "Enhance semantics with I1 integration"),
        ("A3_centroid_validation_optimizer.py", "Validate and optimize concept centroids")
    ]
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for script_name, description in pipeline_scripts:
        if run_script(script_name, description):
            successful += 1
        else:
            failed += 1
            print(f"[WARNING] Continuing despite failure in {script_name}")
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("A-PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Total Scripts: {len(pipeline_scripts)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    
    if failed == 0:
        print("\n[SUCCESS] A-Pipeline completed successfully!")
        print("All document concepts have been extracted and enhanced.")
    else:
        print(f"\n[WARNING] A-Pipeline completed with {failed} failures.")
        print("Check individual script outputs for details.")
    
    print("\nOutput files saved to: A_Concept_pipeline/outputs/")
    print("="*80)

if __name__ == "__main__":
    main()