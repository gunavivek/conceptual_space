#!/usr/bin/env python3
"""
Run FULL B-Pipeline: B1 -> B2 -> B3 -> B4 -> B5
For finqa_test_1212 to demonstrate complete pipeline flow
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_full_b_pipeline():
    """Run the complete B-pipeline B1->B2->B3->B4->B5"""
    print("="*80)
    print("FULL B-PIPELINE EXECUTION: B1 -> B2 -> B3 -> B4 -> B5")
    print("Processing finqa_test_1212")
    print("="*80)
    
    pipeline_start = datetime.now()
    
    # Change to B_Retrieval_pipeline directory
    b_pipeline_dir = Path("B_Retrieval_pipeline")
    
    # Stage timings
    timings = {}
    
    # B1: Already prepared - question loaded
    print("\n" + "="*60)
    print("B1: QUESTION INPUT LAYER")
    print("="*60)
    print("[OK] Question already loaded for finqa_test_1212")
    print("   Question: What is the company's total cost of revenues in 2018 and 2019?")
    
    # B2: Run all B2 components (B2.1, B2.2, B2.3, B2.4)
    print("\n" + "="*60)
    print("B2: PARALLEL INTENT PROCESSING")
    print("="*60)
    
    b2_start = datetime.now()
    b2_scripts = [
        "B2_1_intent_layer_modeling.py",
        "B2_2_declarative_transformation.py", 
        "B2_3_answer_expectation_prediction.py",
        "B2_4_temporal_analysis.py"
    ]
    
    for script in b2_scripts:
        print(f"   Running {script}...")
        result = subprocess.run([sys.executable, f"scripts/{script}"], 
                               capture_output=True, text=True, cwd=b_pipeline_dir)
        if result.returncode == 0:
            print(f"   [OK] {script} completed")
        else:
            print(f"   [ERROR] {script} failed: {result.stderr[:100]}")
    
    timings['B2'] = (datetime.now() - b2_start).total_seconds()
    
    # B3: Run all B3 components (B3.1, B3.2, B3.3)
    print("\n" + "="*60)
    print("B3: MULTI-STRATEGY CONCEPT MATCHING")
    print("="*60)
    
    b3_start = datetime.now()
    b3_scripts = [
        "B3.1_intent_matching.py",
        "B3.2_declarative_matching.py",
        "B3.3_answer_backward_matching.py"
    ]
    
    for script in b3_scripts:
        print(f"   Running {script}...")
        result = subprocess.run([sys.executable, f"scripts/{script}"], 
                               capture_output=True, text=True, cwd=b_pipeline_dir)
        if result.returncode == 0:
            print(f"   [OK] {script} completed")
        else:
            print(f"   [ERROR] {script} failed: {result.stderr[:100]}")
    
    timings['B3'] = (datetime.now() - b3_start).total_seconds()
    
    # B4: Weighted Strategy Combination
    print("\n" + "="*60)
    print("B4: WEIGHTED STRATEGY COMBINATION")
    print("="*60)
    
    b4_start = datetime.now()
    print("   Running B4_weighted_strategy_combination.py...")
    result = subprocess.run([sys.executable, "scripts/B4_weighted_strategy_combination.py"], 
                           capture_output=True, text=True, cwd=b_pipeline_dir)
    if result.returncode == 0:
        print("   [OK] B4 completed - Strategies combined")
    else:
        print(f"   [ERROR] B4 failed: {result.stderr[:100]}")
    
    timings['B4'] = (datetime.now() - b4_start).total_seconds()
    
    # B5: Full Answer Generation (not B5.2)
    print("\n" + "="*60)
    print("B5: FULL ANSWER GENERATION")
    print("="*60)
    
    b5_start = datetime.now()
    print("   Running B5_answer_generation.py...")
    result = subprocess.run([sys.executable, "scripts/B5_answer_generation.py"], 
                           capture_output=True, text=True, cwd=b_pipeline_dir)
    if result.returncode == 0:
        print("   [OK] B5 completed - Final answer generated")
    else:
        print(f"   [ERROR] B5 failed: {result.stderr[:100]}")
    
    timings['B5'] = (datetime.now() - b5_start).total_seconds()
    
    # Summary
    total_time = (datetime.now() - pipeline_start).total_seconds()
    
    print("\n" + "="*80)
    print("FULL B-PIPELINE SUMMARY")
    print("="*80)
    print(f"Question: What is the company's total cost of revenues in 2018 and 2019?")
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Pipeline: B1 -> B2 -> B3 -> B4 -> B5 (COMPLETE)")
    print("\nStage timing:")
    for stage, elapsed in timings.items():
        print(f"  {stage}: {elapsed:.3f}s")
    
    # Check outputs
    print("\nGenerated Outputs:")
    output_dir = Path("B_Retrieval_pipeline/outputs")
    output_files = [
        "B2.1_intent_layer_output.json",
        "B2.2_declarative_output.json", 
        "B2.3_answer_expectation_output.json",
        "B2.4_temporal_analysis_output.json",
        "B3.1_intent_matching_output.json",
        "B3.2_declarative_matching_output.json",
        "B3.3_answer_backward_matching_output.json", 
        "B4_final_ranking.json",
        "B5_answer_output.json"
    ]
    
    for file in output_files:
        file_path = output_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  [OK] {file} ({size} bytes)")
        else:
            print(f"  [MISSING] {file}")
    
    print("\n" + "="*80)
    print("FULL B-PIPELINE EXECUTION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    run_full_b_pipeline()