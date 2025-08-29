#!/usr/bin/env python3
"""
Main orchestrator for running the Conceptual Space Pipeline System
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_script(script_path, description):
    """
    Run a Python script and capture output
    
    Args:
        script_path: Path to the script
        description: Description of what the script does
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path.name}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"✗ {description} failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out")
        return False
    except Exception as e:
        print(f"✗ {description} error: {str(e)}")
        return False

def run_pipeline_a():
    """Run Pipeline A: Concept Building"""
    print("\n" + "="*60)
    print("PIPELINE A: CONCEPT BUILDING")
    print("="*60)
    
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / "A_concept_pipeline" / "scripts"
    
    # Define Pipeline A scripts in order
    scripts = [
        ("A1.1_document_reader.py", "Document Reader"),
        # Add more scripts as they're created:
        # ("A1.2_document_domain_detector.py", "Domain Detection"),
        # ("A2.1_preprocess_document_analysis.py", "Document Preprocessing"),
        # ("A2.2_keyword_phrase_extraction.py", "Keyword Extraction"),
        # ("A2.3_concept_grouping_thematic.py", "Concept Grouping"),
        # etc.
    ]
    
    success_count = 0
    for script_name, description in scripts:
        script_path = scripts_dir / script_name
        if script_path.exists():
            if run_script(script_path, description):
                success_count += 1
        else:
            print(f"⚠ Script not found: {script_name}")
    
    print(f"\nPipeline A completed: {success_count}/{len(scripts)} scripts successful")
    return success_count == len(scripts)

def run_pipeline_b():
    """Run Pipeline B: Retrieval & QA"""
    print("\n" + "="*60)
    print("PIPELINE B: RETRIEVAL & QA")
    print("="*60)
    
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / "B_retrieval_pipeline" / "scripts"
    
    # Define Pipeline B scripts in order
    scripts = [
        ("B1_read_question.py", "Read Question"),
        # Add more scripts as they're created:
        # ("B2_1_intent_layer_modeling.py", "Intent Layer Modeling"),
        # ("B2_2_declarative_transformation.py", "Declarative Transformation"),
        # ("B2_3_answer_expectation_prediction.py", "Answer Expectation"),
        # ("B3_1_intent_based_matching.py", "Intent-Based Matching"),
        # etc.
    ]
    
    success_count = 0
    for script_name, description in scripts:
        script_path = scripts_dir / script_name
        if script_path.exists():
            if run_script(script_path, description):
                success_count += 1
        else:
            print(f"⚠ Script not found: {script_name}")
    
    print(f"\nPipeline B completed: {success_count}/{len(scripts)} scripts successful")
    return success_count == len(scripts)

def main():
    """Main execution"""
    print("="*60)
    print("CONCEPTUAL SPACE PIPELINE SYSTEM")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'a':
            run_pipeline_a()
        elif sys.argv[1].lower() == 'b':
            run_pipeline_b()
        else:
            print("Usage: python run_pipeline.py [a|b|all]")
            print("  a   - Run only Pipeline A")
            print("  b   - Run only Pipeline B")
            print("  all - Run both pipelines (default)")
    else:
        # Run both pipelines by default
        pipeline_a_success = run_pipeline_a()
        pipeline_b_success = run_pipeline_b()
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE")
        print(f"Pipeline A: {'✓ Success' if pipeline_a_success else '✗ Failed'}")
        print(f"Pipeline B: {'✓ Success' if pipeline_b_success else '✗ Failed'}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

if __name__ == "__main__":
    main()