#!/usr/bin/env python3
"""
Test single record with fixed pipeline
"""

import subprocess
import sys
from pathlib import Path

def test_single_record(record_id):
    """Test a single record through the fixed pipeline"""
    base_dir = Path.cwd()
    
    print(f"Testing record: {record_id}")
    print("="*60)
    
    # Run B1 with specific record
    b1_script = base_dir / "B_Retrieval_pipeline" / "scripts" / "B1_read_question.py"
    cmd = [sys.executable, str(b1_script), record_id]
    print(f"Running B1: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(base_dir / "B_Retrieval_pipeline"))
    
    if result.returncode != 0:
        print(f"B1 failed: {result.stderr}")
        return False
    
    print("B1 SUCCESS")
    
    # Run B2 orchestrator
    b2_orch_script = base_dir / "B_Retrieval_pipeline" / "scripts" / "B_pipeline_orchestrator.py"
    cmd = [sys.executable, str(b2_orch_script)]
    print(f"Running B2 orchestrator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(base_dir / "B_Retrieval_pipeline"))
    
    if result.returncode != 0:
        print(f"B2 orchestrator failed: {result.stderr}")
        return False
    
    print("B2 orchestrator SUCCESS")
    
    # Run B5.2
    b52_script = base_dir / "B_Retrieval_pipeline" / "scripts" / "B5.2_generate_answer.py"
    cmd = [sys.executable, str(b52_script), record_id]
    print(f"Running B5.2: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(base_dir / "B_Retrieval_pipeline"))
    
    if result.returncode != 0:
        print(f"B5.2 failed: {result.stderr}")
        return False
    
    print("B5.2 SUCCESS")
    
    # Check the output
    import json
    b52_output = base_dir / "B_Retrieval_pipeline" / "outputs" / "B5.2_direct_answer_output.json"
    with open(b52_output, 'r') as f:
        output = json.load(f)
        print("\nB5.2 Output:")
        print(f"Question: {output.get('question', 'N/A')}")
        print(f"Answer: {output.get('answer', 'N/A')}")
        print(f"Chunks evaluated: {output.get('chunks_evaluated', 0)}")
    
    return True

if __name__ == "__main__":
    # Test with finqa_test_1630
    success = test_single_record("finqa_test_1630")
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")