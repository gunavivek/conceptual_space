#!/usr/bin/env python3
"""
R-Pipeline Orchestrator
Orchestrates the execution of the R-series reference pipeline for concept validation
and alignment with business knowledge standards
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

def run_script(script_path, script_name):
    """
    Run individual R-pipeline script
    
    Args:
        script_path: Path to script file
        script_name: Name of script for logging
        
    Returns:
        tuple: (success, output, error)
    """
    print(f"\n{'='*60}")
    print(f"Executing {script_name}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, timeout=300)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")
        
        success = result.returncode == 0
        
        if success:
            print(f"✓ {script_name} completed successfully")
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
        
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"✗ {script_name} timed out after 5 minutes")
        return False, "", "Script execution timed out"
    
    except Exception as e:
        print(f"✗ {script_name} failed with error: {str(e)}")
        return False, "", str(e)

def run_r_pipeline():
    """
    Execute complete R-pipeline
    
    Returns:
        dict: Execution results and summary
    """
    script_dir = Path(__file__).parent
    
    # R-pipeline script sequence
    r_scripts = [
        ("R1_bizbok_concept_loader.py", "R1: BizBOK Concept Loader"),
        ("R2_concept_validation.py", "R2: Concept Validation"),
        ("R3_reference_alignment.py", "R3: Reference Alignment")
    ]
    
    execution_results = {
        "pipeline": "R-Series Reference Pipeline",
        "start_time": datetime.now().isoformat(),
        "scripts_executed": [],
        "success_count": 0,
        "failure_count": 0,
        "total_scripts": len(r_scripts),
        "overall_success": False
    }
    
    print("🔍 Starting R-Series Reference Pipeline Execution")
    print("=" * 60)
    print("Pipeline: Reference Data Management and Concept Validation")
    print(f"Total Scripts: {len(r_scripts)}")
    print(f"Start Time: {execution_results['start_time']}")
    
    # Execute scripts in sequence
    for script_file, script_name in r_scripts:
        script_path = script_dir / script_file
        
        if not script_path.exists():
            print(f"✗ Script not found: {script_path}")
            execution_results["scripts_executed"].append({
                "script": script_name,
                "file": script_file,
                "success": False,
                "error": "Script file not found"
            })
            execution_results["failure_count"] += 1
            continue
        
        # Run script
        success, output, error = run_script(script_path, script_name)
        
        # Record results
        execution_results["scripts_executed"].append({
            "script": script_name,
            "file": script_file,
            "success": success,
            "output_length": len(output) if output else 0,
            "has_errors": bool(error)
        })
        
        if success:
            execution_results["success_count"] += 1
        else:
            execution_results["failure_count"] += 1
            
            # For R-pipeline, continue execution even if individual scripts fail
            # as each script provides independent validation insights
            print(f"⚠️  Continuing pipeline despite {script_name} failure...")
    
    # Calculate overall success
    execution_results["overall_success"] = execution_results["failure_count"] == 0
    execution_results["end_time"] = datetime.now().isoformat()
    
    return execution_results

def display_pipeline_summary(results):
    """
    Display pipeline execution summary
    
    Args:
        results: Execution results dictionary
    """
    print("\n" + "="*60)
    print("R-PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    print(f"Pipeline: {results['pipeline']}")
    print(f"Total Scripts: {results['total_scripts']}")
    print(f"Successful: {results['success_count']}")
    print(f"Failed: {results['failure_count']}")
    print(f"Success Rate: {(results['success_count']/results['total_scripts'])*100:.1f}%")
    print(f"Overall Status: {'✓ SUCCESS' if results['overall_success'] else '✗ PARTIAL SUCCESS'}")
    
    print(f"\nScript Execution Details:")
    for script_result in results["scripts_executed"]:
        status = "✓" if script_result["success"] else "✗"
        print(f"  {status} {script_result['script']}")
        if not script_result["success"] and "error" in script_result:
            print(f"    Error: {script_result['error']}")
    
    print(f"\nExecution Time: {results['start_time']} to {results.get('end_time', 'N/A')}")
    
    # R-pipeline specific summary
    print(f"\nR-Pipeline Outputs:")
    output_dir = Path(__file__).parent.parent / "output"
    if output_dir.exists():
        output_files = list(output_dir.glob("R*.json"))
        for output_file in sorted(output_files):
            print(f"  📄 {output_file.name}")
    
    if results["overall_success"]:
        print(f"\n🎉 R-Pipeline completed successfully!")
        print("   Reference concepts loaded, validation performed, and alignment created.")
    else:
        print(f"\n⚠️  R-Pipeline completed with some failures.")
        print("   Review failed scripts and check individual outputs.")

def save_execution_log(results):
    """
    Save execution log for monitoring and debugging
    
    Args:
        results: Execution results
    """
    log_dir = Path(__file__).parent.parent / "output"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"r_pipeline_execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 Execution log saved: {log_file}")

def main():
    """Main orchestrator execution"""
    try:
        # Run R-pipeline
        results = run_r_pipeline()
        
        # Display summary
        display_pipeline_summary(results)
        
        # Save execution log
        save_execution_log(results)
        
        # Exit with appropriate code
        sys.exit(0 if results["overall_success"] else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline execution interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ Pipeline orchestration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()