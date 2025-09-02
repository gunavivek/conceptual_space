#!/usr/bin/env python3
"""
I-InterSpace Pipeline Orchestrator
Orchestrates the execution of the I-series Inter-Space Integration Pipeline
for cross-conceptual-space semantic integration
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
import time

class IInterSpacePipelineOrchestrator:
    """Main orchestrator for I-InterSpace Pipeline execution"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.output_dir = self.script_dir.parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.execution_results = {
            "pipeline": "I-Series Inter-Space Integration Pipeline",
            "start_time": None,
            "end_time": None,
            "scripts_executed": [],
            "success_count": 0,
            "failure_count": 0,
            "total_scripts": 3,
            "overall_success": False,
            "performance_summary": {}
        }
    
    def run_script(self, script_path, script_name, timeout_minutes=10):
        """Run individual I-pipeline script with error handling"""
        print(f"\n{'='*60}")
        print(f"Executing {script_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, timeout=timeout_minutes*60)
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"SUCCESS: {script_name} completed successfully!")
                print(f"Execution time: {execution_time:.2f} seconds")
                return {
                    "success": True,
                    "execution_time": execution_time,
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_code": result.returncode
                }
            else:
                print(f"ERROR: {script_name} failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                return {
                    "success": False,
                    "execution_time": execution_time,
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"ERROR: {script_name} timed out after {timeout_minutes} minutes")
            return {
                "success": False,
                "execution_time": execution_time,
                "output": "",
                "error": f"Script execution timed out after {timeout_minutes} minutes",
                "return_code": -1
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"ERROR: {script_name} failed with error: {str(e)}")
            return {
                "success": False,
                "execution_time": execution_time,
                "output": "",
                "error": str(e),
                "return_code": -2
            }
    
    def execute_i_pipeline(self):
        """Execute complete I-InterSpace pipeline in sequence"""
        print("Starting I-Series Inter-Space Integration Pipeline")
        print("="*60)
        print("Mission: Create semantic bridges between R-A-B conceptual spaces")
        print("="*60)
        
        self.execution_results["start_time"] = datetime.now().isoformat()
        
        # I-pipeline script sequence (proper execution order)
        i_scripts = [
            ("I1_cross_pipeline_semantic_integrator.py", "I1: Cross-Pipeline Semantic Integrator", 10),
            ("I2_system_validation.py", "I2: System Validation", 5),
            ("I3_tri_semantic_visualizer.py", "I3: Tri-Semantic Visualizer", 8)
        ]
        
        print(f"Pipeline Overview:")
        for script_file, script_name, timeout in i_scripts:
            print(f"   - {script_name}")
        
        # Execute scripts in sequence
        for script_file, script_name, timeout_minutes in i_scripts:
            script_path = self.script_dir / script_file
            
            if not script_path.exists():
                print(f"WARNING: {script_file} not found, skipping...")
                self.execution_results["scripts_executed"].append({
                    "script": script_name,
                    "success": False,
                    "error": "Script file not found",
                    "execution_time": 0
                })
                self.execution_results["failure_count"] += 1
                continue
            
            # Execute script
            result = self.run_script(script_path, script_name, timeout_minutes)
            
            # Record results
            script_result = {
                "script": script_name,
                "success": result["success"],
                "execution_time": result["execution_time"],
                "return_code": result["return_code"]
            }
            
            if not result["success"]:
                script_result["error"] = result["error"]
                self.execution_results["failure_count"] += 1
            else:
                self.execution_results["success_count"] += 1
            
            self.execution_results["scripts_executed"].append(script_result)
        
        # Calculate overall success
        self.execution_results["overall_success"] = (
            self.execution_results["failure_count"] == 0 and
            self.execution_results["success_count"] > 0
        )
        
        self.execution_results["end_time"] = datetime.now().isoformat()
        
        return self.execution_results
    
    def display_pipeline_summary(self):
        """Display comprehensive pipeline execution summary"""
        print("\n" + "="*80)
        print("I-INTERSPACE PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        print(f"Pipeline: {self.execution_results['pipeline']}")
        print(f"Start Time: {self.execution_results['start_time']}")
        print(f"End Time: {self.execution_results['end_time']}")
        
        print(f"\nExecution Results:")
        print(f"  [OK] Successful: {self.execution_results['success_count']}")
        print(f"  [FAIL] Failed: {self.execution_results['failure_count']}")
        print(f"  [TOTAL] Total Scripts: {self.execution_results['total_scripts']}")
        
        if self.execution_results["overall_success"]:
            print(f"\n[SUCCESS] OVERALL STATUS: SUCCESS")
            print("Inter-space semantic integration pipeline completed successfully!")
        else:
            print(f"\n[WARNING] OVERALL STATUS: PARTIAL SUCCESS OR FAILURE")
            print("Some components failed. Check individual script results.")
        
        print(f"\nScript Details:")
        for script in self.execution_results["scripts_executed"]:
            status = "[OK]" if script["success"] else "[FAIL]"
            print(f"  {status} {script['script']}: {script['execution_time']:.2f}s")
        
        print("="*80)
    
    def save_execution_log(self):
        """Save detailed execution log"""
        log_path = self.output_dir / "I_pipeline_execution_log.json"
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.execution_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nExecution log saved: {log_path}")

def main():
    """Main orchestrator execution"""
    try:
        # Initialize orchestrator
        orchestrator = IInterSpacePipelineOrchestrator()
        
        # Execute I-pipeline
        execution_results = orchestrator.execute_i_pipeline()
        
        # Display summary
        orchestrator.display_pipeline_summary()
        
        # Save execution log
        orchestrator.save_execution_log()
        
        # Exit with appropriate code
        exit_code = 0 if execution_results["overall_success"] else 1
        
        print(f"\nI-InterSpace Pipeline completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\nPipeline execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nCritical error in I-Pipeline orchestrator: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    main()