#!/usr/bin/env python3
"""
R-Pipeline Orchestrator
Orchestrates the execution of the R-series Resource & Reasoning Pipeline
for BIZBOK concept processing, validation, alignment, and semantic ontology building
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
import time

class RPipelineOrchestrator:
    """Main orchestrator for R-Pipeline execution"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.output_dir = self.script_dir.parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.execution_results = {
            "pipeline": "R-Series Resource & Reasoning Pipeline",
            "start_time": None,
            "end_time": None,
            "scripts_executed": [],
            "success_count": 0,
            "failure_count": 0,
            "total_scripts": 4,
            "overall_success": False,
            "performance_summary": {}
        }
    
    def run_script(self, script_path, script_name, timeout_minutes=5):
        """Run individual R-pipeline script with error handling"""
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
            
            # Print output
            if result.stdout:
                print(result.stdout)
            
            if result.stderr:
                print(f"Warnings/Errors: {result.stderr}")
            
            success = result.returncode == 0
            
            if success:
                print(f"SUCCESS: {script_name} completed successfully in {execution_time:.1f}s")
            else:
                print(f"FAILED: {script_name} failed with return code {result.returncode}")
            
            return {
                "success": success,
                "execution_time": execution_time,
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"TIMEOUT: {script_name} timed out after {timeout_minutes} minutes")
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
    
    def execute_r_pipeline(self):
        """Execute complete R-pipeline in sequence"""
        print("Starting R-Series Resource & Reasoning Pipeline")
        print("="*60)
        print("Mission: Build semantic BIZBOK ontology for enhanced concept reasoning")
        print("="*60)
        
        self.execution_results["start_time"] = datetime.now().isoformat()
        
        # R-pipeline script sequence with updated names
        r_scripts = [
            ("R1_bizbok_resource_loader.py", "R1: BIZBOK Resource Loader", 2),
            ("R2_concept_validator.py", "R2: Concept Validator", 3),
            ("R3_reference_alignment.py", "R3: Reference Alignment", 2),
            ("R4L_lexical_ontology_builder.py", "R4L: Lexical Ontology Builder", 5)
            # Optional: Add R4X for tri-semantic integration
            # ("R4X_cross_pipeline_semantic_integrator.py", "R4X: Cross-Pipeline Semantic Integrator", 5)
        ]
        
        print(f"Pipeline Overview:")
        for script_file, script_name, timeout in r_scripts:
            print(f"   - {script_name}")
        
        # Execute scripts in sequence
        for script_file, script_name, timeout_minutes in r_scripts:
            script_path = self.script_dir / script_file
            
            if not script_path.exists():
                print(f"ERROR: Script not found: {script_path}")
                result = {
                    "success": False,
                    "execution_time": 0,
                    "output": "",
                    "error": "Script file not found",
                    "return_code": -3
                }
                self.execution_results["failure_count"] += 1
            else:
                # Run script
                result = self.run_script(script_path, script_name, timeout_minutes)
                
                if result["success"]:
                    self.execution_results["success_count"] += 1
                else:
                    self.execution_results["failure_count"] += 1
            
            # Record results
            self.execution_results["scripts_executed"].append({
                "script": script_name,
                "file": script_file,
                "success": result["success"],
                "execution_time": result["execution_time"],
                "output_length": len(result["output"]) if result["output"] else 0,
                "has_errors": bool(result["error"]),
                "return_code": result["return_code"]
            })
            
            # For R-pipeline, continue execution even if individual scripts fail
            # as each provides independent insights
            if not result["success"]:
                print(f"WARNING: Continuing pipeline despite {script_name} failure...")
                print(f"   Failure reason: {result['error']}")
        
        # Calculate overall success
        self.execution_results["overall_success"] = self.execution_results["failure_count"] == 0
        self.execution_results["end_time"] = datetime.now().isoformat()
        
        return self.execution_results
    
    def analyze_outputs(self):
        """Analyze generated outputs for quality assessment"""
        print("\n[ANALYSIS] Analyzing R-Pipeline Outputs...")
        
        output_analysis = {
            "files_created": [],
            "file_sizes": {},
            "data_quality": {}
        }
        
        # Check for expected output files
        expected_files = [
            "R1_CONCEPTS.json",
            "R1_DOMAINS.json", 
            "R1_KEYWORDS.json",
            "R2_validation_report.json",
            "R3_alignment_mappings.json",
            "R4L_lexical_ontology.json",
            "R4L_integration_api.json"
        ]
        
        for filename in expected_files:
            file_path = self.output_dir / filename
            if file_path.exists():
                file_size = file_path.stat().st_size
                output_analysis["files_created"].append(filename)
                output_analysis["file_sizes"][filename] = file_size
                
                # Quick quality check
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if filename == "R1_CONCEPTS.json":
                        concept_count = len(data.get("concepts", {}))
                        output_analysis["data_quality"]["concept_count"] = concept_count
                        
                    elif filename == "R4L_lexical_ontology.json":
                        ontology = data.get("ontology", {})
                        cluster_count = len(ontology.get("clusters", {}))
                        relationship_count = ontology.get("statistics", {}).get("relationships_total", 0)
                        output_analysis["data_quality"]["cluster_count"] = cluster_count
                        output_analysis["data_quality"]["relationship_count"] = relationship_count
                        
                except Exception as e:
                    output_analysis["data_quality"][filename] = f"Error reading file: {str(e)}"
        
        print(f"   Created {len(output_analysis['files_created'])} output files")
        
        if "concept_count" in output_analysis["data_quality"]:
            print(f"   Processed {output_analysis['data_quality']['concept_count']} BIZBOK concepts")
            
        if "cluster_count" in output_analysis["data_quality"]:
            print(f"   Built {output_analysis['data_quality']['cluster_count']} semantic clusters")
            print(f"   Extracted {output_analysis['data_quality']['relationship_count']} semantic relationships")
        
        return output_analysis
    
    def display_pipeline_summary(self, output_analysis):
        """Display comprehensive pipeline execution summary"""
        print("\n" + "="*60)
        print("R-PIPELINE EXECUTION SUMMARY")
        print("="*60)
        
        print(f"Pipeline: {self.execution_results['pipeline']}")
        print(f"Total Scripts: {self.execution_results['total_scripts']}")
        print(f"Successful: {self.execution_results['success_count']}")
        print(f"Failed: {self.execution_results['failure_count']}")
        
        success_rate = (self.execution_results['success_count'] / self.execution_results['total_scripts']) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.execution_results["overall_success"]:
            print(f"Overall Status: COMPLETE SUCCESS")
        else:
            print(f"Overall Status: PARTIAL SUCCESS")
        
        print(f"\nScript Execution Details:")
        total_time = 0
        for script_result in self.execution_results["scripts_executed"]:
            status = "SUCCESS" if script_result["success"] else "FAILED"
            exec_time = script_result["execution_time"]
            total_time += exec_time
            print(f"  {status} {script_result['script']} ({exec_time:.1f}s)")
            
            if not script_result["success"]:
                print(f"      FAILED - Return code: {script_result['return_code']}")
        
        print(f"\nTiming Summary:")
        print(f"   Total Execution Time: {total_time:.1f} seconds")
        print(f"   Start: {self.execution_results['start_time']}")
        print(f"   End: {self.execution_results['end_time']}")
        
        # R-pipeline specific summary
        print(f"\nGenerated Outputs:")
        for filename in output_analysis["files_created"]:
            file_size_kb = output_analysis["file_sizes"][filename] / 1024
            print(f"   {filename} ({file_size_kb:.1f} KB)")
        
        # Quality metrics
        if output_analysis["data_quality"]:
            print(f"\nData Quality Metrics:")
            for metric, value in output_analysis["data_quality"].items():
                if isinstance(value, int):
                    print(f"   {metric.replace('_', ' ').title()}: {value:,}")
        
        # Success message
        if self.execution_results["overall_success"]:
            print(f"\nSUCCESS: R-Pipeline completed successfully!")
            print("   BIZBOK semantic ontology created with rich relationships")
            print("   Integration API ready for A/B pipeline enhancement")
            print("   Ready for advanced concept reasoning and expansion")
        else:
            print(f"\nWARNING: R-Pipeline completed with some failures.")
            print("   Review failed scripts and check individual outputs")
            print("   Partial results may still be usable for integration")
    
    def save_execution_log(self, output_analysis):
        """Save comprehensive execution log"""
        log_data = {
            "execution_results": self.execution_results,
            "output_analysis": output_analysis,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(Path.cwd())
            }
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"r_pipeline_execution_log_{timestamp}.json"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nExecution log saved: {log_file.name}")

def main():
    """Main orchestrator execution"""
    try:
        # Initialize orchestrator
        orchestrator = RPipelineOrchestrator()
        
        # Execute R-pipeline
        execution_results = orchestrator.execute_r_pipeline()
        
        # Analyze outputs
        output_analysis = orchestrator.analyze_outputs()
        
        # Display summary
        orchestrator.display_pipeline_summary(output_analysis)
        
        # Save execution log
        orchestrator.save_execution_log(output_analysis)
        
        # Exit with appropriate code
        exit_code = 0 if execution_results["overall_success"] else 1
        
        print(f"\nR-Pipeline orchestrator finished (exit code: {exit_code})")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\nWARNING: R-Pipeline execution interrupted by user")
        print("   Graceful shutdown initiated")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\nERROR: R-Pipeline orchestration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()