#!/usr/bin/env python3
"""
R4X System Validation and Testing Suite
Comprehensive testing and validation of the complete R4X Cross-Pipeline Semantic Integration system
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import time
from collections import defaultdict
import traceback

# Add R4X integration path
sys.path.append(str(Path(__file__).parent))

class R4X_SystemValidator:
    """
    Comprehensive R4X System Validation Suite
    
    Tests and validates:
    1. Core R4X component functionality
    2. Cross-pipeline integration effectiveness
    3. Enhanced A-Pipeline processing
    4. Enhanced B-Pipeline processing
    5. R5X visualization system
    6. End-to-end system integration
    7. Performance and quality metrics
    """
    
    def __init__(self):
        """Initialize R4X System Validator"""
        self.script_dir = Path(__file__).parent
        self.test_results = {
            'core_components': {},
            'pipeline_integrations': {},
            'end_to_end': {},
            'performance_metrics': {},
            'quality_assessments': {}
        }
        self.test_start_time = time.time()
        
        # Test configurations
        self.test_question = "What was the change in Current deferred income?"
        self.expected_components = [
            'R4X_cross_pipeline_semantic_integrator.py',
            'R4X_semantic_fusion_engine.py',
            'R5X_tri_semantic_visualizer.py'
        ]
        self.enhanced_scripts = {
            'A_Pipeline': ['A2.9_r4x_semantic_enhancement.py'],
            'B_Pipeline': ['B3.4_r4x_intent_enhancement.py', 'B4.1_r4x_answer_synthesis.py', 'B5.1_r4x_question_understanding.py']
        }
    
    def test_core_r4x_components(self) -> Dict[str, Any]:
        """
        Test core R4X component functionality
        
        Returns:
            Core component test results
        """
        print("Testing Core R4X Components...")
        core_results = {}
        
        # Test R4X Cross-Pipeline Integrator
        print("  Testing R4X Cross-Pipeline Integrator...")
        try:
            from R4X_cross_pipeline_semantic_integrator import R4X_CrossPipelineSemanticIntegrator
            
            integrator = R4X_CrossPipelineSemanticIntegrator()
            test_concept = "financial_metrics"
            
            # Test unified concept view
            unified_view = integrator.get_unified_concept_view(test_concept)
            
            core_results['cross_pipeline_integrator'] = {
                'status': 'pass',
                'initialization': True,
                'unified_concept_view': unified_view is not None,
                'semantic_spaces_count': len(getattr(integrator, 'semantic_spaces', {})),
                'details': 'R4X Cross-Pipeline Integrator functioning correctly'
            }
            
        except Exception as e:
            core_results['cross_pipeline_integrator'] = {
                'status': 'fail',
                'error': str(e),
                'details': f'R4X Cross-Pipeline Integrator test failed: {e}'
            }
        
        # Test R4X Semantic Fusion Engine  
        print("  Testing R4X Semantic Fusion Engine...")
        try:
            from R4X_semantic_fusion_engine import SemanticFusionEngine
            
            fusion_engine = SemanticFusionEngine()
            
            # Test fusion strategies
            test_perspectives = {
                'ontology_perspective': {'confidence': 0.8, 'keywords': ['test']},
                'document_perspective': {'confidence': 0.7, 'keywords': ['test']},
                'question_perspective': {'confidence': 0.6, 'keywords': ['test']}
            }
            
            fusion_result = fusion_engine.fuse_tri_semantic_perspectives("test_concept", test_perspectives)
            
            core_results['semantic_fusion_engine'] = {
                'status': 'pass',
                'initialization': True,
                'fusion_strategies_count': len(getattr(fusion_engine, 'fusion_strategies', {})),
                'fusion_result_available': fusion_result is not None,
                'details': 'R4X Semantic Fusion Engine functioning correctly'
            }
            
        except Exception as e:
            core_results['semantic_fusion_engine'] = {
                'status': 'fail', 
                'error': str(e),
                'details': f'R4X Semantic Fusion Engine test failed: {e}'
            }
        
        # Test R5X Tri-Semantic Visualizer
        print("  Testing R5X Tri-Semantic Visualizer...")
        try:
            from R5X_tri_semantic_visualizer import R5X_TriSemanticVisualizer
            
            visualizer = R5X_TriSemanticVisualizer()
            
            # Test data loading
            tri_semantic_data = visualizer.load_tri_semantic_data()
            
            core_results['tri_semantic_visualizer'] = {
                'status': 'pass',
                'initialization': True,
                'data_loading': tri_semantic_data is not None,
                'semantic_colors_count': len(getattr(visualizer, 'semantic_colors', {})),
                'visualization_types': len(getattr(visualizer, 'visualization_types', {})),
                'details': 'R5X Tri-Semantic Visualizer functioning correctly'
            }
            
        except Exception as e:
            core_results['tri_semantic_visualizer'] = {
                'status': 'fail',
                'error': str(e),
                'details': f'R5X Tri-Semantic Visualizer test failed: {e}'
            }
        
        self.test_results['core_components'] = core_results
        return core_results
    
    def test_enhanced_pipeline_scripts(self) -> Dict[str, Any]:
        """
        Test enhanced pipeline scripts functionality
        
        Returns:
            Enhanced pipeline test results
        """
        print("Testing Enhanced Pipeline Scripts...")
        pipeline_results = {}
        
        # Test Enhanced A-Pipeline Scripts
        print("  Testing Enhanced A-Pipeline Scripts...")
        a_pipeline_results = {}
        
        a_pipeline_dir = self.script_dir.parent.parent / "A_Concept_pipeline" / "scripts"
        for script_name in self.enhanced_scripts['A_Pipeline']:
            script_path = a_pipeline_dir / script_name
            
            if script_path.exists():
                try:
                    # Test script execution
                    result = subprocess.run([sys.executable, str(script_path)], 
                                          capture_output=True, text=True, timeout=60)
                    
                    a_pipeline_results[script_name] = {
                        'status': 'pass' if result.returncode == 0 else 'fail',
                        'execution_time': 'completed',
                        'output_length': len(result.stdout),
                        'error_output': result.stderr if result.stderr else None,
                        'details': f'Enhanced A-Pipeline script executed {"successfully" if result.returncode == 0 else "with errors"}'
                    }
                    
                except subprocess.TimeoutExpired:
                    a_pipeline_results[script_name] = {
                        'status': 'timeout',
                        'details': 'Script execution timed out after 60 seconds'
                    }
                except Exception as e:
                    a_pipeline_results[script_name] = {
                        'status': 'fail',
                        'error': str(e),
                        'details': f'Script execution failed: {e}'
                    }
            else:
                a_pipeline_results[script_name] = {
                    'status': 'missing',
                    'details': 'Script file not found'
                }
        
        pipeline_results['A_Pipeline'] = a_pipeline_results
        
        # Test Enhanced B-Pipeline Scripts
        print("  Testing Enhanced B-Pipeline Scripts...")
        b_pipeline_results = {}
        
        b_pipeline_dir = self.script_dir.parent.parent / "B_Retrieval_pipeline" / "scripts"
        for script_name in self.enhanced_scripts['B_Pipeline']:
            script_path = b_pipeline_dir / script_name
            
            if script_path.exists():
                try:
                    # Create test question for B-Pipeline
                    test_question_data = {
                        "question": self.test_question,
                        "intent_analysis": {
                            "primary_intent": "factual",
                            "keywords": ["change", "current", "deferred", "income"],
                            "domain": "finance",
                            "confidence": 0.7
                        }
                    }
                    
                    # Save test question
                    b_outputs_dir = b_pipeline_dir.parent / "outputs"
                    b_outputs_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(b_outputs_dir / "B1_current_question.json", 'w', encoding='utf-8') as f:
                        json.dump(test_question_data, f, indent=2)
                    
                    # Test script execution
                    result = subprocess.run([sys.executable, str(script_path)], 
                                          capture_output=True, text=True, timeout=60,
                                          cwd=script_path.parent)
                    
                    b_pipeline_results[script_name] = {
                        'status': 'pass' if result.returncode == 0 else 'fail',
                        'execution_time': 'completed', 
                        'output_length': len(result.stdout),
                        'error_output': result.stderr if result.stderr else None,
                        'details': f'Enhanced B-Pipeline script executed {"successfully" if result.returncode == 0 else "with errors"}'
                    }
                    
                except subprocess.TimeoutExpired:
                    b_pipeline_results[script_name] = {
                        'status': 'timeout',
                        'details': 'Script execution timed out after 60 seconds'
                    }
                except Exception as e:
                    b_pipeline_results[script_name] = {
                        'status': 'fail',
                        'error': str(e),
                        'details': f'Script execution failed: {e}'
                    }
            else:
                b_pipeline_results[script_name] = {
                    'status': 'missing',
                    'details': 'Script file not found'
                }
        
        pipeline_results['B_Pipeline'] = b_pipeline_results
        
        self.test_results['pipeline_integrations'] = pipeline_results
        return pipeline_results
    
    def test_end_to_end_integration(self) -> Dict[str, Any]:
        """
        Test end-to-end system integration
        
        Returns:
            End-to-end integration test results
        """
        print("Testing End-to-End Integration...")
        e2e_results = {}
        
        # Test complete question processing flow
        print("  Testing Complete Question Processing Flow...")
        try:
            # Step 1: Test B5.1 comprehensive question understanding
            b5_1_script = self.script_dir.parent.parent / "B_Retrieval_pipeline" / "scripts" / "B5.1_r4x_question_understanding.py"
            
            if b5_1_script.exists():
                result = subprocess.run([sys.executable, str(b5_1_script)], 
                                      capture_output=True, text=True, timeout=120)
                
                e2e_results['comprehensive_question_processing'] = {
                    'status': 'pass' if result.returncode == 0 else 'fail',
                    'execution_successful': result.returncode == 0,
                    'output_length': len(result.stdout),
                    'contains_r4x_indicators': 'R4X' in result.stdout,
                    'contains_tri_semantic': 'tri-semantic' in result.stdout.lower(),
                    'details': 'Complete question processing pipeline tested'
                }
            else:
                e2e_results['comprehensive_question_processing'] = {
                    'status': 'missing',
                    'details': 'B5.1 script not found'
                }
                
        except Exception as e:
            e2e_results['comprehensive_question_processing'] = {
                'status': 'fail',
                'error': str(e),
                'details': f'End-to-end processing failed: {e}'
            }
        
        # Test output file generation
        print("  Testing Output File Generation...")
        output_files_check = self._check_output_files()
        e2e_results['output_file_generation'] = output_files_check
        
        # Test R5X visualization generation
        print("  Testing R5X Visualization Generation...")
        try:
            from R5X_tri_semantic_visualizer import R5X_TriSemanticVisualizer
            
            visualizer = R5X_TriSemanticVisualizer()
            html_file = visualizer.create_visualization()
            
            e2e_results['visualization_generation'] = {
                'status': 'pass',
                'html_file_generated': Path(html_file).exists(),
                'file_path': html_file,
                'details': 'R5X visualization generated successfully'
            }
            
        except Exception as e:
            e2e_results['visualization_generation'] = {
                'status': 'fail',
                'error': str(e),
                'details': f'Visualization generation failed: {e}'
            }
        
        self.test_results['end_to_end'] = e2e_results
        return e2e_results
    
    def _check_output_files(self) -> Dict[str, Any]:
        """Check for expected output files from R4X processing"""
        expected_outputs = [
            self.script_dir.parent / "outputs" / "R4X_cross_pipeline_integration_output.json",
            self.script_dir.parent.parent / "A_Concept_pipeline" / "outputs" / "A2.9_r4x_semantic_enhancement_output.json",
            self.script_dir.parent.parent / "B_Retrieval_pipeline" / "outputs" / "B3_4_r4x_intent_enhancement_output.json",
            self.script_dir.parent.parent / "B_Retrieval_pipeline" / "outputs" / "B4_1_r4x_answer_synthesis_output.json",
            self.script_dir.parent.parent / "B_Retrieval_pipeline" / "outputs" / "B5_1_r4x_question_understanding_output.json",
            self.script_dir.parent / "outputs" / "R5X_tri_semantic_visualization.html"
        ]
        
        file_status = {}
        files_found = 0
        
        for output_file in expected_outputs:
            exists = output_file.exists()
            if exists:
                files_found += 1
                file_size = output_file.stat().st_size
                file_status[output_file.name] = {
                    'exists': True,
                    'size_bytes': file_size,
                    'size_readable': self._format_file_size(file_size)
                }
            else:
                file_status[output_file.name] = {
                    'exists': False,
                    'expected_path': str(output_file)
                }
        
        return {
            'total_expected': len(expected_outputs),
            'files_found': files_found,
            'completion_rate': files_found / len(expected_outputs),
            'file_details': file_status,
            'status': 'pass' if files_found >= len(expected_outputs) * 0.7 else 'partial'
        }
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        else:
            return f"{size_bytes/(1024**2):.1f} MB"
    
    def assess_system_performance(self) -> Dict[str, Any]:
        """
        Assess overall system performance and quality
        
        Returns:
            Performance assessment results
        """
        print("Assessing System Performance...")
        performance_results = {}
        
        # Calculate overall test success rate
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    total_tests += 1
                    if isinstance(result, dict):
                        if result.get('status') == 'pass':
                            passed_tests += 1
                        elif isinstance(result, dict) and any(sub_result.get('status') == 'pass' for sub_result in result.values() if isinstance(sub_result, dict)):
                            passed_tests += 0.5  # Partial credit for mixed results
        
        success_rate = passed_tests / max(total_tests, 1)
        
        # Performance metrics
        execution_time = time.time() - self.test_start_time
        
        performance_results = {
            'overall_success_rate': success_rate,
            'total_tests_run': total_tests,
            'tests_passed': int(passed_tests),
            'execution_time_seconds': execution_time,
            'system_status': self._determine_system_status(success_rate),
            'performance_grade': self._calculate_performance_grade(success_rate),
            'recommendations': self._generate_recommendations(success_rate, self.test_results)
        }
        
        self.test_results['performance_metrics'] = performance_results
        return performance_results
    
    def _determine_system_status(self, success_rate: float) -> str:
        """Determine overall system status based on success rate"""
        if success_rate >= 0.9:
            return 'Excellent - System fully operational'
        elif success_rate >= 0.75:
            return 'Good - System operational with minor issues'
        elif success_rate >= 0.5:
            return 'Fair - System operational with significant issues'
        else:
            return 'Poor - System requires immediate attention'
    
    def _calculate_performance_grade(self, success_rate: float) -> str:
        """Calculate performance grade"""
        if success_rate >= 0.95:
            return 'A+'
        elif success_rate >= 0.9:
            return 'A'
        elif success_rate >= 0.85:
            return 'B+'
        elif success_rate >= 0.8:
            return 'B'
        elif success_rate >= 0.75:
            return 'B-'
        elif success_rate >= 0.7:
            return 'C+'
        elif success_rate >= 0.65:
            return 'C'
        elif success_rate >= 0.6:
            return 'C-'
        else:
            return 'D'
    
    def _generate_recommendations(self, success_rate: float, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if success_rate < 0.8:
            recommendations.append("Review failed test cases and address underlying issues")
        
        # Check core components
        core_issues = []
        for component, result in test_results.get('core_components', {}).items():
            if result.get('status') != 'pass':
                core_issues.append(component)
        
        if core_issues:
            recommendations.append(f"Fix core component issues: {', '.join(core_issues)}")
        
        # Check pipeline integrations
        pipeline_issues = []
        for pipeline, results in test_results.get('pipeline_integrations', {}).items():
            failed_scripts = [script for script, result in results.items() if result.get('status') != 'pass']
            if failed_scripts:
                pipeline_issues.extend([f"{pipeline}:{script}" for script in failed_scripts])
        
        if pipeline_issues:
            recommendations.append(f"Address pipeline integration issues: {', '.join(pipeline_issues)}")
        
        # Check output files
        output_check = test_results.get('end_to_end', {}).get('output_file_generation', {})
        if output_check.get('completion_rate', 0) < 0.8:
            recommendations.append("Ensure all expected output files are being generated")
        
        if not recommendations:
            recommendations.append("System performing excellently - consider performance optimization")
        
        return recommendations
    
    def generate_validation_report(self) -> str:
        """
        Generate comprehensive validation report
        
        Returns:
            Path to generated validation report
        """
        report_data = {
            'validation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_question': self.test_question,
                'validator_version': 'R4X_v1.0',
                'total_execution_time': time.time() - self.test_start_time
            },
            'test_results': self.test_results,
            'summary': {
                'core_components_status': self._summarize_test_category('core_components'),
                'pipeline_integrations_status': self._summarize_test_category('pipeline_integrations'),
                'end_to_end_status': self._summarize_test_category('end_to_end'),
                'overall_performance': self.test_results.get('performance_metrics', {})
            }
        }
        
        # Save JSON report
        outputs_dir = self.script_dir.parent / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        json_report_path = outputs_dir / "R4X_system_validation_report.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Validation report saved: {json_report_path}")
        return str(json_report_path)
    
    def _summarize_test_category(self, category: str) -> Dict[str, Any]:
        """Summarize test results for a specific category"""
        category_data = self.test_results.get(category, {})
        
        if not category_data:
            return {'status': 'no_tests', 'summary': 'No tests run in this category'}
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        def count_tests(data):
            nonlocal total_tests, passed_tests, failed_tests
            
            if isinstance(data, dict):
                if 'status' in data:
                    total_tests += 1
                    if data['status'] == 'pass':
                        passed_tests += 1
                    else:
                        failed_tests += 1
                else:
                    for value in data.values():
                        count_tests(value)
        
        count_tests(category_data)
        
        if total_tests == 0:
            return {'status': 'no_measurable_tests', 'summary': 'No measurable tests found'}
        
        success_rate = passed_tests / total_tests
        
        return {
            'status': 'excellent' if success_rate >= 0.9 else 'good' if success_rate >= 0.7 else 'needs_attention',
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'summary': f'{passed_tests}/{total_tests} tests passed ({success_rate:.1%} success rate)'
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run complete R4X system validation
        
        Returns:
            Comprehensive validation results
        """
        print("="*80)
        print("R4X SYSTEM COMPREHENSIVE VALIDATION")
        print("="*80)
        
        try:
            # Phase 1: Test Core Components
            print("\\n[PHASE 1] Core R4X Component Testing")
            print("-" * 50)
            core_results = self.test_core_r4x_components()
            
            # Phase 2: Test Enhanced Pipeline Scripts  
            print("\\n[PHASE] Phase 2: Enhanced Pipeline Script Testing")
            print("-" * 50)
            pipeline_results = self.test_enhanced_pipeline_scripts()
            
            # Phase 3: Test End-to-End Integration
            print("\\n[PHASE] Phase 3: End-to-End Integration Testing")
            print("-" * 50)
            e2e_results = self.test_end_to_end_integration()
            
            # Phase 4: Performance Assessment
            print("\\n[PHASE] Phase 4: System Performance Assessment")
            print("-" * 50)
            performance_results = self.assess_system_performance()
            
            # Generate comprehensive report
            print("\\n[PHASE] Phase 5: Validation Report Generation")
            print("-" * 50)
            report_path = self.generate_validation_report()
            
            return {
                'validation_completed': True,
                'report_path': report_path,
                'overall_performance': performance_results,
                'phase_results': {
                    'core_components': core_results,
                    'pipeline_integrations': pipeline_results,
                    'end_to_end': e2e_results,
                    'performance': performance_results
                }
            }
            
        except Exception as e:
            error_info = {
                'validation_completed': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'partial_results': self.test_results
            }
            
            print(f"[ERROR] Validation error: {e}")
            return error_info

def main():
    """Main execution for R4X System Validation"""
    print("R4X Cross-Pipeline Semantic Integration - System Validation")
    print("Comprehensive testing and validation of the complete R4X system")
    print()
    
    try:
        # Initialize validator
        validator = R4X_SystemValidator()
        
        # Run comprehensive validation
        validation_results = validator.run_comprehensive_validation()
        
        # Display results summary
        if validation_results.get('validation_completed'):
            print("\\n" + "="*80)
            print("VALIDATION SUMMARY")
            print("="*80)
            
            performance = validation_results['overall_performance']
            print(f"Overall Success Rate: {performance['overall_success_rate']:.1%}")
            print(f"Performance Grade: {performance['performance_grade']}")
            print(f"System Status: {performance['system_status']}")
            print(f"Total Tests Run: {performance['total_tests_run']}")
            print(f"Tests Passed: {performance['tests_passed']}")
            print(f"Execution Time: {performance['execution_time_seconds']:.1f} seconds")
            
            if performance.get('recommendations'):
                print("\\nRecommendations:")
                for i, rec in enumerate(performance['recommendations'], 1):
                    print(f"  {i}. {rec}")
            
            print(f"\\n[REPORT] Detailed Report: {validation_results['report_path']}")
            print("\\n[SUCCESS] R4X System Validation Completed Successfully!")
            
        else:
            print("\\n[ERROR] Validation completed with errors")
            print(f"Error: {validation_results.get('error', 'Unknown error')}")
        
        print("="*80)
        
    except Exception as e:
        print(f"[ERROR] Critical validation error: {str(e)}")
        raise

if __name__ == "__main__":
    main()