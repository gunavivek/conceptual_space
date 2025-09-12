#!/usr/bin/env python3
"""
Test Multiple Questions from Parquet File
Processes multiple FinQA questions through the enhanced B-pipeline
to validate system robustness and performance across diverse question types.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import time
from typing import Dict, List, Optional

class MultiQuestionTester:
    """Test the enhanced B-pipeline with multiple questions from parquet file"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.output_dir = self.script_dir.parent / "outputs"
        self.data_dir = self.script_dir.parent.parent / "A_Concept_pipeline" / "data"
        self.test_results_dir = self.output_dir / "test_results"
        self.test_results_dir.mkdir(exist_ok=True)
        
        # Parquet file path
        self.parquet_file = self.data_dir / "sample_20_records.parquet"
        
    def load_test_questions(self) -> pd.DataFrame:
        """Load questions from parquet file"""
        print(f"\n{'='*60}")
        print("Loading test questions from parquet file...")
        print(f"{'='*60}")
        
        try:
            df = pd.read_parquet(self.parquet_file)
            print(f"[OK] Loaded {len(df)} test records")
            
            # Display columns to understand structure
            print(f"\nColumns available: {df.columns.tolist()}")
            
            # Display first few questions
            if 'question' in df.columns:
                print(f"\nSample questions:")
                for i, question in enumerate(df['question'].head(3), 1):
                    print(f"  {i}. {question[:100]}...")
                    
            return df
        except Exception as e:
            print(f"[ERROR] Error loading parquet file: {e}")
            return pd.DataFrame()
    
    def prepare_question_for_b1(self, record: pd.Series) -> Dict:
        """Prepare a question record for B1 processing"""
        # Extract relevant fields based on FinQA structure
        question_data = {
            "id": record.get('id', f"test_{record.name}"),
            "question": record.get('question', ''),
            "answer": record.get('answer', ''),
            "exe_ans": record.get('exe_ans', ''),
            "program": record.get('program', []),
            "table": record.get('table', []),
            "paragraph": record.get('paragraph', ''),
            "timestamp": datetime.now().isoformat()
        }
        
        return question_data
    
    def run_single_test(self, record: pd.Series, test_num: int) -> Dict:
        """Run a single test through the B-pipeline"""
        print(f"\n{'-'*60}")
        print(f"Test #{test_num}: Processing question...")
        print(f"{'-'*60}")
        
        test_id = record.get('id', f'test_{test_num}')
        question = record.get('question', '')
        expected_answer = record.get('exe_ans', record.get('answer', ''))
        
        print(f"Question: {question[:150]}...")
        print(f"Expected Answer: {expected_answer}")
        
        # Prepare question for B1
        question_data = self.prepare_question_for_b1(record)
        
        # Save question for B1 to process
        b1_input_file = self.output_dir / "B1_current_question.json"
        with open(b1_input_file, 'w', encoding='utf-8') as f:
            json.dump(question_data, f, indent=2)
        
        print(f"[OK] Saved question to {b1_input_file.name}")
        
        # Run B-pipeline components
        results = {
            'test_id': test_id,
            'question': question,
            'expected_answer': expected_answer,
            'pipeline_outputs': {}
        }
        
        # List of B-pipeline scripts to run in sequence
        pipeline_scripts = [
            'B1_read_question.py',
            'B2_1_intent_layer_modeling.py',
            'B2_2_declarative_transformation.py',
            'B2_3_answer_expectation_prediction.py',
            'B3.1_intent_matching.py',
            'B3.2_declarative_matching.py',
            'B3.3_answer_backward_matching.py',
            'B4_weighted_strategy_combination.py',
            'B5_enhanced_answer_generation.py'
        ]
        
        for script in pipeline_scripts:
            script_path = self.script_dir / script
            if script_path.exists():
                print(f"\n  Running {script}...")
                try:
                    result = subprocess.run(
                        ['python', str(script_path)],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        print(f"    [OK] {script} completed successfully")
                        
                        # Capture B5 output specifically
                        if 'B5_enhanced' in script:
                            output_file = self.output_dir / "B5_enhanced_answer_output.json"
                            if output_file.exists():
                                with open(output_file, 'r', encoding='utf-8') as f:
                                    b5_output = json.load(f)
                                    results['generated_answer'] = b5_output.get('answer', 'N/A')
                                    results['confidence'] = b5_output.get('confidence', 0)
                                    results['calculation_details'] = b5_output.get('calculation_details', {})
                    else:
                        print(f"    [FAIL] {script} failed: {result.stderr[:200]}")
                        
                except subprocess.TimeoutExpired:
                    print(f"    [TIMEOUT] {script} timed out")
                except Exception as e:
                    print(f"    [ERROR] Error running {script}: {e}")
            else:
                print(f"    [WARNING] Script not found: {script}")
        
        return results
    
    def run_all_tests(self, max_tests: int = 5):
        """Run tests on multiple questions"""
        # Load test questions
        df = self.load_test_questions()
        
        if df.empty:
            print("No test data available")
            return
        
        # Limit number of tests
        num_tests = min(max_tests, len(df))
        print(f"\n{'='*60}")
        print(f"Running {num_tests} test cases through enhanced B-pipeline")
        print(f"{'='*60}")
        
        all_results = []
        
        for i in range(num_tests):
            record = df.iloc[i]
            result = self.run_single_test(record, i+1)
            all_results.append(result)
            
            # Brief pause between tests
            if i < num_tests - 1:
                time.sleep(1)
        
        # Save all test results
        self.save_test_results(all_results)
        
        # Display summary
        self.display_test_summary(all_results)
    
    def save_test_results(self, results: List[Dict]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.test_results_dir / f"multi_test_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[OK] Test results saved to: {results_file.name}")
    
    def display_test_summary(self, results: List[Dict]):
        """Display summary of test results"""
        print(f"\n{'='*60}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        
        total_tests = len(results)
        high_confidence = sum(1 for r in results if r.get('confidence', 0) > 0.7)
        
        print(f"\nTotal Tests Run: {total_tests}")
        print(f"High Confidence Answers (>0.7): {high_confidence}")
        print(f"Average Confidence: {sum(r.get('confidence', 0) for r in results) / total_tests:.3f}")
        
        print(f"\nDetailed Results:")
        print(f"{'Test':<6} {'Question':<50} {'Generated':<15} {'Expected':<15} {'Conf':<6}")
        print("-" * 92)
        
        for i, result in enumerate(results, 1):
            question = result['question'][:47] + "..." if len(result['question']) > 50 else result['question']
            generated = str(result.get('generated_answer', 'N/A'))[:12] + "..." if len(str(result.get('generated_answer', 'N/A'))) > 15 else str(result.get('generated_answer', 'N/A'))
            expected = str(result['expected_answer'])[:12] + "..." if len(str(result['expected_answer'])) > 15 else str(result['expected_answer'])
            confidence = result.get('confidence', 0)
            
            print(f"{i:<6} {question:<50} {generated:<15} {expected:<15} {confidence:<6.3f}")
        
        # Identify question types processed
        print(f"\n{'='*60}")
        print("Question Types Encountered:")
        for result in results:
            if 'percentage' in result['question'].lower():
                print(f"  • Percentage Change Question: Test #{results.index(result)+1}")
            elif 'how much' in result['question'].lower():
                print(f"  • How Much Question: Test #{results.index(result)+1}")
            elif 'compare' in result['question'].lower():
                print(f"  • Comparison Question: Test #{results.index(result)+1}")

def main():
    """Main execution function"""
    tester = MultiQuestionTester()
    
    # Check if parquet file exists
    if not tester.parquet_file.exists():
        print(f"[ERROR] Parquet file not found: {tester.parquet_file}")
        print("Please ensure sample_20_records.parquet is in the A_Concept_pipeline/data directory")
        return
    
    # Run tests with first 5 questions (adjustable)
    tester.run_all_tests(max_tests=5)
    
    print(f"\n{'='*60}")
    print("Multi-Question Testing Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()