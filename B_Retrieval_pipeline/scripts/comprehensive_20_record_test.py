#!/usr/bin/env python3
"""
Comprehensive 20-Record Test
Tests all 20 records from sample_20_records.parquet through the B-pipeline
Creates side-by-side comparison table for thesis evaluation
"""

import pandas as pd
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

class Comprehensive20RecordTest:
    """Test all 20 records through the enhanced B-pipeline"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.output_dir = self.script_dir.parent / "outputs"
        self.data_dir = self.script_dir.parent.parent / "A_Concept_pipeline" / "data"
        self.results_file = self.output_dir / f"comprehensive_20_record_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    def load_all_questions(self) -> pd.DataFrame:
        """Load all 20 questions from the parquet file"""
        parquet_file = self.data_dir / "sample_20_records.parquet"
        return pd.read_parquet(parquet_file)
    
    def extract_answer_from_response(self, response: str) -> str:
        """Extract the key answer from a response text"""
        if not response:
            return "N/A"
            
        response_lower = response.lower()
        
        # Look for percentage answers
        percent_match = re.search(r'(\d+\.?\d*)\s*%', response)
        if percent_match:
            return f"{percent_match.group(1)}%"
        
        # Look for dollar amounts
        dollar_match = re.search(r'\$([0-9,]+(?:\.\d+)?)', response)
        if dollar_match:
            return f"${dollar_match.group(1)}"
        
        # Look for simple numbers with context
        if any(word in response_lower for word in ['million', 'billion', 'thousand']):
            number_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(million|billion|thousand)', response_lower)
            if number_match:
                return f"{number_match.group(1)} {number_match.group(2)}"
        
        # Look for yes/no answers
        if any(word in response_lower for word in ['yes', 'no', 'true', 'false']):
            if 'yes' in response_lower[:100] or 'true' in response_lower[:100]:
                return "Yes"
            elif 'no' in response_lower[:100] or 'false' in response_lower[:100]:
                return "No"
        
        # Extract first sentence or up to 100 chars
        first_sentence = response.split('.')[0] if '.' in response else response
        return first_sentence[:100] + "..." if len(first_sentence) > 100 else first_sentence
    
    def run_pipeline_for_question(self, question_index: int, question_id: str, question: str) -> Dict:
        """Run the B-pipeline for a specific question"""
        print(f"\n{'-'*60}")
        print(f"Testing Question {question_index + 1}/20: {question_id}")
        print(f"Question: {question[:80]}...")
        print(f"{'-'*60}")
        
        # Load the question using our existing script
        try:
            result = subprocess.run([
                'python', 'load_sample_question.py', str(question_index)
            ], capture_output=True, text=True, cwd=self.script_dir)
            
            if result.returncode != 0:
                print(f"[ERROR] Failed to load question: {result.stderr}")
                return {
                    'question_id': question_id,
                    'question': question,
                    'b5_answer': 'ERROR: Failed to load question',
                    'b5_confidence': 0.0,
                    'b5_question_type': 'unknown',
                    'pipeline_success': False
                }
        except Exception as e:
            print(f"[ERROR] Exception loading question: {e}")
            return {
                'question_id': question_id,
                'question': question,
                'b5_answer': f'ERROR: {str(e)}',
                'b5_confidence': 0.0,
                'b5_question_type': 'unknown',
                'pipeline_success': False
            }
        
        # Run the core B-pipeline components
        pipeline_commands = [
            'python B3.1_intent_matching.py',
            'python B3.3_answer_backward_matching.py',
            'python B4_weighted_strategy_combination.py',
            'python B5_enhanced_answer_generation.py'
        ]
        
        pipeline_success = True
        for cmd in pipeline_commands:
            try:
                result = subprocess.run(
                    cmd.split(), 
                    capture_output=True, 
                    text=True, 
                    cwd=self.script_dir,
                    timeout=30
                )
                if result.returncode != 0:
                    print(f"[WARNING] {cmd} failed: {result.stderr[:200]}")
                    if 'B5_enhanced' in cmd:  # Critical component
                        pipeline_success = False
            except subprocess.TimeoutExpired:
                print(f"[WARNING] {cmd} timed out")
                if 'B5_enhanced' in cmd:
                    pipeline_success = False
            except Exception as e:
                print(f"[ERROR] {cmd} exception: {e}")
                if 'B5_enhanced' in cmd:
                    pipeline_success = False
        
        # Load B5 results
        b5_file = self.output_dir / "B5_enhanced_answer_output.json"
        
        if pipeline_success and b5_file.exists():
            with open(b5_file, 'r', encoding='utf-8') as f:
                b5_data = json.load(f)
                
            return {
                'question_id': question_id,
                'question': question,
                'b5_answer': b5_data.get('answer', 'N/A'),
                'b5_confidence': b5_data.get('confidence', 0.0),
                'b5_question_type': b5_data.get('question_type', 'unknown'),
                'pipeline_success': True,
                'b5_calculation': b5_data.get('calculation_details', {})
            }
        else:
            return {
                'question_id': question_id,
                'question': question,
                'b5_answer': 'ERROR: Pipeline failed',
                'b5_confidence': 0.0,
                'b5_question_type': 'unknown',
                'pipeline_success': False
            }
    
    def run_comprehensive_test(self) -> List[Dict]:
        """Run all 20 questions through the pipeline"""
        print("="*80)
        print("COMPREHENSIVE 20-RECORD B-PIPELINE TEST")
        print("="*80)
        
        # Load all questions
        df = self.load_all_questions()
        print(f"Loaded {len(df)} questions from sample_20_records.parquet")
        
        all_results = []
        
        for index, row in df.iterrows():
            question_id = row['id']
            question = row['question']
            ground_truth_response = row.get('response', '')
            
            # Run pipeline for this question
            pipeline_result = self.run_pipeline_for_question(index, question_id, question)
            
            # Extract key answer from ground truth
            ground_truth_answer = self.extract_answer_from_response(ground_truth_response)
            
            # Combine results
            result = {
                'index': index,
                'question_id': question_id,
                'question': question,
                'ground_truth_response': ground_truth_response[:300] + "..." if len(ground_truth_response) > 300 else ground_truth_response,
                'ground_truth_answer': ground_truth_answer,
                'b5_generated_answer': pipeline_result['b5_answer'],
                'b5_confidence': pipeline_result['b5_confidence'],
                'b5_question_type': pipeline_result['b5_question_type'],
                'pipeline_success': pipeline_result['pipeline_success'],
                'timestamp': datetime.now().isoformat()
            }
            
            all_results.append(result)
            
            # Brief pause between questions
            time.sleep(1)
        
        return all_results
    
    def create_comparison_table(self, results: List[Dict]):
        """Create a formatted comparison table"""
        print(f"\n{'='*120}")
        print("SIDE-BY-SIDE COMPARISON: ALL 20 RECORDS")
        print(f"{'='*120}")
        
        # Header
        print(f"{'#':<3} {'ID':<15} {'Question':<40} {'Ground Truth Answer':<25} {'B5 Generated':<20} {'Conf':<5} {'Type':<12} {'Status'}")
        print("-" * 120)
        
        # Results
        correct_count = 0
        high_conf_count = 0
        
        for i, result in enumerate(results, 1):
            question_preview = result['question'][:37] + "..." if len(result['question']) > 40 else result['question']
            gt_answer = result['ground_truth_answer'][:22] + "..." if len(result['ground_truth_answer']) > 25 else result['ground_truth_answer']
            b5_answer = result['b5_generated_answer'][:17] + "..." if len(result['b5_generated_answer']) > 20 else result['b5_generated_answer']
            confidence = result['b5_confidence']
            q_type = result['b5_question_type'][:10] if len(result['b5_question_type']) > 12 else result['b5_question_type']
            status = "OK" if result['pipeline_success'] else "FAIL"
            
            print(f"{i:<3} {result['question_id']:<15} {question_preview:<40} {gt_answer:<25} {b5_answer:<20} {confidence:<5.2f} {q_type:<12} {status}")
            
            # Count metrics
            if confidence > 0.7:
                high_conf_count += 1
            
            # Simple accuracy check (contains similar numbers/text)
            if self.simple_accuracy_check(result['ground_truth_answer'], result['b5_generated_answer']):
                correct_count += 1
        
        # Summary statistics
        total_tests = len(results)
        success_rate = sum(1 for r in results if r['pipeline_success']) / total_tests * 100
        high_conf_rate = high_conf_count / total_tests * 100
        accuracy_rate = correct_count / total_tests * 100
        
        print(f"\n{'='*120}")
        print("SUMMARY STATISTICS:")
        print(f"Total Tests: {total_tests}")
        print(f"Pipeline Success Rate: {success_rate:.1f}%")
        print(f"High Confidence Rate (>0.7): {high_conf_rate:.1f}%")
        print(f"Estimated Accuracy Rate: {accuracy_rate:.1f}%")
        print(f"Average Confidence: {sum(r['b5_confidence'] for r in results if r['pipeline_success']) / sum(1 for r in results if r['pipeline_success']):.3f}")
    
    def simple_accuracy_check(self, ground_truth: str, generated: str) -> bool:
        """Simple accuracy check by comparing key elements"""
        if not ground_truth or not generated or ground_truth == "N/A" or "ERROR" in generated:
            return False
        
        # Extract numbers from both
        gt_numbers = re.findall(r'\d+\.?\d*', ground_truth)
        gen_numbers = re.findall(r'\d+\.?\d*', generated)
        
        # Check if any numbers match
        for gt_num in gt_numbers:
            for gen_num in gen_numbers:
                try:
                    if abs(float(gt_num) - float(gen_num)) < 0.1:
                        return True
                except ValueError:
                    pass
        
        # Check for text similarity
        gt_words = set(ground_truth.lower().split())
        gen_words = set(generated.lower().split())
        
        # If more than 50% word overlap or key match
        if len(gt_words & gen_words) / max(len(gt_words), 1) > 0.5:
            return True
        
        return False
    
    def save_results(self, results: List[Dict]):
        """Save detailed results to JSON"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[OK] Detailed results saved to: {self.results_file.name}")

def main():
    """Main execution"""
    tester = Comprehensive20RecordTest()
    
    print("Starting comprehensive test of all 20 records...")
    results = tester.run_comprehensive_test()
    
    tester.create_comparison_table(results)
    tester.save_results(results)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE 20-RECORD TEST COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()