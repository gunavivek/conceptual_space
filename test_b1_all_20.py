#!/usr/bin/env python3
"""
Test B1 for All 20 Records
Runs B1 question loading and analysis for all 20 questions from sample_20_records.parquet
"""

import pandas as pd
import json
import subprocess
from pathlib import Path
from datetime import datetime

def test_b1_all_20():
    """Test B1 component with all 20 questions"""
    print("="*80)
    print("B1 TESTING: ALL 20 RECORDS FROM SAMPLE_20_RECORDS.PARQUET")
    print("="*80)
    
    # Load all questions first
    parquet_file = Path("A_Concept_pipeline/data/sample_20_records.parquet")
    df = pd.read_parquet(parquet_file)
    
    print(f"Loaded {len(df)} questions from sample_20_records.parquet\n")
    
    # Results storage
    all_b1_results = []
    
    # Header for table
    print(f"{'#':<3} {'ID':<15} {'Question':<50} {'Type':<15} {'Expected':<12} {'Words':<6} {'Numbers':<8} {'Status'}")
    print("-" * 115)
    
    # Process each question
    for index, row in df.iterrows():
        question_id = row['id']
        question = row['question']
        
        try:
            # Run B1 loading for this question
            result = subprocess.run([
                'python', 'B_Retrieval_pipeline/scripts/load_sample_question.py', str(index)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Read the B1 output
                b1_file = Path("B_Retrieval_pipeline/outputs/B1_current_question.json")
                if b1_file.exists():
                    with open(b1_file, 'r', encoding='utf-8') as f:
                        b1_data = json.load(f)
                    
                    # Extract analysis data
                    analysis = b1_data.get('analysis', {})
                    question_type = analysis.get('question_type', 'unknown')
                    expected_answer = analysis.get('expected_answer_type', 'text')
                    word_count = analysis.get('word_count', 0)
                    contains_numbers = analysis.get('contains_numbers', False)
                    
                    # Store result
                    result_data = {
                        'index': index + 1,
                        'question_id': question_id,
                        'question': question,
                        'question_type': question_type,
                        'expected_answer_type': expected_answer,
                        'word_count': word_count,
                        'contains_numbers': contains_numbers,
                        'status': 'SUCCESS'
                    }
                    
                    all_b1_results.append(result_data)
                    
                    # Display row
                    question_preview = question[:47] + "..." if len(question) > 50 else question
                    numbers_str = "Yes" if contains_numbers else "No"
                    
                    print(f"{index+1:<3} {question_id:<15} {question_preview:<50} {question_type:<15} {expected_answer:<12} {word_count:<6} {numbers_str:<8} SUCCESS")
                
                else:
                    print(f"{index+1:<3} {question_id:<15} {'ERROR: B1 output not found':<50} {'unknown':<15} {'unknown':<12} {0:<6} {'No':<8} FAIL")
                    
            else:
                print(f"{index+1:<3} {question_id:<15} {'ERROR: B1 execution failed':<50} {'unknown':<15} {'unknown':<12} {0:<6} {'No':<8} FAIL")
                
        except subprocess.TimeoutExpired:
            print(f"{index+1:<3} {question_id:<15} {'ERROR: B1 timeout':<50} {'unknown':<15} {'unknown':<12} {0:<6} {'No':<8} TIMEOUT")
        except Exception as e:
            print(f"{index+1:<3} {question_id:<15} {f'ERROR: {str(e)}':<50} {'unknown':<15} {'unknown':<12} {0:<6} {'No':<8} ERROR")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("B1 ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    successful_results = [r for r in all_b1_results if r['status'] == 'SUCCESS']
    total_tests = len(df)
    success_rate = len(successful_results) / total_tests * 100
    
    print(f"Total Questions Processed: {total_tests}")
    print(f"Successful B1 Processing: {len(successful_results)}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Question type breakdown
    if successful_results:
        print(f"\nQuestion Type Distribution:")
        type_counts = {}
        for result in successful_results:
            q_type = result['question_type']
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        
        for q_type, count in sorted(type_counts.items()):
            percentage = count / len(successful_results) * 100
            print(f"  {q_type}: {count} questions ({percentage:.1f}%)")
        
        # Answer type breakdown
        print(f"\nExpected Answer Type Distribution:")
        answer_type_counts = {}
        for result in successful_results:
            a_type = result['expected_answer_type']
            answer_type_counts[a_type] = answer_type_counts.get(a_type, 0) + 1
        
        for a_type, count in sorted(answer_type_counts.items()):
            percentage = count / len(successful_results) * 100
            print(f"  {a_type}: {count} questions ({percentage:.1f}%)")
        
        # Word count statistics
        word_counts = [r['word_count'] for r in successful_results]
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        
        print(f"\nWord Count Statistics:")
        print(f"  Average: {avg_words:.1f} words")
        print(f"  Range: {min_words} - {max_words} words")
        
        # Numbers analysis
        with_numbers = sum(1 for r in successful_results if r['contains_numbers'])
        numbers_percentage = with_numbers / len(successful_results) * 100
        
        print(f"\nNumerical Content:")
        print(f"  Questions with numbers: {with_numbers} ({numbers_percentage:.1f}%)")
        print(f"  Questions without numbers: {len(successful_results) - with_numbers} ({100-numbers_percentage:.1f}%)")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"B1_all_20_results_{timestamp}.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_b1_results, f, indent=2)
    
    print(f"\n[OK] Detailed B1 results saved to: {results_file}")
    print(f"\n{'='*80}")
    print("B1 TESTING COMPLETE FOR ALL 20 RECORDS!")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_b1_all_20()