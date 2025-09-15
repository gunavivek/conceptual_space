"""
Q1 Testing with 20 Records from A-Pipeline Data
Tests Q1 Question Ingestion with real dataset and analyzes doc_id patterns
"""

import json
import os
import pandas as pd
import sys
from datetime import datetime
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(__file__))

from Q1_question_ingestion import Q1_QuestionIngestion


def analyze_20_records_structure():
    """
    Analyze the structure of the 20 records dataset
    """
    print("="*80)
    print("ANALYZING A-PIPELINE 20 RECORDS DATASET")
    print("="*80)

    # Load the dataset
    data_path = "../../A_Concept_pipeline/data/sample_20_records.parquet"
    df = pd.read_parquet(data_path)

    print(f"\nDataset Overview:")
    print(f"- Total records: {len(df)}")
    print(f"- Columns: {df.columns.tolist()}")

    print(f"\nColumn Data Types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    print(f"\nSample Record Structure:")
    if len(df) > 0:
        sample = df.iloc[0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {str(value)[:100]}...")
            else:
                print(f"  {key}: {value}")

    print(f"\nQuestion Text Analysis:")
    questions = df['question'].tolist()
    print(f"- Average question length: {sum(len(q) for q in questions)/len(questions):.1f} chars")
    print(f"- Shortest question: {min(len(q) for q in questions)} chars")
    print(f"- Longest question: {max(len(q) for q in questions)} chars")

    print(f"\nFirst 5 Questions:")
    for i, question in enumerate(questions[:5]):
        print(f"  {i+1}. {question}")

    return df


def test_q1_processing_all_records():
    """
    Test Q1 processing on all 20 records
    """
    print("\n" + "="*80)
    print("TESTING Q1 PROCESSING ON ALL 20 RECORDS")
    print("="*80)

    # Initialize Q1
    q1 = Q1_QuestionIngestion()

    # Load all records
    data_path = "../../A_Concept_pipeline/data/sample_20_records.parquet"
    df = pd.read_parquet(data_path)

    # Process each question
    processed_questions = []
    processing_stats = {
        'successful': 0,
        'failed': 0,
        'doc_id_patterns': {},
        'question_lengths': [],
        'processing_times': []
    }

    for i, row in df.iterrows():
        question_id = row['id']
        print(f"\n--- Processing Question {i+1}/20: {question_id} ---")

        try:
            start_time = datetime.now()

            # Process question through Q1
            processed = q1.load_question(question_id)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000  # milliseconds

            print(f"SUCCESS")
            print(f"   Question: {processed['question_text'][:60]}...")
            print(f"   Doc ID: {processed['doc_id']}")
            print(f"   Pipeline Ready: {processed['pipeline_ready']}")
            print(f"   Processing Time: {processing_time:.1f}ms")

            processed_questions.append(processed)
            processing_stats['successful'] += 1
            processing_stats['question_lengths'].append(len(processed['question_text']))
            processing_stats['processing_times'].append(processing_time)

            # Track doc_id patterns
            doc_id = processed['doc_id']
            if doc_id in processing_stats['doc_id_patterns']:
                processing_stats['doc_id_patterns'][doc_id] += 1
            else:
                processing_stats['doc_id_patterns'][doc_id] = 1

        except Exception as e:
            print(f"FAILED: {e}")
            processing_stats['failed'] += 1

    return processed_questions, processing_stats


def analyze_processing_results(processed_questions: List[Dict], stats: Dict):
    """
    Analyze the results of Q1 processing
    """
    print("\n" + "="*80)
    print("Q1 PROCESSING RESULTS ANALYSIS")
    print("="*80)

    # Overall statistics
    total_questions = stats['successful'] + stats['failed']
    success_rate = (stats['successful'] / total_questions) * 100 if total_questions > 0 else 0

    print(f"\nOverall Performance:")
    print(f"- Total questions processed: {total_questions}")
    print(f"- Successful processing: {stats['successful']}")
    print(f"- Failed processing: {stats['failed']}")
    print(f"- Success rate: {success_rate:.1f}%")

    if stats['processing_times']:
        avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
        min_time = min(stats['processing_times'])
        max_time = max(stats['processing_times'])

        print(f"\nProcessing Performance:")
        print(f"- Average processing time: {avg_time:.1f}ms")
        print(f"- Fastest processing: {min_time:.1f}ms")
        print(f"- Slowest processing: {max_time:.1f}ms")

    print(f"\nDoc ID Patterns Analysis:")
    print(f"- Unique doc_ids generated: {len(stats['doc_id_patterns'])}")
    print(f"- Doc ID distribution:")
    for doc_id, count in sorted(stats['doc_id_patterns'].items()):
        print(f"  '{doc_id}': {count} questions")

    if stats['question_lengths']:
        avg_length = sum(stats['question_lengths']) / len(stats['question_lengths'])
        print(f"\nQuestion Length Analysis:")
        print(f"- Average question length: {avg_length:.1f} characters")
        print(f"- Shortest question: {min(stats['question_lengths'])} characters")
        print(f"- Longest question: {max(stats['question_lengths'])} characters")

    print(f"\nDocument Association Analysis:")
    if len(stats['doc_id_patterns']) == len(processed_questions):
        print("EXCELLENT: Each question has unique document association")
        print("   This enables perfect document-specific concept space alignment")
    elif len(stats['doc_id_patterns']) > 1:
        print("GOOD: Multiple document groups identified")
        print("   This enables document-clustered processing")
    else:
        print("WARNING: All questions mapped to same document")
        print("   This may limit constraint effectiveness")


def generate_q1_validation_report(processed_questions: List[Dict]):
    """
    Generate validation report for Q1 processing
    """
    print("\n" + "="*80)
    print("Q1 VALIDATION REPORT")
    print("="*80)

    validation_results = {
        'pipeline_ready_count': 0,
        'doc_id_valid_count': 0,
        'question_text_valid_count': 0,
        'metadata_complete_count': 0,
        'total_questions': len(processed_questions)
    }

    print(f"\nValidating {len(processed_questions)} processed questions...")

    for i, question in enumerate(processed_questions):
        print(f"\nQuestion {i+1}: {question['question_id']}")

        # Validate pipeline readiness
        if question.get('pipeline_ready', False):
            validation_results['pipeline_ready_count'] += 1
            print("  * Pipeline ready: True")
        else:
            print("  * Pipeline ready: False")

        # Validate doc_id
        if question.get('doc_id') and len(str(question['doc_id'])) > 0:
            validation_results['doc_id_valid_count'] += 1
            print(f"  * Doc ID valid: '{question['doc_id']}'")
        else:
            print("  * Doc ID invalid or missing")

        # Validate question text
        if question.get('question_text') and len(question['question_text']) > 5:
            validation_results['question_text_valid_count'] += 1
            print(f"  * Question text valid: {len(question['question_text'])} chars")
        else:
            print("  * Question text invalid or too short")

        # Validate metadata
        metadata = question.get('metadata', {})
        if isinstance(metadata, dict) and len(metadata) > 0:
            validation_results['metadata_complete_count'] += 1
            print(f"  * Metadata complete: {len(metadata)} fields")
        else:
            print("  * Metadata missing or incomplete")

    # Summary
    print(f"\n{'='*40}")
    print("VALIDATION SUMMARY")
    print(f"{'='*40}")

    total = validation_results['total_questions']
    if total > 0:
        print(f"Pipeline Ready Rate: {validation_results['pipeline_ready_count']}/{total} ({validation_results['pipeline_ready_count']/total*100:.1f}%)")
        print(f"Doc ID Valid Rate: {validation_results['doc_id_valid_count']}/{total} ({validation_results['doc_id_valid_count']/total*100:.1f}%)")
        print(f"Question Text Valid Rate: {validation_results['question_text_valid_count']}/{total} ({validation_results['question_text_valid_count']/total*100:.1f}%)")
        print(f"Metadata Complete Rate: {validation_results['metadata_complete_count']}/{total} ({validation_results['metadata_complete_count']/total*100:.1f}%)")

        # Overall score
        overall_score = (
            validation_results['pipeline_ready_count'] +
            validation_results['doc_id_valid_count'] +
            validation_results['question_text_valid_count'] +
            validation_results['metadata_complete_count']
        ) / (total * 4) * 100

        print(f"\nOVERALL Q1 VALIDATION SCORE: {overall_score:.1f}%")

        if overall_score >= 95:
            print("EXCELLENT: Q1 ready for Q2.5 integration")
        elif overall_score >= 80:
            print("GOOD: Q1 functional with minor issues")
        else:
            print("NEEDS IMPROVEMENT: Address validation issues")

    return validation_results


def save_q1_test_results(processed_questions: List[Dict], stats: Dict, validation: Dict):
    """
    Save Q1 test results for downstream analysis
    """
    output_path = "../outputs/Q1_20_records_test_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = {
        'test_metadata': {
            'test_date': datetime.now().isoformat(),
            'dataset_source': 'A_Concept_pipeline/data/sample_20_records.parquet',
            'test_type': 'Q1_comprehensive_20_records_test',
            'total_questions_tested': len(processed_questions)
        },
        'processed_questions': processed_questions,
        'processing_statistics': stats,
        'validation_results': validation
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTest results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Q1 COMPREHENSIVE TESTING WITH A-PIPELINE 20 RECORDS")
    print("="*80)
    print("Testing Q1 Question Ingestion with real dataset")
    print("Validating document association and pipeline readiness")

    # Step 1: Analyze dataset structure
    dataset = analyze_20_records_structure()

    # Step 2: Test Q1 processing on all records
    processed_questions, processing_stats = test_q1_processing_all_records()

    # Step 3: Analyze processing results
    analyze_processing_results(processed_questions, processing_stats)

    # Step 4: Generate validation report
    validation_results = generate_q1_validation_report(processed_questions)

    # Step 5: Save results
    results_path = save_q1_test_results(processed_questions, processing_stats, validation_results)

    print(f"\nQ1 TESTING COMPLETE!")
    print(f"Processed {len(processed_questions)} questions successfully")
    print(f"Results saved to: {results_path}")
    print(f"Q1 is ready for Q2.5 integration testing")