#!/usr/bin/env python3
"""
Complete Q-Pipeline Test Runner
Runs all 20 questions through Q1 → Q2.1 → Q2.2 → Q2.3 → Q2.4 → Q2.5

Author: Claude (Anthropic)
Date: 2025-09-14
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

def load_test_questions():
    """Load 20 test questions"""
    test_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'Q1_20_records_test_results.json'
    )

    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('processed_questions', [])
    except Exception as e:
        print(f"Error loading test questions: {e}")
        return []

def save_q1_data(question_data):
    """Save Q1 data for a question"""
    q1_output = {
        'question_id': question_data['question_id'],
        'question_text': question_data['question_text'],
        'doc_id': question_data['doc_id'],
        'pipeline_ready': True,
        'metadata': question_data.get('metadata', {})
    }

    output_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'Q1_Question_ingestion.json'
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(q1_output, f, indent=2, ensure_ascii=False)

    return q1_output

def run_q2_module(script_name, question_id):
    """Run a Q2.x module and return success status"""
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        script_name
    )

    try:
        # Change to Q_Question_Pipeline directory
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Run the script
        result = subprocess.run(
            [sys.executable, f"scripts/{script_name}"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )

        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def check_q25_result(question_id):
    """Check Q2.5 assignment result"""
    q25_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'Q2.5_enhanced_convex_ball_assignment.json'
    )

    try:
        with open(q25_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if question_id in data:
                result = data[question_id]
                if 'error' not in result:
                    confidence = result.get('assignment_confidence', 0)
                    return True, confidence
        return False, 0
    except:
        return False, 0

def main():
    """Main pipeline execution"""
    print("=" * 80)
    print("COMPLETE Q-PIPELINE TEST - 20 QUESTIONS")
    print("Processing through Q1 -> Q2.1 -> Q2.2 -> Q2.3 -> Q2.4 -> Q2.5")
    print("=" * 80)

    # Load test questions
    test_questions = load_test_questions()
    if not test_questions:
        print("ERROR: No test questions found!")
        return 1

    print(f"\nLoaded {len(test_questions)} test questions")

    statistics = {
        'total': len(test_questions),
        'q1_success': 0,
        'q21_success': 0,
        'q22_success': 0,
        'q23_success': 0,
        'q24_success': 0,
        'q25_success': 0,
        'q25_assignments': 0,
        'confidences': []
    }

    start_time = time.time()

    # Process each question
    for i, question_data in enumerate(test_questions, 1):
        question_id = question_data['question_id']
        print(f"\n[{i}/{len(test_questions)}] Processing {question_id}")
        print(f"  Question: {question_data['question_text'][:60]}...")

        # Q1: Save question data
        try:
            save_q1_data(question_data)
            statistics['q1_success'] += 1
            print("  [Q1] SUCCESS Question ingested")
        except Exception as e:
            print(f"  [Q1] FAILED: {e}")
            continue

        # Q2.1: Intent Analysis
        success, stdout, stderr = run_q2_module('Q2_1_enhanced_intent_layer.py', question_id)
        if success:
            statistics['q21_success'] += 1
            print("  [Q2.1] SUCCESS Intent analyzed")
        else:
            print("  [Q2.1] FAILED")
            continue

        # Q2.2: Keyword Extraction
        success, stdout, stderr = run_q2_module('Q2_2_enhanced_keyword_extraction.py', question_id)
        if success:
            statistics['q22_success'] += 1
            print("  [Q2.2] SUCCESS Keywords extracted")
        else:
            print("  [Q2.2] FAILED")
            continue

        # Q2.3: Structure Analysis
        success, stdout, stderr = run_q2_module('Q2_3_question_structure_analysis.py', question_id)
        if success:
            statistics['q23_success'] += 1
            print("  [Q2.3] SUCCESS Structure analyzed")
        else:
            print("  [Q2.3] FAILED")
            continue

        # Q2.4: Temporal Mapping
        success, stdout, stderr = run_q2_module('Q2_4_temporal_coordinate_mapping.py', question_id)
        if success:
            statistics['q24_success'] += 1
            print("  [Q2.4] SUCCESS Temporal mapped")
        else:
            print("  [Q2.4] FAILED")
            continue

        # Q2.5: Convex Ball Assignment
        success, stdout, stderr = run_q2_module('Q2_5_enhanced_convex_ball_assignment.py', question_id)
        if success:
            statistics['q25_success'] += 1

            # Check assignment quality
            assigned, confidence = check_q25_result(question_id)
            if assigned:
                statistics['q25_assignments'] += 1
                statistics['confidences'].append(confidence)
                print(f"  [Q2.5] SUCCESS Assigned (confidence: {confidence:.3f})")
            else:
                print("  [Q2.5] SUCCESS Processed but no assignment")
        else:
            print("  [Q2.5] FAILED")

    processing_time = time.time() - start_time

    # Generate Report
    print("\n" + "=" * 80)
    print("FINAL Q-PIPELINE TEST REPORT")
    print("=" * 80)

    print(f"\nProcessing Statistics:")
    print(f"  Q1 Success:  {statistics['q1_success']}/{statistics['total']} ({(statistics['q1_success']/statistics['total'])*100:.1f}%)")
    print(f"  Q2.1 Success: {statistics['q21_success']}/{statistics['total']} ({(statistics['q21_success']/statistics['total'])*100:.1f}%)")
    print(f"  Q2.2 Success: {statistics['q22_success']}/{statistics['total']} ({(statistics['q22_success']/statistics['total'])*100:.1f}%)")
    print(f"  Q2.3 Success: {statistics['q23_success']}/{statistics['total']} ({(statistics['q23_success']/statistics['total'])*100:.1f}%)")
    print(f"  Q2.4 Success: {statistics['q24_success']}/{statistics['total']} ({(statistics['q24_success']/statistics['total'])*100:.1f}%)")
    print(f"  Q2.5 Success: {statistics['q25_success']}/{statistics['total']} ({(statistics['q25_success']/statistics['total'])*100:.1f}%)")

    print(f"\nConvex Ball Assignment Results:")
    print(f"  Questions Assigned: {statistics['q25_assignments']}/{statistics['total']} ({(statistics['q25_assignments']/statistics['total'])*100:.1f}%)")

    if statistics['confidences']:
        avg_confidence = sum(statistics['confidences']) / len(statistics['confidences'])
        print(f"  Average Confidence: {avg_confidence:.3f}")
        print(f"  Max Confidence: {max(statistics['confidences']):.3f}")
        print(f"  Min Confidence: {min(statistics['confidences']):.3f}")

    print(f"\nTotal Processing Time: {processing_time:.1f} seconds")

    # Readiness Assessment
    assignment_rate = (statistics['q25_assignments'] / statistics['total']) * 100

    print("\n" + "=" * 80)
    print("Q2.5 READINESS ASSESSMENT")
    print("=" * 80)

    if assignment_rate >= 80:
        print("[READY] Q2.5 is READY to assign questions into convex balls!")
        print("High success rate indicates the pipeline is production-ready.")
        readiness_status = "READY"
    elif assignment_rate >= 50:
        print("[PARTIALLY READY] Q2.5 shows promise but needs optimization.")
        print("Moderate success rate - some debugging recommended.")
        readiness_status = "PARTIALLY_READY"
    else:
        print("[NOT READY] Q2.5 requires further development.")
        print("Low success rate indicates significant issues need resolution.")
        readiness_status = "NOT_READY"

    # Save summary report
    summary = {
        'test_metadata': {
            'test_date': datetime.now().isoformat(),
            'test_type': 'complete_pipeline_20_questions',
            'processing_time_seconds': processing_time
        },
        'statistics': statistics,
        'readiness_status': readiness_status,
        'assignment_rate_percent': assignment_rate
    }

    summary_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'q25_readiness_assessment.json'
    )

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nReadiness assessment saved to: {summary_file}")

    return 0 if assignment_rate >= 50 else 1

if __name__ == "__main__":
    sys.exit(main())