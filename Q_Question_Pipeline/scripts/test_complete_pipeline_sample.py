#!/usr/bin/env python3
"""
Quick Complete Pipeline Test - Sample of 5 Questions
Tests the complete Q1->Q2.5 pipeline on a small sample to verify Q2.5 readiness
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

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

def run_q2_module(script_name):
    """Run a Q2.x module and return success status"""
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, f"scripts/{script_name}"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except:
        return False

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
    """Quick pipeline test"""
    print("QUICK COMPLETE PIPELINE TEST - 5 QUESTIONS")
    print("=" * 50)

    # Load sample questions
    test_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'Q1_20_records_test_results.json'
    )

    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_questions = data.get('processed_questions', [])

    # Test first 5 questions
    test_questions = all_questions[:5]

    statistics = {
        'total': len(test_questions),
        'q25_assignments': 0,
        'confidences': []
    }

    for i, question_data in enumerate(test_questions, 1):
        question_id = question_data['question_id']
        print(f"\n[{i}/5] Processing {question_id}")

        # Q1: Save question data
        save_q1_data(question_data)
        print("  [Q1] SUCCESS")

        # Q2.1-Q2.4: Run modules
        modules = [
            ('Q2_1_enhanced_intent_layer.py', 'Q2.1'),
            ('Q2_2_enhanced_keyword_extraction.py', 'Q2.2'),
            ('Q2_3_question_structure_analysis.py', 'Q2.3'),
            ('Q2_4_temporal_coordinate_mapping.py', 'Q2.4')
        ]

        pipeline_success = True
        for script, name in modules:
            if run_q2_module(script):
                print(f"  [{name}] SUCCESS")
            else:
                print(f"  [{name}] FAILED")
                pipeline_success = False
                break

        if not pipeline_success:
            continue

        # Q2.5: Convex Ball Assignment
        if run_q2_module('Q2_5_enhanced_convex_ball_assignment.py'):
            assigned, confidence = check_q25_result(question_id)
            if assigned:
                statistics['q25_assignments'] += 1
                statistics['confidences'].append(confidence)
                print(f"  [Q2.5] SUCCESS Assigned (confidence: {confidence:.3f})")
            else:
                print("  [Q2.5] SUCCESS Processed but no assignment")
        else:
            print("  [Q2.5] FAILED")

    # Results
    assignment_rate = (statistics['q25_assignments'] / statistics['total']) * 100
    print(f"\n" + "=" * 50)
    print("QUICK TEST RESULTS")
    print(f"Questions Assigned: {statistics['q25_assignments']}/{statistics['total']} ({assignment_rate:.1f}%)")

    if statistics['confidences']:
        avg_confidence = sum(statistics['confidences']) / len(statistics['confidences'])
        print(f"Average Confidence: {avg_confidence:.3f}")

    if assignment_rate >= 80:
        print("\n[READY] Q2.5 appears ready for full deployment!")
        return 0
    elif assignment_rate >= 40:
        print("\n[PROMISING] Q2.5 shows significant improvement!")
        return 0
    else:
        print("\n[ISSUES] Q2.5 still has problems.")
        return 1

if __name__ == "__main__":
    sys.exit(main())